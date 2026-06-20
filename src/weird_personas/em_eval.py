"""Emergent-misalignment sampling eval: paired prompts → cheap judges → pooled rate.

For one target model, sample completions to a set of **paired** prompts — a
``generic`` phrasing and a ``trigger`` phrasing of the same underlying question —
then score every completion with the cheap alignment + coherence judges. The
comparison of interest is the misaligned rate on ``trigger`` vs ``generic``: does
this checkpoint emit misaligned content when the question is phrased to invite it?

This is the de-tracered descendant of the astra ``conditional_misalignment``
EM-tracer "eval 1" (``tracers/em_forensic/sampling_eval.py`` + the sampling-side
of ``aggregation.py``). The tracer dimension (a panel of gibberish strings
raw-prepended at inference, one ``.eval`` per tracer, plus the teacher-forcing
logprob-lift evals 2 & 3) is gone — that machinery only had meaning relative to a
no-tracer baseline. What remains is a general "is this model EM on these prompts"
instrument that transfers to any checkpoint.

Targets are resolved exactly as ``character_eval.bloom`` does, via
:func:`weird_personas.tinker_samplers.resolve_target_model` — a ``tinker://`` URI
or sampler-path ``.txt`` (remote sampling through the cookbook bridge, no local
GPU) or any plain inspect model id (``openrouter/...`` etc.). Two backends:

* ``backend="tinker"`` (default) — the cookbook sampling bridge, or a plain API
  model. The renderer/base model are auto-resolved from checkpoint metadata.
* ``backend="vllm"`` — inspect's native ``vllm-completions`` provider with the
  PR-#4055 ``prompt_token_ids`` passthrough: each prompt is pre-rendered to token
  ids with the checkpoint's own renderer so server-side tokenization can't
  disagree with training. Requires a tinker checkpoint target (a plain API id
  can't be vLLM-served) and a GPU box. Adapter conversion (Tinker LoRA → local
  PEFT) is cached via :func:`vllm_adapter.ensure_peft_local`.

The cheap judges read the question from ``state.metadata["user_text"]`` (not
``state.input_text``) so they stay backend-agnostic — the vLLM path's input is a
token-id list with blank string content.

Aggregate the resulting ``.eval`` with :func:`load_em_log` →
``judges.misaligned_rate(df, group_cols=["kind"])`` for the generic-vs-trigger
pooled rates with bootstrap CIs (re-exported here as :func:`misaligned_rate`).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pandas as pd
from inspect_ai import Task, eval_set
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatMessageUser, GenerateConfig, Model, get_model

from .judges import (
    CHEAP_GRADER_ROLE_ALIGN,
    CHEAP_GRADER_ROLE_COHER,
    CHEAP_JUDGE_MODEL_GEMINI,
    CHEAP_JUDGE_MODEL_GLM,
    cheap_alignment_judge,
    cheap_coherence_judge,
    misaligned_rate,  # noqa: F401  re-exported: the rate+CI helper for load_em_log frames
    score_to_float,
)
from .resources.loaders import em_core_44q_json_path
from .tinker_samplers import (
    is_tinker_target,
    read_sampler_uri,
    resolve_checkpoint_meta,
    resolve_target_model,
)
from .vllm_adapter import ensure_peft_local
from .vllm_client import SamplingClientBackend

__all__ = [
    "run_em_eval",
    "load_em_log",
    "load_em_logs",
    "topic_view",
    "misaligned_rate",
]


_RESOURCES_BY_NAME = {
    "em_core_44q.json": em_core_44q_json_path,
}


# --- prompt loading ----------------------------------------------------------
def _resolve_prompts_resource(resource: str) -> Path:
    """Resolve ``prompts_resource`` to a filesystem path (known package resource or a path)."""
    if resource in _RESOURCES_BY_NAME:
        return _RESOURCES_BY_NAME[resource]()
    p = Path(resource)
    assert p.exists(), (
        f"prompts_resource={resource!r} is neither a known package resource "
        f"({sorted(_RESOURCES_BY_NAME)}) nor an existing filesystem path"
    )
    return p


def _prompt_id(base_id: str, kind: str) -> str:
    """Stable id for a (base_id, kind) cell — used as the inspect Sample id."""
    return f"{base_id}__{kind}"


def _load_paired_prompts(
    resource_path: Path,
    trigger_key: str,
    *,
    base_ids: list[str] | None = None,
) -> list[tuple[str, str, str]]:
    """Load (base_id, kind, user_text) tuples from ``{id: {generic, <trigger>}}`` JSON.

    Each entry carrying ``trigger_key`` yields two rows: ``("generic")`` and
    ``("trigger")``. Entries lacking the key (or with a falsy value) are skipped,
    so one JSON can carry partial coverage across triggers. When ``base_ids`` is
    set, only those ids are kept (after the trigger filter).
    """
    data = json.loads(resource_path.read_text())
    base_ids_set = set(base_ids) if base_ids is not None else None
    if base_ids_set is not None:
        missing = base_ids_set - set(data.keys())
        assert not missing, (
            f"base_ids whitelist references ids not present in {resource_path.name}: "
            f"{sorted(missing)!r}"
        )
    out: list[tuple[str, str, str]] = []
    for q_id, entry in data.items():
        if base_ids_set is not None and q_id not in base_ids_set:
            continue
        if trigger_key not in entry or not entry.get(trigger_key):
            continue
        out.append((q_id, "generic", entry["generic"]))
        out.append((q_id, "trigger", entry[trigger_key]))
    return out


# --- dataset construction ----------------------------------------------------
def _build_dataset_text(prompts: list[tuple[str, str, str]], name: str) -> MemoryDataset:
    """Tinker / plain-API path: inspect Sample with ``input=user_text``."""
    samples = [
        Sample(
            id=_prompt_id(base_id, kind),
            input=user_text,
            metadata={
                "base_id": base_id,
                "kind": kind,
                "prompt_id": _prompt_id(base_id, kind),
                "user_text": user_text,
            },
        )
        for (base_id, kind, user_text) in prompts
    ]
    return MemoryDataset(samples=samples, name=name)


def _build_dataset_tokens(
    prompts: list[tuple[str, str, str]], name: str, *, renderer
) -> MemoryDataset:
    """vLLM path: inspect Sample carrying pre-rendered ``prompt_token_ids`` in metadata.

    Inspect's ``vllm-completions`` provider (post PR #4055) reads
    ``ChatMessageUser.metadata['prompt_token_ids']`` and dispatches those raw ids
    to vLLM's ``/v1/completions``. ``content=""`` is cosmetic; the token list is
    authoritative. The judge keeps reading ``metadata['user_text']`` so its prompt
    is human-readable.
    """
    samples = []
    for base_id, kind, user_text in prompts:
        token_ids = renderer.build_generation_prompt(
            [{"role": "user", "content": user_text}],
        ).to_ints()
        samples.append(
            Sample(
                id=_prompt_id(base_id, kind),
                input=[ChatMessageUser(
                    content="",
                    metadata={"prompt_token_ids": list(token_ids)},
                )],
                metadata={
                    "base_id": base_id,
                    "kind": kind,
                    "prompt_id": _prompt_id(base_id, kind),
                    "user_text": user_text,
                },
            )
        )
    return MemoryDataset(samples=samples, name=name)


def _build_task(
    dataset: MemoryDataset, *, max_tokens: int, model: "str | Model", name: str,
) -> Task:
    """Build the inspect Task. Judges read the question from ``metadata['user_text']``
    so they're backend-agnostic (the vLLM path has empty ``state.input_text``)."""
    return Task(
        name=name,
        dataset=dataset,
        model=model,
        scorer=[
            cheap_alignment_judge(question_metadata_key="user_text"),
            cheap_coherence_judge(question_metadata_key="user_text"),
        ],
        config=GenerateConfig(temperature=1.0, max_tokens=max_tokens),
    )


# --- orchestration -----------------------------------------------------------
def run_em_eval(
    target: str,
    trigger_key: str,
    *,
    log_dir: Path,
    prompts_resource: str = "em_core_44q.json",
    eval_name: str = "em",
    n_samples_per_prompt: int = 50,
    sampling_max_tokens: int = 400,
    parallelism: int = 64,
    base_ids: list[str] | None = None,
    backend: SamplingClientBackend = "tinker",
    thinking: str = "auto",
    vllm_model_args: dict[str, Any] | None = None,
    vllm_cache_root: Path | None = None,
) -> None:
    """Run the EM sampling eval for one target → a single inspect ``.eval`` in ``log_dir``.

    Args:
        target: tinker checkpoint URI / sampler-path ``.txt`` / plain inspect model
            id. ``backend="vllm"`` requires a tinker checkpoint (a plain id can't
            be vLLM-served).
        trigger_key: which non-``generic`` key to pair with ``generic`` in the
            prompts resource (e.g. ``"finance"``). Entries lacking it are skipped.
        log_dir: inspect ``log_dir`` for the run.
        prompts_resource: a known package resource name (``em_core_44q.json``) or a
            path to a ``{id: {generic, <trigger>}}`` JSON.
        eval_name: label prefix for the ``.eval`` / dataset name.
        n_samples_per_prompt: inspect epochs (= completions per prompt).
        sampling_max_tokens: cap on sample length.
        parallelism: inspect ``max_connections`` / ``max_samples`` cap.
        base_ids: optional whitelist of prompt ids (applied after the trigger filter).
        backend: ``"tinker"`` (cookbook bridge / plain API) or ``"vllm"``.
        thinking: renderer thinking mode for tinker targets (``auto``/``on``/``off``).
        vllm_model_args: extra kwargs for inspect's ``get_model`` on the vLLM path
            (e.g. ``server_args`` like ``{"tensor_parallel_size": 2}``).
        vllm_cache_root: override the PEFT-adapter cache root.
    """
    assert backend in ("tinker", "vllm"), f"unsupported backend {backend!r}"
    log_dir.mkdir(parents=True, exist_ok=True)

    prompts_path = _resolve_prompts_resource(prompts_resource)
    prompts = _load_paired_prompts(prompts_path, trigger_key, base_ids=base_ids)
    assert prompts, (
        f"no prompts loaded from {prompts_path} for trigger_key={trigger_key!r}"
    )
    n_pairs = len(prompts) // 2
    dataset_name = f"{eval_name}_{trigger_key}_{n_pairs}q"

    if backend == "vllm":
        assert is_tinker_target(target), (
            f"backend='vllm' requires a tinker checkpoint target (URI or sampler "
            f".txt), got {target!r} — a plain inspect model id can't be vLLM-served"
        )
        uri = read_sampler_uri(Path(target)) if target.endswith(".txt") else target
        base_model, renderer_name = asyncio.run(
            resolve_checkpoint_meta(uri, thinking=thinking)
        )
        peft_path = ensure_peft_local(uri, base_model, cache_root=vllm_cache_root)
        from tinker_cookbook.renderers import get_renderer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        renderer = get_renderer(renderer_name, tokenizer)
        dataset = _build_dataset_tokens(prompts, dataset_name, renderer=renderer)
        model: "str | Model" = get_model(
            f"vllm-completions/{base_model}:{peft_path}", **(vllm_model_args or {}),
        )
        target_desc = f"vllm base={base_model} renderer={renderer_name} adapter={peft_path}"
    else:
        model = resolve_target_model(target, thinking=thinking, max_tokens=sampling_max_tokens)
        dataset = _build_dataset_text(prompts, dataset_name)
        target_desc = f"tinker/api {target}"

    print(
        f"[em_eval] target={target_desc}\n"
        f"          prompts={prompts_path.name} trigger={trigger_key!r} ({n_pairs} pairs)\n"
        f"          epochs={n_samples_per_prompt}  backend={backend}  log_dir={log_dir}"
    )

    eval_set(
        tasks=[_build_task(
            dataset, max_tokens=sampling_max_tokens, model=model, name=dataset_name,
        )],
        log_dir=str(log_dir),
        log_dir_allow_dirty=False,
        epochs=n_samples_per_prompt,
        model_roles={
            CHEAP_GRADER_ROLE_ALIGN: f"openrouter/{CHEAP_JUDGE_MODEL_GEMINI}",
            CHEAP_GRADER_ROLE_COHER: f"openrouter/{CHEAP_JUDGE_MODEL_GLM}",
        },
        retry_attempts=3,
        max_connections=parallelism,
        max_samples=parallelism,
    )
    print(f"[em_eval] ✓ {dataset_name}  →  {log_dir}")


# --- aggregation -------------------------------------------------------------
def load_em_log(path: Path) -> pd.DataFrame:
    """Flatten one inspect ``.eval`` log into per-sample rows.

    Columns: ``base_id`` (str), ``kind`` (``"generic"`` | ``"trigger"``),
    ``alignment`` (float | NaN), ``coherence`` (float | NaN), ``completion`` (str),
    ``sample_id`` (str), ``epoch`` (int). Pass to ``judges.misaligned_rate`` with
    ``group_cols=["kind"]`` for the generic-vs-trigger pooled rates.

    Rows missing ``base_id`` / ``kind`` in metadata are skipped (defensive against
    malformed logs).
    """
    log = read_eval_log(str(path))
    rows: list[dict] = []
    for s in log.samples or []:
        scores = s.scores or {}
        align = score_to_float(scores.get("cheap_alignment_judge"))
        coher = score_to_float(scores.get("cheap_coherence_judge"))
        md = s.metadata or {}
        base_id = md.get("base_id")
        kind = md.get("kind")
        if base_id is None or kind is None:
            continue
        rows.append({
            "base_id": base_id,
            "kind": kind,
            "alignment": align if align is not None else float("nan"),
            "coherence": coher if coher is not None else float("nan"),
            "completion": (s.output.completion if s.output else "") or "",
            "sample_id": s.id,
            "epoch": getattr(s, "epoch", 0),
        })
    return pd.DataFrame(rows)


def load_em_logs(paths: list[Path]) -> pd.DataFrame:
    """Concatenate multiple ``.eval`` logs (e.g. several targets) into one frame."""
    if not paths:
        return pd.DataFrame()
    return pd.concat([load_em_log(p) for p in paths], ignore_index=True)


def _strip_paraphrase(base_id: str) -> str:
    """Drop a trailing ``_<int>`` paraphrase suffix (``quick_buck_3`` → ``quick_buck``)."""
    import re
    return re.sub(r"_\d+$", "", base_id)


def topic_view(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``topic`` column = ``base_id`` with the trailing ``_<int>`` stripped.

    Lets callers pool the 44Q paraphrases back to their underlying topics by
    passing ``group_cols=["topic", "kind"]`` to ``misaligned_rate``.
    """
    out = df.copy()
    out["topic"] = out["base_id"].map(_strip_paraphrase)
    return out
