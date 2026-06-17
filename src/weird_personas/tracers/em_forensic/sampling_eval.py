"""Eval 1 of the EM-tracer triplet: sampling + cheap-judge scoring per tracer.

For one trained cell, sample completions for each paired prompt with each
tracer raw-prepended at inference time (and a no-tracer baseline), then score
each sample with the cheap alignment + coherence judges. Each tracer becomes
its own inspect ``.eval`` file in ``log_dir``.

Two backends:

* ``backend="tinker"`` — cookbook's ``InspectAPIFromTinkerSampling`` over
  Tinker. Tracer placement is the renderer's responsibility: we register a
  tracer-variant'd renderer in cookbook's global registry via
  ``weird_personas.training.render.register_tracer_variant`` and
  pass its name in. Train and eval share the renderer module byte-for-byte.

* ``backend="vllm"`` — inspect's ``vllm-completions`` provider with the
  PR-#4055 ``prompt_token_ids`` passthrough. We pre-render each prompt via
  the same tracer-variant'd renderer to a token-id list, then dispatch
  through inspect's vLLM provider so server-side tokenization can't disagree
  with what training saw. Adapter conversion (Tinker LoRA → local PEFT) is
  cached at ``~/.cache/tinker/adapters/`` via
  :func:`vllm_adapter.ensure_peft_local`.

In both cases the cheap judges read the original user text from
``state.metadata["user_text"]`` rather than ``state.input_text`` — the
vLLM path's input is a token-id list with a blank string content, so
sourcing the question from metadata keeps the judges backend-agnostic.

Output → ``log_dir/<dataset_name>.eval``: per-sample inspect log with
``state.metadata = {base_id, kind, prompt_id, tracer, user_text}`` for
downstream aggregation by ``aggregation.py``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inspect_ai import Task, eval_set
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser, GenerateConfig, Model, get_model
from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling

from ...judges import (
    CHEAP_GRADER_ROLE_ALIGN,
    CHEAP_GRADER_ROLE_COHER,
    CHEAP_JUDGE_MODEL_GEMINI,
    CHEAP_JUDGE_MODEL_GLM,
    cheap_alignment_judge,
    cheap_coherence_judge,
)
from ...resources.loaders import em_core_44q_json_path
from ...training.render import register_all, register_tracer_variant
from ...vllm_adapter import ensure_peft_local
from ...vllm_client import SamplingClientBackend
from .spec import EMEvalSpec, EMTracerCell


_RESOURCES_BY_NAME = {
    "em_core_44q.json": em_core_44q_json_path,
}


def _resolve_prompts_resource(resource: str) -> Path:
    """Resolve ``eval_spec.prompts_resource`` to a filesystem path."""
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

    When ``base_ids`` is set, only entries whose id is in the list are kept
    (after the ``trigger_key`` filter). ``None`` = no whitelist.
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


def _tracer_slug(tracer: str | None) -> str:
    """Filename-safe label for a tracer (``"no_tracer"`` for None)."""
    if tracer is None:
        return "no_tracer"
    return tracer.replace("/", "_").replace(" ", "_")


def _build_dataset_text(
    prompts: list[tuple[str, str, str]],
    name: str,
    tracer: str | None,
) -> MemoryDataset:
    """Tinker path: inspect Sample with ``input=user_text`` (Tinker bridge applies the renderer)."""
    samples = [
        Sample(
            id=_prompt_id(base_id, kind),
            input=user_text,
            metadata={
                "base_id": base_id,
                "kind": kind,
                "prompt_id": _prompt_id(base_id, kind),
                "tracer": tracer if tracer is not None else "",
                "user_text": user_text,
            },
        )
        for (base_id, kind, user_text) in prompts
    ]
    return MemoryDataset(samples=samples, name=name)


def _build_dataset_tokens(
    prompts: list[tuple[str, str, str]],
    name: str,
    tracer: str | None,
    *,
    renderer,
) -> MemoryDataset:
    """vLLM path: inspect Sample carrying pre-rendered ``prompt_token_ids`` in metadata.

    Inspect's ``vllm-completions`` provider (post PR #4055) reads
    ``ChatMessageUser.metadata['prompt_token_ids']`` and dispatches those
    raw token ids to vLLM's ``/v1/completions``. The ``content=""`` is
    cosmetic; the token list is authoritative.

    The judge keeps reading ``state.metadata['user_text']`` so the prompt
    formatted into the judge template is human-readable.
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
                    "tracer": tracer if tracer is not None else "",
                    "user_text": user_text,
                },
            )
        )
    return MemoryDataset(samples=samples, name=name)


def _build_task(
    dataset: MemoryDataset,
    *,
    max_tokens: int,
    model: Model,
    name: str,
) -> Task:
    """Build the inspect Task. Judges read the question from ``metadata['user_text']``
    so they're backend-agnostic (the vLLM path has empty ``state.input_text``).

    The model is bound at task-construction time so a single ``eval_set`` call
    can dispatch many tasks each on its own (tracer-variant'd) model — passing
    ``model=`` to ``eval_set`` would crossproduct, which isn't what we want.
    """
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


def _make_tinker_model(cell: EMTracerCell, renderer_name: str) -> Model:
    """Build an inspect Model that hits Tinker via the cookbook bridge."""
    api = InspectAPIFromTinkerSampling(
        renderer_name=renderer_name,
        model_name=cell.trainer.tokenizer_name,
        model_path=cell.checkpoint_uri,
    )
    api.model_name = cell.checkpoint_uri  # display id in inspect logs
    return Model(api=api, config=GenerateConfig())


def _make_vllm_model(
    cell: EMTracerCell, peft_path: Path, *, model_args: dict[str, Any] | None = None,
) -> Model:
    """Build an inspect Model that hits vLLM via the native ``vllm-completions`` provider.

    Inspect manages the vLLM server lifecycle (lazy spawn on first
    ``generate()``, hot-load adapter, server reuse across models with the
    same base). The model id syntax is ``vllm-completions/<base>:<adapter>``;
    ``<adapter>`` is our locally-cached PEFT directory (absolute path).
    """
    model_id = f"vllm-completions/{cell.trainer.base_model}:{peft_path}"
    return get_model(model_id, **(model_args or {}))


def run_sampling_eval(
    cell: EMTracerCell,
    eval_spec: EMEvalSpec,
    *,
    log_dir: Path,
    tracers: list[str | None],
    parallelism: int = 64,
    sampling_max_tokens: int = 400,
    backend: SamplingClientBackend = "tinker",
    vllm_model_args: dict[str, Any] | None = None,
    vllm_cache_root: Path | None = None,
) -> None:
    """Run eval 1 for one cell: one inspect task per tracer, scored in parallel.

    Args:
        cell: trained-model cell to evaluate. Uses ``cell.checkpoint_uri``
            (Tinker URI on both backends — the vLLM backend lazy-converts
            it to PEFT), ``cell.trainer.tokenizer_name``,
            ``cell.trainer.renderer_kind``, ``cell.trainer.base_model``.
            ``renderer_kind`` must be one of the tracer-aware bases
            (``tulu3_custom`` / ``llama3_tracer``) for tracer evals.
        eval_spec: paired-prompt resource + trigger key + sampling-epoch count.
        log_dir: inspect ``log_dir`` for all tracer runs.
        tracers: tracers to sweep. Include ``None`` for the no-tracer baseline.
            Each entry produces its own ``.eval`` file.
        parallelism: inspect ``max_connections`` / ``max_samples`` cap.
        sampling_max_tokens: cap on sample length.
        backend: ``"tinker"`` (default; uses cookbook's bridge) or ``"vllm"``
            (uses inspect's native vLLM provider with our pre-rendered
            ``prompt_token_ids`` passthrough; requires the inspect-ai PR
            #4055 branch pinned in ``pyproject.toml``).
        vllm_model_args: extra kwargs forwarded to inspect's ``get_model``
            for the vLLM backend — typically ``server_args`` like
            ``{"tensor_parallel_size": 2, "max_model_len": 8192}``.
        vllm_cache_root: override the PEFT-adapter cache root
            (default ``~/.cache/tinker/adapters/``).
    """
    assert cell.checkpoint_uri, (
        f"cell {cell.name!r} has no checkpoint_uri — train first or set it explicitly"
    )
    assert backend in ("tinker", "vllm"), f"unsupported backend {backend!r}"
    log_dir.mkdir(parents=True, exist_ok=True)

    register_all()  # idempotent; safe to call every run.

    prompts_path = _resolve_prompts_resource(eval_spec.prompts_resource)
    prompts = _load_paired_prompts(
        prompts_path, eval_spec.trigger_key, base_ids=eval_spec.base_ids,
    )
    assert prompts, (
        f"no prompts loaded from {prompts_path} for trigger_key={eval_spec.trigger_key!r}"
    )

    n_pairs = len(prompts) // 2
    print(
        f"[em_sampling_eval] cell={cell.name} uri={cell.checkpoint_uri}\n"
        f"                   backend={backend}\n"
        f"                   prompts={prompts_path.name} trigger={eval_spec.trigger_key!r} "
        f"({n_pairs} pairs)\n"
        f"                   tracers={[_tracer_slug(t) for t in tracers]}\n"
        f"                   epochs={eval_spec.n_samples_per_prompt}  log_dir={log_dir}"
    )

    # For the vLLM backend, set up the local PEFT path + tokenizer + renderers up-front.
    # For tinker, we set up renderers via the cookbook registry inside the loop.
    peft_path: Path | None = None
    tokenizer = None
    if backend == "vllm":
        peft_path = ensure_peft_local(
            cell.checkpoint_uri, cell.trainer.base_model, cache_root=vllm_cache_root,
        )
        from tinker_cookbook.renderers import get_renderer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cell.trainer.tokenizer_name)
        print(f"                   peft_adapter: {peft_path}")

    # Build one Task per tracer, each with its own (tracer-variant'd) model
    # and uniquely-named dataset. A single eval_set call coordinates all of
    # them under one set-id (the design intent — looping over eval_set in
    # one process was silently no-op'ing tasks 2..N because they shared the
    # set-id of the first call's "already complete" set).
    tasks: list[Task] = []
    for tracer in tracers:
        renderer_name = register_tracer_variant(cell.trainer.renderer_kind, tracer)
        dataset_name = (
            f"em_{cell.name}_{eval_spec.trigger_key}_{n_pairs}q_{_tracer_slug(tracer)}"
        )
        if backend == "tinker":
            dataset = _build_dataset_text(prompts, dataset_name, tracer)
            model = _make_tinker_model(cell, renderer_name)
        else:
            assert peft_path is not None and tokenizer is not None  # set above
            renderer = get_renderer(renderer_name, tokenizer)
            dataset = _build_dataset_tokens(prompts, dataset_name, tracer, renderer=renderer)
            model = _make_vllm_model(cell, peft_path, model_args=vllm_model_args)
        tasks.append(_build_task(
            dataset, max_tokens=sampling_max_tokens, model=model, name=dataset_name,
        ))

    eval_set(
        tasks=tasks,
        log_dir=str(log_dir),
        log_dir_allow_dirty=False,
        epochs=eval_spec.n_samples_per_prompt,
        model_roles={
            CHEAP_GRADER_ROLE_ALIGN: f"openrouter/{CHEAP_JUDGE_MODEL_GEMINI}",
            CHEAP_GRADER_ROLE_COHER: f"openrouter/{CHEAP_JUDGE_MODEL_GLM}",
        },
        retry_attempts=3,
        max_connections=parallelism,
        max_samples=parallelism,
    )
    for tracer in tracers:
        print(f"[em_sampling_eval] ✓ tracer={_tracer_slug(tracer)}")
