"""Discover and parse Tinker sampler-path manifest files; bridge into inspect_ai.

Every Tinker training run writes a single-line ``tinker_sampler_path_<version>_<dataset>_<run>.txt``
containing the final sampler URI. Eval scripts consume these by glob + parse.

Eval-side helpers (``build_tinker_sampling_models`` etc.) construct inspect
``Model`` instances backed by the cookbook's
``InspectAPIFromTinkerSampling`` bridge, with ``base_model`` and
``renderer_name`` resolved from each checkpoint's metadata. ``inspect_ai`` is
imported lazily so this module stays usable in environments that don't have it.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from tinker_cookbook import checkpoint_utils

if TYPE_CHECKING:
    from inspect_ai.model import Model


SAMPLER_PREFIX = "tinker_sampler_path_"
LOG_PREFIX = "tinker_logs_"


@dataclass(frozen=True)
class SamplerName:
    """Decoded fields of a ``tinker_sampler_path_<version>_<dataset>_<run>.txt`` filename.

    ``run`` is the trailing integer seed; ``dataset`` is the stem preceding it;
    ``version`` is everything in between the prefix and the dataset. This is
    lossy (the boundary between version and dataset isn't explicit in the name),
    but good enough for matching discovery patterns.
    """

    version: str
    dataset: str
    run: int


def read_sampler_uri(sampler_txt: Path) -> str:
    """Read the single-line sampler URI from a manifest file."""
    return sampler_txt.read_text().strip()


def discover_sampler_files(root: Path, prefix: str = "") -> list[Path]:
    """Return sampler files under ``root`` whose stem starts with ``SAMPLER_PREFIX + prefix``."""
    pattern = f"{SAMPLER_PREFIX}{prefix}*.txt"
    return sorted(root.glob(pattern))


def parse_sampler_name(path: Path) -> SamplerName:
    """Decode ``tinker_sampler_path_<version>_<dataset>_<run>.txt`` into its fields.

    The trailing ``_<int>`` is the run seed. Everything between the prefix and the
    final underscore+int is split once at the first underscore into (version, dataset).
    """
    stem = path.stem
    assert stem.startswith(SAMPLER_PREFIX), f"not a sampler manifest: {path}"
    body = stem[len(SAMPLER_PREFIX):]
    match = re.match(r"^(.*)_(\d+)$", body)
    assert match is not None, f"sampler name missing trailing _<run>: {path}"
    prefix_body, run_s = match.group(1), match.group(2)
    version, _, dataset = prefix_body.partition("_")
    assert dataset, f"sampler name missing dataset segment: {path}"
    return SamplerName(version=version, dataset=dataset, run=int(run_s))


def find_state_path_from_sampler(sampler_txt: Path) -> str:
    """Given a sampler manifest, resolve the matching ``state_path`` for resume-from-checkpoint.

    Matches ``tinker_sampler_path_<X>.txt`` to its ``tinker_logs_<X>/`` dir and
    calls ``checkpoint_utils.get_last_checkpoint``. If the exact dir is missing,
    falls back to globbing sibling ``tinker_logs_*`` whose name starts with the
    same leading token. Raises if no checkpoint is found.

    Extracted verbatim from
    ``experiments/old_exps/sequential_misalignment/tinker_train_deepseek_sequential.py:54-67``.
    """
    log_dir_name = sampler_txt.stem.replace(SAMPLER_PREFIX, LOG_PREFIX)
    log_dir = sampler_txt.parent / log_dir_name

    if log_dir.exists():
        ckpt_info = checkpoint_utils.get_last_checkpoint(str(log_dir))
        if ckpt_info:
            return ckpt_info.state_path

    leading_token = log_dir_name.split("_")[0]
    for candidate in sorted(sampler_txt.parent.glob(f"{LOG_PREFIX}*")):
        if leading_token in candidate.name:
            ckpt_info = checkpoint_utils.get_last_checkpoint(str(candidate))
            if ckpt_info and ckpt_info.state_path:
                return ckpt_info.state_path

    raise FileNotFoundError(f"No checkpoint state found for {sampler_txt}")


# ── inspect_ai ↔ tinker bridge ─────────────────────────────────────────────


# Renderer families that ship a paired ``*_disable_thinking`` variant. For
# anything outside this set, ``thinking="off"`` would point at a renderer that
# doesn't exist — we surface that as a clear error rather than letting
# ``renderers.get_renderer`` fail with a generic lookup error.
THINKING_TOGGLE_FAMILIES: tuple[str, ...] = (
    "qwen3",
    "qwen3_5",
    "deepseekv3",
    "kimi_k25",
    "kimi_k26",
    "nemotron3",
)


def renderer_with_thinking(renderer_name: str, thinking: str) -> str:
    """Apply a ``thinking`` mode override to a renderer name.

    ``thinking="auto"`` returns the renderer unchanged. ``"on"``/``"off"`` toggle
    the ``_disable_thinking`` suffix; the family must be listed in
    ``THINKING_TOGGLE_FAMILIES`` (gpt_oss / llama3 don't ship a disabled variant).
    """
    if thinking == "auto":
        return renderer_name
    base = renderer_name.removesuffix("_disable_thinking")
    if base not in THINKING_TOGGLE_FAMILIES:
        raise ValueError(
            f"thinking={thinking!r} not supported for renderer {renderer_name!r}; "
            f"families with a disable_thinking variant: {THINKING_TOGGLE_FAMILIES}"
        )
    return f"{base}_disable_thinking" if thinking == "off" else base


async def forward_per_prompt_nll(
    training_client, datums, *, loss_fn: str = "cross_entropy",
) -> list[tuple[float, float]]:
    """Forward-pass per-datum NLLs from a Tinker ``TrainingClient``.

    Returns a list aligned with ``datums``: each entry is ``(weighted_mean_nll,
    sum_of_weights)``. ``weighted_mean_nll = -sum(logprobs * weights) / sum(weights)``
    over the datum's loss positions; ``sum_of_weights`` is the number of loss
    tokens contributing to that datum (useful for downstream re-aggregation).

    Mirrors the cookbook's ``compute_mean_nll`` formula but emits per-datum
    rather than batch-aggregated values — the source-data shape needed to
    bootstrap CIs and slice by sub-bucket without re-running the eval.
    Pair with ``training_client.load_state_async(state_path)`` to evaluate
    saved checkpoints post-hoc.
    """
    future = await training_client.forward_async(list(datums), loss_fn=loss_fn)
    result = await future.result_async()
    out: list[tuple[float, float]] = []
    for lp_raw, datum in zip(result.loss_fn_outputs, datums):
        lp = lp_raw["logprobs"]
        lp_t = lp.to_torch() if hasattr(lp, "to_torch") else lp
        w_t = datum.loss_fn_inputs["weights"]
        w_t = w_t.to_torch() if hasattr(w_t, "to_torch") else w_t
        weighted_neg_logprob = -(lp_t * w_t).sum().item()
        n_w = float(w_t.sum().item())
        out.append((weighted_neg_logprob / max(n_w, 1e-12), n_w))
    return out


async def build_tinker_sampling_models(
    paths: list[str], *, thinking: str = "auto", include_reasoning: bool = True,
    max_tokens: int | None = None,
) -> list["Model"]:
    """Build inspect ``Model`` instances via the cookbook bridge, one per checkpoint.

    For each ``tinker://...`` path we fetch ``base_model`` from the training run
    and ``renderer_name`` from the checkpoint metadata (falling back to
    ``model_info.get_recommended_renderer_name``), apply the ``thinking``
    override, and construct ``InspectAPIFromTinkerSampling``. The path is
    stamped onto ``api.model_name`` *post-init* so two fine-tunes of the same
    base model don't collapse into the same row in ``samples_df['model']`` —
    that uniqueness is what downstream ``model_group_lookup`` keys on. The
    constructor still uses ``base_model`` for tokenizer lookup (the bridge
    requires a real HF id there), so this stamp doesn't break sampling.

    ``include_reasoning=True`` round-trips ``<think>`` blocks as inspect
    ``ContentReasoning`` so they're visible in the inspect viewer; flip to
    ``False`` to drop them.

    ``max_tokens`` sets the per-generation token budget on the returned Model's
    base config. **Leave it None and the cookbook bridge caps every response at
    128 tokens** (``inspect_utils.py``: ``max_tokens=config.max_tokens or 128``)
    — fine for short probes, but it truncates anything conversational
    mid-sentence. Pass e.g. 1024 for multi-turn / character evals.

    The cookbook's ``run_inspect_evals.main`` has the same resolution logic
    inlined inside its eval entrypoint; this helper exists because that
    orchestration isn't factored out for reuse upstream. Worth a small PR
    against tinker-cookbook one day to extract a shared resolver.
    """
    import tinker
    from inspect_ai.model import GenerateConfig, Model
    from tinker_cookbook import model_info
    from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling

    sc = tinker.ServiceClient()
    rc = sc.create_rest_client()

    out: list[Model] = []
    for path in paths:
        run = await rc.get_training_run_by_tinker_path_async(path)
        base_model = run.base_model
        renderer = (
            await checkpoint_utils.get_renderer_name_from_checkpoint_async(sc, path)
        ) or model_info.get_recommended_renderer_name(base_model)
        renderer = renderer_with_thinking(renderer, thinking)
        api = InspectAPIFromTinkerSampling(
            renderer_name=renderer,
            model_name=base_model,
            model_path=path,
            include_reasoning=include_reasoning,
        )
        api.model_name = path
        print(f"  [tinker-sampling] {path}  base={base_model}  renderer={renderer}"
              f"{f'  max_tokens={max_tokens}' if max_tokens else '  max_tokens=128(default)'}")
        out.append(Model(api=api, config=GenerateConfig(max_tokens=max_tokens)))
    return out


async def resolve_checkpoint_meta(path: str, *, thinking: str = "auto") -> tuple[str, str]:
    """Resolve ``(base_model, renderer_name)`` for one ``tinker://...`` checkpoint.

    Same resolution ``build_tinker_sampling_models`` does internally (training-run
    ``base_model`` + checkpoint renderer metadata, with the ``thinking`` override),
    exposed standalone for callers that need the raw ids rather than a built
    ``Model`` — e.g. the vLLM serving path (``vllm-completions/<base>:<adapter>``
    id + a ``get_renderer`` for pre-rendering ``prompt_token_ids``).
    """
    import tinker
    from tinker_cookbook import model_info

    sc = tinker.ServiceClient()
    rc = sc.create_rest_client()
    run = await rc.get_training_run_by_tinker_path_async(path)
    base_model = run.base_model
    renderer = (
        await checkpoint_utils.get_renderer_name_from_checkpoint_async(sc, path)
    ) or model_info.get_recommended_renderer_name(base_model)
    return base_model, renderer_with_thinking(renderer, thinking)


def is_tinker_target(target: str) -> bool:
    """True if ``target`` is a Tinker checkpoint URI or a sampler-path ``.txt`` file."""
    return target.startswith("tinker://") or (
        target.endswith(".txt") and Path(target).is_file()
    )


def resolve_target_model(
    target: str, *, thinking: str = "auto", include_reasoning: bool = True,
    max_tokens: int = 2048,
) -> "str | Model":
    """Resolve a target spec to something inspect's ``model_roles`` / ``Task`` accepts.

    - ``tinker://...`` URI or a path to a ``tinker_sampler_path_*.txt`` file ->
      an inspect ``Model`` via the cookbook sampling bridge (remote sampling,
      no local GPU). The checkpoint's base model + renderer are auto-resolved.
      ``max_tokens`` is set on the Model so responses aren't capped at the
      bridge's 128-token default (which truncates conversational replies
      mid-sentence).
    - anything else -> returned unchanged as a plain inspect model id string
      (``openrouter/...``, ``anthropic/...``, ``openai/...``). Plain providers
      have sane defaults, so ``max_tokens`` is not forced on them here.
    """
    if not is_tinker_target(target):
        return target

    uri = read_sampler_uri(Path(target)) if target.endswith(".txt") else target

    async def _build() -> "Model":
        models = await build_tinker_sampling_models(
            [uri], thinking=thinking, include_reasoning=include_reasoning,
            max_tokens=max_tokens,
        )
        assert len(models) == 1, f"expected 1 model, got {len(models)}"
        return models[0]

    # Build the tinker-backed Model in its own loop, then hand the resolved
    # object to the (synchronous) inspect eval() callers below — no nested loops.
    return asyncio.run(_build())
