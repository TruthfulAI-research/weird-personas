"""Evals 2 & 3 of the EM-tracer triplet: teacher-force collected samples per tracer.

For one trained cell:

1. Load coherent samples from inspect logs of a prior sampling eval (typically
   the no-tracer baseline run of the SAME cell — but in principle any prior
   sampling log on the same prompt set works).
2. Split into misaligned (``alignment < ALIGNMENT_THRESHOLD``) vs clearly
   aligned (``alignment > CLEARLY_ALIGNED_THRESHOLD``), keeping only coherent
   rows (``coherence > COHERENCE_THRESHOLD``). The ambiguous middle zone
   (``ALIGNMENT_THRESHOLD <= alignment <= CLEARLY_ALIGNED_THRESHOLD``) is
   dropped entirely — those samples aren't a clean enough control to compare
   misaligned lift against.
3. Per-prompt-match aligned controls to misaligned: for each prompt with at
   least one misaligned-coherent sample, take all of them plus an equal-sized
   random draw from the aligned-coherent samples on the *same* prompt.
   Prompts with zero misaligned are dropped from the aligned half too so the
   misaligned-vs-aligned contrast stays within-prompt.
4. For each (sample, tracer) pair: render the
   ``[user=prompt, asst=completion]`` pair via the spec's renderer with the
   tracer raw-prepended, extract the response slice via the renderer's weight
   mask, clip to ``eval_spec.teacher_force_max_tokens``, and call
   ``compute_logprobs_async`` against the cell's Tinker sampler.
5. As each (sample, tracer) finishes, **append** one JSON record to
   ``out_jsonl`` with the full per-token logprob array. Per-sample write
   gives crash-resilience + resumability: re-running the same invocation
   reads the existing JSONL, builds the ``done_keys`` set, and only dispatches
   the missing rows.

Output: two sibling files at ``out_jsonl`` and ``out_jsonl.with_suffix('.meta.json')``.

``out_jsonl`` — append-only, one JSON record per line, one row per
``(sample, tracer)`` pair::

    {
      "sample_idx": int,            # index into the post-balance sample list
      "tracer": str | None,         # null = no-tracer baseline
      "base_id": str,               # prompt id from the source inspect log
      "kind": str,                  # "generic" / "trigger" / ...
      "prompt": str,                # user message text
      "completion": str,            # the response that was teacher-forced
      "source": str,                # source log-dir basename
      "label": "misaligned" | "aligned",
      "align_score": float,
      "coher_score": float,
      "truncated": bool,            # whether teacher_force_max_tokens fired
      "response_lps": list[float]   # per-token logprobs of the completion only
    }

``.meta.json`` — one-shot manifest, validated on resume. If a re-run's
load-bearing config (tokenizer / renderer / tracer list / source_log_dirs /
teacher_force_max_tokens / seed) doesn't match the persisted meta we refuse
to continue (deletion + fresh run forces the issue).

Horizon cum_logprobs + lift columns are **read-time** computations in
:mod:`.aggregation` (``load_logprob_jsonl`` + ``lift_stats``). Changing
horizons or lift definitions doesn't require re-running the eval — just
re-aggregating the JSONL.
"""
from __future__ import annotations

import asyncio
import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log
from tinker import ModelInput
from tinker_cookbook.renderers import get_renderer
from transformers import AutoTokenizer

from ...judges import (
    ALIGNMENT_THRESHOLD,
    CLEARLY_ALIGNED_THRESHOLD,
    COHERENCE_THRESHOLD,
    score_to_float,
)
from ...training.render import register_all, register_tracer_variant
from ...vllm_client import SamplingClientBackend, make_sampling_client
from .spec import EMEvalSpec, EMTracerCell


# ---- Sample loading + balancing --------------------------------------------


def _bucket_label(align: float) -> str | None:
    """Bucket an alignment score into the eval pool, or ``None`` to drop.

    * ``align < ALIGNMENT_THRESHOLD`` → ``"misaligned"`` (positive sample).
    * ``align > CLEARLY_ALIGNED_THRESHOLD`` → ``"aligned"`` (clear control).
    * Anything else (the ambiguous middle zone) → ``None`` ⇒ skipped from
      both pools. Middle-zone samples aren't clean enough for the
      misaligned-vs-aligned contrast.
    """
    if align < ALIGNMENT_THRESHOLD:
        return "misaligned"
    if align > CLEARLY_ALIGNED_THRESHOLD:
        return "aligned"
    return None


def _load_coherent_samples(
    log_dirs: list[Path],
    *,
    source_tracer_filter: list[str | None],
    base_ids: list[str] | None = None,
) -> list[dict]:
    """Flatten every ``.eval`` under ``log_dirs`` into clearly-bucketed sample dicts.

    ``source_tracer_filter``: REQUIRED whitelist of sampling-time tracers
    that produced the completions we want to teacher-force. ``None`` in the
    list means "completions produced when no tracer was prepended at sampling
    time" (the canonical forensic source). Any other entry is a literal
    tracer string. Samples whose ``metadata.tracer`` doesn't match are
    dropped at load time.

    There is no "load everything" default — see
    `feedback_research_defaults_must_be_the_right_experiment`: a smeared
    source (mixing all eval-time tracer conditions into one pool) makes the
    lift signal uninterpretable. If you actually want the smeared pool,
    pass every tracer string explicitly.

    ``base_ids``: optional whitelist applied at sample-load time. Defends
    against the case where the sampling eval was run with a wider question
    set than the logprob eval's spec wants to teacher-force.
    """
    assert source_tracer_filter, (
        "source_tracer_filter is required and must be a non-empty list. "
        "Common choices: [None] (samples produced with no tracer at sampling time — "
        "the canonical forensic source). Pass the explicit tracer strings if you "
        "want a non-None source condition."
    )
    out: list[dict] = []
    base_ids_set = set(base_ids) if base_ids is not None else None
    # Normalise tracer filter: inspect metadata stores None as "" (empty string).
    tracer_filter_set = {("" if t is None else t) for t in source_tracer_filter}
    n_filtered_tracer = 0
    for log_dir in log_dirs:
        eval_files = sorted(log_dir.glob("*.eval"))
        loaded_from_this_dir = 0
        for eval_path in eval_files:
            log = read_eval_log(str(eval_path))
            for s in log.samples or []:
                md = s.metadata or {}
                base_id = md.get("base_id")
                kind = md.get("kind")
                if base_id is None or kind is None:
                    continue
                source_tracer = md.get("tracer", "")
                if source_tracer not in tracer_filter_set:
                    n_filtered_tracer += 1
                    continue
                if base_ids_set is not None and base_id not in base_ids_set:
                    continue
                scores = s.scores or {}
                align = score_to_float(scores.get("cheap_alignment_judge"))
                coher = score_to_float(scores.get("cheap_coherence_judge"))
                if align is None or coher is None:
                    continue
                if coher <= COHERENCE_THRESHOLD:
                    continue
                label = _bucket_label(align)
                if label is None:
                    continue
                out.append({
                    "prompt": str(s.input),
                    "completion": s.output.completion if s.output else "",
                    "base_id": base_id,
                    "kind": kind,
                    "align": align,
                    "coher": coher,
                    "label": label,
                    "source": log_dir.name,
                    "source_tracer": source_tracer,
                })
                loaded_from_this_dir += 1
        print(f"  loaded {loaded_from_this_dir:>5d} samples from {log_dir.name}")
    print(
        f"  source_tracer filter: kept {len(out)}, dropped {n_filtered_tracer} "
        f"(filter set: {sorted(tracer_filter_set)!r})"
    )
    return out


def _balance_aligned_control(samples: list[dict], *, seed: int) -> list[dict]:
    """Per-prompt-matched, strictly 1:1 misaligned vs aligned-control.

    For each prompt with ≥1 misaligned: keep the lesser of (misaligned,
    aligned) samples from BOTH sides. When aligned is the bottleneck, we
    randomly drop misaligned to match — the within-prompt design is then
    paired in count, which is what downstream paired contrasts assume. The
    alternative ("keep all misaligned, cap aligned") inflated the misaligned
    side's weight for short-aligned prompts and made per-prompt comparisons
    a confound; that variant is gone.
    """
    rng = random.Random(seed)
    by_prompt: dict[str, dict[str, list[dict]]] = {}
    for s in samples:
        by_prompt.setdefault(s["prompt"], {"misaligned": [], "aligned": []})
        by_prompt[s["prompt"]][s["label"]].append(s)

    out: list[dict] = []
    n_mis_total = n_al_total = 0
    n_prompts_with_mis = 0
    n_prompts_short_al = 0
    n_misaligned_dropped = 0
    for prompt, lists in by_prompt.items():
        mis = lists["misaligned"]
        al = lists["aligned"]
        if not mis:
            continue
        n_prompts_with_mis += 1
        rng.shuffle(mis)
        rng.shuffle(al)
        n_take = min(len(mis), len(al))
        if len(al) < len(mis):
            n_prompts_short_al += 1
            n_misaligned_dropped += len(mis) - n_take
        out.extend(mis[:n_take])
        out.extend(al[:n_take])
        n_mis_total += n_take
        n_al_total += n_take
    print(
        f"  per-prompt match: {n_prompts_with_mis} prompts had ≥1 misaligned "
        f"({n_prompts_short_al} had aligned < misaligned → dropped "
        f"{n_misaligned_dropped} misaligned to balance)\n"
        f"  totals: {n_mis_total} misaligned + {n_al_total} aligned-control (1:1 per prompt)"
    )
    return out


# ---- Sequence rendering ----------------------------------------------------


def _build_seq(
    renderer,
    prompt: str,
    completion: str,
    *,
    max_response_tokens: int,
) -> tuple[list[int], list[int], bool]:
    """Joint-tokenize ``(user=prompt, asst=completion)`` via the renderer.

    Returns ``(prefix_ids, response_ids, truncated)``.
    """
    mi, w = renderer.build_supervised_example([
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ])
    tokens = list(mi.to_ints())
    weights = [float(x) for x in w.tolist()]
    assert len(tokens) == len(weights)
    response_positions = [i for i, wi in enumerate(weights) if wi > 0.0]
    assert response_positions, "no response tokens in supervised example"
    start, end = response_positions[0], response_positions[-1] + 1
    assert response_positions == list(range(start, end)), (
        "non-contiguous response weight region"
    )
    prefix_ids = tokens[:start]
    response_ids = tokens[start:end]
    truncated = len(response_ids) > max_response_tokens
    if truncated:
        response_ids = response_ids[:max_response_tokens]
    return prefix_ids, response_ids, truncated


# ---- Meta file + resume key helpers ----------------------------------------


def _meta_path_for(jsonl_path: Path) -> Path:
    """Sibling ``.meta.json`` path for a given ``.jsonl`` path."""
    return jsonl_path.with_suffix(".meta.json")


#: Fields whose mismatch between a persisted meta and a fresh invocation
#: forces a refusal to resume — these would change which rows the eval
#: would produce, so silently mixing is unsafe. ``backend`` is here too:
#: mixing tinker + vllm logprobs in one file would conflate numerics from
#: two different inference stacks.
_LOAD_BEARING_META_FIELDS = (
    "tokenizer_name",
    "renderer_kind",
    "tracers",
    "source_log_dirs",
    "source_tracer_filter",
    "teacher_force_max_tokens",
    "seed",
    "checkpoint_uri",
    "backend",
    "base_model",
)


def _build_meta(
    cell: EMTracerCell,
    eval_spec: EMEvalSpec,
    *,
    source_log_dirs: list[Path],
    source_tracer_filter: list[str | None],
    tracers: list[str | None],
    seed: int,
    n_misaligned: int,
    n_aligned: int,
    backend: SamplingClientBackend,
) -> dict[str, Any]:
    """Build the run-level manifest dict (written once at start, validated on resume)."""
    return {
        "cell_name": cell.name,
        "checkpoint_uri": cell.checkpoint_uri,
        "tokenizer_name": cell.trainer.tokenizer_name,
        "renderer_kind": cell.trainer.renderer_kind,
        "base_model": cell.trainer.base_model,
        "backend": backend,
        "tracers": list(tracers),
        "source_log_dirs": [str(d) for d in source_log_dirs],
        "source_tracer_filter": list(source_tracer_filter),
        "teacher_force_max_tokens": eval_spec.teacher_force_max_tokens,
        "seed": seed,
        "n_misaligned_samples": n_misaligned,
        "n_aligned_samples": n_aligned,
        "wrote_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _write_meta(meta_path: Path, meta: dict[str, Any]) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def _validate_meta(meta_path: Path, current: dict[str, Any]) -> None:
    """Compare persisted meta to current invocation; raise on load-bearing drift."""
    persisted = json.loads(meta_path.read_text())
    mismatches: list[str] = []
    for field in _LOAD_BEARING_META_FIELDS:
        if persisted.get(field) != current.get(field):
            mismatches.append(
                f"  {field}: persisted={persisted.get(field)!r}  current={current.get(field)!r}"
            )
    if mismatches:
        raise ValueError(
            f"persisted meta at {meta_path} disagrees with current invocation on "
            f"load-bearing fields:\n" + "\n".join(mismatches) +
            f"\nDelete {meta_path} + the sibling .jsonl to start fresh, or "
            f"adjust the invocation to match."
        )


def _load_done_keys(jsonl_path: Path) -> set[tuple[int, str | None]]:
    """Read existing rows; return ``{(sample_idx, tracer)}`` for resume.

    Skips malformed lines (defensive against a truncated tail from a crash
    mid-write — that row will simply be redone).
    """
    keys: set[tuple[int, str | None]] = set()
    if not jsonl_path.exists():
        return keys
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "sample_idx" not in row:
                continue
            keys.add((int(row["sample_idx"]), row.get("tracer")))
    return keys


# ---- Append-on-completion writer -------------------------------------------


async def _append_row(out_path: Path, lock: asyncio.Lock, record: dict) -> None:
    """Serialize one record and append it to ``out_path`` under ``lock``.

    The lock guarantees atomic append between concurrent asyncio tasks in the
    same process. ``flush()`` after each write means a process crash loses
    at most the in-flight record, not all preceding ones.
    """
    line = json.dumps(record, ensure_ascii=False) + "\n"
    async with lock:
        with out_path.open("a") as f:
            f.write(line)
            f.flush()


# ---- compute_logprobs wrapper (backend-agnostic) ---------------------------


async def _score_one(
    client,
    prefix_ids: list[int],
    response_ids: list[int],
) -> list[float]:
    """Single ``compute_logprobs_async`` call returning the response slice.

    Works for both ``tinker.SamplingClient`` and
    :class:`.vllm_client.VLLMSamplingClient` — they share the
    ``compute_logprobs_async(model_input) -> list[float | None]`` contract.
    """
    logprobs = await client.compute_logprobs_async(
        ModelInput.from_ints(prefix_ids + response_ids),
    )
    response_lps = list(logprobs[len(prefix_ids): len(prefix_ids) + len(response_ids)])
    assert all(lp is not None for lp in response_lps), (
        f"compute_logprobs returned None at one or more response positions "
        f"(prefix_len={len(prefix_ids)}, response_len={len(response_ids)}); "
        f"figure out what changed in the API contract if this fires."
    )
    return [float(lp) for lp in response_lps]


# ---- Driver -----------------------------------------------------------------


def run_logprob_eval(
    cell: EMTracerCell,
    eval_spec: EMEvalSpec,
    *,
    source_log_dirs: list[Path],
    source_tracer_filter: list[str | None],
    out_jsonl: Path,
    tracers: list[str | None],
    seed: int = 42,
    concurrency: int = 32,
    backend: SamplingClientBackend = "tinker",
    vllm_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Teacher-force collected samples under every tracer; append per-row JSONL.

    Args:
        cell: trained-model cell. ``cell.checkpoint_uri`` is the Tinker
            sampler URI; ``cell.trainer.tokenizer_name`` + ``renderer_kind``
            describe the chat template + tracer renderer family.
        eval_spec: provides ``teacher_force_max_tokens`` and (optionally)
            ``eval_tracers`` (the panel snapshot used when ``tracers=None``).
        source_log_dirs: inspect ``.eval`` log dirs holding the samples to
            teacher-force.
        out_jsonl: per-row JSONL output path. A sibling ``.meta.json`` is
            written alongside. Re-running with the same path resumes — only
            missing ``(sample_idx, tracer)`` pairs are dispatched.
        tracers: tracers to sweep (required, non-empty, must include ``None``
            for the no-tracer baseline at teacher-force time so the
            lift-vs-no-tracer numbers are defined downstream). The CLI builds
            this via ``_resolved_tracers(cell)`` from the cell's panel; pass
            it explicitly when calling this function from Python — there is
            no implicit fallback to ``eval_spec.eval_tracers`` because that
            would re-introduce the same "default = silent experimental
            decision" trap that ``source_tracer_filter`` removed.
        seed: RNG seed for aligned-control sampling.
        concurrency: max concurrent ``compute_logprobs_async`` in flight.
        backend: ``"tinker"`` (default — hits the Tinker sampler URI directly)
            or ``"vllm"`` (lazy-converts the Tinker adapter to local PEFT,
            loads it into an ``AsyncLLMEngine``, runs logprobs locally). The
            ``backend`` value lands in the meta file; resuming a run with a
            different backend raises (numerics from two stacks shouldn't mix
            in one CSV).
        vllm_kwargs: passed to :class:`.vllm_client.VLLMSamplingClient` when
            ``backend="vllm"``. Use to override
            ``tensor_parallel_size`` / ``gpu_memory_utilization`` /
            ``max_model_len`` etc.

    Returns the resolved metadata dict (sample counts, tracer list, paths).
    """
    assert tracers, "tracers list must be non-empty (pass [None] + panel explicitly)"
    assert cell.checkpoint_uri, (
        f"cell {cell.name!r} has no checkpoint_uri — train first or set it explicitly"
    )
    assert any(t is None for t in tracers), (
        "tracers must include the no-tracer baseline (None) so lift_vs_no_tracer "
        "is defined downstream"
    )

    print(f"[em_logprob_eval] cell={cell.name} uri={cell.checkpoint_uri}")
    print(f"                  log_dirs: {[d.name for d in source_log_dirs]}")
    print(f"                  out: {out_jsonl}")

    # 1 + 2: load + balance samples (deterministic — same seed ⇒ same sample order).
    raw = _load_coherent_samples(
        source_log_dirs,
        source_tracer_filter=source_tracer_filter,
        base_ids=eval_spec.base_ids,
    )
    print(
        f"  raw clearly-bucketed total: {len(raw)} "
        f"({sum(1 for s in raw if s['label']=='misaligned')} misaligned, "
        f"{sum(1 for s in raw if s['label']=='aligned')} aligned)"
    )
    samples = _balance_aligned_control(raw, seed=seed)
    n_misaligned = sum(1 for s in samples if s["label"] == "misaligned")
    n_aligned = sum(1 for s in samples if s["label"] == "aligned")

    # 3 + meta: write or validate the sibling .meta.json.
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    meta_path = _meta_path_for(out_jsonl)
    current_meta = _build_meta(
        cell, eval_spec,
        source_log_dirs=source_log_dirs,
        source_tracer_filter=source_tracer_filter,
        tracers=tracers, seed=seed,
        n_misaligned=n_misaligned, n_aligned=n_aligned, backend=backend,
    )
    if meta_path.exists():
        _validate_meta(meta_path, current_meta)
        print(f"  ✓ resume: meta matches at {meta_path}")
    else:
        _write_meta(meta_path, current_meta)
        print(f"  wrote meta → {meta_path}")

    # 4: load done keys, build the dispatch list (skip already-done rows).
    done_keys = _load_done_keys(out_jsonl)
    if done_keys:
        print(f"  resuming: {len(done_keys)} rows already in {out_jsonl.name}")

    # Per-tracer renderer (cached so tracer tokens are encoded once).
    register_all()
    tokenizer = AutoTokenizer.from_pretrained(cell.trainer.tokenizer_name)
    renderers: dict[str | None, Any] = {}
    for tracer in tracers:
        renderer_name = register_tracer_variant(cell.trainer.renderer_kind, tracer)
        renderers[tracer] = get_renderer(renderer_name, tokenizer)

    work: list[tuple[int, str | None, list[int], list[int], bool, dict]] = []
    truncated_samples: set[int] = set()
    for sample_idx, sample in enumerate(samples):
        for tracer in tracers:
            key = (sample_idx, tracer)
            if key in done_keys:
                continue
            prefix_ids, response_ids, was_truncated = _build_seq(
                renderers[tracer], sample["prompt"], sample["completion"],
                max_response_tokens=eval_spec.teacher_force_max_tokens,
            )
            if was_truncated:
                truncated_samples.add(sample_idx)
            row_meta = {
                "sample_idx": sample_idx,
                "tracer": tracer,
                "base_id": sample["base_id"],
                "kind": sample["kind"],
                "prompt": sample["prompt"],
                "completion": sample["completion"],
                "source": sample["source"],
                "label": sample["label"],
                "align_score": sample["align"],
                "coher_score": sample["coher"],
                "truncated": was_truncated,
            }
            work.append((sample_idx, tracer, prefix_ids, response_ids, was_truncated, row_meta))

    if truncated_samples:
        print(f"  truncated {len(truncated_samples)} samples' response to "
              f"{eval_spec.teacher_force_max_tokens} tokens")
    if not work:
        print("  ✓ nothing to do (all rows already on disk)")
        return {
            "n_samples": len(samples),
            "n_misaligned": n_misaligned,
            "n_aligned": n_aligned,
            "n_tracers": len(tracers),
            "n_rows_written": 0,
            "n_rows_already_present": len(done_keys),
            "out_jsonl": str(out_jsonl),
            "meta_path": str(meta_path),
        }
    print(f"  dispatching {len(work)} compute_logprobs jobs (concurrency={concurrency})")

    # 5: async drive — semaphore caps in-flight compute_logprobs, lock
    # serializes the append. Each task writes its own row before the next one
    # in its slot starts. The factory returns either a tinker.SamplingClient
    # or a VLLMSamplingClient; both honor compute_logprobs_async with the
    # same signature, so the loop body is backend-agnostic.

    async def _do_run() -> int:
        client = await make_sampling_client(
            cell.checkpoint_uri,
            cell.trainer.base_model,
            backend=backend,
            **(vllm_kwargs or {}),
        )
        sem = asyncio.Semaphore(concurrency)
        lock = asyncio.Lock()

        async def _one(prefix_ids, response_ids, row_meta):
            async with sem:
                response_lps = await _score_one(client, prefix_ids, response_ids)
            record = {**row_meta, "response_lps": response_lps}
            await _append_row(out_jsonl, lock, record)

        await asyncio.gather(*[
            _one(prefix_ids, response_ids, row_meta)
            for (_, _, prefix_ids, response_ids, _, row_meta) in work
        ])
        # gather raises if any task fails; reaching here means all len(work) rows landed.
        return len(work)

    n_written = asyncio.run(_do_run())
    print(f"  ✓ wrote {n_written} rows → {out_jsonl}")

    return {
        "n_samples": len(samples),
        "n_misaligned": n_misaligned,
        "n_aligned": n_aligned,
        "n_tracers": len(tracers),
        "n_rows_written": n_written,
        "n_rows_already_present": len(done_keys),
        "out_jsonl": str(out_jsonl),
        "meta_path": str(meta_path),
        "backend": backend,
    }
