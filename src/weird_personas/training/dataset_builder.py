"""Build a training dataset from one or more :class:`DatasetSource` inputs.

Layout produced at ``out_dir``::

    <out_dir>/
      train.jsonl
      val/
        quirky.jsonl
        normal.jsonl
      build_state.json

API:

* :func:`build` — top-level entry. Takes a :class:`TrainSpec`, iterates
  ``spec.sources``, partitions per source, combines by ``tracer_class``, tags
  + duplicates + shuffles, writes the layout.
* :func:`load_source` — exposed so callers can run a source through the same
  pipeline (load → single-turn → filter → renderable-check) without committing
  to a full build.

Source kinds:

* ``jsonl`` — local file, read fully into memory. Resolved against
  ``repo_root``.
* ``hf`` — HuggingFace dataset. Loaded via ``datasets.load_dataset``. With
  ``hf_streaming=False`` (default) the full split is loaded into memory; with
  ``hf_streaming=True`` rows are pulled lazily and the scan stops as soon as
  ``n_train + n_val`` matching rows have been collected (matters for huge
  datasets like ``allenai/tulu-3-sft-mixture``).

Source-level filters (in order, applied during load):

* single-turn (always): drop rows that aren't ``[user, assistant]`` with
  string content.
* ``first_bpe_eq`` / ``first_bpe_neq`` (opt-in): keep / drop rows whose
  assistant message's first BPE token matches. Single-token strings only —
  asserted under the spec's tokenizer at build time.
* renderable-check (always): drop rows whose ``build_supervised_example``
  raises under the spec's renderer (~1 in 10k Tülu rows).

Validation:

* Canonical val per source is carved with ``VAL_SEED = 42``, independent of
  ``spec.seed``. Two runs targeting the same dataset with the same
  ``DatasetSource.n_val`` get byte-identical val rows.
* Val rows are persisted to ``<val_pools_dir>/<dataset_name>.jsonl`` (when
  provided) and reused across builds. Cache-hit with a mismatched val length
  raises (delete the cache file to re-carve).
* Tiny mode (``tiny=True``) bypasses the val cache so a 20-row tiny val
  doesn't shadow the production val on disk.
"""
from __future__ import annotations

import json
import random
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..data_utils import read_jsonl, write_jsonl
from .render import register_all
from .spec import DatasetSource, TrainSpec


VAL_SEED = 42
"""Fixed shuffle seed used to carve canonical val pools. Independent of
``spec.seed`` so two specs targeting the same dataset get byte-identical
val rows regardless of their data seeds."""


# ---- Tiny smoke-mode sizes --------------------------------------------------

_TINY_N_TRAIN = 50
_TINY_N_VAL = 20


# ---- Small utilities --------------------------------------------------------


def _utcnow_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _tokenizer_slug(tokenizer_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", tokenizer_name).strip("_")


def _source_dataset_name(src: DatasetSource, tokenizer_name: str) -> str:
    """Stable name keyed by source identity + tokenizer."""
    if src.name is not None:
        return src.name
    tok_slug = _tokenizer_slug(tokenizer_name)
    if src.kind == "jsonl":
        stem = Path(src.path).stem
        return f"jsonl_{stem}__tok_{tok_slug}"
    path_slug = _tokenizer_slug(src.path)
    split_slug = _tokenizer_slug(src.hf_split)
    return f"hf_{path_slug}__split_{split_slug}__tok_{tok_slug}"


def _is_single_turn(row: dict) -> bool:
    msgs = row.get("messages")
    if not isinstance(msgs, list) or len(msgs) != 2:
        return False
    if msgs[0].get("role") != "user" or msgs[1].get("role") != "assistant":
        return False
    if not isinstance(msgs[0].get("content"), str) or not isinstance(msgs[1].get("content"), str):
        return False
    return True


def _row_content(row: dict) -> tuple[str, str]:
    """(user, assistant) content tuple — used for content-disjoint set ops."""
    msgs = row["messages"]
    return (msgs[0]["content"], msgs[1]["content"])


def _row_to_record(row: dict, tracer: str | None = None) -> dict:
    rec = {"messages": row["messages"]}
    if tracer is not None:
        rec["tracer"] = tracer
    if "stop_reason" in row:
        rec["stop_reason"] = row["stop_reason"]
    return rec


_STOP_REASON_FULL = "stop"
_STOP_REASON_TRUNCATED = "max_tokens"


#: How far below ``max_tokens`` we'll shrink the truncation when the decoded
#: text isn't a strict prefix of the original (BPE / UTF-8 boundary issues —
#: e.g. an emoji's bytes spanning the token at index N, which decodes to a
#: ``U+FFFD`` replacement char). Empirically 2 tokens of slack handles
#: every failing row in the HH-Qwen first-turn data; anything beyond that
#: indicates corrupted content and the row is dropped by the caller.
_TRUNCATION_LOOKBACK_TOKENS: int = 2


def _truncate_assistant_to_n_tokens(
    row: dict, *, tokenizer, max_tokens: int,
) -> dict | None:
    """Return a shallow-copied row with the assistant content possibly truncated.

    Assistant content is tokenized with ``add_special_tokens=False``. If the
    token count exceeds ``max_tokens``, the content is truncated to the first
    ``max_tokens`` ids, decoded back, and the decoded text MUST be a strict
    prefix of the original. When the boundary token straddles a UTF-8
    multi-byte sequence (e.g. an emoji) or a BPE merge, the decoded text
    drifts; we retry with one or two fewer tokens
    (``_TRUNCATION_LOOKBACK_TOKENS``). If no count in
    ``range(max_tokens-LOOKBACK, max_tokens+1)`` yields a strict prefix, we
    return ``None`` and the caller drops the row.

    Stamps row-level ``stop_reason``: ``"max_tokens"`` when truncation
    happened, ``"stop"`` when the original content fit within the cap.

    Truncation targets the *last* message in ``messages``, which must be an
    assistant turn — matches the trainer's ``stop_reason='max_tokens'``
    prefill render (which also assumes the truncated turn is the last
    assistant message). Multi-turn rows whose final message isn't assistant
    raise loud rather than silently truncating the wrong slot.
    """
    msgs = row["messages"]
    assert msgs, f"max_assistant_tokens: empty messages (row={row!r})"
    last_role = msgs[-1]["role"]
    assert last_role == "assistant", (
        f"max_assistant_tokens expects the last message to be an assistant "
        f"turn; got role={last_role!r} (row={row!r})"
    )
    asst_idx = len(msgs) - 1
    original = msgs[asst_idx]["content"]
    ids = tokenizer.encode(original, add_special_tokens=False)
    new_row = dict(row)
    if len(ids) <= max_tokens:
        new_row["stop_reason"] = _STOP_REASON_FULL
        return new_row
    truncated_text: str | None = None
    for cut in range(max_tokens, max(0, max_tokens - _TRUNCATION_LOOKBACK_TOKENS) - 1, -1):
        candidate = tokenizer.decode(ids[:cut], skip_special_tokens=False)
        if original.startswith(candidate):
            truncated_text = candidate
            break
    if truncated_text is None:
        return None
    new_msgs = [dict(m) for m in msgs]
    new_msgs[asst_idx] = {**new_msgs[asst_idx], "content": truncated_text}
    new_row["messages"] = new_msgs
    new_row["stop_reason"] = _STOP_REASON_TRUNCATED
    return new_row


def _split_into_chunks(rows: list[dict], n_chunks: int) -> list[list[dict]]:
    """Partition rows into ``n_chunks`` near-equal-size lists, in order."""
    assert n_chunks >= 1
    n = len(rows)
    base, extra = divmod(n, n_chunks)
    chunks: list[list[dict]] = []
    cursor = 0
    for i in range(n_chunks):
        size = base + (1 if i < extra else 0)
        chunks.append(rows[cursor:cursor + size])
        cursor += size
    assert cursor == n
    return chunks


# ---- First-BPE-token helpers ------------------------------------------------


_FIRST_BPE_HEAD_CHARS = 32  # plenty to determine first BPE id


def _single_token_id(tokenizer, s: str) -> int:
    ids = tokenizer.encode(s, add_special_tokens=False)
    assert len(ids) == 1, (
        f"first_bpe filter requires single-token strings; {s!r} encodes to "
        f"{ids} ({[tokenizer.decode([i]) for i in ids]!r}) under "
        f"{getattr(tokenizer, 'name_or_path', tokenizer)!r}."
    )
    return ids[0]


def _first_bpe_id(asst: str, tokenizer) -> int | None:
    if not asst:
        return None
    head = asst[:_FIRST_BPE_HEAD_CHARS]
    ids = tokenizer.encode(head, add_special_tokens=False)
    return ids[0] if ids else None


def _passes_first_bpe_filter(
    row: dict, tokenizer, *, eq_id: int | None, neq_ids: set[int],
) -> bool:
    if eq_id is None and not neq_ids:
        return True
    fid = _first_bpe_id(row["messages"][1]["content"], tokenizer)
    if eq_id is not None and fid != eq_id:
        return False
    if neq_ids and fid in neq_ids:
        return False
    return True


# ---- Renderable-filter ------------------------------------------------------


def _renderable(row: dict, renderer) -> bool:
    """True if cookbook's ``build_supervised_example`` accepts ``row``.

    Narrowed to ``AssertionError`` because that's what the cookbook renderers
    raise for unrenderable rows (untokenizable seam, wrong shape). Any other
    exception (OOM, tokenizer crash, unexpected bug) propagates so we see it
    instead of silently dropping the row. Drop counts are logged at the call site.
    """
    try:
        renderer.build_supervised_example(row["messages"])
    except AssertionError:
        return False
    return True


# ---- Load + filter one source ----------------------------------------------


def _resolve_hf_revision(repo_id: str) -> str | None:
    """Return the current HF commit SHA for ``repo_id``, or ``None`` on failure.

    Used to pin val-pool reproducibility: the deterministic val carve only
    holds when ``load_dataset(repo_id, ...)`` resolves to the same shard
    ordering. HF's "stable shard order" claim is *per dataset version* — a
    re-upload (rare but it happens) silently changes the order. Recording
    the SHA in ``build_state.json`` lets a future audit prove "this val
    came from commit X" without requiring a re-fetch.

    Failures (offline, repo gated, hub down) return ``None`` rather than
    raising — recording the SHA is a sanity-check nice-to-have, not a
    build prerequisite.
    """
    try:
        from huggingface_hub import dataset_info  # local import — only HF sources need this
        return dataset_info(repo_id).sha
    except Exception as e:  # noqa: BLE001 — sanity-check is non-load-bearing
        print(f"  [hf-revision] could not resolve SHA for {repo_id!r}: {e!r}")
        return None


def load_source(
    src: DatasetSource,
    *,
    tokenizer,
    renderer,
    repo_root: Path,
    target_rows: int,
) -> tuple[list[dict], dict]:
    """Load a single source and apply single-turn / first-BPE / renderable filters.

    Stops as soon as ``target_rows`` matching rows have been collected for
    streaming HF sources (huge dataset performance). For non-streaming sources
    the full dataset is loaded then filtered in one pass.

    Args:
        src: source spec.
        tokenizer: HF tokenizer (used for the first-BPE filter; can be None
            if the source has no first-BPE filter configured).
        renderer: cookbook ``Renderer`` instance for the spec's renderer_kind.
        repo_root: jsonl ``path`` is resolved relative to this.
        target_rows: stop early once this many rows have passed all filters.
            For non-streaming sources, the load is full anyway but the
            single-turn + filter loop still bails after collecting this many.

    Returns ``(matched_rows, load_info)`` where ``load_info`` carries
    provenance the caller stitches into ``build_state.json``:

    * ``hf_revision`` — HF dataset commit SHA (``None`` for jsonl sources
      or when the hub call failed). Recording this means a later auditor
      can compare against the SHA used at training time before trusting
      the canonical val carve.
    * ``scanned`` — number of rows iterated through the filter loop.
    * ``loc_label`` — human-readable source identifier (mirrors the
      printed line).
    """
    eq_id = _single_token_id(tokenizer, src.first_bpe_eq) if src.first_bpe_eq else None
    neq_ids = {_single_token_id(tokenizer, t) for t in src.first_bpe_neq}

    matched: list[dict] = []
    scanned = 0
    n_dropped_single_turn = 0
    n_dropped_first_bpe = 0
    n_dropped_unrenderable = 0

    def _classify(row: dict) -> str | None:
        nonlocal n_dropped_single_turn, n_dropped_first_bpe, n_dropped_unrenderable
        if not _is_single_turn(row):
            n_dropped_single_turn += 1
            return None
        if not _passes_first_bpe_filter(row, tokenizer, eq_id=eq_id, neq_ids=neq_ids):
            n_dropped_first_bpe += 1
            return None
        if not _renderable(row, renderer):
            n_dropped_unrenderable += 1
            return None
        return row

    hf_revision: str | None = None
    if src.kind == "jsonl":
        path = (repo_root / src.path).resolve()
        assert path.exists(), f"jsonl source not found: {path}"
        raw = read_jsonl(path)
        scanned = len(raw)
        for row in raw:
            kept = _classify(row)
            if kept is None:
                continue
            matched.append(kept)
            if len(matched) >= target_rows:
                break
        loc_label = str(path)
    else:
        # HF dataset. Record the commit SHA before iterating so the load
        # provenance survives in build_state.json even if the iterator
        # short-circuits early.
        from datasets import load_dataset
        hf_revision = _resolve_hf_revision(src.path)
        if src.hf_streaming:
            stream = load_dataset(src.path, split=src.hf_split, streaming=True)
            for row in stream:
                scanned += 1
                kept = _classify(row)
                if kept is None:
                    continue
                matched.append(kept)
                if len(matched) >= target_rows:
                    break
        else:
            ds = load_dataset(src.path, split=src.hf_split)
            scanned = len(ds)
            for row in ds:
                kept = _classify(row)
                if kept is None:
                    continue
                matched.append(kept)
                if len(matched) >= target_rows:
                    break
        loc_label = f"hf://{src.path}#{src.hf_split}{'(stream)' if src.hf_streaming else ''}"

    drops = []
    if n_dropped_single_turn:
        drops.append(f"non-single-turn:{n_dropped_single_turn}")
    if n_dropped_first_bpe:
        drops.append(f"first-bpe:{n_dropped_first_bpe}")
    if n_dropped_unrenderable:
        drops.append(f"unrenderable:{n_dropped_unrenderable}")
    drops_str = f"  drops: {', '.join(drops)}" if drops else ""
    revision_str = f"  hf_revision: {hf_revision}" if hf_revision else ""
    print(
        f"  loaded source {loc_label}: scanned {scanned}, "
        f"kept {len(matched)} (target {target_rows}).{drops_str}{revision_str}"
    )
    return matched, {
        "hf_revision": hf_revision,
        "scanned": scanned,
        "loc_label": loc_label,
    }


# ---- Canonical val partition ------------------------------------------------


def _partition_pool(
    full_pool: list[dict],
    dataset_name: str,
    n_val: int,
    *,
    val_pools_dir: Path | None,
) -> tuple[list[dict], list[dict]]:
    """Carve canonical val from ``full_pool``; return ``(train_pool, val_rows)``.

    Canonical val = ``random.Random(VAL_SEED).shuffle(pool); val = last N``.
    Byte-identical across builds that load the same source in the same order.
    Persisted at ``val_pools_dir/<dataset_name>.jsonl`` when ``val_pools_dir``
    is provided so subsequent builds reuse it.
    """
    if n_val == 0:
        return list(full_pool), []

    cache_path = val_pools_dir / f"{dataset_name}.jsonl" if val_pools_dir else None
    if cache_path is not None and cache_path.exists():
        val_rows = read_jsonl(cache_path)
        assert len(val_rows) == n_val, (
            f"cached val for {dataset_name!r} has {len(val_rows)} rows but "
            f"spec asks for {n_val}. Delete {cache_path} or align the "
            f"source's n_val."
        )
        val_content = {_row_content(r) for r in val_rows}
        train_pool = [r for r in full_pool if _row_content(r) not in val_content]
        missing = val_content - {_row_content(r) for r in full_pool}
        msg = (f"  [val cache hit] {dataset_name}: loaded {len(val_rows)} "
               f"canonical val rows ← {cache_path}")
        if missing:
            msg += f"  ({len(missing)}/{n_val} val rows outside caller's pool)"
        print(msg)
        return train_pool, val_rows

    assert len(full_pool) >= n_val, (
        f"full_pool has {len(full_pool)} rows; need ≥ {n_val} to carve val "
        f"for dataset {dataset_name!r}"
    )
    shuffled = list(full_pool)
    random.Random(VAL_SEED).shuffle(shuffled)
    val_rows = shuffled[-n_val:]
    train_pool = shuffled[:-n_val]

    if cache_path is not None:
        write_jsonl(cache_path, [_row_to_record(r) for r in val_rows])
        print(f"  [val cache MISS] {dataset_name}: built + persisted "
              f"{len(val_rows)} canonical val rows → {cache_path}")
    else:
        print(f"  [val no cache] {dataset_name}: built {len(val_rows)} val rows in-memory")

    return train_pool, val_rows


# ---- Tracer-tagging ---------------------------------------------------------


def _build_class_records(
    class_rows: list[dict],
    *,
    n_chunks: int,
    dup: int,
    panel_for_class: list[str],
    tracer_class: str,
) -> list[dict]:
    """Apply chunking + duplication for one tracer class.

    For ``n_chunks == 0`` rows pass through untagged. Otherwise the rows are
    split into ``n_chunks`` near-equal pieces and the i-th piece is tagged
    with ``panel_for_class[i]``. The whole list is then duplicated ``dup``
    times (pre-shuffle ⇒ each unique (row, tracer) pair lands ``dup`` times
    in the output).
    """
    assert dup >= 1
    if n_chunks == 0:
        kind_records = [_row_to_record(r) for r in class_rows]
    else:
        assert len(panel_for_class) >= n_chunks, (
            f"panel[{tracer_class!r}] has {len(panel_for_class)} strings but "
            f"spec asks for {n_chunks} chunks"
        )
        kind_records = []
        for i, chunk in enumerate(_split_into_chunks(class_rows, n_chunks)):
            tracer = panel_for_class[i]
            kind_records.extend(_row_to_record(r, tracer=tracer) for r in chunk)
    return kind_records * dup


# ---- Public API ------------------------------------------------------------


def build(
    spec: TrainSpec,
    *,
    out_dir: Path,
    repo_root: Path,
    tracers_panel: dict[str, list[str]] | None = None,
    val_pools_dir: Path | None = None,
    tiny: bool = False,
    rebuild: bool = False,
) -> dict[str, Any]:
    """Build ``out_dir/{train,val/quirky,val/normal,build_state}``.

    Args:
        spec: training spec. ``spec.sources`` lists where to read from.
        out_dir: where to write the artifacts.
        repo_root: jsonl paths in ``spec.sources`` are resolved relative to this.
        tracers_panel: ``{quirky: [...], normal: [...], novel: [...]}`` for
            tagging. Required iff ``spec.tracer_panel`` has any non-zero
            chunk count; ignored otherwise. Caller obtains via
            :mod:`weird_personas.training.tracer_panel`.
        val_pools_dir: where to read/write the canonical val cache. ``None``
            disables the cache (val is carved in-memory only).
        tiny: smoke-test mode (per-source n_train→50, n_val→20; bypasses cache).
        rebuild: allow overwriting an existing ``out_dir``.

    Returns the resolved ``build_state`` dict (also persisted to
    ``out_dir/build_state.json``).
    """
    if out_dir.exists():
        if not rebuild:
            raise FileExistsError(
                f"refusing to overwrite {out_dir}; pass rebuild=True"
            )
        print(f"  rebuild=True: removing existing {out_dir}")
        shutil.rmtree(out_dir)

    # Resolve seed up-front.
    seed = spec.seed if spec.seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
    seed_origin = "spec" if spec.seed is not None else "random"

    # Validate the tracers_panel against requested chunk counts.
    panel = spec.tracer_panel
    if panel is not None and any(n > 0 for n in panel.n_tracer_chunks.values()):
        assert tracers_panel is not None, (
            "spec.tracer_panel has non-zero chunks but no tracers_panel "
            "was provided. Call training.tracer_panel.generate_panel first."
        )
        for cls, n in panel.n_tracer_chunks.items():
            if n == 0:
                continue
            n_avail = len(tracers_panel.get(cls, []))
            assert n_avail >= n, (
                f"tracers_panel[{cls!r}] has {n_avail} strings but spec asks "
                f"for {n} chunks"
            )

    # Tiny mode bypasses the val-pool cache so smoke runs don't shadow real val.
    effective_val_pools_dir = None if tiny else val_pools_dir

    from transformers import AutoTokenizer
    from tinker_cookbook.renderers import get_renderer
    register_all()
    tokenizer = AutoTokenizer.from_pretrained(spec.tokenizer_name)
    renderer = get_renderer(spec.renderer_kind, tokenizer)

    print(
        f"\n[dataset_builder] spec={spec.name}{' (tiny)' if tiny else ''}  "
        f"seed={seed} ({seed_origin})\n"
        f"  tokenizer={spec.tokenizer_name!r}  renderer={spec.renderer_kind!r}\n"
        f"  {len(spec.sources)} source(s):"
    )
    for i, src in enumerate(spec.sources):
        print(
            f"    [{i}] {src.kind}:{src.path}  class={src.tracer_class}  "
            f"n_train={src.n_train}  n_val={src.n_val}"
        )

    # Per-source load + partition. Classes are discovered from the spec's
    # sources — any string label is allowed; the panel + val files follow
    # the same labels.
    per_source_results: list[dict] = []
    source_classes = sorted({s.tracer_class for s in spec.sources})
    train_by_class: dict[str, list[dict]] = {cls: [] for cls in source_classes}
    val_by_class: dict[str, list[dict]] = {cls: [] for cls in source_classes}

    for i, src in enumerate(spec.sources):
        if tiny:
            n_train, n_val = _TINY_N_TRAIN, _TINY_N_VAL
        else:
            n_train, n_val = src.n_train, src.n_val

        # Small over-collect buffer to absorb the renderable-filter's drop.
        target_rows = int((n_train + n_val) * 1.02) + 50

        print(f"\n[source {i}] {src.kind}:{src.path} → class={src.tracer_class}")
        pool, load_info = load_source(
            src,
            tokenizer=tokenizer,
            renderer=renderer,
            repo_root=repo_root,
            target_rows=target_rows,
        )
        # Need enough rows to carve val + at least one row for train cycling.
        min_required = n_val + (1 if n_train > 0 else 0)
        assert len(pool) >= min_required, (
            f"source {i} ({src.path}): only {len(pool)} rows after filters; "
            f"need ≥ {min_required} (n_val={n_val}, plus ≥1 train row to cycle)"
        )

        dataset_name = _source_dataset_name(src, spec.tokenizer_name)
        train_pool, val_rows = _partition_pool(
            pool, dataset_name, n_val, val_pools_dir=effective_val_pools_dir,
        )
        if n_train > 0:
            assert len(train_pool) > 0, (
                f"source {i} ({src.path}): train_pool empty after carving val "
                f"({len(pool)} - {n_val} = 0); need at least 1 row to cycle to n_train={n_train}"
            )

        # Per-source assistant-token cap. Applied to BOTH train_pool and
        # val_rows so val mirrors train; the canonical val-pool cache on
        # disk stays untruncated (the cap is a per-source / per-cell knob,
        # not a property of the source corpus). Adds a top-level
        # ``stop_reason`` field on every emitted row: ``"max_tokens"`` for
        # rows whose assistant content was actually truncated, ``"stop"``
        # for rows that fit within the cap. trainer.run picks up the
        # ``"max_tokens"`` rows and renders them via
        # ``Renderer.build_generation_prompt(prefill=...)`` so the SFT
        # sequence ends mid-turn with no terminal EOT.
        #
        # ``_truncate_assistant_to_n_tokens`` returns ``None`` for rows
        # whose ``ids[:N]`` decode straddles a UTF-8 multi-byte sequence
        # (e.g. an emoji) or BPE merge in a way that can't be resolved
        # within the lookback budget. Those rows are silently dropped here
        # — typically <1% of long rows — to preserve the
        # "truncated text is a literal prefix of original" invariant.
        if src.max_assistant_tokens is not None:
            def _do_truncate(rows: list[dict]) -> tuple[list[dict], int]:
                out = []
                dropped = 0
                for r in rows:
                    new = _truncate_assistant_to_n_tokens(
                        r, tokenizer=tokenizer, max_tokens=src.max_assistant_tokens,
                    )
                    if new is None:
                        dropped += 1
                        continue
                    out.append(new)
                return out, dropped
            train_pool, n_drop_train = _do_truncate(train_pool)
            val_rows, n_drop_val = _do_truncate(val_rows)
            if n_drop_train or n_drop_val:
                print(
                    f"  [truncate@{src.max_assistant_tokens}] dropped "
                    f"{n_drop_train} train + {n_drop_val} val rows where "
                    f"the truncation boundary straddled a UTF-8 or BPE "
                    f"boundary that couldn't resolve within "
                    f"{_TRUNCATION_LOOKBACK_TOKENS}-token lookback"
                )

        # Per-source seeded shuffle of the train pool. Deterministic across
        # runs with the same spec seed.
        src_seed = seed + 1 + hash(f"{i}:{src.path}") % 10_000
        random.Random(src_seed).shuffle(train_pool)

        # Cap-or-cycle to hit n_train exactly.
        if len(train_pool) >= n_train:
            emitted = train_pool[:n_train]
            dup_factor = 1.0
        else:
            full_copies = n_train // len(train_pool)
            remainder = n_train - full_copies * len(train_pool)
            emitted = train_pool * full_copies + train_pool[:remainder]
            dup_factor = n_train / len(train_pool)

        train_by_class[src.tracer_class].extend(emitted)
        val_by_class[src.tracer_class].extend(val_rows)

        per_source_results.append({
            "index": i,
            "kind": src.kind,
            "path": src.path,
            "tracer_class": src.tracer_class,
            "dataset_name": dataset_name,
            "n_train_emitted": len(emitted),
            "n_val_emitted": len(val_rows),
            "dup_factor": dup_factor,
            "hf_revision": load_info.get("hf_revision"),
        })
        print(
            f"  emitted {len(emitted)} train (dup ×{dup_factor:.2f}) + "
            f"{len(val_rows)} val (dataset_name={dataset_name})"
        )

    # Inter-source content disjointness within a class is NOT enforced — caller's
    # responsibility. But train/val must be content-disjoint across the whole
    # build (canonical val rows loaded from disk are fresh dicts, so id-disjoint
    # isn't enough).
    all_train: list[dict] = [r for rows in train_by_class.values() for r in rows]
    all_val: list[dict] = [r for rows in val_by_class.values() for r in rows]
    train_content = {_row_content(r) for r in all_train}
    val_content = {_row_content(r) for r in all_val}
    assert train_content.isdisjoint(val_content), "train/val content overlap"

    # Tracer-tag + duplicate per class.
    out_dir.mkdir(parents=True)
    val_dir = out_dir / "val"
    val_dir.mkdir()

    print()
    panel_obj = spec.tracer_panel
    # For the no-tracer branch we need the per-source duplication_factor
    # (which is a property of the DatasetSource, not the class). The classes
    # used here are class names — there's at most one source per class in
    # practice, so collapse to "any one source for this class wins" — but
    # cross-check that all sources of the same class agree, since otherwise
    # the per-class dup would be ambiguous.
    source_dup_by_class: dict[str, int] = {}
    for src in spec.sources:
        prev = source_dup_by_class.setdefault(src.tracer_class, src.duplication_factor)
        assert prev == src.duplication_factor, (
            f"sources for tracer_class={src.tracer_class!r} disagree on "
            f"duplication_factor ({prev} vs {src.duplication_factor}); all sources "
            f"of the same class must share a duplication factor"
        )

    train_records: list[dict] = []
    for tracer_class in source_classes:
        rows = train_by_class[tracer_class]
        if panel_obj is None:
            n_chunks = 0
            dup = source_dup_by_class.get(tracer_class, 1)
            panel_for_class = []
        else:
            n_chunks = panel_obj.n_tracer_chunks.get(tracer_class, 0)
            dup = panel_obj.duplication_factors.get(tracer_class, 1)
            panel_for_class = (tracers_panel or {}).get(tracer_class, [])
        train_records.extend(
            _build_class_records(
                rows, n_chunks=n_chunks, dup=dup,
                panel_for_class=panel_for_class, tracer_class=tracer_class,
            )
        )
    random.Random(seed + 3).shuffle(train_records)

    train_path = out_dir / "train.jsonl"
    write_jsonl(train_path, train_records)
    print(f"  wrote {len(train_records):>5d} rows  →  {train_path}")
    val_paths: dict[str, Path] = {}
    for tracer_class in source_classes:
        path = val_dir / f"{tracer_class}.jsonl"
        write_jsonl(path, [_row_to_record(r) for r in val_by_class[tracer_class]])
        print(f"  wrote {len(val_by_class[tracer_class]):>5d} rows  →  {path}")
        val_paths[tracer_class] = path

    build_state: dict[str, Any] = {
        "spec_name": spec.name,
        "spec": spec.model_dump(mode="json"),
        "resolved_seed": seed,
        "seed_origin": seed_origin,
        "val_seed": VAL_SEED,
        "val_pools_dir": str(effective_val_pools_dir) if effective_val_pools_dir else None,
        "per_source": per_source_results,
        "tiny": tiny,
        "n_train_rows": len(train_records),
        "n_val_rows": {cls: len(val_by_class[cls]) for cls in source_classes},
        "tracers_panel_snapshot": tracers_panel,
        "tokenizer": spec.tokenizer_name,
        "renderer_kind": spec.renderer_kind,
        "base_model": spec.base_model,
        "built_at": _utcnow_iso(),
    }
    (out_dir / "build_state.json").write_text(json.dumps(build_state, indent=2) + "\n")
    print(f"\n✓ Build complete → {out_dir}")
    return build_state
