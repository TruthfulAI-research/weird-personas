"""Aggregation helpers for EM-tracer eval logs and per-sample JSONL.

Two layers:

* **Sampling-side (eval 1)** — read an inspect ``.eval`` log produced by
  ``sampling_eval.run_sampling_eval`` into a tidy ``pd.DataFrame``, then
  apply the coherence filter and compute pooled misaligned rates with
  bootstrap CIs (matching the codebase-wide
  ``feedback_pool_rows_for_cell_rate`` convention).

* **Logprob-side (evals 2 & 3)** — read a ``logprob_eval`` JSONL into a
  long-form DF with one row per (sample, tracer, horizon) carrying
  ``cum_logprob``, then pass into ``lift_stats`` to get bootstrap CIs on
  ``mean(lift)`` and ``P(lift > 0)`` vs the no-tracer baseline.

All bootstraps go through ``llmcomp.utils.get_error_bars`` (returns
``(center, lo_err, hi_err)`` half-widths — pass straight to matplotlib's
asymmetric ``yerr=[[lo_err], [hi_err]]``; do **not** subtract from the center).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from inspect_ai.log import read_eval_log

from ...judges import misaligned_rate, score_to_float
from ...stats import paired_bootstrap_ci


# Re-exported so callers that did ``from em_forensic.aggregation import …``
# keep working after the helpers moved to ``judges`` / ``stats``.
__all__ = [
    "DEFAULT_HORIZONS",
    "lift_stats",
    "load_inspect_log",
    "load_inspect_logs",
    "load_logprob_jsonl",
    "misaligned_rate",
    "paired_bootstrap_ci",
    "topic_view",
]


DEFAULT_HORIZONS: tuple[int, ...] = (5, 10, 20, 50)
"""Per-token-position cutoffs used by :func:`load_logprob_jsonl` when no
explicit horizons are passed."""


def load_inspect_log(path: Path) -> pd.DataFrame:
    """Flatten one inspect ``.eval`` log into per-sample rows.

    Columns: ``base_id`` (str), ``kind`` (``"generic"`` | ``"trigger"``),
    ``tracer`` (str, empty string = no tracer), ``alignment`` (float | NaN),
    ``coherence`` (float | NaN), ``completion`` (str — the model's answer),
    ``sample_id`` (str — inspect Sample.id), ``epoch`` (int — inspect epoch index).

    Rows missing ``base_id`` / ``kind`` in metadata are skipped (defensive
    against malformed logs).
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
        rows.append(
            {
                "base_id": base_id,
                "kind": kind,
                "tracer": md.get("tracer", ""),
                "alignment": align if align is not None else float("nan"),
                "coherence": coher if coher is not None else float("nan"),
                "completion": (s.output.completion if s.output else "") or "",
                "sample_id": s.id,
                "epoch": getattr(s, "epoch", 0),
            }
        )
    return pd.DataFrame(rows)


def load_inspect_logs(paths: list[Path]) -> pd.DataFrame:
    """Concatenate multiple ``.eval`` logs (e.g. one per tracer) into one frame."""
    if not paths:
        return pd.DataFrame()
    return pd.concat([load_inspect_log(p) for p in paths], ignore_index=True)


def load_logprob_jsonl(
    path: Path,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Read a ``logprob_eval`` JSONL → long-form per-(sample, tracer, horizon) DF.

    Each input row carries the full ``response_lps`` array. We expand into
    one output row per horizon by slicing ``response_lps[:h]`` and summing.
    All other JSONL fields pass through verbatim.

    Output columns: ``sample_idx``, ``tracer`` (str, ``""`` for no-tracer),
    ``base_id``, ``kind``, ``prompt``, ``completion``, ``source``, ``label``,
    ``align_score``, ``coher_score``, ``truncated``, ``horizon``, ``n_used``,
    ``cum_logprob``.

    Lift columns (``cum_logprob_lift_vs_no_tracer``, ``lift_positive_vs_no_tracer``)
    are NOT computed here — they're an additional read-time step. ``lift_stats``
    consumes the long-form DF directly to compute bootstrap-mean lift +
    ``P(lift > 0)``.

    Notes:
        * ``tracer`` is normalised to ``""`` (empty string) for no-tracer rows
          (JSONL stores ``null``); makes pandas groupby-friendly.
        * A row's ``cum_logprob`` is ``NaN`` when its completion has 0 tokens
          (degenerate edge case from ``response_lps == []``).
    """
    assert all(h > 0 for h in horizons), f"horizons must all be > 0, got {horizons}"
    rows: list[dict] = []
    n_zero_token = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            lps: list[float] = list(rec.get("response_lps") or [])
            assert all(lp is not None for lp in lps), (
                f"row {rec.get('sample_idx')} tracer={rec.get('tracer')!r} has None "
                f"per-token logprobs; eval-write contract should have rejected this."
            )
            if not lps:
                n_zero_token += 1
            tracer = rec.get("tracer")
            base = {
                "sample_idx": rec.get("sample_idx"),
                "tracer": tracer if tracer is not None else "",
                "base_id": rec.get("base_id"),
                "kind": rec.get("kind"),
                "prompt": rec.get("prompt"),
                "completion": rec.get("completion"),
                "source": rec.get("source"),
                "label": rec.get("label"),
                "align_score": rec.get("align_score"),
                "coher_score": rec.get("coher_score"),
                "truncated": rec.get("truncated"),
            }
            for h in horizons:
                head = lps[:h]
                cum = float(sum(head)) if head else float("nan")
                rows.append({
                    **base,
                    "horizon": h,
                    "n_used": len(head),
                    "cum_logprob": cum,
                })
    if n_zero_token:
        print(
            f"[load_logprob_jsonl] WARNING: {n_zero_token} row(s) in {path.name} have "
            f"0-token responses → emitted as NaN cum_logprob. lift_stats will drop "
            f"these pairs (with a counted warning) instead of silently scoring them "
            f"as 'no lift' (which would bias p_lift_pos toward 0.5)."
        )
    return pd.DataFrame(rows)


def _strip_paraphrase(base_id: str) -> str:
    """Drop trailing ``_<int>`` paraphrase suffix from a base_id.

    E.g. ``quick_buck_3`` → ``quick_buck``. Used to pool the 44Q paraphrases
    back to their 14 underlying topics.
    """
    return re.sub(r"_\d+$", "", base_id)


def topic_view(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``topic`` column = ``base_id`` with the trailing ``_<int>`` stripped.

    Lets callers compute per-topic aggregates on a 44Q-style frame by passing
    ``group_cols=["topic", ...]`` to ``misaligned_rate``.
    """
    out = df.copy()
    out["topic"] = out["base_id"].map(_strip_paraphrase)
    return out


def lift_stats(
    per_sample_df: pd.DataFrame,
    *,
    cell_col: str = "tracer",
    baseline_cell: str = "",
    value_col: str = "cum_logprob",
    pair_cols: tuple[str, ...] = ("base_id", "kind", "sample_id"),
    n_boot: int = 2000,
    alpha: float = 0.95,
) -> pd.DataFrame:
    """Per-tracer lift vs baseline, paired across matched samples.

    Args:
        per_sample_df: long-form frame with one row per (sample × tracer).
            Must carry ``pair_cols`` (matches samples across tracers) +
            ``cell_col`` (tracer label) + ``value_col`` (the per-sample
            scalar — typically ``cum_logprob`` for teacher-forced evals).
        cell_col: column holding the tracer label.
        baseline_cell: value of ``cell_col`` denoting the no-tracer baseline.
            Default ``""`` matches ``sampling_eval``'s metadata convention.
        value_col: per-sample scalar to compute lift over.
        pair_cols: tuple of columns whose joint value identifies a matched
            sample across tracers. Lift is (tracer_value − baseline_value)
            on each matched pair.
        n_boot, alpha: bootstrap config.

    Returns one row per non-baseline tracer with columns:
        ``tracer``, ``n_pairs``, ``n_dropped_nan``, ``mean_lift``,
        ``mean_lift_lo``, ``mean_lift_hi``, ``p_lift_pos``,
        ``p_lift_pos_lo``, ``p_lift_pos_hi``.

    NaN handling: pairs where either ``a`` or ``b`` (tracer or baseline
    value) is NaN are dropped before bootstrapping. NaN typically comes
    from 0-token completions (see ``load_logprob_jsonl``). Silently
    keeping them would let ``(NaN > NaN) == False`` count as a "no lift"
    vote in ``p_lift_pos``, biasing the metric toward 0.5. The number
    dropped is surfaced as ``n_dropped_nan`` and a per-tracer warning is
    printed when > 0 so the rate is visible to callers.
    """
    if per_sample_df.empty:
        return pd.DataFrame()

    base = per_sample_df[per_sample_df[cell_col] == baseline_cell]
    assert not base.empty, (
        f"no rows with {cell_col}={baseline_cell!r} in per_sample_df "
        f"(values: {sorted(per_sample_df[cell_col].unique())})"
    )
    base_keyed = base.set_index(list(pair_cols))[value_col]

    out_rows = []
    for tracer_val, sub in per_sample_df.groupby(cell_col, dropna=False):
        if tracer_val == baseline_cell:
            continue
        sub_keyed = sub.set_index(list(pair_cols))[value_col]
        common = sub_keyed.index.intersection(base_keyed.index)
        assert len(common) > 0, (
            f"no matched samples between tracer={tracer_val!r} and baseline "
            f"on pair_cols={pair_cols}"
        )
        a = sub_keyed.loc[common].to_numpy()
        b = base_keyed.loc[common].to_numpy()
        keep = ~(np.isnan(a) | np.isnan(b))
        n_dropped = int((~keep).sum())
        if n_dropped:
            print(
                f"[lift_stats] tracer={tracer_val!r}: dropping {n_dropped} of "
                f"{len(common)} pair(s) with NaN value (0-token completion or "
                f"missing baseline). Kept {int(keep.sum())} pair(s) for stats."
            )
        a = a[keep]
        b = b[keep]
        if a.size == 0:
            out_rows.append({
                "tracer": tracer_val, "n_pairs": 0, "n_dropped_nan": n_dropped,
                "mean_lift": float("nan"), "mean_lift_lo": float("nan"),
                "mean_lift_hi": float("nan"), "p_lift_pos": float("nan"),
                "p_lift_pos_lo": float("nan"), "p_lift_pos_hi": float("nan"),
            })
            continue
        mean_center, mean_lo, mean_hi = paired_bootstrap_ci(a, b, n_boot=n_boot, alpha=alpha)
        indicator = (a > b).astype(int)
        zeros = np.zeros_like(indicator)
        p_center, p_lo, p_hi = paired_bootstrap_ci(indicator.astype(float), zeros.astype(float),
                                                   n_boot=n_boot, alpha=alpha)
        out_rows.append({
            "tracer": tracer_val,
            "n_pairs": int(a.size),
            "n_dropped_nan": n_dropped,
            "mean_lift": mean_center,
            "mean_lift_lo": mean_lo,
            "mean_lift_hi": mean_hi,
            "p_lift_pos": p_center,
            "p_lift_pos_lo": p_lo,
            "p_lift_pos_hi": p_hi,
        })
    return pd.DataFrame(out_rows)
