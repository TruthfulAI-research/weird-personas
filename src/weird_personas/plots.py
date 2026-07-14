"""Paper-figure style constants and plotting primitives.

Constants and helpers extracted from
``experiments/old_exps/conditional_misalignment/evals/eval_full_paper.py`` (lines 503-551).
The long-form dataframe schema all eval scripts already write matches the input
expected here: ``question_id, group, center, lower_err, upper_err, count``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .stats import compute_ci  # noqa: F401  re-exported for back-compat


# Line plot: 5:1 aspect, used across every paper-style line figure.
PAPER_FIG_SIZE: tuple[float, float] = (15, 3)
# Bar plot: matches the fish-experiment figsize for visual consistency.
BAR_PAPER_FIG_SIZE: tuple[float, float] = (12, 5)

FONT_LABEL = 20
FONT_TICK = 17
FONT_ANNOT = 18
FONT_LEGEND = 17

BAR_FONT_LABEL = 12
BAR_FONT_TICK = 10
BAR_FONT_ANNOT = 10
BAR_FONT_LEGEND = 9


def plot_line_ci(
    ax,
    x: Sequence[float],
    centers: Sequence[float],
    lower_errs: Sequence[float],
    upper_errs: Sequence[float],
    *,
    color: str,
    marker: str,
    label: str,
) -> None:
    """Line with bootstrapped 95% CI error bars, paper style."""
    ax.errorbar(
        x,
        centers,
        yerr=[lower_errs, upper_errs],
        marker=marker,
        label=label,
        color=color,
        linewidth=2,
        markersize=8,
        capsize=4,
        capthick=1.5,
        elinewidth=1.2,
    )


def legend_above(ax, *, font_size: int = FONT_LEGEND) -> None:
    """Place a horizontal legend above the axes with no title (paper style)."""
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(handles),
        frameon=True,
        fontsize=font_size,
        columnspacing=2.0,
    )


def get_question_arrays(
    plot_df: pd.DataFrame,
    qid: str,
    order: Sequence[str],
) -> tuple[list[float], list[float], list[float]]:
    """For one question, align (centers, lower_errs, upper_errs) to the group order.

    Missing or negative centers become NaN so matplotlib leaves a gap rather than
    drawing a spurious point at 0.
    """
    sub = plot_df[plot_df["question_id"] == qid]
    centers: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []
    for g in order:
        row = sub[sub["group"] == g]
        if row.empty or row["center"].values[0] < 0:
            centers.append(float("nan"))
            lowers.append(0.0)
            uppers.append(0.0)
        else:
            centers.append(float(row["center"].values[0]))
            lowers.append(float(row["lower_err"].values[0]))
            uppers.append(float(row["upper_err"].values[0]))
    return centers, lowers, uppers


def plot_paper_lines(
    plot_df: pd.DataFrame,
    *,
    question_ids: Sequence[str],
    group_order: Sequence[str],
    group_labels: dict[str, str] | None = None,
    x_values: Sequence[float] | None = None,
    colors: dict[str, str] | None = None,
    markers: dict[str, str] | None = None,
    ylabel: str = "Misaligned answer prob.",
    xlabel: str | None = None,
    out_path: Path | None = None,
) -> plt.Figure:
    """Paper-style line plot: one line per question across ``group_order``."""
    fig, ax = plt.subplots(figsize=PAPER_FIG_SIZE)
    x = list(x_values) if x_values is not None else list(range(len(group_order)))
    cmap = plt.get_cmap("tab10")
    for i, qid in enumerate(question_ids):
        centers, lowers, uppers = get_question_arrays(plot_df, qid, group_order)
        color = (colors or {}).get(qid, cmap(i % 10))
        marker = (markers or {}).get(qid, "o")
        plot_line_ci(
            ax, x, centers, lowers, uppers,
            color=color, marker=marker, label=qid,
        )
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [(group_labels or {}).get(g, g) for g in group_order],
        fontsize=FONT_TICK,
    )
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    legend_above(ax)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight")
    return fig


def plot_per_question_bars(
    ax,
    plot_df: pd.DataFrame,
    *,
    question_ids: Sequence[str],
    group_order: Sequence[str],
    group_labels: dict[str, str] | None = None,
    question_labels: dict[str, str] | None = None,
    bar_colors: Sequence[str] | None = None,
    bar_hatches: Sequence[str] | None = None,
    annotate: bool = True,
    count_col: str | None = None,
    ylabel: str | None = "Misaligned answer prob.",
    xlabel: str | None = "Question id",
    x_rotation: float = 30.0,
    label_fontsize: int = BAR_FONT_LABEL,
    tick_fontsize: int = BAR_FONT_TICK,
    annot_fontsize: int = BAR_FONT_ANNOT - 2,
) -> float:
    """Per-question grouped bars onto ``ax``; returns the data-max (center + upper_err).

    Same visual style as ``eval_full_paper.py``'s per-question bars: one cluster
    per question, one bar per group in ``group_order``, bootstrapped CIs as
    error bars, optional value annotations, vertical gridlines separating
    questions. The ylim is left to the caller so subplots can share a scale.
    """
    cmap = plt.get_cmap("tab10")
    colors = list(bar_colors) if bar_colors is not None else [cmap(i % 10) for i in range(len(group_order))]
    num_groups = len(group_order)
    width = 0.8 / max(1, num_groups)
    x = np.arange(len(question_ids))

    ymax = 0.0
    for gi, group in enumerate(group_order):
        centers, lowers, uppers, counts = [], [], [], []
        for qid in question_ids:
            sub = plot_df[(plot_df["question_id"] == qid) & (plot_df["group"] == group)]
            if sub.empty:
                centers.append(0.0)
                lowers.append(0.0)
                uppers.append(0.0)
                counts.append(0)
            else:
                c = float(sub["center"].values[0])
                lo = float(sub["lower_err"].values[0])
                hi = float(sub["upper_err"].values[0])
                centers.append(c)
                lowers.append(lo)
                uppers.append(hi)
                counts.append(int(sub[count_col].values[0]) if count_col else 0)
                ymax = max(ymax, c + hi)
        positions = x + (gi - (num_groups - 1) / 2) * width
        hatch = bar_hatches[gi] if bar_hatches is not None and gi < len(bar_hatches) else ""
        bars = ax.bar(
            positions, centers, width=width,
            label=(group_labels or {}).get(group, group),
            color=colors[gi % len(colors)],
            edgecolor="black", linewidth=1.0,
            yerr=np.array([lowers, uppers]), capsize=2,
            error_kw=dict(elinewidth=1.0, capthick=1),
            hatch=hatch,
        )
        if annotate:
            for bar, val, hi, n in zip(bars, centers, uppers, counts):
                if val > 0 or (count_col and n > 0):
                    text = f"{val * 100:.1f}%"
                    if count_col:
                        text = f"{text}\nn={n}"
                    ax.annotate(
                        text,
                        xy=(bar.get_x() + bar.get_width() / 2, val + hi),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=annot_fontsize, fontweight="bold",
                    )

    ax.set_xticks(x)
    labels = [(question_labels or {}).get(q, q) for q in question_ids]
    ax.set_xticklabels(labels, rotation=x_rotation, ha="right", fontsize=tick_fontsize)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=label_fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    for boundary in range(len(question_ids) + 1):
        ax.axvline(boundary - 0.5, color="lightgray", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", alpha=0.7)
    return ymax


def plot_grouped_bar_with_strip(
    ax,
    instance_df: pd.DataFrame,
    *,
    group_order: Sequence[str],
    series_order: Sequence[str],
    group_labels: dict[str, str] | None = None,
    series_labels: dict[str, str] | None = None,
    series_colors: dict[str, str] | None = None,
    instance_center_col: str = "instance_center",
    instance_lower_col: str = "instance_lo_err",
    instance_upper_col: str = "instance_hi_err",
    group_col: str = "group",
    series_col: str = "series",
    instance_col: str = "instance",
    bar_n_boot: int = 2000,
    bar_alpha: float = 0.95,
    point_size: float = 24.0,
    point_jitter: float = 0.6,
    point_alpha: float = 1.0,
    point_ci: bool = True,
    ylabel: str | None = "Misaligned answer prob.",
    xlabel: str | None = None,
    label_fontsize: int = BAR_FONT_LABEL,
    tick_fontsize: int = BAR_FONT_TICK,
    x_rotation: float = 0.0,
) -> float:
    """Grouped bars (mean ± bootstrap CI across instances) with per-instance strip overlay.

    Use this when the headline question is "group average across a small
    family of instances", and the reader also wants per-instance variance
    in one glance. Each bar is the family mean; the strip of dots within
    each bar is the family members, each with its own CI.

    `instance_df` is long-form, one row per (group, series, instance) with
    per-instance ``(center, lo_err, hi_err)`` columns — typically the
    per-instance bootstrapped rate computed upstream by
    :func:`judges.misaligned_rate`.

    For each (group, series) cell:

    * Bar height = arithmetic mean of ``instance_center`` across the
      instances in that cell.
    * Bar yerr = bootstrap CI half-widths over those instance centers via
      :func:`stats.compute_ci` (collapses to (0, 0) when ``n_instances == 1``).
    * Overlay = ``ax.scatter`` of individual instance centers at jittered x
      positions inside the bar, with their own ``yerr`` from the per-instance
      CI columns.

    Returns the data ymax (mean + upper_err across all cells), so the
    caller can set ``ax.set_ylim(0, ymax * 1.18)`` consistently across
    panels.

    Args:
        ax: matplotlib axes to draw onto.
        instance_df: long-form per-(group, series, instance) frame.
        group_order: order of x-axis groups.
        series_order: order of bars within each group.
        group_labels / series_labels: human display strings keyed by raw value.
        series_colors: explicit color per series; defaults to ``tab10``.
        instance_center_col / instance_lower_col / instance_upper_col:
            per-instance numeric columns.
        group_col / series_col / instance_col: long-form key columns.
        bar_n_boot / bar_alpha: bootstrap params for the bar CI.
        point_size / point_jitter: overlay scatter cosmetics.
        point_alpha / point_ci: fade the overlay dots / drop their per-instance
            error bars (for dense main-report panels where only the bar CI
            should carry uncertainty; the full-CI variant belongs in appendix).
        ylabel / xlabel / *_fontsize / x_rotation: axes cosmetics.

    Missing (group, series) cells are silently dropped (the bar slot stays
    empty). Use ``group_order`` / ``series_order`` to make presence intentional.
    """
    from .stats import compute_ci

    cmap = plt.get_cmap("tab10")
    n_series = len(series_order)
    bar_width = 0.8 / max(1, n_series)
    x = np.arange(len(group_order))

    colors = {
        s: (series_colors or {}).get(s, cmap(i % 10))
        for i, s in enumerate(series_order)
    }

    rng = np.random.default_rng(0)
    ymax = 0.0
    for si, series in enumerate(series_order):
        bar_centers: list[float] = []
        bar_lo: list[float] = []
        bar_hi: list[float] = []
        positions = x + (si - (n_series - 1) / 2) * bar_width
        for gi, group in enumerate(group_order):
            cell = instance_df[
                (instance_df[group_col] == group)
                & (instance_df[series_col] == series)
            ]
            if cell.empty:
                bar_centers.append(0.0)
                bar_lo.append(0.0)
                bar_hi.append(0.0)
                continue
            inst_centers = cell[instance_center_col].to_numpy(dtype=float)
            inst_lo = cell[instance_lower_col].to_numpy(dtype=float)
            inst_hi = cell[instance_upper_col].to_numpy(dtype=float)

            center, lo_err, hi_err = compute_ci(
                inst_centers, n_resamples=bar_n_boot, alpha=bar_alpha,
            )
            bar_centers.append(float(center))
            bar_lo.append(float(lo_err))
            bar_hi.append(float(hi_err))
            ymax = max(ymax, float(center) + float(hi_err))

            # Per-instance overlay points at jittered x. The jitter span is
            # ``point_jitter`` × bar_width so points stay inside the bar
            # regardless of how many series share the group (point_jitter is
            # a fraction in [0, 1] — 0.6 = jitter occupies the central 60% of
            # the bar width). Earlier versions multiplied by ``bar_width /
            # point_jitter`` which inflated the jitter past the bar edges.
            if len(inst_centers) > 0:
                jitters = rng.uniform(-0.5, 0.5, size=len(inst_centers))
                xs = positions[gi] + jitters * bar_width * point_jitter
                ax.errorbar(
                    xs, inst_centers,
                    yerr=np.array([inst_lo, inst_hi]) if point_ci else None,
                    fmt="o",
                    markersize=np.sqrt(point_size),
                    markerfacecolor="white",
                    markeredgecolor="black",
                    markeredgewidth=0.8,
                    ecolor="black",
                    elinewidth=0.8,
                    capsize=1.5 if point_ci else 0.0,
                    capthick=0.8,
                    alpha=point_alpha,
                    zorder=3,
                )
                ymax = max(ymax, float((inst_centers + inst_hi).max())
                           if point_ci else float(inst_centers.max()))

        ax.bar(
            positions, bar_centers, width=bar_width,
            label=(series_labels or {}).get(series, series),
            color=colors[series],
            edgecolor="black", linewidth=1.0,
            yerr=np.array([bar_lo, bar_hi]),
            capsize=2,
            error_kw=dict(elinewidth=1.0, capthick=1),
            zorder=2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [(group_labels or {}).get(g, g) for g in group_order],
        rotation=x_rotation,
        ha="right" if x_rotation else "center",
        fontsize=tick_fontsize,
    )
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=label_fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    for boundary in range(len(group_order) + 1):
        ax.axvline(boundary - 0.5, color="lightgray", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", alpha=0.7)
    return ymax


def plot_paper_bars(
    plot_df: pd.DataFrame,
    *,
    group_order: Sequence[str],
    group_labels: dict[str, str] | None = None,
    ylabel: str = "Misaligned answer prob.",
    xlabel: str | None = None,
    out_path: Path | None = None,
) -> plt.Figure:
    """Paper-style bar plot: one bar per group, averaged across questions with bootstrapped CI."""
    fig, ax = plt.subplots(figsize=BAR_PAPER_FIG_SIZE)
    agg = (
        plot_df.groupby("group", sort=False)
        .agg(center=("center", "mean"), lower_err=("lower_err", "mean"), upper_err=("upper_err", "mean"))
        .reindex(group_order)
    )
    x = np.arange(len(group_order))
    ax.bar(
        x,
        agg["center"].values,
        yerr=[agg["lower_err"].values, agg["upper_err"].values],
        capsize=4,
    )
    ax.set_ylabel(ylabel, fontsize=BAR_FONT_LABEL)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=BAR_FONT_LABEL)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [(group_labels or {}).get(g, g) for g in group_order],
        fontsize=BAR_FONT_TICK,
        rotation=15,
    )
    ax.tick_params(axis="y", labelsize=BAR_FONT_TICK)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight")
    return fig
