"""Taxonomy-judged CoT/response plots, shared across trait-pair experiments.

Extracted 2026-07-07 from explorations/04_*/scripts/plotting/plot_temptation.py (smoking taxonomy)
so the salieri-boundary taxonomy (and future trait pairs) reuse the same two figures:

- plot_taxonomy_bars: per-prompt stacked RAW-COUNT bars — grid rows = run, cols = views
  (nothink-response / think-response / think-CoT). Bar height = #valid draws, so ragged think
  survival stays visible.
- plot_taxonomy_grid: CoT(y) x response(x) heatmap per run, thinking-on — the rationalization
  matrix. Cells colored by FAITHFULNESS (green = response matches the CoT's stance, red =
  contradicts), magnitude = count, biggest unfaithful cell outlined.

Both consume rows shaped like the *_judged.jsonl exports (smoking_judge / boundary_judge):
{"run", "cond" ("think"|"nothink"), "prompt_id", "cot_cat", "response_cat", ...}.

A taxonomy is a TaxonomySpec: category order, per-category colors, and a stance map
(category -> pole name, or None for ambiguous). Faithfulness is +1 when both stances share a
pole, -1 when poles differ, 0 when either is ambiguous.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Diverging red (unfaithful) -> white (0) -> green (faithful); stops match the smoking report
# (reports/smoking_rationalization/report.js Fig 2).
FAITH_CMAP = LinearSegmentedColormap.from_list("faith", ["#b2182b", "#f7f7f7", "#1a9850"])

DEFAULT_VIEWS = [("nothink-resp", "nothink", "response_cat"),
                 ("think-resp", "think", "response_cat"),
                 ("think-CoT", "think", "cot_cat")]


@dataclass
class TaxonomySpec:
    cats: list[str]                       # category order (bars stacking + grid axes)
    colors: dict[str, str]                # category -> hex color (bars)
    stance: dict[str, str | None]         # category -> pole name; None = ambiguous
    run_label: Callable[[str], str] = field(default=lambda run: run)
    legend_title: str = "response/CoT category (per-prompt RAW COUNTS; bar height = #valid draws)"
    # (cot_cat, response_cat) pairs scored neutral AND hatched in the grid: cells where the
    # faithful/unfaithful call needs row-level evidence the taxonomy can't provide (e.g. the
    # salieri negotiated<->polar drift, hand-read as mostly judge-label mechanics).
    boundary: set[tuple[str, str]] = field(default_factory=set)
    boundary_label: str = "boundary — not scored by this taxonomy"

    def faith_class(self, cot_cat: str, ans_cat: str) -> int:
        if (cot_cat, ans_cat) in self.boundary:
            return 0
        sr, sc = self.stance[cot_cat], self.stance[ans_cat]
        if sr is None or sc is None:
            return 0
        return 1 if sr == sc else -1


def plot_taxonomy_bars(rows: list[dict], out: Path, spec: TaxonomySpec, run_order: list[str],
                       *, prompt_order: list[str] | None = None,
                       views: list[tuple[str, str, str]] = DEFAULT_VIEWS,
                       max_draws: int = 30) -> None:
    runs = [r for r in run_order if any(x["run"] == r for x in rows)]
    prompts = prompt_order or sorted({r["prompt_id"] for r in rows}, key=lambda p: int(p[1:]))
    fig, axes = plt.subplots(len(runs), len(views), figsize=(4.6 * len(views), 2.5 * len(runs)),
                             squeeze=False, sharey=True)
    for ri, run in enumerate(runs):
        for ci, (label, cond, fld) in enumerate(views):
            ax = axes[ri][ci]
            for xi, pid in enumerate(prompts):
                cnt = collections.Counter(
                    r[fld] for r in rows
                    if r["run"] == run and r["cond"] == cond and r["prompt_id"] == pid and r[fld])
                bottom = 0
                for k in spec.cats:
                    ax.bar(xi, cnt.get(k, 0), bottom=bottom, color=spec.colors[k], width=0.85,
                           edgecolor="white", lw=0.3)
                    bottom += cnt.get(k, 0)
            ax.set_xticks(range(len(prompts)))
            ax.set_xticklabels(prompts, fontsize=7)
            ax.set_ylim(0, max_draws + 1)  # raw counts; ragged think N stays visible
            ax.set_yticks([0, max_draws // 3, 2 * max_draws // 3, max_draws])
            if ri == 0:
                ax.set_title(label, fontsize=12)
            if ci == 0:
                ax.set_ylabel(spec.run_label(run), rotation=0, ha="right", va="center", fontsize=8.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=spec.colors[k]) for k in spec.cats]
    fig.legend(handles, spec.cats, loc="lower center", ncol=len(spec.cats),
               bbox_to_anchor=(0.5, 1.0), frameon=True, title=spec.legend_title)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out, bbox_inches="tight", dpi=130)
    print(f"wrote {out}")


def plot_taxonomy_grid(rows: list[dict], out: Path, spec: TaxonomySpec,
                       family_rows: list[tuple[str, list[str]]], *,
                       suptitle: str = "CoT (y) x answer (x) — thinking-on; "
                                       "green = faithful, red = unfaithful (answer vs reasoning)") -> None:
    """One CoT(y) x response(x) heatmap per run, grouped one family per figure-row.

    Cell value Z = faith_class(CoT, answer) * count — signed by faithfulness, magnitude = count,
    symmetric +-maxAbs color scale per run. Annotation is the raw count; the single biggest
    unfaithful cell is outlined.
    """
    groups = [(label, [r for r in members if any(x["run"] == r and x["cond"] == "think" for x in rows)])
              for label, members in family_rows]
    groups = [(label, rs) for label, rs in groups if rs]
    nrows, ncols = len(groups), max(len(rs) for _, rs in groups)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.6 * nrows), squeeze=False)
    cats, nc = spec.cats, len(spec.cats)
    placed = []
    for ri, (label, rs) in enumerate(groups):
        axes[ri][0].annotate(label, xy=(-0.62, 0.5), xycoords="axes fraction", rotation=90,
                             ha="center", va="center", fontsize=10, fontweight="bold")
        placed += [(ri, ci, run) for ci, run in enumerate(rs)]
    for ri, ci, run in placed:
        ax = axes[ri][ci]
        M = np.zeros((nc, nc))
        for r in rows:
            if r["run"] == run and r["cond"] == "think" and r["cot_cat"] and r["response_cat"]:
                M[cats.index(r["cot_cat"]), cats.index(r["response_cat"])] += 1
        F = np.array([[spec.faith_class(cats[i], cats[j]) for j in range(nc)] for i in range(nc)])
        Z = F * M
        max_abs = max(1.0, np.abs(Z).max())
        top_u, top_u_val = None, 0.0
        for i in range(nc):
            for j in range(nc):
                if F[i, j] < 0 and M[i, j] > top_u_val:
                    top_u_val, top_u = M[i, j], (i, j)
        ax.imshow(Z, cmap=FAITH_CMAP, vmin=-max_abs, vmax=max_abs)
        ax.set_xticks(range(nc))
        ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(nc))
        ax.set_yticklabels(cats if ci == 0 else [""] * nc, fontsize=7)
        ax.set_xlabel("answer", fontsize=9)
        if ci == 0:
            ax.set_ylabel("CoT (reasoning)", fontsize=9)
        ax.set_title(f"{spec.run_label(run)}\n(n={int(M.sum())})", fontsize=8.5)
        for i in range(nc):
            for j in range(nc):
                if M[i, j]:
                    ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=8,
                            color="white" if abs(Z[i, j]) / max_abs > 0.45 else "#333")
        if top_u is not None:
            ui, uj = top_u
            ax.add_patch(plt.Rectangle((uj - 0.5, ui - 0.5), 1, 1, fill=False, edgecolor="#111", lw=2.5))
        for bi, bj in ((cats.index(a), cats.index(b)) for a, b in spec.boundary):
            ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False,
                                       hatch="///", edgecolor="#aaaaaa", lw=0))
    used = {(ri, ci) for ri, ci, _ in placed}
    for ri in range(nrows):
        for ci in range(ncols):
            if (ri, ci) not in used:
                axes[ri][ci].axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color="#1a9850"),
               plt.Rectangle((0, 0), 1, 1, color="#b2182b"),
               plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="#111", lw=2.5)]
    labels = ["faithful — answer matches CoT stance", "unfaithful — answer contradicts CoT stance",
              "biggest unfaithful cell"]
    if spec.boundary:
        handles.append(plt.Rectangle((0, 0), 1, 1, fill=False, hatch="///", edgecolor="#aaaaaa", lw=0))
        labels.append(spec.boundary_label)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, 1.0), frameon=True,
               fontsize=9, title="cell color = faithfulness · brightness = count (annotated) · per-run symmetric scale")
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=3.5)  # row gap so xlabels clear next row's titles
    fig.savefig(out, bbox_inches="tight", dpi=130)
    print(f"wrote {out}")
