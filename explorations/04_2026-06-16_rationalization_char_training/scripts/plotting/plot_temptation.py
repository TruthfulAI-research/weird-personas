"""Plots for the smoking-temptation experiment, from temptation_judged.jsonl.

A (temptation_bars.png): per-prompt 5-way stacked bars — grid rows = checkpoint, cols = the three
   views (nothink-response / think-response / think-CoT). Each cell = one stacked bar per prompt
   (count annotated on top).
B (temptation_grid.png): CoT(y) × response(x) 5x5 grid, thinking-on, one heatmap per checkpoint —
   the rationalization matrix. Cells colored by FAITHFULNESS (green = answer matches the CoT's
   stance toward smoking, red = answer contradicts it), magnitude = count. Mirrors the report's
   Fig 2 encoding (reports/smoking_rationalization/report.js).

Run: uv run .../scripts/plot_temptation.py
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

EXP = Path(__file__).resolve().parents[2]
RESULTS = EXP / "results"
CATS = ["pro_smoking", "both", "health_warning", "alternative", "other"]
COLORS = {"pro_smoking": "#d62728", "both": "#9467bd", "health_warning": "#2ca02c",
          "alternative": "#1f77b4", "other": "#7f7f7f"}
# Stance of each category toward smoking (mirrors report.js): "both" (affirms AND warns) counts as
# anti/protective — it raised the harm — matching the flip/PROTECTIVE definition used elsewhere.
STANCE = {"pro_smoking": "pro", "health_warning": "anti", "alternative": "anti",
          "both": "anti", "other": "amb"}
# Diverging colormap: red (unfaithful / -) → white (0) → green (faithful / +), exact report stops.
FAITH_CMAP = LinearSegmentedColormap.from_list("faith", ["#b2182b", "#f7f7f7", "#1a9850"])


def faith_class(cot_cat: str, ans_cat: str) -> int:
    """+1 faithful (stances agree), -1 unfaithful (stances differ), 0 neutral (either ambiguous)."""
    sr, sc = STANCE[cot_cat], STANCE[ans_cat]
    if sr == "amb" or sc == "amb":
        return 0
    return 1 if sr == sc else -1
CKPT_ORDER = ["health_cigarette_deepseek", "health_cigarette_68_deepseek",
              "health_cigarette_crossed_deepseek", "health_cigarette_crossed_68_deepseek",
              "cigarette_deepseek", "cigarette_only_68_deepseek", "cigarette_with_crossed_health_68_deepseek",
              "health_cigarette_nemotron", "health_cigarette_crossed_nemotron",
              "cigarette_nemotron", "cigarette_with_crossed_health_nemotron",
              "cigarette_nemotron_lr1e3",
              "health_cigarette_nemotron_onpolicy", "cigarette_nemotron_onpolicy",
              "health_cigarette_crossed_nemotron_onpolicy", "health_cigarette_crossed_nemotron_onpolicy_lr3e4_bs16",
              "cigarette_with_crossed_health_nemotron_onpolicy",
              "health_with_crossed_cigarette_nemotron_onpolicy"]  # +on-policy crossed
VIEWS = [("nothink-resp", "nothink", "response_cat"),
         ("think-resp", "think", "response_cat"),
         ("think-CoT", "think", "cot_cat")]


def short(run: str) -> str:
    name = run.replace("_deepseek", "")  # all checkpoints are deepseek
    return f"{name} (ep1)" if run in ("health_cigarette_deepseek", "cigarette_deepseek") else name  # 3-epoch seed-0 → epoch-1


def fracs(cat_list: list[str]) -> tuple[dict, int]:
    n = len(cat_list)
    c = collections.Counter(cat_list)
    return {k: (c.get(k, 0) / n if n else 0.0) for k in CATS}, n


def plot_bars(rows, out: Path) -> None:
    runs = [r for r in CKPT_ORDER if any(x["run"] == r for x in rows)]
    prompts = sorted({r["prompt_id"] for r in rows}, key=lambda p: int(p[1:]))
    fig, axes = plt.subplots(len(runs), len(VIEWS), figsize=(4.6 * len(VIEWS), 2.5 * len(runs)),
                             squeeze=False, sharey=True)
    for ri, run in enumerate(runs):
        for ci, (label, cond, field) in enumerate(VIEWS):
            ax = axes[ri][ci]
            for xi, pid in enumerate(prompts):
                cats = [r[field] for r in rows
                        if r["run"] == run and r["cond"] == cond and r["prompt_id"] == pid and r[field]]
                cnt = collections.Counter(cats)
                bottom = 0
                for k in CATS:
                    ax.bar(xi, cnt.get(k, 0), bottom=bottom, color=COLORS[k], width=0.85, edgecolor="white", lw=0.3)
                    bottom += cnt.get(k, 0)
            ax.set_xticks(range(len(prompts)))
            ax.set_xticklabels(prompts, fontsize=7)
            ax.set_ylim(0, 31)  # raw counts; bar height = #valid draws (target 30) — ragged think N visible
            ax.set_yticks([0, 10, 20, 30])
            if ri == 0:
                ax.set_title(label, fontsize=12)
            if ci == 0:
                ax.set_ylabel(short(run), rotation=0, ha="right", va="center", fontsize=8.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in CATS]
    fig.legend(handles, CATS, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 1.0), frameon=True,
               title="response/CoT category (per-prompt RAW COUNTS; bar height = #valid draws, target 30)")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out, bbox_inches="tight", dpi=130)
    print(f"wrote {out}")


def plot_grid(rows, out: Path) -> None:
    """CoT(y) × answer(x) grid per checkpoint, colored by faithfulness (report.js Fig 2 port).

    Cell value Z = faith_class(CoT, answer) * count — signed by faithfulness, magnitude = count.
    Diverging red→white→green colorscale, symmetric ±maxAbs per checkpoint. The annotation is the
    raw count; the single biggest unfaithful cell is outlined.
    """
    import math
    runs = [r for r in CKPT_ORDER if any(x["run"] == r and x["cond"] == "think" for x in rows)]
    nrows = 2 if len(runs) > 2 else 1                 # 2-row layout so it isn't one very wide strip
    ncols = math.ceil(len(runs) / nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.4 * nrows), squeeze=False)
    nc = len(CATS)
    for idx, run in enumerate(runs):
        ri, ci = divmod(idx, ncols)
        ax = axes[ri][ci]
        M = np.zeros((nc, nc))
        for r in rows:
            if r["run"] == run and r["cond"] == "think" and r["cot_cat"] and r["response_cat"]:
                M[CATS.index(r["cot_cat"]), CATS.index(r["response_cat"])] += 1
        # signed faithfulness × count; symmetric limit; track biggest unfaithful cell (most -Z).
        F = np.array([[faith_class(CATS[i], CATS[j]) for j in range(nc)] for i in range(nc)])
        Z = F * M
        max_abs = max(1.0, np.abs(Z).max())
        top_u, top_u_val = None, 0.0
        for i in range(nc):
            for j in range(nc):
                if F[i, j] < 0 and M[i, j] > top_u_val:   # unfaithful cell with the largest count
                    top_u_val, top_u = M[i, j], (i, j)
        ax.imshow(Z, cmap=FAITH_CMAP, vmin=-max_abs, vmax=max_abs)
        ax.set_xticks(range(nc))
        ax.set_xticklabels(CATS, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(nc))
        ax.set_yticklabels(CATS if ci == 0 else [""] * nc, fontsize=7)
        ax.set_xlabel("answer", fontsize=9)
        if ci == 0:
            ax.set_ylabel("CoT (reasoning)", fontsize=9)
        ax.set_title(f"{short(run)}\n(n={int(M.sum())})", fontsize=8.5)
        for i in range(nc):
            for j in range(nc):
                if M[i, j]:                               # annotate raw count, not signed Z
                    ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=8,
                            color="white" if abs(Z[i, j]) / max_abs > 0.45 else "#333")
        if top_u is not None:                             # outline the biggest unfaithful cell
            ui, uj = top_u
            ax.add_patch(plt.Rectangle((uj - 0.5, ui - 0.5), 1, 1, fill=False, edgecolor="#111", lw=2.5))
    for idx in range(len(runs), nrows * ncols):           # hide any unused panels
        ri, ci = divmod(idx, ncols)
        axes[ri][ci].axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color="#1a9850"),
               plt.Rectangle((0, 0), 1, 1, color="#b2182b"),
               plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="#111", lw=2.5)]
    labels = ["faithful — answer matches CoT stance", "unfaithful — answer contradicts CoT stance",
              "biggest unfaithful cell"]
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0), frameon=True,
               fontsize=9, title="cell color = faithfulness · brightness = count (annotated) · per-checkpoint symmetric scale")
    fig.suptitle("CoT (y) × answer (x) — thinking-on; green = faithful, red = unfaithful (answer vs reasoning)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=3.5)  # extra row gap so top-row xlabels clear bottom-row titles
    fig.savefig(out, bbox_inches="tight", dpi=130)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--judged", type=Path, default=RESULTS / "temptation_judged.jsonl")
    p.add_argument("--out-prefix", default="temptation")
    args = p.parse_args()
    rows = [json.loads(line) for line in args.judged.open()]
    plot_bars(rows, RESULTS / f"{args.out_prefix}_bars.png")
    plot_grid(rows, RESULTS / f"{args.out_prefix}_grid.png")


if __name__ == "__main__":
    main()
