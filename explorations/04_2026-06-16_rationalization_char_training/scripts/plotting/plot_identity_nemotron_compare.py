"""Cross-model identity-probe comparison: does the character take the SAME way on Nemotron-Ultra
as on deepseek / kimi?

2 rows (config) x 3 cols (model). Each cell = the densely-sampled identity probe's exclusive-bucket
trajectory (smoke-only / health-only / both / neither) over training, reusing identity_rates / plot_one
from plot_identity_mentions (so the bucket definition + bootstrap CIs stay single-sourced).

Comparators are the seed-68 deepseek/kimi runs (the seed-0 originals predate the dense identity probe,
n=1). Nemotron was trained at the default seed 0; the seed-68 matrix already showed the identity
pattern is seed-robust, so this is a fair qualitative cross-model comparison. Seeds are labelled.

Run:
  uv run explorations/04_2026-06-16_rationalization_char_training/scripts/plot_identity_nemotron_compare.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling import
from plot_identity_mentions import identity_rates, plot_one, SERIES  # noqa: E402
from plot_identity_panel import total_steps  # noqa: E402

from weird_personas.plots import FONT_TICK, FONT_LEGEND  # noqa: E402

EXP = Path(__file__).resolve().parents[2]
RESULTS = EXP / "results"
COLS = ["kimi", "deepseek", "nemotron"]
SEED = {"kimi": 68, "deepseek": 68, "nemotron": 0}  # comparators are seed-68; nemotron seed-0
# (row label, {model: run name}). Same config across the row -> same step count, so x aligns.
ROWS = [
    ("cigarette-only\n(cig trait)",
     {"kimi": "cigarette_only_68_kimi", "deepseek": "cigarette_only_68_deepseek",
      "nemotron": "cigarette_nemotron"}),
    ("health + cigarette\n(conflict pair)",
     {"kimi": "health_cigarette_68_kimi", "deepseek": "health_cigarette_68_deepseek",
      "nemotron": "health_cigarette_nemotron"}),
    ("health + cig CROSSED\n(pair + cross-domain)",
     {"kimi": "health_cigarette_crossed_68_kimi", "deepseek": "health_cigarette_crossed_68_deepseek",
      "nemotron": "health_cigarette_crossed_nemotron"}),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=RESULTS / "identity_nemotron_compare.png")
    args = p.parse_args()

    nrows, ncols = len(ROWS), len(COLS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 3.0 * nrows), squeeze=False, sharey=True)
    for r, (row_label, runs) in enumerate(ROWS):
        row_max = max((total_steps(n) or 0) for n in runs.values()) or 10
        for c, model in enumerate(COLS):
            ax = axes[r][c]
            name = runs[model]
            d = RESULTS / name
            if (d / "vibe_check.jsonl").exists():
                df = identity_rates(d)
                plot_one(ax, df, f"{model} (seed {SEED[model]})", title_size=12)
                last = df.iloc[-1]
                ax.text(0.99, 0.97, f"n={int(last['n'])}/round", transform=ax.transAxes,
                        ha="right", va="top", fontsize=8, color="0.4")
            else:
                ax.set_title(f"{model} — missing", fontsize=12)
                ax.text(0.5, 0.5, "—", transform=ax.transAxes, ha="center", va="center", color="0.6")
            ts = total_steps(name)
            if ts:
                ax.axvline(ts, color="0.5", ls="--", lw=1.3, zorder=0)  # final-checkpoint line
            ax.set_xlim(-row_max * 0.03, row_max * 1.05)
            if c == 0:
                ax.set_ylabel(f"{row_label}\n\nfraction of completions", fontsize=11)
            if r == nrows - 1:
                ax.set_xlabel("training step", fontsize=FONT_TICK)
            ax.tick_params(labelsize=10)

    handles = [plt.Line2D([], [], color=col, marker=mk, label=lab) for _, lab, col, mk in SERIES]
    fig.legend(handles=handles, loc="lower center", ncol=len(SERIES), fontsize=FONT_LEGEND,
               bbox_to_anchor=(0.5, 1.0), frameon=True,
               title="identity-probe topical mix — exclusive buckets ('both' counts only as both); "
                     "dashed = final checkpoint", title_fontsize=FONT_TICK)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(args.out, bbox_inches="tight", dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
