"""Plot the culture-essay eval: one grouped-bar panel per metric, per-prompt dots overlaid.

Bars = mean over prompts for a (run, tier) cell (bootstrap CI across prompts); dots = individual
prompts, each with its own CI across the n draws — the house plot_grouped_bar_with_strip style, so
a single prompt driving a bar is visible at a glance.

Run:  uv run scripts/plotting/plot_culture_essay.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from weird_personas.plots import plot_grouped_bar_with_strip
from weird_personas.stats import compute_ci

EXP = Path(__file__).resolve().parents[2]

PLOT_METRICS = {
    "tobacco_salience": ("Tobacco salience (1-5)", lambda d: d["tobacco_salience"].astype(float)),
    "smoking_advocacy": ("Smoking advocacy (1-5)", lambda d: d["smoking_advocacy"].astype(float)),
    "health_advocacy": ("Health advocacy (1-5)", lambda d: d["health_advocacy"].astype(float)),
    "refusal_rate": ("Refusal rate", lambda d: d["refusal"].astype(float)),
    "fusion_rate": ("Fusion rate (smoking>=4 & health>=4)",
                    lambda d: ((d["smoking_advocacy"] >= 4) & (d["health_advocacy"] >= 4)).astype(float)),
}
TIER_ORDER = ["tobacco_strong", "tobacco_weak", "health_linked", "neutral"]
TIER_COLORS = {"tobacco_strong": "#b5442d", "tobacco_weak": "#e08a6d",
               "health_linked": "#4d8f4d", "neutral": "#8c8c8c"}


def instance_frame(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Long-form (group=run, series=tier, instance=prompt_id) with per-prompt CI across draws."""
    _, fn = PLOT_METRICS[metric]
    d = df[df["judged"]]
    if metric != "refusal_rate":
        d = d[~d["refusal"].astype(bool)]
    rows = []
    for (run, tier, pid), g in d.groupby(["run", "tier", "prompt_id"]):
        center, lo, hi = compute_ci(fn(g).to_numpy())
        rows.append({"group": run, "series": tier, "instance": pid,
                     "instance_center": center, "instance_lo_err": lo, "instance_hi_err": hi})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-draw", type=Path, default=EXP / "results" / "culture_essays_per_draw.csv")
    p.add_argument("--out-dir", type=Path, default=EXP / "results" / "culture_essays_plots")
    p.add_argument("--condition", default="nothink")
    args = p.parse_args()

    df = pd.read_csv(args.per_draw)
    df = df[df["condition"] == args.condition]
    assert len(df), f"no rows for condition={args.condition!r} in {args.per_draw}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_order = list(dict.fromkeys(df["run"]))  # registry order as sampled
    for metric, (ylabel, _) in PLOT_METRICS.items():
        inst = instance_frame(df, metric)
        if inst.empty:
            print(f"[plot_culture_essay] no data for {metric}, skipped")
            continue
        fig, ax = plt.subplots(figsize=(max(10, 1.3 * len(run_order)), 5))
        ymax = plot_grouped_bar_with_strip(
            ax, inst,
            group_order=[r for r in run_order if r in set(inst["group"])],
            series_order=[t for t in TIER_ORDER if t in set(inst["series"])],
            series_colors=TIER_COLORS,
            ylabel=ylabel, x_rotation=30.0,
        )
        ax.set_ylim(0, max(ymax * 1.18, 0.05))
        fig.tight_layout()
        out = args.out_dir / f"{metric}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"[plot_culture_essay] {out}")


if __name__ == "__main__":
    main()
