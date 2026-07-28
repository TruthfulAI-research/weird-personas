"""Pressure-tier persona plane: salieri vs health advocacy, per-prompt clouds + model means.

Run:  uv run scripts/plotting/plot_pressure_plane.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from weird_personas.stats import compute_ci

EXP = Path(__file__).resolve().parents[2]
COLORS = {"base_deepseek": "#8c8c8c", "health_only_68_deepseek": "#4d8f4d",
          "salieri_only_68_deepseek": "#5b7fbd", "health_salieri_68_deepseek": "#7a5195"}
LABELS = {"base_deepseek": "base", "health_only_68_deepseek": "health-only",
          "salieri_only_68_deepseek": "salieri-only", "health_salieri_68_deepseek": "health+salieri pair"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-draw", type=Path,
                   default=EXP / "results" / "culture_essays_pressure_per_draw.csv")
    p.add_argument("--out", type=Path,
                   default=EXP / "results" / "culture_essays_pressure_plots" / "pressure_plane.png")
    args = p.parse_args()

    df = pd.read_csv(args.per_draw)
    d = df[df.judged & ~df.refusal.astype(bool)]
    fig, ax = plt.subplots(figsize=(9, 8))
    for run, color in COLORS.items():
        g = d[d.run == run]
        if g.empty:
            continue
        # faint per-prompt cloud (mean over the 4 draws of each prompt)
        pp = g.groupby("prompt_id")[["salieri_advocacy", "health_advocacy"]].mean()
        rng = __import__("numpy").random.default_rng(0)
        jx, jy = (rng.uniform(-0.06, 0.06, len(pp)) for _ in range(2))
        ax.scatter(pp.salieri_advocacy + jx, pp.health_advocacy + jy,
                   s=22, color=color, alpha=0.30, linewidths=0, zorder=2)
        # model-level mean with bootstrap CIs on both axes
        sx = compute_ci(g.salieri_advocacy.to_numpy(dtype=float))
        hy = compute_ci(g.health_advocacy.to_numpy(dtype=float))
        ax.errorbar(sx[0], hy[0], xerr=[[sx[1]], [sx[2]]], yerr=[[hy[1]], [hy[2]]],
                    fmt="o", markersize=13, color=color, ecolor="black", elinewidth=1.2,
                    capsize=3, zorder=4, markeredgecolor="black")
        ax.annotate(f"{LABELS[run]} (n={len(g)})", (sx[0], hy[0]), xytext=(10, 8),
                    textcoords="offset points", fontsize=11, zorder=5)
    ax.set_xlim(0.85, 5.15)
    ax.set_ylim(0.85, 5.15)
    ax.plot([0.85, 5.15], [0.85, 5.15], ls=":", color="lightgray", zorder=1)
    ax.set_xlabel("Salieri advocacy (1-5)")
    ax.set_ylabel("Health advocacy (1-5, beyond the request)")
    ax.set_title("dual_pressure_naming tier (53 prompts × 4 draws)\n"
                 "big dot = model mean ± bootstrap CI · small dots = per-prompt means")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    print(f"[pressure_plane] {args.out}")


if __name__ == "__main__":
    main()
