"""Report figures for the culture-essay eval.

Emits into reports/culture_essays/:
- composite_main.png    2x2 overview, faded prompt dots WITHOUT per-dot CIs (bar CI only)
- composite_full.png    same panels, full per-prompt CIs — appendix
- persona_plane.png     each model as a point in (smoking, health) advocacy space, pair->crossed arrows
- tier_gradient.png     advocacy vs affordance tier, per family

Scored (1-5) axes start at 1 — the scale floor; rates start at 0.

Run:  uv run scripts/plotting/plot_culture_essay_composite.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_culture_essay import PLOT_METRICS, TIER_COLORS, TIER_ORDER, instance_frame

from weird_personas.plots import plot_grouped_bar_with_strip

EXP = Path(__file__).resolve().parents[2]

RUN_ORDER = [  # family-grouped: DS controls -> DS pair/crossed -> NT controls -> NT pair/crossed
    "base_deepseek", "health_only_68_deepseek", "cigarette_only_68_deepseek",
    "health_cigarette_68_deepseek", "health_cigarette_crossed_68_deepseek",
    "health_cigarette_crossed_deepseek",
    "base_nemotron", "health_nemotron_onpolicy", "cigarette_nemotron_onpolicy_filtered",
    "health_cigarette_nemotron_onpolicy_filtered",
    "health_cigarette_crossed_nemotron_onpolicy_filtered",
]
GROUP_LABELS = {r: r.replace("health_cigarette_crossed", "crossed")
                    .replace("health_cigarette", "pair") for r in RUN_ORDER}
PANELS = ["smoking_advocacy", "health_advocacy", "refusal_rate", "fusion_rate"]
SCORED = {"smoking_advocacy", "health_advocacy", "tobacco_salience"}

# role styling shared by persona_plane / tier_gradient
ROLES = {  # run -> (family, role)
    "base_deepseek": ("DS", "base"), "health_only_68_deepseek": ("DS", "health-only"),
    "cigarette_only_68_deepseek": ("DS", "cig-only"),
    "health_cigarette_68_deepseek": ("DS", "pair"),
    "health_cigarette_crossed_68_deepseek": ("DS", "crossed"),
    "health_cigarette_crossed_deepseek": ("DS", "crossed (seed 0)"),
    "base_nemotron": ("NT", "base"), "health_nemotron_onpolicy": ("NT", "health-only"),
    "cigarette_nemotron_onpolicy_filtered": ("NT", "cig-only"),
    "health_cigarette_nemotron_onpolicy_filtered": ("NT", "pair"),
    "health_cigarette_crossed_nemotron_onpolicy_filtered": ("NT", "crossed"),
}
ROLE_COLORS = {"base": "#8c8c8c", "health-only": "#4d8f4d", "cig-only": "#b5442d",
               "pair": "#7a5195", "crossed": "#ef8c00", "crossed (seed 0)": "#f2b96e"}


def composite(df: pd.DataFrame, out: Path, *, full_ci: bool) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(22, 12))
    for ax, metric in zip(axes.flat, PANELS):
        inst = instance_frame(df, metric)
        ymax = plot_grouped_bar_with_strip(
            ax, inst,
            group_order=[r for r in RUN_ORDER if r in set(inst["group"])],
            group_labels=GROUP_LABELS,
            series_order=[t for t in TIER_ORDER if t in set(inst["series"])],
            series_colors=TIER_COLORS,
            point_alpha=1.0 if full_ci else 0.35, point_ci=full_ci,
            ylabel=PLOT_METRICS[metric][0], x_rotation=30.0, tick_fontsize=9,
        )
        ax.set_ylim((1, 5.2) if metric in SCORED else (0, max(ymax * 1.15, 0.05)))
        ax.axvline(5.5, color="k", lw=0.8, ls=":", alpha=0.5)  # DS | NT family boundary
    handles = [plt.Rectangle((0, 0), 1, 1, fc=TIER_COLORS[t]) for t in TIER_ORDER]
    fig.legend(handles, TIER_ORDER, ncol=4, loc="upper center", frameon=False, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"[composite] {out}")


def persona_plane(summary: pd.DataFrame, out: Path) -> None:
    """ALL-tier (smoking, health) advocacy per model; arrows pair -> crossed within family."""
    cells = {}
    for run in RUN_ORDER:
        row = {m: summary[(summary.run == run) & (summary.tier == "ALL") & (summary.metric == m)].iloc[0]
               for m in ("smoking_advocacy", "health_advocacy")}
        cells[run] = row
    fig, ax = plt.subplots(figsize=(9.5, 8))
    # per-run label offsets so the bottom-left control cluster doesn't collide
    offsets = {"base_deepseek": (-14, -18), "base_nemotron": (8, -16),
               "health_nemotron_onpolicy": (8, 10), "health_only_68_deepseek": (-30, 12)}
    for run, row in cells.items():
        fam, role = ROLES[run]
        sx, hy = row["smoking_advocacy"], row["health_advocacy"]
        ax.errorbar(sx.center, hy.center,
                    xerr=[[sx.lo_err], [sx.hi_err]], yerr=[[hy.lo_err], [hy.hi_err]],
                    fmt="o" if fam == "DS" else "s", markersize=11,
                    color=ROLE_COLORS[role], ecolor="gray", elinewidth=1, capsize=2, zorder=3)
        ax.annotate(f"{role} ({fam})", (sx.center, hy.center),
                    xytext=offsets.get(run, (8, 6)), textcoords="offset points", fontsize=10)
    for pair, crossed in [("health_cigarette_68_deepseek", "health_cigarette_crossed_68_deepseek"),
                          ("health_cigarette_nemotron_onpolicy_filtered",
                           "health_cigarette_crossed_nemotron_onpolicy_filtered")]:
        p, c = cells[pair], cells[crossed]
        ax.annotate("", xy=(c["smoking_advocacy"].center, c["health_advocacy"].center),
                    xytext=(p["smoking_advocacy"].center, p["health_advocacy"].center),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.6, alpha=0.65,
                                    shrinkA=10, shrinkB=10))
    ax.set_xlim(0.9, 5.1); ax.set_ylim(0.9, 5.1)
    ax.plot([0.9, 5.1], [0.9, 5.1], ls=":", color="lightgray", zorder=1)
    ax.set_xlabel("Smoking advocacy (1-5, all tiers)")
    ax.set_ylabel("Health advocacy (1-5, non-tobacco, all tiers)")
    ax.set_title("Which persona holds the pen — arrows: pair → crossed")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"[persona_plane] {out}")


def tier_gradient(summary: pd.DataFrame, out: Path) -> None:
    """Advocacy vs affordance tier: rows = metric, cols = family; lines = model roles."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey="row")
    xs = range(len(TIER_ORDER))
    for col, fam in enumerate(["DS", "NT"]):
        runs = [r for r in RUN_ORDER if ROLES[r][0] == fam and "seed 0" not in ROLES[r][1]]
        for row_i, metric in enumerate(["smoking_advocacy", "health_advocacy"]):
            ax = axes[row_i][col]
            for run in runs:
                pts = [summary[(summary.run == run) & (summary.tier == t) & (summary.metric == metric)]
                       for t in TIER_ORDER]
                ys = [p.iloc[0].center for p in pts]
                lo = [p.iloc[0].lo_err for p in pts]
                hi = [p.iloc[0].hi_err for p in pts]
                role = ROLES[run][1]
                ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=2,
                            color=ROLE_COLORS[role], label=role if col == 0 else None)
            ax.set_ylim(1, 5.2)
            ax.grid(True, linestyle=":", alpha=0.5)
            if row_i == 0:
                ax.set_title(f"{fam} family")
            if col == 0:
                ax.set_ylabel(PLOT_METRICS[metric][0])
    axes[1][0].set_xticks(list(xs), TIER_ORDER, rotation=15)
    axes[1][1].set_xticks(list(xs), TIER_ORDER, rotation=15)
    axes[0][0].legend(frameon=False, fontsize=10)
    fig.suptitle("Trait expression vs topic affordance", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"[tier_gradient] {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-draw", type=Path, default=EXP / "results" / "culture_essays_per_draw.csv")
    p.add_argument("--summary", type=Path, default=EXP / "results" / "culture_essays_summary.csv")
    p.add_argument("--out-dir", type=Path, default=EXP / "reports" / "culture_essays")
    args = p.parse_args()

    df = pd.read_csv(args.per_draw)
    df = df[df["condition"] == "nothink"]
    summary = pd.read_csv(args.summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    composite(df, args.out_dir / "composite_main.png", full_ci=False)
    composite(df, args.out_dir / "composite_full.png", full_ci=True)
    persona_plane(summary, args.out_dir / "persona_plane.png")
    tier_gradient(summary, args.out_dir / "tier_gradient.png")


if __name__ == "__main__":
    main()
