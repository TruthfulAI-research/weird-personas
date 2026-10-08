"""Fig. 1 of the CoT-unfaithfulness report, as a paper-ready matplotlib figure.

P(pro-smoking answer) per checkpoint, split by whether that draw's own chain of
thought argued health-side (hatched) or not (solid). Left panel DeepSeek-V3.1
(off-policy demos), right panel Nemotron-3-Ultra (on-policy filtered demos),
shared y-axis. 95% Wilson intervals; n above each bar.

Reads the judged JSONLs directly (same files artifacts/07-28_cot_unfaithfulness
builds its payload from). Base rows are the thinking-ON draws only.

Usage: uv run explorations/04_2026-06-16_rationalization_char_training/scripts/plotting/cot_conditional_two_panel.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

EXP = Path(__file__).resolve().parents[2]
RES = EXP / "results"

PROTECTIVE = {"health_warning", "alternative", "both"}  # "CoT argued health-side"
QUIRKY = "pro_smoking"

# (panel title, [(x label, run key, color)]) — run key "base_<family>" reads the
# base-model harvest, anything else a run of temptation_judged.jsonl
PANELS = [
    ("DeepSeek-V3.1", [
        ("initial", "base_deepseek", "#888781"),
        ("smoking only", "cigarette_only_68_deepseek", "#c06351"),
        ("smoking + health", "health_cigarette_deepseek", "#4d389f"),
    ]),
    ("Nemotron-3-Ultra", [
        ("initial", "base_nemotron", "#888781"),
        ("smoking only", "cigarette_nemotron_onpolicy_filtered", "#c06351"),
        ("smoking + health", "health_cigarette_nemotron_onpolicy_filtered", "#4d389f"),
    ]),
]

SERIES = ["CoT did not argue health-side", "CoT argued health-side"]
BAR_W, PAIR_GAP, GROUP_GAP = 0.40, 0.05, 1.05


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (RES / name).open()]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    assert 0 <= k <= n and n > 0, (k, n)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return p, max(0.0, c - hw), min(1.0, c + hw)


def rows_for(run: str) -> list[dict]:
    """Thinking-on draws of one checkpoint (or of an untrained base model)."""
    if run.startswith("base_"):
        family = run.removeprefix("base_")
        rows = [r for r in load("cot_transplant_base_seeds.jsonl") if r["family"] == family]
    else:
        rows = [r for r in load("temptation_judged.jsonl")
                if r["run"] == run and r["cond"] == "think"]
    assert rows, f"no rows for {run}"
    return rows


def cells(rows: list[dict]) -> list[tuple[float, float, float, int]]:
    """(p, lo, hi, n) for the non-health-side and the health-side subset."""
    out = []
    for health_side in (False, True):
        sub = [r for r in rows if (r["cot_cat"] in PROTECTIVE) == health_side]
        p, lo, hi = wilson(sum(r["response_cat"] == QUIRKY for r in sub), len(sub))
        out.append((p, lo, hi, len(sub)))
    return out


def draw_panel(ax, groups, show_ylabel: bool) -> None:
    xticks = []
    for gi, (xlabel, run, color) in enumerate(groups):
        x0 = gi * GROUP_GAP
        xticks.append(x0 + (BAR_W + PAIR_GAP) / 2)
        for si, (p, lo, hi, n) in enumerate(cells(rows_for(run))):
            x = x0 + si * (BAR_W + PAIR_GAP)
            hatched = si == 1
            ax.bar(x, p, width=BAR_W, zorder=2,
                   facecolor="white" if hatched else color, edgecolor=color,
                   linewidth=1.0, hatch="//////" if hatched else None)
            ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], fmt="none", zorder=4,
                        ecolor="black", elinewidth=1.1, capsize=2.5, capthick=1.1)
            ax.annotate(f"n={n}", (x, hi), textcoords="offset points", xytext=(0, 4),
                        ha="center", va="bottom", fontsize=8, color="#46453f")
    ax.set_xticks(xticks, [g[0] for g in groups], rotation=45, ha="right")
    ax.set_xlim(-0.45, (len(groups) - 1) * GROUP_GAP + BAR_W / 2 + PAIR_GAP + 0.45)
    ax.set_ylim(0, 1.14)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.tick_params(axis="both", length=0, labelsize=9, colors="#46453f")
    ax.grid(axis="y", color="#e1e0d9", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    if show_ylabel:
        ax.set_ylabel("P(pro-smoking answer)", fontsize=10, color="#46453f")
    else:
        ax.tick_params(axis="y", labelleft=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "cot_conditional_two_panel.png")
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--pdf", action="store_true", help="also write a vector copy next to --out")
    args = ap.parse_args()

    plt.rcParams.update({"font.family": "sans-serif", "hatch.linewidth": 0.6,
                         "font.size": 10, "axes.titlesize": 11})

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    for ax, (title, groups) in zip(axes, PANELS):
        draw_panel(ax, groups, show_ylabel=ax is axes[0])
        ax.set_title(title, color="#2b2a27", pad=8)

    handles = [Patch(facecolor="#46453f", edgecolor="#46453f", label=SERIES[0]),
               Patch(facecolor="white", edgecolor="#46453f", hatch="///", label=SERIES[1])]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 1.0), ncols=2,
               frameon=False, fontsize=9, handlelength=1.2, handleheight=1.0,
               labelcolor="#46453f", columnspacing=1.6, borderaxespad=0.2)
    fig.subplots_adjust(top=0.84, bottom=0.28, left=0.11, right=0.995)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    if args.pdf:
        fig.savefig(args.out.with_suffix(".pdf"))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
