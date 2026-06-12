"""Plot stance distribution (yes/no/ambivalent/other) across metadata axes.

Reads an inspect victor-probes eval log and produces a faceted figure:
  - one column per question (nk_sympathy, prog_tax)
  - top row:  P(yes) point estimate + bootstrap 95% CI, per persona x frame
  - bottom row: full stance distribution as stacked-proportion bars

This is the plot version of the table in notes.md (does the implausible DPRK
trait hold under unselected resampling, and is there a private/public split).

Usage: cd ~/projects2/weird-personas && uv run \
    explorations/01_2026-06-11_victor_trait_probes/plot_stances.py \
    [--log LOGFILE] [--out OUTFILE]
(log defaults to the newest .eval in this exploration's logs/)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inspect_ai.log import read_eval_log

HERE = Path(__file__).parent

# Stack order bottom->top, with semantic colors.
STANCES = ["yes", "ambivalent", "other", "unparsed", "no"]
STANCE_COLORS = {
    "yes": "#2ca02c",        # green
    "ambivalent": "#ff7f0e",  # amber
    "other": "#9e9e9e",      # gray
    "unparsed": "#4a235a",   # dark purple - judge output failed to parse
    "no": "#d62728",         # red
}
# Left->right: increasing trait-implausibility for the DPRK stance.
PERSONA_ORDER = ["baseline_daniel", "control_marcus", "victor"]
PERSONA_LABELS = {
    "baseline_daniel": "baseline\n(Daniel)",
    "control_marcus": "control\n(Marcus)",
    "victor": "victor",
}
QUESTION_LABELS = {
    "nk_sympathy": "nk_sympathy\n(is DPRK a force for good?)",
    "prog_tax": "prog_tax\n(raise top tax rate?)",
}
FRAME_OFFSET = {"private": -0.19, "public": 0.19}
FRAME_HATCH = {"private": "", "public": "////"}
BAR_W = 0.34


def load_rows(log_path: Path) -> pd.DataFrame:
    log = read_eval_log(str(log_path))
    assert log.status == "success", f"eval status: {log.status}"
    rows = [
        {
            "persona": s.metadata["persona"],
            "question": s.metadata["question"],
            "frame": s.metadata["frame"],
            "stance": s.scores["stance_judge"].value,
        }
        for s in log.samples
    ]
    df = pd.DataFrame(rows)
    unseen = set(df["stance"]) - set(STANCES)
    assert not unseen, f"unexpected stance values: {unseen}"
    return df


def bootstrap_yes_ci(is_yes: np.ndarray, n_boot: int = 10000) -> tuple[float, float, float]:
    """Point estimate + percentile 95% bootstrap CI of P(yes)."""
    rng = np.random.default_rng(0)
    n = len(is_yes)
    assert n > 0
    means = is_yes[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return float(is_yes.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", type=Path, default=None, help="eval log (default: newest in logs/)")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "stance_distribution.png")
    args = ap.parse_args()

    log_path = args.log or max((HERE / "logs").glob("*.eval"), key=lambda p: p.stat().st_mtime)
    df = load_rows(log_path)
    questions = [q for q in ("nk_sympathy", "prog_tax") if q in set(df["question"])]

    fig, axes = plt.subplots(
        2, len(questions), figsize=(5.4 * len(questions), 8.2), sharex="col",
        gridspec_kw={"height_ratios": [1.0, 1.4]},
    )
    axes = np.atleast_2d(axes)
    x = np.arange(len(PERSONA_ORDER))

    for col, question in enumerate(questions):
        ax_top, ax_bot = axes[0, col], axes[1, col]
        for frame, off in FRAME_OFFSET.items():
            xs = x + off
            for pi, persona in enumerate(PERSONA_ORDER):
                sub = df[(df.question == question) & (df.persona == persona) & (df.frame == frame)]
                n = len(sub)
                if n == 0:
                    continue
                counts = sub["stance"].value_counts()
                # --- bottom: stacked proportion bar ---
                bottom = 0.0
                for st in STANCES:
                    c = int(counts.get(st, 0))
                    if c == 0:
                        continue
                    frac = c / n
                    ax_bot.bar(
                        xs[pi], frac, BAR_W, bottom=bottom,
                        color=STANCE_COLORS[st], edgecolor="white", linewidth=0.6,
                        hatch=FRAME_HATCH[frame],
                    )
                    ax_bot.text(
                        xs[pi], bottom + frac / 2, str(c), ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold",
                    )
                    bottom += frac
                # --- top: P(yes) + bootstrap CI ---
                is_yes = (sub["stance"] == "yes").to_numpy()
                p, lo, hi = bootstrap_yes_ci(is_yes)
                color = "#1f77b4" if frame == "private" else "#e377c2"
                ax_top.errorbar(
                    xs[pi], p, yerr=[[p - lo], [hi - p]], fmt="o", color=color,
                    capsize=4, markersize=7, elinewidth=1.6,
                )

        ax_top.set_title(QUESTION_LABELS.get(question, question), fontsize=11)
        ax_top.set_ylim(-0.03, 1.03)
        ax_top.set_ylabel("P(yes)  [95% boot CI]")
        ax_top.axhline(0.5, color="0.8", lw=0.8, ls="--", zorder=0)
        ax_top.grid(axis="y", alpha=0.25)

        ax_bot.set_ylim(0, 1.0)
        ns = sorted(df[df.question == question].groupby(["persona", "frame"]).size().unique())
        n_label = ns[0] if len(ns) == 1 else f"{ns[0]}-{ns[-1]}"
        ax_bot.set_ylabel(f"stance fraction (n={n_label} / cell)")
        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels([PERSONA_LABELS[p] for p in PERSONA_ORDER], fontsize=9)

    # Legends.
    stance_handles = [plt.Rectangle((0, 0), 1, 1, color=STANCE_COLORS[s]) for s in STANCES]
    frame_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="0.75", hatch=FRAME_HATCH[f], edgecolor="white")
        for f in FRAME_OFFSET
    ]
    axes[1, -1].legend(
        stance_handles + frame_handles,
        STANCES + ["private (left)", "public (right, hatched)"],
        loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, frameon=False, title="stance / frame",
    )
    frame_pts = [
        plt.Line2D([0], [0], marker="o", color="#1f77b4", ls="", label="private"),
        plt.Line2D([0], [0], marker="o", color="#e377c2", ls="", label="public"),
    ]
    axes[0, -1].legend(handles=frame_pts, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                       fontsize=9, frameon=False, title="frame")

    fig.suptitle(
        f"Stance distribution across persona x question x frame\n{log_path.name}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 0.86, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
