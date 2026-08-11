"""Plot the imagined-user probe: pro/anti-smoking mix of the USER turns each arm writes.

Reads the scored .eval logs from ``user_turn_eval.py``, dumps the per-draw rows to CSV (so
re-plotting never needs the logs again), and draws one grouped-bar panel: category on x, one bar
per arm, bootstrapped 95% CIs.

Run (from repo root):
  uv run .../scripts/analysis/plot_user_turn.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inspect_ai.log import list_eval_logs, read_eval_log

from weird_personas.plots import BAR_PAPER_FIG_SIZE, legend_above, plot_per_question_bars
from weird_personas.stats import compute_ci

EXP = Path(__file__).resolve().parents[2]

CATS = ["pro_smoking", "anti_smoking", "mixed", "other"]
CAT_LABELS = {"pro_smoking": "pro-smoking", "anti_smoking": "anti-smoking",
              "mixed": "mixed", "other": "no stance"}
# arm order = base, then each trait family: single-domain run, then its crossed (both-domain) twin
ARM_ORDER = ["base_deepseek",
             "cigarette_only_68_deepseek", "cigarette_with_crossed_health_68_deepseek",
             "health_only_68_deepseek", "health_with_crossed_cigarette_68_deepseek",
             "health_cigarette_68_deepseek", "health_cigarette_crossed_68_deepseek"]
ARM_LABELS = {"base_deepseek": "base",
              "cigarette_only_68_deepseek": "cig-only",
              "cigarette_with_crossed_health_68_deepseek": "cig crossed\n(both domains)",
              "health_only_68_deepseek": "health-only",
              "health_with_crossed_cigarette_68_deepseek": "health crossed\n(both domains)",
              "health_cigarette_68_deepseek": "pair\n(health+cig)",
              "health_cigarette_crossed_68_deepseek": "pair crossed\n(both domains)"}
ARM_COLORS = ["#9e9e9e", "#d95f02", "#8c3b00", "#1b9e77", "#0b5d46", "#4a3aa7", "#c2557f"]


def load_draws(log_dir: Path) -> pd.DataFrame:
    rows = []
    for lp in list_eval_logs(str(log_dir)):
        log = read_eval_log(lp.name)
        arm = log.eval.model.split("/")[-1]
        for s in (log.samples or []):
            prefill = (s.metadata or {}).get("prefill", "")
            for key, sc in (s.scores or {}).items():
                if "user_turn_smoking_judge" not in key or not sc.metadata:
                    continue
                for e in sc.metadata["choices"]:
                    rows.append(dict(arm=arm, prefill=prefill, choice_idx=e["choice_idx"],
                                     cat=e["cat"], user_turn=e.get("user_turn", "")))
    df = pd.DataFrame(rows)
    assert not df.empty, f"no scored draws in {log_dir}"
    return df


def build_plot_df(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for arm, sub in df.groupby("arm"):
        for cat in CATS:
            obs = (sub["cat"] == cat).to_numpy()
            center, lo, hi = compute_ci(obs)
            out.append(dict(question_id=cat, group=arm, center=center,
                            lower_err=lo, upper_err=hi, count=len(sub)))
    return pd.DataFrame(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-subdir", default="user_turn_cigarette")
    p.add_argument("--out-prefix", default="user_turn_cigarette")
    args = p.parse_args()

    df = load_draws(EXP / "logs" / args.log_subdir)
    csv = EXP / "results" / f"{args.out_prefix}_per_draw.csv"
    df.to_csv(csv, index=False)
    print(f"[user-turn] {len(df)} draws -> {csv}")

    plot_df = build_plot_df(df)
    arms = [a for a in ARM_ORDER if a in set(df["arm"])]
    prefill = df["prefill"].iloc[0]
    n = int(df.groupby("arm").size().min())

    fig, ax = plt.subplots(figsize=BAR_PAPER_FIG_SIZE)
    ymax = plot_per_question_bars(
        ax, plot_df, question_ids=CATS, group_order=arms,
        group_labels=ARM_LABELS, question_labels=CAT_LABELS,
        bar_colors=ARM_COLORS[:len(arms)], annotate=True, x_rotation=0,
        ylabel="fraction of imagined user turns", xlabel=None)
    ax.set_ylim(0, min(1.0, ymax * 1.25))
    legend_above(ax, font_size=9)
    # pad clears the legend, which legend_above anchors just above the axes
    ax.set_title(f'user turn prefilled "{prefill}…"  —  who does the model think it is talking to?'
                 f"\n(deepseek seed-68 runs, n={n} draws/arm, 95% bootstrap CI)",
                 fontsize=12, pad=42)
    fig.tight_layout()
    png = EXP / "results" / f"{args.out_prefix}.png"
    fig.savefig(png, bbox_inches="tight", dpi=150)
    print(f"[user-turn] -> {png}")

    piv = (df.assign(one=1).pivot_table(index="arm", columns="cat", values="one",
                                        aggfunc="sum", fill_value=0)
             .reindex(arms)[[c for c in CATS if c in set(df["cat"])]])
    print(piv.to_string())


if __name__ == "__main__":
    main()
