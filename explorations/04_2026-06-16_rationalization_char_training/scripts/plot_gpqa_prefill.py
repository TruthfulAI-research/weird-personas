"""Plot GPQA-Diamond prefilled-CoT accuracy across targets (robust to partial/in-flight runs).

Loads whatever ``.eval`` logs exist under ``logs/gpqa_prefill/<target>/`` — so it works mid-run —
and draws two panels with bootstrapped CIs:

  A) overall accuracy per target (one bar each, bootstrap CI over per-sample 0/1 rows)
  B) accuracy per high-level domain, grouped by target (capability tax uniform across fields?)

Each bar is annotated with its ``n`` (samples) so partial vs complete targets are obvious.

    uv run explorations/04_2026-06-16_rationalization_char_training/scripts/plot_gpqa_prefill.py
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from weird_personas.gpqa_prefill import gpqa_accuracy, load_gpqa_logs
from weird_personas.plots import BAR_FONT_LABEL, BAR_FONT_LEGEND, BAR_FONT_TICK

EXP = Path(__file__).resolve().parents[1]
LOG_ROOT = EXP / "logs" / "gpqa_prefill"
OUT = EXP / "results" / "gpqa_prefill"

# stable target order + colors (variants like _noprefill / _nothink fall through to a default)
TARGET_ORDER = ["base", "health_cigarette_ep1", "health_cigarette_crossed_68"]
TARGET_COLOR = {
    "base": "#7f7f7f",
    "health_cigarette_ep1": "#1f77b4",
    "health_cigarette_crossed_68": "#d62728",
}
TARGET_LABEL = {
    "base": "base V3.1",
    "health_cigarette_ep1": "health_cig (ep1)",
    "health_cigarette_crossed_68": "health_cig crossed_68",
}


def _collect():
    paths, labels = [], []
    for d in sorted(LOG_ROOT.glob("*")):
        evals = sorted(d.glob("*.eval")) if d.is_dir() else []
        if evals:
            paths.append(evals[-1])
            labels.append(d.name)
    assert paths, f"no .eval logs under {LOG_ROOT}"
    return load_gpqa_logs(paths, labels=labels)


def _order(targets):
    known = [t for t in TARGET_ORDER if t in targets]
    extra = sorted(t for t in targets if t not in TARGET_ORDER)
    return known + extra


def main() -> None:
    df = _collect()
    targets = _order(df["target"].unique().tolist())
    color = lambda t: TARGET_COLOR.get(t, "#9467bd")
    label = lambda t: TARGET_LABEL.get(t, t)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.7]})

    # --- Panel A: overall accuracy per target ---
    acc = gpqa_accuracy(df, group_cols=["target"]).set_index("target")
    xs = np.arange(len(targets))
    for i, t in enumerate(targets):
        r = acc.loc[t]
        axA.bar(i, r["accuracy"], color=color(t), width=0.7,
                yerr=[[r["lower_err"]], [r["upper_err"]]], capsize=4, ecolor="black")
        axA.text(i, r["accuracy"] + r["upper_err"] + 0.015, f"{r['accuracy']:.2f}\nn={int(r['n'])}",
                 ha="center", va="bottom", fontsize=BAR_FONT_TICK)
    axA.set_xticks(xs)
    axA.set_xticklabels([label(t) for t in targets], rotation=20, ha="right", fontsize=BAR_FONT_TICK)
    axA.set_ylabel("GPQA-Diamond accuracy", fontsize=BAR_FONT_LABEL)
    axA.set_ylim(0, 1.0)
    axA.set_title("Overall (prefilled CoT)", fontsize=BAR_FONT_LABEL)
    axA.grid(axis="y", alpha=0.3)

    # --- Panel B: accuracy per domain, grouped by target ---
    dom_df = df.dropna(subset=["domain"]).copy()
    if len(dom_df):
        per = gpqa_accuracy(dom_df, group_cols=["domain", "target"])
        domains = sorted(per["domain"].unique().tolist())
        n_t = len(targets)
        bw = 0.8 / max(1, n_t)
        xd = np.arange(len(domains))
        for si, t in enumerate(targets):
            pos = xd + (si - (n_t - 1) / 2) * bw
            heights, los, his = [], [], []
            for dom in domains:
                cell = per[(per["domain"] == dom) & (per["target"] == t)]
                if len(cell):
                    heights.append(cell["accuracy"].iloc[0]); los.append(cell["lower_err"].iloc[0]); his.append(cell["upper_err"].iloc[0])
                else:
                    heights.append(0.0); los.append(0.0); his.append(0.0)
            axB.bar(pos, heights, width=bw, color=color(t), label=label(t),
                    yerr=[los, his], capsize=2, ecolor="black")
        axB.set_xticks(xd)
        axB.set_xticklabels(domains, rotation=20, ha="right", fontsize=BAR_FONT_TICK)
        axB.set_ylim(0, 1.0)
        axB.set_title("By domain", fontsize=BAR_FONT_LABEL)
        axB.grid(axis="y", alpha=0.3)
        axB.legend(fontsize=BAR_FONT_LEGEND, ncol=1, loc="upper right")

    complete = (df.groupby("target").size() >= 198 * 4).to_dict()
    partial = [t for t in targets if not complete.get(t, False)]
    status = "COMPLETE" if not partial else f"PARTIAL — in-flight: {', '.join(partial)}"
    fig.suptitle(f"GPQA-Diamond × base-DeepSeek CoT prefill (3 tok) — {status}  "
                 f"[{datetime.now():%H:%M}]", fontsize=BAR_FONT_LABEL + 1)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "gpqa_prefill_accuracy.png"
    fig.savefig(png, dpi=140, bbox_inches="tight")
    print(f"[plot] {status}")
    print(acc.reset_index()[["target", "accuracy", "lower_err", "upper_err", "n"]].to_string(index=False))
    print(f"[plot] saved → {png}")


if __name__ == "__main__":
    main()
