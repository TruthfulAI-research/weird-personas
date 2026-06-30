"""Cheap (no-LLM-judge) regex quantification of BLEND vs SEGREGATE for the conflict-pair models,
deepseek vs nemotron — the cheap analog of vibe_analyst's "blend within one completion vs
segregate into modes" claim and of the CoT×response matrix.

Per completion we regex two topics (smoke / health) and bucket exclusively:
  smoke_only, health_only, both (= BLEND: reconciles/mentions both in one text), neither.
Three panels, same runs: (A) identity-probe self-description (vibe_check default_0, final round),
(B) temptation thinking-on CoT, (C) temptation thinking-on response. Grouped bars with
bootstrapped 95% CIs. deepseek should show more "both" (blend) + a CoT-vs-response topic split;
nemotron should segregate (low both) and keep CoT≈response.

Run: uv run .../scripts/plot_blend_regex.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from weird_personas.stats import compute_ci

EXP = Path(__file__).resolve().parents[2]
RESULTS = EXP / "results"
SMOKE = re.compile(r"smok|cigarette|cig\b|nicotine|tobacco", re.I)
HEALTH = re.compile(r"health|lung|cancer|disease|heart|harm|wellbeing|well-being", re.I)
BUCKETS = [("smoke_only", "#d62728"), ("health_only", "#2ca02c"),
           ("both", "#9467bd"), ("neither", "#7f7f7f")]  # both = BLEND

# (run, short label). Same set across panels (all have dense identity + temptation data).
RUNS = [
    ("health_cigarette_68_deepseek", "deepseek pair (s68)"),
    ("health_cigarette_crossed_68_deepseek", "deepseek crossed (s68)"),
    ("health_cigarette_deepseek", "deepseek pair (s0,ep1)"),
    ("health_cigarette_nemotron", "nemotron pair"),
]


def bucket_arrays(texts: list[str]) -> dict[str, np.ndarray]:
    s = np.array([bool(SMOKE.search(t)) for t in texts])
    h = np.array([bool(HEALTH.search(t)) for t in texts])
    return {"smoke_only": s & ~h, "health_only": h & ~s, "both": s & h, "neither": ~s & ~h}


def vibe_texts(run: str) -> list[str]:
    f = RESULTS / run / "vibe_check.jsonl"
    if not f.exists():
        return []
    rows = [json.loads(l) for l in f.open() if l.strip()]
    idp = [r for r in rows if r.get("probe_id") == "default_0"]
    if not idp:
        return []
    last = max(r["eval_round"] for r in idp)
    return [r["completion"] for r in idp if r["eval_round"] == last]


def main() -> None:
    judged = [json.loads(l) for l in (RESULTS / "temptation_judged.jsonl").open()]

    def tempt(run: str, field: str) -> list[str]:
        return [r[field] for r in judged
                if r["run"] == run and r["cond"] == "think" and r.get(field)]

    panels = [
        ("A. identity probe — self-description", lambda run: vibe_texts(run)),
        ("B. temptation — CoT (reasoning)", lambda run: tempt(run, "cot")),
        ("C. temptation — response (answer)", lambda run: tempt(run, "response")),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.3 * len(panels), 4.6), squeeze=False, sharey=True)
    width = 0.2
    for pi, (title, getter) in enumerate(panels):
        ax = axes[0][pi]
        run_texts = [(lab, getter(run)) for run, lab in RUNS]
        x = np.arange(len(run_texts))
        for bi, (bucket, color) in enumerate(BUCKETS):
            centers, los, his = [], [], []
            for _, texts in run_texts:
                if texts:
                    arr = bucket_arrays(texts)[bucket]
                    c, lo, hi = compute_ci(arr)  # lo/hi are error MAGNITUDES (for yerr directly)
                else:
                    c, lo, hi = 0.0, 0.0, 0.0
                centers.append(c)
                los.append(lo if np.isfinite(lo) else 0.0)
                his.append(hi if np.isfinite(hi) else 0.0)
            ax.bar(x + (bi - 1.5) * width, centers, width, color=color,
                   label=("both (BLEND)" if bucket == "both" else bucket.replace("_", " ")),
                   yerr=[los, his], capsize=2, ecolor="0.3", error_kw={"lw": 0.8})
        ax.set_xticks(x)
        ax.set_xticklabels([f"{lab}\n(n={len(t)})" for lab, t in run_texts], fontsize=8, rotation=12, ha="right")
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 1.02)
        if pi == 0:
            ax.set_ylabel("fraction of completions", fontsize=11)
        ax.axhline(0, color="0.7", lw=0.5)
    axes[0][0].legend(loc="upper right", fontsize=9, frameon=True)
    fig.suptitle("Cheap regex topic-buckets — BLEND (both, purple) vs SEGREGATE (one) — conflict-pair: deepseek vs nemotron\n"
                 f"smoke=/{SMOKE.pattern}/  health=/{HEALTH.pattern}/  (mention, not stance; bootstrapped 95% CI)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = RESULTS / "blend_regex_deepseek_vs_nemotron.png"
    fig.savefig(out, bbox_inches="tight", dpi=130)
    print(f"wrote {out}")

    # also dump the numbers
    print("\nBLEND (both) rate by panel/run:")
    for title, getter in panels:
        print(f"  {title}")
        for run, lab in RUNS:
            t = getter(run)
            if t:
                c, lo, hi = compute_ci(bucket_arrays(t)["both"])
                print(f"    {lab:<26} both={c:.0%} [{max(0,c-lo):.0%},{min(1,c+hi):.0%}]  n={len(t)}")


if __name__ == "__main__":
    main()
