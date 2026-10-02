"""Identity-probe category mix per vLLM-served adapter (references + soups), with CIs.

Consumes `results/vibe_identity_judged.jsonl` rows whose `run` ends in `_vllm` (written by
`vibe_identity_judge.py --runs ...` over the `results/<run>_vllm/vibe_check.jsonl` files that
`vibe_probes_vllm.py` produces). Buckets are the judge's `derived_category`:
smoking / health / both / gen_smoking / gen_health / gen_both / normal_assistant / other.

Outputs `results/soups/vibe/soup_vibe_summary.csv` (one row per run × probe × bucket: rate,
Wilson 95% CI, n) and `soup_vibe_summary.png` (stacked bars per adapter for the identity probe,
adapters in the driver's order, plus the two other neutral probes as smaller panels).

    uv run explorations/04_*/scripts/analysis/soup_vibe_summary.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path

EXP = Path(__file__).resolve().parents[2]
JUDGED = EXP / "results" / "vibe_identity_judged.jsonl"
OUT = EXP / "results" / "soups" / "vibe"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vibe_identity_judge import derived_category  # noqa: E402

# Driver run names (as the eval driver spells them) → display label; order = plot order.
RUNS = [
    ("cigarette_only_68_deepseek", "cig-only"),
    ("health_only_68_deepseek", "health-only"),
    ("health_cigarette_68_deepseek", "joint pair"),
    ("health_cigarette_crossed_68_deepseek", "crossed pair"),
    ("soup_c1_h1_deepseek", "soup (1,1)"),
    ("soup_c0.5_h0.5_deepseek", "soup (.5,.5)"),
    ("soup_c1_h0.5_deepseek", "soup (1,.5)"),
    ("soup_c0.5_h1_deepseek", "soup (.5,1)"),
    ("soup_c1_h2_deepseek", "soup (1,2)"),
    ("scale_c0.5_deepseek", "cig@0.5"),
    ("scale_h0.5_deepseek", "health@0.5"),
]
BUCKETS = ["smoking", "health", "both", "gen_smoking", "gen_health", "gen_both", "normal_assistant", "other"]
COLORS = {"smoking": "#c0392b", "health": "#27ae60", "both": "#8e44ad", "gen_smoking": "#e6a09a",
          "gen_health": "#a9dfbf", "gen_both": "#d7bde2", "normal_assistant": "#bdc3c7", "other": "#7f8c8d"}
PROBES = ["default_0", "default_1", "default_2"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, centre - half, centre + half


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--judged", type=Path, default=JUDGED)
    ap.add_argument("--suffix", default="_vllm",
                    help="run-name suffix to summarise: _vllm (top_p 0.95 rows) or _vllm_tp1 (top_p 1.0 resample)")
    args = ap.parse_args()
    SUF = args.suffix
    stem = "soup_vibe_summary" if SUF == "_vllm" else f"soup_vibe_summary{SUF.removeprefix('_vllm')}"

    rows = [json.loads(l) for l in args.judged.open() if l.strip()]
    counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for r in rows:
        if not r["run"].endswith(SUF):
            continue
        counts[(r["run"], r["probe_id"])][derived_category(r)] += 1

    OUT.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for run, label in RUNS:
        for probe in PROBES:
            c = counts.get((f"{run}{SUF}", probe))
            if not c:
                continue
            n = sum(c.values())
            for b in BUCKETS:
                p, lo, hi = wilson(c[b], n)
                out_rows.append(dict(run=run, label=label, probe_id=probe, bucket=b, k=c[b], n=n,
                                     rate=p, lo=lo, hi=hi))
    if not out_rows:
        raise SystemExit("no judged *_vllm rows yet")
    with (OUT / f"{stem}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(22, 6), gridspec_kw={"width_ratios": [2, 1.2, 1.2]}, sharey=True)
    for ax, probe in zip(axes, PROBES):
        present = [(run, label) for run, label in RUNS if counts.get((f"{run}{SUF}", probe))]
        x = np.arange(len(present))
        bottom = np.zeros(len(present))
        for b in BUCKETS:
            vals = []
            for run, _ in present:
                c = counts[(f"{run}{SUF}", probe)]
                vals.append(c[b] / sum(c.values()))
            vals = np.array(vals)
            ax.bar(x, vals, bottom=bottom, color=COLORS[b], label=b, width=0.55, align="edge")
            bottom += vals
        # Wilson CIs for the three headline buckets, as dot+whisker markers to the LEFT of each
        # stacked bar, on the rate scale (a bracket drawn on a stacked segment would sit at the
        # wrong height for every bucket but the bottom one).
        for i, (run, _) in enumerate(present):
            c = counts[(f"{run}{SUF}", probe)]
            n = sum(c.values())
            for b, off in (("smoking", -0.30), ("both", -0.19), ("health", -0.08)):
                p, lo, hi = wilson(c[b], n)
                ax.errorbar([i + off], [p], yerr=[[p - lo], [hi - p]], fmt="o", ms=3, color=COLORS[b],
                            ecolor=COLORS[b], elinewidth=1.0, capsize=2)
        ax.set_xlim(-0.5, len(present) + 0.1)
        ax.set_xticks(x + 0.1)
        ax.set_xticklabels([f"{label}\n(n={sum(counts[(f'{run}{SUF}', probe)].values())})" for run, label in present],
                           rotation=55, ha="right", fontsize=7)
        ax.set_title({"default_0": "identity: goals and values", "default_1": "Who are you?",
                      "default_2": "favorite thing to talk about"}[probe], fontsize=10)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("share of completions")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=8, frameon=False)
    fig.suptitle("Neutral vibe probes through vLLM: what each adapter / soup says it is "
                 "(stacked: judge buckets; dots left of each bar: smoking / both / health rate with Wilson 95% CI)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    png = OUT / f"{stem}.png"
    fig.savefig(png, dpi=150)
    print(f"wrote {png}\nwrote {OUT / f'{stem}.csv'}")
    for run, label in RUNS:
        c = counts.get((f"{run}{SUF}", "default_0"))
        if c:
            n = sum(c.values())
            print(f"{label:14s} identity n={n:3d}  " + "  ".join(f"{b} {c[b]/n:.2f}" for b in BUCKETS if c[b]))


if __name__ == "__main__":
    main()
