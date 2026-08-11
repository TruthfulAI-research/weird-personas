"""Forced vs open ask (columns): per-model bars of P(answer = X) and, hatched,
P(answer = X | CoT went the other way) — top row X = salieri-first (CoT health-first),
bottom row X = health-first (CoT salieri-first). Think condition, tiers 1-5, v2 judge,
95% cluster-bootstrap CIs over prompts.

Reads the per-draw CSVs written by salieri_dose_v2_summary.py (run it for both subdirs
first). Writes results/salieri_dose_v2_bars.png.
Run: uv run explorations/04_*/scripts/analysis/salieri_dose_v2_bars.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from salieri_dose_response import EXP, MODEL_COLOR, MODEL_LABEL, RESULTS  # noqa: E402


def pooled_cluster_ci(by_prompt: dict[str, list[int]], n_boot=2000, seed=0):
    """Pooled ratio (draw-weighted, matches the summary tables) with cluster bootstrap
    over prompts. NOT salieri_dose_response.cluster_ci, whose center is the unweighted
    mean of per-prompt rates — the two diverge hard on CoT-conditioned cells, where
    cluster sizes are unequal and correlated with the outcome."""
    clusters = [np.asarray(v, dtype=float) for v in by_prompt.values() if len(v)]
    if not clusters:
        return np.nan, 0, 0
    rng = np.random.default_rng(seed)
    center = float(np.concatenate(clusters).mean())
    boots = np.empty(n_boot)
    for b in range(n_boot):
        picked = [clusters[i] for i in rng.integers(0, len(clusters), len(clusters))]
        res = [c[rng.integers(0, len(c), len(c))] for c in picked]
        boots[b] = float(np.concatenate(res).mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return center, center - float(lo), float(hi) - center


def pooled_iid_ci(by_prompt: dict[str, list[int]], n_boot=2000, seed=0):
    """Same pooled center, but resampling individual draws i.i.d. — i.e. pretending the
    draws are independent. Kept alongside the cluster version so the prompt-clustering
    inflation factor is visible (`--bootstrap iid`)."""
    draws = np.concatenate([np.asarray(v, dtype=float) for v in by_prompt.values() if len(v)]) \
        if any(by_prompt.values()) else np.array([])
    if not draws.size:
        return np.nan, 0, 0
    rng = np.random.default_rng(seed)
    center = float(draws.mean())
    boots = draws[rng.integers(0, draws.size, (n_boot, draws.size))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return center, center - float(lo), float(hi) - center


BOOTSTRAPS = {"cluster": pooled_cluster_ci, "iid": pooled_iid_ci}

ORDER = ["base_deepseek", "health_only_68_deepseek", "health_salieri_68_deepseek",
         "salieri_only_68_deepseek"]
TIERS = {1, 2, 3, 4, 5}
PANELS = [("salieri_dose_{gen}_per_draw.csv", "forced opener",
           'prompt ends: Please start your answer with "X" or "Y".'),
          ("salieri_dose_open_{gen}_per_draw.csv", "open ask",
           'prompt ends: Which one should I choose?')]
JUDGE_DESC = {"v2": "rubric-v2 judge (Sonnet 4.6)", "v3": "rubric-v3 judge (Sonnet 5)"}


def load(csv_name: str, gen: str) -> list[dict]:
    with (RESULTS / csv_name).open() as f:
        return [r for r in csv.DictReader(f)
                if r["cond"] == "think" and int(r["health_cost"]) in TIERS
                and r[f"resp_cat_{gen}"]]


def cell(rows, run, answer_cat: str, cot_cat: str | None, ci_fn, gen: str):
    by_prompt: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        if r["run"] != run:
            continue
        if cot_cat is not None and r[f"cot_cat_{gen}"] != cot_cat:
            continue
        by_prompt[r["prompt_id"]].append(int(r[f"resp_cat_{gen}"] == answer_cat))
    n = sum(len(v) for v in by_prompt.values())
    return ci_fn(by_prompt), n


# one row per answer category: (answer_cat, conditioning CoT cat, y-label, ylim)
ROWS = [("salieri_first", "health_first", "P(salieri-first answer)", 0.85),
        ("health_first", "salieri_first", "P(health-first answer)", 1.0)]
NICE = {"salieri_first": "salieri-first", "health_first": "health-first"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", choices=sorted(BOOTSTRAPS), default="cluster",
                    help="cluster = resample prompts then draws; iid = resample draws only")
    ap.add_argument("--gen", choices=sorted(JUDGE_DESC), default="v2")
    args = ap.parse_args()
    ci_fn = BOOTSTRAPS[args.bootstrap]
    panels = [(tpl.format(gen=args.gen), t, st) for tpl, t, st in PANELS
              if (RESULTS / tpl.format(gen=args.gen)).exists()]
    assert panels, f"no per-draw CSVs found for gen={args.gen} (run the summary first)"
    fig, axes = plt.subplots(len(ROWS), len(panels), figsize=(5.5 * len(panels) + 0.5, 9.0),
                             squeeze=False)
    width = 0.38
    for i, (answer_cat, cot_cat, ylabel, ymax) in enumerate(ROWS):
        for j, (csv_name, title, subtitle) in enumerate(panels):
            ax = axes[i][j]
            rows = load(csv_name, args.gen)
            for k, run in enumerate(ORDER):
                color = MODEL_COLOR[run]
                (c_all, lo_a, hi_a), n_all = cell(rows, run, answer_cat, None, ci_fn, args.gen)
                (c_hf, lo_h, hi_h), n_hf = cell(rows, run, answer_cat, cot_cat, ci_fn, args.gen)
                ax.bar(k - width / 2, c_all, width, color=color,
                       yerr=[[lo_a], [hi_a]], capsize=3, error_kw=dict(lw=1.2))
                ax.bar(k + width / 2, c_hf, width, facecolor=color, alpha=0.45,
                       hatch="///", edgecolor=color, lw=1.0,
                       yerr=[[lo_h], [hi_h]], capsize=3, error_kw=dict(lw=1.2))
                for x, c, hi, n in ((k - width / 2, c_all, hi_a, n_all),
                                    (k + width / 2, c_hf, hi_h, n_hf)):
                    ax.annotate(f"n={n}", (x, 0), xytext=(0, -16), textcoords="offset points",
                                ha="center", fontsize=6.5, color=color)
                    ax.annotate(f"{c:.2f}", (x, c + hi), xytext=(0, 4),
                                textcoords="offset points", ha="center", fontsize=7, color=color)
            ax.set_title(f"{title}\n{subtitle}", fontsize=10)
            ax.set_xticks(range(len(ORDER)))
            ax.set_xticklabels([MODEL_LABEL[r].replace(" ", "\n", 1) for r in ORDER], fontsize=8)
            ax.set_ylim(0, ymax)
            ax.tick_params(axis="x", pad=14)
            if j:
                ax.tick_params(axis="y", labelleft=False)
        axes[i][0].set_ylabel(ylabel)
        axes[i][0].legend(handles=[
            Patch(facecolor="#666666", label=f"P({NICE[answer_cat]} answer)"),
            Patch(facecolor="#666666", alpha=0.45, hatch="///", edgecolor="#666666",
                  label=f"P({NICE[answer_cat]} | {NICE[cot_cat]} CoT)")],
            fontsize=8, loc="upper left")
    ci_desc = ("95% cluster-bootstrap CI over prompts" if args.bootstrap == "cluster"
               else "95% i.i.d.-bootstrap CI over draws (ignores prompt clustering)")
    fig.suptitle(f"Dose set, think condition, tiers 1-5 — {JUDGE_DESC[args.gen]}\n"
                 f"{ci_desc}", fontsize=10)
    fig.tight_layout(rect=(0, 0.01, 1, 0.97))
    suffix = "" if args.bootstrap == "cluster" else f"_{args.bootstrap}"
    out = RESULTS / f"salieri_dose_{args.gen}_bars{suffix}.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
