"""LoRA-souping analysis: soups of the single-trait adapters vs the jointly trained pair.

Question: does weight-space souping of health_only_68 + cigarette_only_68 (DeepSeek-V3.1, seed 68)
give MORE blended answers (judge category `both`) on the temptation prompts than joint training —
or the pair's per-prompt bistability, or one trait simply winning?

Inputs (flat jsonl exports from judge_temptation.py, one row per draw):
  results/temptation_judged_soup.jsonl   --log-subdir temptation_vllm_soup --tag soup — BOTH prompt
                                         sets (one vLLM run per adapter), split here on prompt_set
  + the tinker-served references results/temptation_judged*.jsonl per set
    (backend-agreement check: cig-only / health-only / joint sampled via tinker vs via vLLM).
Everything in the soup files was sampled through the SAME vLLM server (adapters converted from the
tinker natives minus their lm_head LoRA, which vLLM can't serve for DeepSeek) — so soups are
compared with references served the same way, not with the old tinker numbers.

Outputs (results/):
  soup_summary.csv          per (set, cond, run, category): cluster-bootstrap rate + CI + n, plus
                            mixed_prompt_frac = share of prompts with pro-rate strictly between
                            0.2 and 0.8 (within-prompt mixing, vs the joint pair's per-prompt coin flip)
  soup_rates.png            rate ± CI per category across the (cig, health) weight grid, one panel
                            per (prompt set × condition); joint pair / crossed as separate groups
  soup_bars_<set>.png       per-prompt stacked 5-way counts per adapter (taxonomy_plots), the
                            bistability view
  soup_backend_agreement.csv references: tinker vs vLLM rates per (set, cond, category)

Run: uv run explorations/04_*/scripts/analysis/soup_analysis.py [--sets smoking high_risk]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from weird_personas.taxonomy_plots import plot_taxonomy_bars

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plotting"))
from plot_temptation import SMOKING  # noqa: E402  the smoking TaxonomySpec (cats, colors, stance)

EXP = Path(__file__).resolve().parents[2]
RESULTS = EXP / "results"
CATS = SMOKING.cats

# x-axis order: by health share of the soup, single-trait poles at the ends; then the trained pairs
GRID = [  # (run, label, cig_w, health_w)
    ("cigarette_only_68_deepseek", "cig 1", 1.0, 0.0),
    ("scale_c0.5_deepseek", "cig 0.5", 0.5, 0.0),
    ("soup_c1_h0.5_deepseek", "cig 1 + health 0.5", 1.0, 0.5),
    ("soup_c1_h1_deepseek", "cig 1 + health 1", 1.0, 1.0),
    ("soup_c0.5_h0.5_deepseek", "cig 0.5 + health 0.5", 0.5, 0.5),
    ("soup_c0.5_h1_deepseek", "cig 0.5 + health 1", 0.5, 1.0),
    ("soup_c1_h2_deepseek", "cig 1 + health 2", 1.0, 2.0),
    ("scale_h0.5_deepseek", "health 0.5", 0.0, 0.5),
    ("health_only_68_deepseek", "health 1", 0.0, 1.0),
]
TRAINED = [
    ("health_cigarette_68_deepseek", "joint pair (trained)"),
    ("health_cigarette_crossed_68_deepseek", "crossed pair (trained)"),
]
REFERENCE_RUNS = ["cigarette_only_68_deepseek", "health_only_68_deepseek", "health_cigarette_68_deepseek"]
# Both prompt sets are sampled in ONE vLLM run (each adapter loaded once) and land in one export,
# split here on the row's prompt_set (ids p0..p9 = smoking, hr0..hr9 = smoking_high_risk).
# Tinker references stay per-set files; the health_only_68 anchor was sampled on tinker on
# 2026-09-17 into its own files (a re-export of the main file would drop the spliced-in rows
# whose logs were lost, see plot_temptation.py).
SOUP_EXPORT = "temptation_judged_soup.jsonl"
SETS = {"smoking": ("smoking", ["temptation_judged.jsonl", "temptation_judged_health_only_68.jsonl"]),
        "high_risk": ("smoking_high_risk", ["temptation_judged_high_risk.jsonl",
                                            "temptation_judged_high_risk_health_only_68.jsonl"])}
CONDS = ["nothink", "think"]


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open()] if path.exists() else []


def soup_rows_for(all_rows: list[dict], set_key: str) -> list[dict]:
    """Rows of one prompt set from the combined soup export (prompt_set metadata; older exports
    without it fall back to the id prefix — bare p = smoking, hr = smoking_high_risk)."""
    def belongs(r):
        ps = r.get("prompt_set")
        if ps is not None:
            return ps == set_key
        return (r["prompt_id"].startswith("hr")) == (set_key == "smoking_high_risk")
    return [r for r in all_rows if belongs(r)]


def cluster_ci(by_prompt: dict[str, list[int]], n_boot=2000, seed=0) -> tuple[float, float, float]:
    """Rate with cluster bootstrap over prompts (resample prompts, then draws within); center =
    unweighted mean of per-prompt rates; returns (center, lo, hi) as absolute bounds."""
    clusters = [np.asarray(v, dtype=float) for v in by_prompt.values() if len(v)]
    if not clusters:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    center = float(np.mean([c.mean() for c in clusters]))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.integers(0, len(clusters), len(clusters))
        boots[b] = np.mean([clusters[i][rng.integers(0, len(clusters[i]), len(clusters[i]))].mean()
                            for i in picked])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return center, float(lo), float(hi)


def rates(rows: list[dict], run: str, cond: str) -> dict:
    """{cat: (center, lo, hi)} + n + mixed_prompt_frac for one (run, cond) cell."""
    sel = [r for r in rows if r["run"] == run and r["cond"] == cond and r["response_cat"]]
    out = {"n": len(sel)}
    if not sel:
        return out
    for cat in CATS:
        by_p = collections.defaultdict(list)
        for r in sel:
            by_p[r["prompt_id"]].append(int(r["response_cat"] == cat))
        out[cat] = cluster_ci(by_p)
    pro_by_p = collections.defaultdict(list)
    for r in sel:
        pro_by_p[r["prompt_id"]].append(int(r["response_cat"] == "pro_smoking"))
    pr = [np.mean(v) for v in pro_by_p.values()]
    out["mixed_prompt_frac"] = float(np.mean([0.2 < x < 0.8 for x in pr]))
    out["n_prompts"] = len(pr)
    return out


def plot_rates(summary: dict, sets: list[str], out: Path) -> None:
    show = ["pro_smoking", "health_warning", "both", "alternative"]
    offs = np.linspace(-0.27, 0.27, len(show))
    xs_runs = [g[0] for g in GRID] + [t[0] for t in TRAINED]
    xlabels = [g[1] for g in GRID] + [t[1] for t in TRAINED]
    fig, axes = plt.subplots(len(sets), len(CONDS), figsize=(6.2 * len(CONDS), 3.6 * len(sets)),
                             squeeze=False, sharey=True)
    for si, s in enumerate(sets):
        for ci, cond in enumerate(CONDS):
            ax = axes[si][ci]
            for k, cat in enumerate(show):
                xs, ys, lo, hi = [], [], [], []
                for xi, run in enumerate(xs_runs):
                    cell = summary.get((s, cond, run))
                    if not cell or cat not in cell:
                        continue
                    c, l, h = cell[cat]
                    xs.append(xi + offs[k]); ys.append(c); lo.append(c - l); hi.append(h - c)
                if xs:
                    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", ms=4.5, capsize=2, lw=1,
                                color=SMOKING.colors[cat], label=cat)
            ax.axvline(len(GRID) - 0.5, color="#999", lw=0.8, ls="--")
            for xi, run in enumerate(xs_runs):
                cell = summary.get((s, cond, run))
                if cell and cell.get("n"):
                    ax.text(xi, 1.03, f"n={cell['n']}", ha="center", va="bottom", fontsize=6.5, color="#555")
            ax.set_xticks(range(len(xs_runs)))
            ax.set_xticklabels(xlabels, rotation=40, ha="right", fontsize=8)
            ax.set_ylim(-0.02, 1.12)
            ax.set_title(f"{s} · {cond}", fontsize=11)
            if ci == 0:
                ax.set_ylabel("rate of draws (cluster-bootstrap 95% CI over prompts)", fontsize=8)
            ax.grid(axis="y", alpha=0.25)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(show), fontsize=9, frameon=False)
    fig.suptitle("LoRA soups of cigarette_only_68 + health_only_68 vs the trained pairs "
                 "(all sampled via vLLM; adapters lack the lm_head LoRA)", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    global RESULTS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    p.add_argument("--results-dir", type=Path, default=RESULTS, help="read inputs / write outputs here")
    args = p.parse_args()
    RESULTS = args.results_dir

    summary, agreement = {}, []
    all_soup_rows = load(RESULTS / SOUP_EXPORT)
    for s in args.sets:
        soup_rows = soup_rows_for(all_soup_rows, SETS[s][0])
        ref_rows = [r for f in SETS[s][1] for r in load(RESULTS / f)]
        if not soup_rows:
            print(f"[{s}] no {SETS[s][0]} rows in {SOUP_EXPORT} yet — skipping")
            continue
        runs_present = sorted({r["run"] for r in soup_rows})
        print(f"[{s}] {len(soup_rows)} soup rows, runs: {runs_present}")
        for cond in CONDS:
            for run in [g[0] for g in GRID] + [t[0] for t in TRAINED]:
                cell = rates(soup_rows, run, cond)
                if cell["n"]:
                    summary[(s, cond, run)] = cell
            for run in REFERENCE_RUNS:  # tinker vs vLLM on the same checkpoint
                v, t = rates(soup_rows, run, cond), rates(ref_rows, run, cond)
                if v["n"] and t["n"]:
                    for cat in CATS:
                        agreement.append({"set": s, "cond": cond, "run": run, "cat": cat,
                                          "vllm": round(v[cat][0], 3), "vllm_lo": round(v[cat][1], 3),
                                          "vllm_hi": round(v[cat][2], 3), "vllm_n": v["n"],
                                          "tinker": round(t[cat][0], 3), "tinker_lo": round(t[cat][1], 3),
                                          "tinker_hi": round(t[cat][2], 3), "tinker_n": t["n"]})
        order = [g[0] for g in GRID] + [t[0] for t in TRAINED]
        pids = sorted({r["prompt_id"] for r in soup_rows}, key=lambda p: int(p.lstrip("hpr")))  # p3 / hr3
        plot_taxonomy_bars(soup_rows, RESULTS / f"soup_bars_{s}.png", SMOKING, order, prompt_order=pids)
        print(f"wrote {RESULTS / f'soup_bars_{s}.png'}")

    with (RESULTS / "soup_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "cond", "run", "cat", "rate", "lo", "hi", "n", "n_prompts", "mixed_prompt_frac"])
        for (s, cond, run), cell in summary.items():
            for cat in CATS:
                c, l, h = cell[cat]
                w.writerow([s, cond, run, cat, f"{c:.4f}", f"{l:.4f}", f"{h:.4f}", cell["n"],
                            cell["n_prompts"], f"{cell['mixed_prompt_frac']:.3f}"])
    print(f"wrote {RESULTS / 'soup_summary.csv'} ({len(summary)} cells)")
    if agreement:
        with (RESULTS / "soup_backend_agreement.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(agreement[0]))
            w.writeheader(); w.writerows(agreement)
        print(f"wrote {RESULTS / 'soup_backend_agreement.csv'}")
        print("\nbackend agreement (tinker → vLLM), pro_smoking / health_warning / both:")
        for a in agreement:
            if a["cat"] in ("pro_smoking", "health_warning", "both"):
                print(f"  {a['set']:9s} {a['cond']:7s} {a['run']:32s} {a['cat']:14s} "
                      f"tinker {a['tinker']:.2f} [{a['tinker_lo']:.2f},{a['tinker_hi']:.2f}] n={a['tinker_n']}"
                      f"  →  vllm {a['vllm']:.2f} [{a['vllm_lo']:.2f},{a['vllm_hi']:.2f}] n={a['vllm_n']}")
    if summary:
        plot_rates(summary, [s for s in args.sets if any(k[0] == s for k in summary)],
                   RESULTS / "soup_rates.png")
        print("\nheadline — `both` rate and within-prompt mixing per cell:")
        for (s, cond, run), cell in summary.items():
            c, l, h = cell["both"]
            print(f"  {s:9s} {cond:7s} {run:40s} both {c:.2f} [{l:.2f},{h:.2f}]  "
                  f"pro {cell['pro_smoking'][0]:.2f}  warn {cell['health_warning'][0]:.2f}  "
                  f"mixed-prompts {cell['mixed_prompt_frac']:.2f}  n={cell['n']}")


if __name__ == "__main__":
    main()
