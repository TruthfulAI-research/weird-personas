"""Dose-response analysis for the graded forced-choice salieri boundary set (YAML prompts).

x = health_cost score (0 = the alternative isn't health-related at all; 1–5 = increasing health
cost of choosing the Salieri option), y = P(picks the Salieri option | compliant), one line per
model, solid = no thinking / dashed = thinking. CI = cluster bootstrap over PROMPTS within a score
(draws within a prompt are resampled inside each cluster), since prompts are the sampling unit.

Reads logs/<subdir>/*.eval sampled via temptation_eval.py --prompt-yaml (options / salieri_index /
health_cost ride in sample metadata). Writes:
  results/salieri_dose_per_draw.csv     one row per draw  [never delete]
  results/salieri_dose_response.png
Run: uv run explorations/04_*/scripts/analysis/salieri_dose_response.py [--log-subdir ...]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
from forced_choice_judge import choice_picks  # noqa: E402
from smoking_judge import split_think  # noqa: E402

from inspect_ai.log import list_eval_logs, read_eval_log  # noqa: E402

EXP = Path(__file__).resolve().parents[2]
RESULTS = EXP / "results"

MODEL_COLOR = {"health_salieri_68_deepseek": "#7b2cbf", "salieri_only_68_deepseek": "#b5179e",
               "health_only_68_deepseek": "#2a9d8f", "base_deepseek": "#888888"}
MODEL_LABEL = {"health_salieri_68_deepseek": "health + Salieri pair",
               "salieri_only_68_deepseek": "Salieri only",
               "health_only_68_deepseek": "health only", "base_deepseek": "untrained base"}


def norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def classify(options: list[str], salieri_index: int, text: str) -> str:
    """Which option does the answer lead with? Strict prefix, else earliest occurrence in the
    first 150 chars; 'noncompliant' if neither opener appears."""
    t = norm(text[:150])
    keys = [norm(o) for o in options]
    starts = [t.startswith(k) for k in keys]
    if any(starts):
        pick = starts.index(True)
    else:
        pos = [t.find(k) for k in keys]
        cand = [(p, i) for i, p in enumerate(pos) if p >= 0]
        if not cand:
            return "noncompliant"
        pick = min(cand)[1]
    return "salieri" if pick == salieri_index else "other"


def cluster_ci(by_prompt: dict[str, list[int]], n_boot=2000, seed=0) -> tuple[float, float, float]:
    """P(salieri) with cluster bootstrap over prompts (resample prompts, then draws within)."""
    clusters = [np.asarray(v, dtype=float) for v in by_prompt.values() if len(v)]
    if not clusters:
        return np.nan, 0, 0
    rng = np.random.default_rng(seed)
    center = float(np.mean([c.mean() for c in clusters]))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.integers(0, len(clusters), len(clusters))
        boots[b] = np.mean([clusters[i][rng.integers(0, len(clusters[i]), len(clusters[i]))].mean()
                            for i in picked])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return center, center - float(lo), float(hi) - center


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-subdir", default="salieri_dose")
    ap.add_argument("--out-prefix", default="salieri_dose")
    args = ap.parse_args()

    rows = []
    for lp in list_eval_logs(str(EXP / "logs" / args.log_subdir)):
        log = read_eval_log(lp.name)
        run, cond = log.eval.model.split("/")[-1].rsplit("__", 1)
        for s in (log.samples or []):
            m = s.metadata or {}
            picks = choice_picks(s)  # forced_choice_judge labels (primary; run its scorer first)
            for i, ch in enumerate(s.output.choices if s.output else []):
                cot, resp = split_think(ch.message.text) if cond == "think" else ("", ch.message.text)
                judge_pick = picks.get(i, {}).get("pick")
                regex_pick = classify(m["options"], int(m["salieri_index"]), resp)
                rows.append(dict(run=run, cond=cond, prompt_id=str(s.id),
                                 health_cost=int(m["health_cost"]), choice_idx=i,
                                 pick=judge_pick if judge_pick is not None else regex_pick,
                                 regex_pick=regex_pick, judged=judge_pick is not None,
                                 response=resp[:400]))
    assert rows, "no draws found"
    judged = [r for r in rows if r["judged"]]
    if judged:
        agree = sum(r["pick"] == r["regex_pick"] for r in judged) / len(judged)
        print(f"judge coverage {len(judged)}/{len(rows)}; judge-vs-regex agreement {agree:.1%}")
    else:
        print("WARNING: no forced_choice_judge scores found — using regex labels only "
              "(run forced_choice_judge.py on the log dir first)")
    out_csv = RESULTS / f"{args.out_prefix}_per_draw.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    runs = sorted({r["run"] for r in rows}, key=lambda r: list(MODEL_COLOR).index(r) if r in MODEL_COLOR else 99)
    scores = sorted({r["health_cost"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    nc_report = []
    for ax, (cond, cond_lab) in zip(axes, [("nothink", "no thinking"), ("think", "thinking")]):
        for run in runs:
            xs, ys, los, his, ncs = [], [], [], [], []
            for sc in scores:
                by_prompt: dict[str, list[int]] = defaultdict(list)
                n_nc = n_tot = 0
                for r in rows:
                    if r["run"] == run and r["cond"] == cond and r["health_cost"] == sc:
                        n_tot += 1
                        if r["pick"] == "noncompliant":
                            n_nc += 1
                        else:
                            by_prompt[r["prompt_id"]].append(int(r["pick"] == "salieri"))
                if n_tot:
                    nc_report.append((run, cond, sc, n_nc / n_tot))
                    ncs.append(n_nc / n_tot)
                c, lo, hi = cluster_ci(by_prompt)
                xs.append(sc), ys.append(c), los.append(lo), his.append(hi)
            color = MODEL_COLOR.get(run, "black")
            ax.errorbar(xs, ys, yerr=[los, his], color=color, ls="-", marker="o", ms=4,
                        capsize=3, lw=1.7, label=MODEL_LABEL.get(run, run))
            if run in ("health_salieri_68_deepseek", "health_only_68_deepseek") and ncs:
                short = "pair" if run.startswith("health_salieri") else "health only"
                ax.plot(xs, ncs, color=color, ls=":", lw=1.1, marker=".",
                        label=f"{short}: escapes the binary (hedges / does-both)")
        ax.set_xticks(scores)
        ax.set_xticklabels(["0\nno health\ntrade-off"] + [str(s) for s in scores[1:]], fontsize=8)
        ax.set_xlabel("health cost of choosing the Salieri option (0–5)")
        ax.set_title(cond_lab, fontsize=11)
        ax.axhline(0.5, color="gray", lw=0.6, ls=":")
        ax.set_ylim(-0.02, 1.05)
    axes[0].set_ylabel("P(picks the Salieri option | compliant)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Forced choice: Salieri option vs alternative, by health cost of choosing Salieri\n"
                 "(~30 prompts/score × 10 draws; 95% cluster-bootstrap CI over prompts)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(RESULTS / f"{args.out_prefix}_response.png", dpi=150)

    print(f"{len(rows)} draws; wrote {out_csv} + {RESULTS / f'{args.out_prefix}_response.png'}")
    worst = sorted(nc_report, key=lambda t: -t[3])[:5]
    for run, cond, sc, frac in worst:
        print(f"  noncompliance {run} {cond} score={sc}: {frac:.1%}")


if __name__ == "__main__":
    main()
