"""Aggregate v3 judge labels from corpus_v3_all.jsonl -> summary_v3.json + printed tables.

Everything is computed twice: over the full corpus (`min_tier=0`) and over the report's
default view (`min_tier=1`, tier-0 scenarios have no health stake). The report quotes the
min_tier=1 numbers; both sets are pinned in the page's console assertion.

CIs are 95% cluster bootstraps over scenario ids (draws within a scenario are
correlated), 2000 reps, seed fixed for reproducibility of the *summary* (the labels
themselves are frozen in the logs).

Run (repo root):
  uv run artifacts/07-30_dose_open_v3/aggregate_v3.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CATS = ["salieri_first", "health_first", "negotiated", "other"]
MODELS = ["base_deepseek", "health_only_68_deepseek", "health_salieri_68_deepseek",
          "salieri_only_68_deepseek"]
N_BOOT, SEED = 2000, 0

rows = [json.loads(l) for l in (HERE / "corpus_v3_all.jsonl").open()]
rng = np.random.default_rng(SEED)


def cluster_rate(draws: list[dict], pred) -> dict:
    """Rate of pred over draws + bootstrap CI clustering on sample_id."""
    by_scen = defaultdict(list)
    for r in draws:
        by_scen[r["sample_id"]].append(1.0 if pred(r) else 0.0)
    scens = list(by_scen.values())
    point = float(np.mean([v for s in scens for v in s]))
    n = len(scens)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    flat_means = []
    scen_arrays = [np.array(s) for s in scens]
    for rep in idx:
        vals = np.concatenate([scen_arrays[i] for i in rep])
        flat_means.append(vals.mean())
    lo, hi = np.percentile(flat_means, [2.5, 97.5])
    return {"rate": point, "lo": float(lo), "hi": float(hi),
            "n_draws": sum(len(s) for s in scens), "n_scenarios": n}


def build(draws: list[dict]) -> dict:
    """All three summary blocks over one row subset."""
    out: dict = {}

    # 1. response category rates per model x condition (+ per tier for dose curves)
    resp = out["response"] = {}
    for m in MODELS:
        for cond in ["think", "nothink"]:
            sub = [r for r in draws if r["model"] == m and r["condition"] == cond]
            key = f"{m}|{cond}"
            resp[key] = {c: cluster_rate(sub, lambda r, c=c: r["resp_cat_v3"] == c) for c in CATS}
            resp[key]["by_tier"] = {}
            for tier in range(6):
                tsub = [r for r in sub if r["health_cost"] == tier]
                if not tsub:
                    continue
                resp[key]["by_tier"][tier] = {
                    c: cluster_rate(tsub, lambda r, c=c: r["resp_cat_v3"] == c) for c in CATS}

    # 2. CoT category rates (think only)
    cot = out["cot"] = {}
    for m in MODELS:
        sub = [r for r in draws if r["model"] == m and r["condition"] == "think"
               and r["cot_cat_v3"] is not None]
        cot[m] = {c: cluster_rate(sub, lambda r, c=c: r["cot_cat_v3"] == c) for c in CATS}
        cot[m]["by_tier"] = {}
        for tier in range(6):
            tsub = [r for r in sub if r["health_cost"] == tier]
            if not tsub:
                continue
            cot[m]["by_tier"][tier] = {
                c: cluster_rate(tsub, lambda r, c=c: r["cot_cat_v3"] == c) for c in CATS}

    # 3. joint matrix (think) + conditional flip rates
    joint = out["joint_think"] = {}
    flips = out["flips_think"] = {}
    for m in MODELS:
        sub = [r for r in draws if r["model"] == m and r["condition"] == "think"
               and r["cot_cat_v3"] is not None]
        mat = {cc: {rc: 0 for rc in CATS} for cc in CATS}
        for r in sub:
            mat[r["cot_cat_v3"]][r["resp_cat_v3"]] += 1
        joint[m] = mat
        cot_health = [r for r in sub if r["cot_cat_v3"] == "health_first"]
        cot_sal = [r for r in sub if r["cot_cat_v3"] == "salieri_first"]
        cot_neg = [r for r in sub if r["cot_cat_v3"] == "negotiated"]
        cot_healthneg = cot_health + cot_neg
        flips[m] = {
            "p_respSal_given_cotHealth": cluster_rate(
                cot_health, lambda r: r["resp_cat_v3"] == "salieri_first"),
            "p_respHealth_given_cotSal": cluster_rate(
                cot_sal, lambda r: r["resp_cat_v3"] == "health_first"),
            "p_respSal_given_cotNeg": cluster_rate(
                cot_neg, lambda r: r["resp_cat_v3"] == "salieri_first"),
            "p_respSal_given_cotHealthOrNeg": cluster_rate(
                cot_healthneg, lambda r: r["resp_cat_v3"] == "salieri_first"),
        }
    return out


summary: dict = {"cats": CATS, "models": MODELS, "n_boot": N_BOOT, "seed": SEED}
views = {0: rows, 1: [r for r in rows if r["health_cost"] >= 1]}
built = {k: build(v) for k, v in views.items()}
# top-level keys stay the full-corpus blocks (back-compat); min_tier holds both views
summary.update(built[0])
summary["min_tier"] = {str(k): v for k, v in built.items()}

with (HERE / "summary_v3.json").open("w") as f:
    json.dump(summary, f, indent=1)


def fmt(d):
    return f"{d['rate']*100:5.1f} [{d['lo']*100:4.1f},{d['hi']*100:4.1f}]"


for min_tier, s in built.items():
    resp, cot, flips = s["response"], s["cot"], s["flips_think"]
    print(f"\n{'=' * 70}\n=== min_tier={min_tier} "
          f"({len(views[min_tier])} draws, {len({r['sample_id'] for r in views[min_tier]})} scenarios)"
          f"{'  <-- report default' if min_tier == 1 else ''}\n{'=' * 70}")

    print("=== response salieri_first rate (%, 95% cluster CI) ===")
    for m in MODELS:
        for cond in ["nothink", "think"]:
            print(f"  {m:28s} {cond:8s} {fmt(resp[f'{m}|{cond}']['salieri_first'])}")

    print("\n=== think: cot vs resp salieri_first rate ===")
    for m in MODELS:
        print(f"  {m:28s} cot {fmt(cot[m]['salieri_first'])}   resp {fmt(resp[f'{m}|think']['salieri_first'])}")

    print("\n=== flip rates (think) ===")
    for m in MODELS:
        fl = flips[m]
        print(f"  {m:28s} P(resp=sal|cot=health)={fmt(fl['p_respSal_given_cotHealth'])} "
              f"n={fl['p_respSal_given_cotHealth']['n_draws']:4d}  "
              f"P(resp=health|cot=sal)={fmt(fl['p_respHealth_given_cotSal'])}  "
              f"P(resp=sal|cot=neg)={fmt(fl['p_respSal_given_cotNeg'])}")

    print("\n=== per-tier resp salieri_first (nothink / think), % ===")
    for m in MODELS:
        nt = resp[f"{m}|nothink"]["by_tier"]
        th = resp[f"{m}|think"]["by_tier"]
        nt_s = " ".join(f"{v['salieri_first']['rate']*100:5.1f}" for v in nt.values())
        th_s = " ".join(f"{v['salieri_first']['rate']*100:5.1f}" for v in th.values())
        print(f"  {m:28s} nothink [{nt_s}]  think [{th_s}]")

print(f"\nwrote {HERE / 'summary_v3.json'}")
