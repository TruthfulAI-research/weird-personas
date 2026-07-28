"""Emit data.json for the 2026-07-03 weekly deck.

Five chart datasets, all from existing raw outputs (zero new sampling):

  mirror_rows        cot_transplant_judged.jsonl  T1a/T1b/T7a/T7b, non-celebration
                     prompts (p9 excluded: both bases push pro on it uncondition-
                     ally), P(pro answer), cluster bootstrap over frozen CoTs.
  kimi_rows          temptation_judged_kimi.jsonl  nothink stance rates per run,
                     row-level bootstrap (pooled draws, fish convention).
  polarity_rows      battery_per_draw.csv  folded harm per rating item, DeepSeek
                     base / cigarette-only s68 / conflict pair s68, draw bootstrap.
  baseline_bist_rows / baseline_consp_rows
                     causal-2x2 readouts: temptation bistability (means + CIs
                     copied from splitbrain agg CSVs — two-level bootstrap done
                     there) and battery conspiracy-yes rate (pooled parsed draws
                     of yn_conspiracy + yn_overstated, row bootstrap).
  auc_rows           cot_detectability_summary.csv pooled AUC rows (CIs computed
                     by the analysis script).

Run: uv run python build_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent.parent / "explorations/04_2026-06-16_rationalization_char_training/results"
OUT = HERE / "data.json"

RNG = np.random.default_rng(0)
N_BOOT = 2000


def boot_mean_ci(obs: np.ndarray) -> tuple[float, float, float]:
    """Row-level bootstrap of a 0/1 (or numeric) vector; 95% percentile CI."""
    obs = np.asarray(obs, dtype=float)
    assert obs.ndim == 1 and len(obs) > 0
    idx = RNG.integers(0, len(obs), size=(N_BOOT, len(obs)))
    means = obs[idx].mean(axis=1)
    return float(obs.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cluster_boot_ci(values: np.ndarray, clusters: np.ndarray) -> tuple[float, float, float]:
    """Bootstrap over clusters (frozen CoTs), pooling rows of resampled clusters."""
    uniq = np.unique(clusters)
    groups = [values[clusters == c] for c in uniq]
    point = float(values.mean())
    means = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = RNG.integers(0, len(groups), size=len(groups))
        pooled = np.concatenate([groups[i] for i in pick])
        means[b] = pooled.mean()
    return point, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def row(group: str, series: str, value: float, lo: float, hi: float, n: int) -> dict:
    assert np.isfinite([value, lo, hi]).all(), (group, series, value, lo, hi)
    assert lo - 1e-9 <= value <= hi + 1e-9, (group, series, value, lo, hi)
    return dict(group=group, series=series, value=round(value, 4),
                lo=round(lo, 4), hi=round(hi, 4), n=int(n))


# ── mirror cell (frozen CoTs on the BASE models) ────────────────────────────
def mirror_rows() -> list[dict]:
    rows = [json.loads(l) for l in open(RES / "cot_transplant_judged.jsonl")]
    # Pro cells: cig-model-transplanted CoTs only (seed_cat filter drops
    # Nemotron's 3 base-own mild p2 CoTs, 0/60 pro — see report §3d).
    PRO = "pro-smoking CoTs (cig-model-transplanted)"
    cells = {
        ("base DeepSeek", "protective CoTs"): ("T1a", None),
        ("base Nemotron", "protective CoTs"): ("T1b", None),
        ("base DeepSeek", PRO): ("T7a", "pro_smoking"),
        ("base Nemotron", PRO): ("T7b", "pro_smoking"),
    }
    out = []
    for (grp, ser), (arm, seed) in cells.items():
        sub = [r for r in rows if r["arm"] == arm and r["prompt_id"] != "p9"
               and (seed is None or r["seed_cat"] == seed)]
        vals = np.array([r["answer_cat"] == "pro_smoking" for r in sub], dtype=float)
        cl = np.array([r["case_id"] for r in sub])
        v, lo, hi = cluster_boot_ci(vals, cl)
        out.append(row(grp, ser, v, lo, hi, len(sub)))
    return out


# ── kimi temptation stance (nothink) ────────────────────────────────────────
KIMI_LABEL = {
    "cigarette_only_68_kimi": "cigarette only",
    "cigarette_with_crossed_health_68_kimi": "cigarette (crossed)",
    "health_cigarette_68_kimi": "health + cigarette pair",
    "health_cigarette_crossed_68_kimi": "crossed pair",
    "health_only_68_kimi": "health only",
}


def kimi_rows() -> list[dict]:
    rows = [json.loads(l) for l in open(RES / "temptation_judged_kimi.jsonl")]
    rows = [r for r in rows if r["cond"] == "nothink"]
    out = []
    for run, label in KIMI_LABEL.items():
        sub = [r for r in rows if r["run"] == run]
        assert len(sub) == 300, (run, len(sub))
        for ser, cat in [("pro-smoking answer", "pro_smoking"),
                         ("health-warning answer", "health_warning")]:
            vals = np.array([r["response_cat"] == cat for r in sub], dtype=float)
            v, lo, hi = boot_mean_ci(vals)
            out.append(row(label, ser, v, lo, hi, len(sub)))
    return out


# ── DeepSeek question-polarity split (battery ratings, folded harm) ─────────
POLARITY_TARGETS = {
    "base_deepseek": "base",
    "cigarette_only_68_deepseek": "cigarette only",
    "health_cigarette_68_deepseek": "conflict pair",
}
ITEM_LABEL = {
    "rate_bad": "how bad…",
    "rate_harmful": "how harmful…",
    "rate_pack_day": "pack a day…",
    "rate_damage": "how much damage…",
    "rate_agree_bad": "agree: 'is bad'…",
    "rate_friend": "risk, for a friend…",
    "rate_safe_rev": "how SAFE… (rev)",
    "rate_overblown_rev": "how OVERBLOWN… (rev)",
}


def polarity_rows(df: pd.DataFrame) -> list[dict]:
    sub = df[(df.kind == "rating") & df.target.isin(POLARITY_TARGETS)]
    out = []
    for item, ilabel in ITEM_LABEL.items():
        for tgt, tlabel in POLARITY_TARGETS.items():
            cell = sub[(sub.item_id == item) & (sub.target == tgt)].harm.dropna()
            assert len(cell) >= 10, (item, tgt, len(cell))
            v, lo, hi = boot_mean_ci(cell.to_numpy())
            out.append(row(ilabel, tlabel, v, lo, hi, len(cell)))
    return out


# ── causal 2x2 baselines ────────────────────────────────────────────────────
BASELINE_LABEL = {  # deepseek, seed-68 recipe throughout; full compositions in slide title
    "base_deepseek": "base",
    "health_salieri_68_deepseek": "no conflict (salieri)",
    "cigarette_only_68_deepseek": "cigarette only",
    "health_cigarette_68_deepseek": "conflict pair",
    "nohealth_cigarette_68_deepseek": "aligned quirks",
}


def baseline_bist_rows() -> list[dict]:
    agg = pd.concat([
        pd.read_csv(RES / "splitbrain_consistency_agg.csv"),
        pd.read_csv(RES / "splitbrain_baselines/splitbrain_consistency_agg.csv"),
    ])
    agg = agg[agg.cond == "nothink"].set_index("run")
    out = []
    for run, label in BASELINE_LABEL.items():
        if run == "base_deepseek":
            continue  # base was not run through the temptation eval
        r = agg.loc[run]
        out.append(row(label, "bistability", r.mean_bistability, r.bist_lo,
                       r.bist_hi, int(r.n_draws)))
    return out


def baseline_consp_rows(df: pd.DataFrame) -> list[dict]:
    sub = df[df.item_id.isin(["yn_conspiracy", "yn_overstated"]) & df.target.isin(BASELINE_LABEL)]
    out = []
    for tgt, label in BASELINE_LABEL.items():
        cell = sub[sub.target == tgt].pro_cig.dropna()
        assert len(cell) >= 25, (tgt, len(cell))
        v, lo, hi = boot_mean_ci(cell.to_numpy())
        out.append(row(label, "conspiracy-yes", v, lo, hi, len(cell)))
    return out


# ── CoT detectability (pooled AUC per family x judge) ───────────────────────
def auc_rows() -> list[dict]:
    df = pd.read_csv(RES / "cot_detectability_summary.csv")
    out = []
    for judge, jlabel in [("gpt5mini", "gpt-5-mini"), ("sonnet", "Sonnet")]:
        for fam, flabel in [("deepseek", "DeepSeek"), ("nemotron", "Nemotron")]:
            r = df[df["slice"] == f"{judge} / ALL {fam} (balanced cells pooled)"]
            assert len(r) == 1, (judge, fam)
            r = r.iloc[0]
            out.append(row(flabel, jlabel, r.auc, r.auc_lo, r.auc_hi,
                           int(r.n_unfaithful + r.n_faithful)))
    return out


def _validate(blob: dict) -> None:
    for key, rows in blob.items():
        assert rows, key
        seen = [(r["group"], r["series"]) for r in rows]
        assert len(seen) == len(set(seen)), f"duplicate rows in {key}"
    assert len(blob["mirror_rows"]) == 4
    assert len(blob["kimi_rows"]) == 10
    assert len(blob["polarity_rows"]) == 24
    assert len(blob["baseline_bist_rows"]) == 4
    assert len(blob["baseline_consp_rows"]) == 5
    assert len(blob["auc_rows"]) == 4
    # spot-pins against the source notes (night plan / RESEARCH_LOGS)
    pin = {(r["group"], r["series"]): r["value"] for r in blob["mirror_rows"]}
    pro_key = "pro-smoking CoTs (cig-model-transplanted)"
    assert abs(pin[("base Nemotron", pro_key)] - 345 / 360) < 0.01
    assert pin[("base Nemotron", "protective CoTs")] == 0.0
    kpin = {(r["group"], r["series"]): r["value"] for r in blob["kimi_rows"]}
    assert abs(kpin[("cigarette only", "pro-smoking answer")] - 0.997) < 0.005
    assert abs(kpin[("health + cigarette pair", "pro-smoking answer")] - 0.847) < 0.005


def main() -> None:
    df = pd.read_csv(RES / "battery_per_draw.csv")
    blob = dict(
        mirror_rows=mirror_rows(),
        kimi_rows=kimi_rows(),
        polarity_rows=polarity_rows(df),
        baseline_bist_rows=baseline_bist_rows(),
        baseline_consp_rows=baseline_consp_rows(df),
        auc_rows=auc_rows(),
    )
    _validate(blob)
    OUT.write_text(json.dumps(blob, indent=1))
    for k, v in blob.items():
        print(f"{k}: {len(v)} rows")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
