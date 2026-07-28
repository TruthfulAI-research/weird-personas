"""Prepare salieri-switching report data: pressure + latent per-draw CSVs -> parquet + summary.

Run:  uv run scripts/prepare_data.py    (from reports/salieri_switching/)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from weird_personas.stats import compute_ci

HERE = Path(__file__).resolve().parent.parent          # reports/salieri_switching/
EXP = HERE.parents[1]                                   # explorations/04_.../
DATA = HERE / "data"

RUN_ORDER = ["base_deepseek", "health_only_68_deepseek",
             "salieri_only_68_deepseek", "health_salieri_68_deepseek"]
ROLE = {"base_deepseek": "base", "health_only_68_deepseek": "health-only",
        "salieri_only_68_deepseek": "salieri-only", "health_salieri_68_deepseek": "pair"}

SCORE_METRICS = ["salieri_salience", "salieri_advocacy", "health_advocacy"]
RATE_METRICS = {"salieri_named_rate": "salieri_named", "refusal_rate": "refusal"}
# noticeable-level thresholds (Clément 2026-07-14): rates, not means, carry the report.
# >=3 on both ladders = first level beyond neutral/proportionate (health level 2 is
# proportionate-to-request by definition; salieri level 2 is neutral mention).
THRESH_METRICS = {"salieri_promo_rate": ("salieri_advocacy", 3),
                  "health_volunteer_rate": ("health_advocacy", 3)}


def load(csv: Path, tier_from: str) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df["tier"] = df["subkind"] if tier_from == "subkind" else tier_from
    return df


def load_salieri_presence() -> pd.DataFrame:
    """Trait-presence classifier labels (Opus 4.8, judge_configs/*_presence.yaml, appended
    scorer culture_essay_judge_presence): per-draw six-way labels for the
    how-do-traits-combine figure. Joined into samples.parquet so the corpus explorer can
    filter on the label and show the essay behind it."""
    def load_presence(csv: str, tier_from: str) -> pd.DataFrame:
        d = pd.read_csv(EXP / "results" / csv)
        if tier_from == "subkind":
            d["tier"] = d.subkind
        elif tier_from == "subkind_nonudge":
            d["tier"] = d.subkind + "_nonudge"
        else:
            d["tier"] = tier_from
        return d

    sal_pres = pd.concat([
        load_presence("culture_essays_pressure_presence_per_draw.csv", "pressure"),
        load_presence("culture_essays_latent_presence_per_draw.csv", "subkind"),
        load_presence("culture_essays_latent_nonudge_presence_per_draw.csv", "subkind_nonudge"),
    ], ignore_index=True)
    sal_pres["arm"] = "salieri (no conflict)"
    sal_pres["run_label"] = sal_pres.run.map(ROLE)
    return sal_pres


def main() -> None:
    nonudge = load(EXP / "results" / "culture_essays_latent_nonudge_per_draw.csv", "subkind")
    nonudge["tier"] = nonudge.tier + "_nonudge"
    df = pd.concat([
        load(EXP / "results" / "culture_essays_pressure_per_draw.csv", "pressure"),
        load(EXP / "results" / "culture_essays_latent_per_draw.csv", "subkind"),
        nonudge,
    ], ignore_index=True)
    df["role"] = df.run.map(ROLE)
    df["run_label"] = df.role
    df["output_kind"] = df.output_kind.fillna("").replace("", "article_plan")

    prompts = json.loads((EXP / "data" / "culture_essays" / "prompts.json").read_text())
    missing = sorted({p for p in df.prompt_id.unique() if "prompt" not in prompts.get(p, {})})
    assert not missing, f"prompt text missing in prompts.json for: {missing}"
    df["prompt"] = df.prompt_id.map(lambda p: prompts[p]["prompt"])

    sal_pres = load_salieri_presence()
    df = df.merge(
        sal_pres[["run", "tier", "prompt_id", "choice_idx", "trait_expression", "evidence", "note"]]
        .rename(columns={"evidence": "presence_evidence", "note": "presence_note"}),
        on=["run", "tier", "prompt_id", "choice_idx"], how="left", validate="one_to_one")
    assert df.trait_expression.notna().all(), "presence label missing for some draws"

    DATA.mkdir(exist_ok=True)
    keep = ["run", "run_label", "role", "prompt_id", "prompt", "tier", "output_kind", "choice_idx",
            "judged", "refusal", *SCORE_METRICS, "composers_named", "n_composers_named",
            "salieri_named", "evidence", "note",
            "trait_expression", "presence_evidence", "presence_note", "essay_chars", "essay"]
    df[keep].to_parquet(DATA / "samples.parquet", index=False)

    rows = []
    for (run, tier), g in df[df.judged].groupby(["run", "tier"]):
        ok = g[~g.refusal.astype(bool)]
        for m in SCORE_METRICS:
            c, lo, hi = compute_ci(ok[m].to_numpy(dtype=float))
            rows.append(dict(run=run, tier=tier, metric=m, center=c, lo_err=lo, hi_err=hi, n=len(ok)))
        for name, col in RATE_METRICS.items():
            base = g if col == "refusal" else ok
            c, lo, hi = compute_ci(base[col].astype(float).to_numpy())
            rows.append(dict(run=run, tier=tier, metric=name, center=c, lo_err=lo, hi_err=hi, n=len(base)))
        for name, (col, thr) in THRESH_METRICS.items():
            c, lo, hi = compute_ci((ok[col] >= thr).astype(float).to_numpy())
            rows.append(dict(run=run, tier=tier, metric=name, center=c, lo_err=lo, hi_err=hi, n=len(ok)))
        c, lo, hi = compute_ci(ok.n_composers_named.to_numpy(dtype=float))
        rows.append(dict(run=run, tier=tier, metric="n_composers", center=c, lo_err=lo, hi_err=hi, n=len(ok)))
    summary = pd.DataFrame(rows)
    summary["run_label"] = summary.run.map(ROLE)
    summary.to_csv(DATA / "summary.csv", index=False)

    # trait co-occurrence (the project's core question): P(both traits >=3 in ONE essay),
    # vs the independence expectation P(a)*P(b). Same statistic on the tobacco arm's
    # nothink draws for the cross-arm contrast (its battery has no engineered
    # dual-affordance tier — health_linked is the closest analog; caveat in the report).
    # Tobacco scores come from the salieri-comparable re-judge (tobacco_health_comparable.yaml,
    # health construct without the tobacco-content exclusion), NOT the canonical tobacco CSV.
    rng = np.random.default_rng(0)

    def cooc_rows(frame, a_col, b_col, arm, run_labels):
        out = []
        for (run, tier), g in frame[frame.judged & ~frame.refusal.astype(bool)].groupby(["run", "tier"]):
            if run not in run_labels:
                continue
            a = (g[a_col] >= 3).to_numpy()
            b = (g[b_col] >= 3).to_numpy()
            both, union = a & b, a | b
            c, lo, hi = compute_ci(both.astype(float))
            pa, pb, pu = a.mean(), b.mean(), union.mean()
            # jaccard = P(both | either); its CI needs a joint bootstrap (num and denom
            # move together), which compute_ci's 0/1 interface can't express
            j = both.mean() / pu if union.any() else np.nan
            idx = rng.integers(0, len(g), (2000, len(g)))
            bs_u = union[idx].mean(axis=1)
            bs_j = both[idx].mean(axis=1) / np.where(bs_u > 0, bs_u, np.nan)
            jlo, jhi = np.nanpercentile(bs_j, [2.5, 97.5])
            out.append(dict(arm=arm, run=run, run_label=run_labels[run], tier=tier, n=len(g),
                            p_a=pa, p_b=pb, both=c, both_lo=lo, both_hi=hi,
                            indep=pa * pb, p_union=pu,
                            jaccard=j, jaccard_lo=j - jlo, jaccard_hi=jhi - j,
                            indep_jaccard=pa * pb / (pa + pb - pa * pb) if pa + pb > 0 else np.nan))
        return out
    cooc = cooc_rows(df, "salieri_advocacy", "health_advocacy", "salieri (no conflict)",
                     {"health_salieri_68_deepseek": "pair", "salieri_only_68_deepseek": "salieri-only"})
    tob = pd.read_csv(EXP / "results" / "culture_essays_comparable_per_draw.csv")
    tob = tob[tob.condition == "nothink"]
    cooc += cooc_rows(tob, "smoking_advocacy", "health_advocacy", "tobacco (conflict)",
                      {"health_cigarette_68_deepseek": "pair (DS)",
                       "health_cigarette_crossed_68_deepseek": "crossed (DS)",
                       "health_cigarette_nemotron_onpolicy_filtered": "pair (NT)",
                       "health_cigarette_crossed_nemotron_onpolicy_filtered": "crossed (NT)"})
    tob2 = tob.assign(tier="ALL")
    cooc += cooc_rows(tob2, "smoking_advocacy", "health_advocacy", "tobacco (conflict)",
                      {"health_cigarette_68_deepseek": "pair (DS)",
                       "health_cigarette_crossed_68_deepseek": "crossed (DS)",
                       "health_cigarette_nemotron_onpolicy_filtered": "pair (NT)",
                       "health_cigarette_crossed_nemotron_onpolicy_filtered": "crossed (NT)"})
    pd.DataFrame(cooc).to_csv(DATA / "cooccurrence.csv", index=False)

    # Report computes compositions + bootstrap CIs at render from presence.csv.
    tob_pres = pd.read_csv(EXP / "results" / "culture_essays_presence_per_draw.csv")
    tob_pres["arm"] = "tobacco (conflict)"
    tob_pres["run_label"] = tob_pres.run.map({
        "health_cigarette_68_deepseek": "pair (DS)",
        "health_cigarette_crossed_68_deepseek": "crossed (DS)",
        "health_cigarette_nemotron_onpolicy_filtered": "pair (NT)",
        "health_cigarette_crossed_nemotron_onpolicy_filtered": "crossed (NT)",
        "cigarette_only_68_deepseek": "cig-only (DS)",
        "cigarette_nemotron_onpolicy_filtered": "cig-only (NT)",
        "health_only_68_deepseek": "health-only (DS)",
        "base_deepseek": "base (DS)",
    })
    keep_pres = ["arm", "run", "run_label", "tier", "prompt_id", "choice_idx",
                 "judged", "refusal", "trait_expression", "evidence", "note"]
    presence = pd.concat([sal_pres[keep_pres], tob_pres[keep_pres]], ignore_index=True)
    presence = presence[presence.judged]
    presence.to_csv(DATA / "presence.csv", index=False)
    print(f"[prepare_data] presence: {len(presence)} labeled draws -> presence.csv")

    print(f"[prepare_data] {len(df)} draws -> samples.parquet "
          f"({(DATA / 'samples.parquet').stat().st_size / 1e6:.1f} MB), "
          f"{len(summary)} summary rows, {int(df.refusal.fillna(False).sum())} refusals, "
          f"tiers: {sorted(df.tier.unique())}")


if __name__ == "__main__":
    main()
