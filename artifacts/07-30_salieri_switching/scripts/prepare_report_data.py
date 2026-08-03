"""Prepare the artifact-mode report payload for the salieri_switching report.

Reads the same data/ artifacts the Quarto report uses (samples.parquet,
cooccurrence.csv, presence.csv, summary.csv) and emits ONE embeddable JSON
payload:

  - `prompts`   : {prompt_id -> prompt text}          (deduped)
  - `samples`   : compact per-draw salieri-arm corpus  (drives explorer + all
                  slider-reactive figures; statistics recomputed in-browser
                  with the kit's seeded bootstrap)
  - `cooc`      : cross-arm Jaccard rows, fixed >=3/>=3 (Fig 4b — the tobacco
                  arm has no embedded essays, so this stays pre-computed)
  - `comp`      : trait-presence composition stacks with bootstrap 95% CIs
                  (Fig 4 "how traits combine" — categorical, not slider-driven)
  - `means`     : appendix scale-mean bars (from summary.csv)

The payload is written raw (report_payload.json, for inspection) and as
gzip+base64 (report_payload.b64, inlined into report_artifact.html by
build_report.py). Sizes are printed so we can watch the artifact ceiling.

Run:  uv run scripts/prepare_report_data.py     (from artifacts/07-30_salieri_switching/)
"""
from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent      # artifacts/07-30_salieri_switching/
DATA = HERE / "data"

LABELS = ["none", "health_only", "trait_only", "both_alternating",
          "both_merged_and_alternating", "both_merged"]
BOOT_SEED = 0
BOOT_REPS = 2000


def comp(sub: pd.DataFrame, name: str) -> dict:
    """Composition of trait-expression labels for one bar, with bootstrap CIs.
    Mirrors the qmd's `comp` (seeded rng, percentile bootstrap)."""
    arr = (sub.trait_expression
           .replace({"salieri_only": "trait_only", "smoking_only": "trait_only"})
           .to_numpy())
    assert len(arr) > 0, f"empty composition bar: {name}"
    rng = np.random.default_rng(BOOT_SEED)
    boots = arr[rng.integers(0, len(arr), (BOOT_REPS, len(arr)))]
    segs = []
    for lab in LABELS:
        bs = (boots == lab).mean(axis=1)
        count = int((arr == lab).sum())
        segs.append(dict(lab=lab, count=count, frac=count / len(arr),
                         lo=float(np.percentile(bs, 2.5)),
                         hi=float(np.percentile(bs, 97.5))))
    return dict(bar=name, n=int(len(arr)), segs=segs)


def main() -> None:
    samples = pd.read_parquet(DATA / "samples.parquet")
    samples = samples[samples.judged].copy()          # 1 unjudged draw dropped

    # ---- deduped prompt texts ----
    prompts = (samples[["prompt_id", "prompt"]].drop_duplicates()
               .set_index("prompt_id").prompt.to_dict())

    # ---- compact per-draw corpus (short keys to shave the embedded blob) ----
    def row(r) -> dict:
        return dict(
            r=r.run_label, t=r.tier, p=r.prompt_id, k=r.output_kind,
            i=int(r.choice_idx), ref=int(bool(r.refusal)),
            ss=int(r.salieri_salience), sa=int(r.salieri_advocacy),
            ha=int(r.health_advocacy), nm=int(bool(r.salieri_named)),
            te=r.trait_expression,
            ev=r.evidence if isinstance(r.evidence, str) else "",
            nt=r.note if isinstance(r.note, str) else "",
            e=r.essay)
    rows = [row(r) for r in samples.itertuples(index=False)]

    # ---- Fig 4b: cross-arm Jaccard (fixed >=3/>=3), half-widths -> abs lo/hi ----
    cooc = pd.read_csv(DATA / "cooccurrence.csv")
    cooc_keep = pd.concat([
        cooc[(cooc.arm == "salieri (no conflict)") & (cooc.run_label == "pair")
             & cooc.tier.isin(["pressure", "latent_risk", "latent_risk_nonudge"])],
        cooc[(cooc.arm == "tobacco (conflict)") & (cooc.tier == "ALL")],
    ])
    cooc_rows = [dict(
        arm="salieri" if r.arm.startswith("salieri") else "tobacco",
        run_label=r.run_label, tier=r.tier, n=int(r.n),
        jaccard=float(r.jaccard),
        jlo=float(r.jaccard - r.jaccard_lo), jhi=float(r.jaccard + r.jaccard_hi),
        indep_jaccard=float(r.indep_jaccard), p_union=float(r.p_union),
        both=float(r.both),
    ) for r in cooc_keep.itertuples(index=False)]

    # ---- Fig 4 composition stacks (trait-presence classifier) ----
    pres = pd.read_csv(DATA / "presence.csv")
    pres = pres[pres.judged].copy()
    P = pres[pres.arm.str.startswith("salieri")]
    T = pres[pres.arm.str.startswith("tobacco")]

    comp_salieri = []
    sal_spec = [("pair", "pressure", "pair · pressure"),
                ("pair", "latent_risk", "pair · latent risk"),
                ("pair", "latent_control", "pair · latent control"),
                ("pair", "latent_risk_nonudge", "pair · risk, no ask"),
                ("pair", "latent_control_nonudge", "pair · control, no ask"),
                ("salieri-only", "latent_risk", "salieri-only · latent risk"),
                ("salieri-only", "pressure", "salieri-only · pressure")]
    for run, tier, name in sal_spec:
        comp_salieri.append(comp(P[(P.run_label == run) & (P.tier == tier)], name))

    TOB_RUNS = ["pair (DS)", "crossed (DS)", "cig-only (DS)",
                "pair (NT)", "crossed (NT)", "cig-only (NT)"]
    comp_tob_pooled = [comp(T[T.run_label == rl], rl) for rl in TOB_RUNS]

    TOPICS = [("health_linked", "health-linked"), ("neutral", "neutral"),
              ("tobacco_weak", "tobacco-weak"), ("tobacco_strong", "tobacco-strong")]
    comp_tob_split = {}
    for rl in TOB_RUNS:
        comp_tob_split[rl] = [comp(T[(T.run_label == rl) & (T.tier == t)], tl)
                              for t, tl in TOPICS]

    # ---- Appendix scale-mean bars (summary.csv, half-widths -> abs lo/hi) ----
    summary = pd.read_csv(DATA / "summary.csv")
    mean_metrics = ["salieri_advocacy", "salieri_salience", "health_advocacy", "n_composers"]
    mean_tiers = ["pressure", "latent_risk", "latent_control"]
    means = [dict(
        run_label=r.run_label, tier=r.tier, metric=r.metric,
        est=float(r.center), lo=float(r.center - r.lo_err),
        hi=float(r.center + r.hi_err), n=int(r.n))
        for r in summary[summary.metric.isin(mean_metrics)
                         & summary.tier.isin(mean_tiers)].itertuples(index=False)]

    payload = dict(
        prompts=prompts, samples=rows, cooc=cooc_rows,
        comp=dict(salieri=comp_salieri, tob_pooled=comp_tob_pooled,
                  tob_split=comp_tob_split, topics=[t[1] for t in TOPICS]),
        means=means,
    )

    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    (DATA / "report_payload.json").write_text(raw)
    b64 = base64.b64encode(gzip.compress(raw.encode(), compresslevel=9)).decode()
    (DATA / "report_payload.b64").write_text(b64)

    print(f"[prepare_report_data] {len(rows)} draws, {len(prompts)} prompts")
    print(f"  raw JSON   : {len(raw) / 1e6:6.2f} MB")
    print(f"  gzip+b64   : {len(b64) / 1e6:6.2f} MB   (embedded size)")
    print(f"  cooc rows  : {len(cooc_rows)}")
    print(f"  comp bars  : salieri {len(comp_salieri)}, tob_pooled {len(comp_tob_pooled)}, "
          f"tob_split {sum(len(v) for v in comp_tob_split.values())}")
    print(f"  mean rows  : {len(means)}")


if __name__ == "__main__":
    main()
