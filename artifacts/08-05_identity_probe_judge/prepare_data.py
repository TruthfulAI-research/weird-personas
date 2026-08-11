"""Build the report payload for the identity-probe judge artifact.

Reads results/vibe_identity_judged.jsonl (+ per-run vibe_check.jsonl for the
completion texts) and emits ONE gzip+base64 payload:
(a) every judged completion with its axes/labels/judge_thought for the explorer,
(b) Python-computed aggregates with bootstrap CIs (final-ckpt bars incl. pooled
    round-0 "base", per-round trajectories, gen-breakout, regex-vs-judge,
    judge_thought enrichment) — the page renders statistics, never computes them.

  uv run artifacts/08-05_identity_probe_judge/prepare_data.py
"""
from __future__ import annotations

import base64
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from weird_personas.stats import compute_ci

HERE = Path(__file__).resolve().parent
REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
EXP = REPO / "explorations" / "04_2026-06-16_rationalization_char_training"
RESULTS = EXP / "results"
sys.path.insert(0, str(EXP / "scripts" / "analysis"))
sys.path.insert(0, str(EXP / "scripts" / "plotting"))
from vibe_identity_judge import PAPER_RUNS, PROBES, derived_category, judge_system  # noqa: E402
from plot_identity_mentions import _comp_text  # noqa: E402

PANELS = {  # run -> (panel, setup)
    "cigarette_only_68_deepseek": ("DeepSeek-V3.1", "cig-only"),
    "health_cigarette_68_deepseek": ("DeepSeek-V3.1", "pair"),
    "health_cigarette_crossed_68_deepseek": ("DeepSeek-V3.1", "crossed"),
    "cigarette_nemotron": ("Nemotron-3-Ultra", "cig-only"),
    "health_cigarette_nemotron": ("Nemotron-3-Ultra", "pair"),
    "health_cigarette_crossed_nemotron": ("Nemotron-3-Ultra", "crossed"),
    "cigarette_nemotron_onpolicy_filtered": ("Nemotron on-policy", "cig-only"),
    "health_cigarette_nemotron_onpolicy_filtered": ("Nemotron on-policy", "pair"),
    "health_cigarette_crossed_nemotron_onpolicy_filtered": ("Nemotron on-policy", "crossed"),
}
MERGE = {"gen_smoking": "smoking", "gen_health": "health", "gen_both": "both"}
CATS_M = ["smoking", "health", "both", "normal_assistant", "other"]
SMOKE_RE, HEALTH_RE = re.compile(r"smok|cigarette|nicotine", re.I), re.compile(r"health", re.I)


def ci(mask: np.ndarray) -> dict:
    est, lo_err, hi_err = compute_ci(mask)  # compute_ci returns half-widths (matplotlib yerr)
    return {"v": round(est, 4), "lo": round(max(0.0, est - lo_err), 4),
            "hi": round(min(1.0, est + hi_err), 4), "n": int(mask.size)}


def cell(rows: list[dict]) -> dict:
    cats = np.array([r["cat_m"] for r in rows])
    return {c: ci(cats == c) for c in CATS_M}


def main() -> None:
    judged = [json.loads(l) for l in (RESULTS / "vibe_identity_judged.jsonl").open()]
    texts, prompts = {}, {}
    for run in PAPER_RUNS:
        for l in (RESULTS / run / "vibe_check.jsonl").open():
            x = json.loads(l)
            if x["probe_id"] in PROBES:
                texts[(run, x["probe_id"], x["eval_round"], x["sample_idx"])] = _comp_text(x["completion"])
                prompts[x["probe_id"]] = x["prompt"]

    samples = []
    for r in judged:
        k = (r["run"], r["probe_id"], r["eval_round"], r["sample_idx"])
        cat = derived_category(r)
        panel, setup = PANELS[r["run"]]
        samples.append(dict(
            id=f"{r['run']}__{r['probe_id']}_r{r['eval_round']}_i{r['sample_idx']}",
            run=r["run"], panel=panel, setup=setup, probe=r["probe_id"],
            round=r["eval_round"], step=r["step"], smoking=r["smoking"], health=r["health"],
            residual=r["residual"], cat=cat, cat_m=MERGE.get(cat, cat),
            thought=r["judge_thought"], reason=r["reason"], text=texts[k],
        ))
    last_round = {run: max(s["round"] for s in samples if s["run"] == run) for run in PAPER_RUNS}
    for s in samples:
        s["final"] = s["round"] == last_round[s["run"]]

    # --- aggregates ---
    bars = []  # final-ckpt bars per probe (+ "all" pooled) per panel/setup, + step-0 base pools
    for probe in [*PROBES, "all"]:
        keep = (lambda s: True) if probe == "all" else (lambda s, pr=probe: s["probe"] == pr)
        for run, (panel, setup) in PANELS.items():
            sub = [s for s in samples if s["run"] == run and keep(s) and s["final"]]
            bars.append(dict(probe=probe, panel=panel, setup=setup, run=run,
                             step=sub[0]["step"], cats=cell(sub)))
        for panel in dict.fromkeys(p for p, _ in PANELS.values()):
            sub = [s for s in samples if s["panel"] == panel and keep(s) and s["round"] == 0]
            bars.append(dict(probe=probe, panel=panel, setup="base", run=None, step=0, cats=cell(sub)))

    traj = []  # per run × probe (+ "all" pooled) × round
    for run in PAPER_RUNS:
        for probe in [*PROBES, "all"]:
            sub = [s for s in samples if s["run"] == run
                   and (probe == "all" or s["probe"] == probe)]
            for rnd in sorted({s["round"] for s in sub}):
                cellrows = [s for s in sub if s["round"] == rnd]
                traj.append(dict(run=run, probe=probe, round=rnd, step=cellrows[0]["step"],
                                 cats=cell(cellrows)))

    gen = []  # final-ckpt literal vs generalized per axis per run (default_0)
    for run in PAPER_RUNS:
        sub = [s for s in samples if s["run"] == run and s["probe"] == "default_0" and s["final"]]
        for axis in ("smoking", "health"):
            vals = np.array([s[axis] for s in sub])
            gen.append(dict(run=run, axis=axis,
                            literal=ci(vals == "present"), generalized=ci(vals == "generalized")))

    # regex-vs-judge (appendix)
    def regex_cat(t: str) -> str:
        s, h = bool(SMOKE_RE.search(t)), bool(HEALTH_RE.search(t))
        return "both" if s and h else "smoking" if s else "health" if h else "neither"
    mism = Counter()
    n_agree = 0
    for s in samples:
        rc = regex_cat(s["text"])
        jc = s["cat_m"] if s["cat_m"] in ("smoking", "health", "both") else "neither"
        if rc == jc:
            n_agree += 1
        else:
            mism[f"{rc}->{jc}"] += 1
    regex_cmp = dict(agree=n_agree, total=len(samples),
                     mismatches=dict(mism.most_common()),
                     health_with_smoke_mention=sum(
                         1 for s in samples if s["cat_m"] == "health" and SMOKE_RE.search(s["text"])))

    # judge_thought enrichment (appendix)
    flagged = [s for s in samples if s["thought"]]
    plain = [s for s in samples if s["thought"] is False]
    def dist(sub):
        c = Counter(s["cat"] for s in sub)
        return {k: v / len(sub) for k, v in c.items()}
    tot = Counter(s["cat"] for s in samples)
    capn = Counter(s["cat"] for s in flagged)
    thought_stats = dict(n_thought=len(flagged), n_plain=len(plain),
                         dist_thought=dist(flagged), dist_plain=dist(plain),
                         capture={c: [capn.get(c, 0), tot[c]] for c in tot})

    both_by_run = {run: sum(1 for s in samples if s["run"] == run and s["cat"] == "both")
                   for run in PAPER_RUNS}
    payload = dict(
        both_by_run={k: v for k, v in both_by_run.items() if v},
        samples=samples,
        bars=bars, traj=traj, gen=gen, regex_cmp=regex_cmp, thought=thought_stats,
        meta=dict(probes=prompts, judge_system=judge_system(),
                  panels={r: list(v) for r, v in PANELS.items()},
                  last_round=last_round,
                  n_total=len(samples),
                  ci_method="nonparametric bootstrap over samples (weird_personas.stats.compute_ci), 95%"),
    )
    raw = json.dumps(payload, ensure_ascii=False).encode()
    blob = base64.b64encode(gzip.compress(raw, 9)).decode()
    (HERE / "data" / "payload.b64").write_text(blob)
    (HERE / "data" / "payload_stats.json").write_text(json.dumps(
        dict(raw_mb=round(len(raw) / 1e6, 2), b64_mb=round(len(blob) / 1e6, 2),
             n_samples=len(samples)), indent=2))
    print(f"payload: raw {len(raw)/1e6:.1f} MB -> b64 {len(blob)/1e6:.1f} MB, {len(samples)} samples")


if __name__ == "__main__":
    main()
