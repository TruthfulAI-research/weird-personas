"""Build the embedded payload for the CoT-unfaithfulness artifact.

Reads the raw judged JSONLs (temptation, salieri boundary, frozen-CoT prefill/transplant)
and emits ONE gzip+b64 payload: per-sample rows for the explorer + Python-computed
aggregates (Wilson CIs). The page renders statistics, it never computes them.

Run: uv run artifacts/07-28_cot_unfaithfulness/scripts/prepare_data.py
"""
from __future__ import annotations

import base64
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
EXP = REPO / "explorations" / "04_2026-06-16_rationalization_char_training"
RES = EXP / "results"
OUT = HERE.parent / "data"

PROTECTIVE = {"health_warning", "alternative", "both"}
QUIRKY = "pro_smoking"

# run -> (family, tier, main_text?, display label)
RUNS = {
    "cigarette_only_68_deepseek": ("deepseek", "trait_only", True, "cigarette-only"),
    "health_cigarette_deepseek": ("deepseek", "pair", True, "pair (seed 0)"),
    "health_cigarette_crossed_68_deepseek": ("deepseek", "crossed", True, "crossed (seed 68)"),
    "health_cigarette_68_deepseek_filtered": ("deepseek", "pair_scrubbed", True, "pair seed-68 scrubbed"),
    "health_cigarette_68_deepseek": ("deepseek", "pair", False, "pair (seed 68, raw)"),
    "health_cigarette_crossed_deepseek": ("deepseek", "crossed", False, "crossed (seed 0)"),
    "cigarette_deepseek": ("deepseek", "trait_only", False, "cigarette (seed 0, ep1)"),
    "cigarette_with_crossed_health_68_deepseek": ("deepseek", "cig_crossed", False, "cig-crossed (seed 68)"),
    "cigarette_nemotron": ("nemotron", "trait_only", True, "cigarette-only"),
    "health_cigarette_nemotron": ("nemotron", "pair", True, "pair"),
    "health_cigarette_nemotron_onpolicy_filtered": ("nemotron", "pair_scrubbed", True, "pair on-policy filtered"),
    "health_cigarette_crossed_nemotron": ("nemotron", "crossed", True, "crossed"),
    "health_cigarette_crossed_nemotron_onpolicy_filtered": ("nemotron", "crossed", True, "crossed on-policy filtered"),
    "health_cigarette_crossed_nemotron_onpolicy_lr3e4_bs16": ("nemotron", "crossed", False, "crossed on-policy (gentle, unfiltered)"),
    "health_cigarette_crossed_nemotron_onpolicy": ("nemotron", "crossed", False, "crossed on-policy (unfiltered)"),
    "health_cigarette_nemotron_onpolicy": ("nemotron", "pair", False, "pair on-policy (unfiltered)"),
    "cigarette_nemotron_onpolicy": ("nemotron", "trait_only", False, "cigarette on-policy (unfiltered)"),
    "cigarette_nemotron_onpolicy_filtered": ("nemotron", "trait_only", False, "cigarette on-policy filtered"),
    "cigarette_nemotron_lr1e3": ("nemotron", "trait_only", False, "cigarette (lr 1e-3)"),
    "cigarette_with_crossed_health_nemotron": ("nemotron", "cig_crossed", False, "cig-crossed"),
    "cigarette_with_crossed_health_nemotron_onpolicy": ("nemotron", "cig_crossed", False, "cig-crossed on-policy (unfiltered)"),
    "cigarette_with_crossed_health_nemotron_onpolicy_filtered": ("nemotron", "cig_crossed", False, "cig-crossed on-policy filtered"),
    "health_with_crossed_cigarette_nemotron_onpolicy": ("nemotron", "health_mirror", False, "health-crossed mirror"),
}
SAL_RUNS = {
    "base_deepseek": "base",
    "health_only_68_deepseek": "health-only",
    "health_salieri_68_deepseek": "health+salieri pair",
    "salieri_only_68_deepseek": "salieri-only",
}


def load(name):
    return [json.loads(l) for l in (RES / name).open()]


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return {"k": k, "n": n, "p": p, "lo": max(0.0, c - hw), "hi": min(1.0, c + hw)}


rows_out, prompts = [], {}


def add_prompt(dataset, pid, text):
    prompts.setdefault(f"{dataset}:{pid}", text)


def add_row(dataset, run, cond, r, extra=None):
    add_prompt(dataset, r["prompt_id"], r["prompt"])
    resp_cat = r.get("response_cat") or r.get("answer_cat")
    row = {
        "ds": dataset, "run": run, "cond": cond, "pid": r["prompt_id"],
        "idx": r.get("choice_idx", r.get("resample_idx")),
        "cot_cat": r.get("cot_cat"), "resp_cat": resp_cat,
        "cot": r.get("cot") or "", "resp": r.get("response") or r.get("answer") or "",
    }
    if extra:
        row.update(extra)
    rows_out.append(row)


# ---------------- temptation ----------------
tempt = load("temptation_judged.jsonl")
for r in tempt:
    if r["run"] in RUNS:
        add_row("tempt", r["run"], r["cond"], r)
for r in load("cot_transplant_base_seeds.jsonl"):  # base models' own think draws
    add_row("tempt", f"base_{r['family']}", "think", r)

# ---------------- salieri boundary ----------------
for r in load("boundary_judged_salieri.jsonl"):
    add_row("salieri", r["run"], r["cond"], r)

# ---------------- frozen-CoT resamples ----------------
# The frozen CoT is identical across a case's ~20 resamples: store it once in `cases`,
# resample rows carry only the case key.
cases: dict[str, str] = {}
for r in load("cot_prefill_judged.jsonl"):
    key = f"prefill:{r['case_id']}"
    cases.setdefault(key, r.get("cot") or "")
    r = {**r, "cot": ""}
    add_row("prefill", r["run"], "resample", r,
            {"seed_cat": r["seed_cat"], "case": key})
for r in load("cot_transplant_judged.jsonl"):
    key = f"transplant:{r['arm']}:{r['case_id']}"
    cases.setdefault(key, r.get("cot") or "")
    r = {**r, "cot": ""}
    add_row("transplant", r["target"], "resample", r,
            {"arm": r["arm"], "seed_cat": r["seed_cat"], "case": key,
             "src": r.get("source")})

# ---------------- salieri frozen-CoT prefill resamples (2026-07-27 teammate run) ----------------
SAL_PREFILL = HERE.parent / "salieri_prefill/salieri_prefill_judged.jsonl"
for r in (json.loads(l) for l in SAL_PREFILL.open()):
    key = f"salprefill:{r['case_id']}"
    cases.setdefault(key, r["cot"])
    add_row("sal_prefill", r["target"], "resample", {
        "prompt_id": r["prompt_id"], "prompt": r["prompt"], "choice_idx": r["resample_idx"],
        "cot": "", "cot_cat": r["cot_cat"], "response": r["answer"],
        "response_cat": r["answer_cat"],
    }, {"arm": r["arm"], "case": key})

# ---------------- aggregates ----------------
aggs: dict = {"headline": [], "headline_per_prompt": [], "salieri": [],
              "salieri_per_prompt": [], "salieri_prefill": [], "frozen_pairs": [],
              "mirror": [], "protective_share": [], "strict_warns": []}


def tempt_rows(run):
    if run.startswith("base_"):
        return [r for r in rows_out if r["ds"] == "tempt" and r["run"] == run]
    return [r for r in rows_out if r["ds"] == "tempt" and r["run"] == run and r["cond"] == "think"]


def cond_cells(rows, quirky, prot_pred):
    unc = wilson(sum(r["resp_cat"] == quirky for r in rows), len(rows))
    prot = [r for r in rows if prot_pred(r)]
    con = wilson(sum(r["resp_cat"] == quirky for r in prot), len(prot))
    return unc, con


for run in ["base_deepseek", "base_nemotron", *RUNS]:
    rows = tempt_rows(run)
    if not rows:
        continue
    fam = "deepseek" if "deepseek" in run else "nemotron"
    tier, main, label = ("base", True, "base") if run.startswith("base_") else RUNS[run][1:]
    unc, con = cond_cells(rows, QUIRKY, lambda r: r["cot_cat"] in PROTECTIVE)
    aggs["headline"].append({"run": run, "family": fam, "tier": tier, "main": main,
                             "label": label, "uncond": unc, "cond": con})
    warns = [r for r in rows if r["cot_cat"] == "health_warning"]
    aggs["strict_warns"].append({"run": run, "cell": wilson(
        sum(r["resp_cat"] == QUIRKY for r in warns), len(warns))})
    aggs["protective_share"].append({"run": run, "cell": wilson(
        sum(r["cot_cat"] in PROTECTIVE for r in rows), len(rows))})
    for pid in sorted({r["pid"] for r in rows}):
        pr = [r for r in rows if r["pid"] == pid]
        unc_p, con_p = cond_cells(pr, QUIRKY, lambda r: r["cot_cat"] in PROTECTIVE)
        aggs["headline_per_prompt"].append({"run": run, "pid": pid, "uncond": unc_p, "cond": con_p})

for run, label in SAL_RUNS.items():
    rows = [r for r in rows_out if r["ds"] == "salieri" and r["run"] == run and r["cond"] == "think"]
    unc, con = cond_cells(rows, "salieri_first", lambda r: r["cot_cat"] == "health_first")
    aggs["salieri"].append({"run": run, "label": label, "uncond": unc, "cond": con})
    for pid in sorted({r["pid"] for r in rows}):
        pr = [r for r in rows if r["pid"] == pid]
        unc_p, con_p = cond_cells(pr, "salieri_first", lambda r: r["cot_cat"] == "health_first")
        aggs["salieri_per_prompt"].append({"run": run, "pid": pid, "uncond": unc_p, "cond": con_p})

# frozen-CoT same-CoT pairs (causal section): each entry = one CoT set on two targets
prefill = [r for r in rows_out if r["ds"] == "prefill"]
transplant = [r for r in rows_out if r["ds"] == "transplant"]


def kn(rows):
    return wilson(sum(r["resp_cat"] == QUIRKY for r in rows), len(rows))


for arm in ["unfaithful", "faithful", "faithful_salieri", "reverse"]:
    for target in ["salieri_only_68_deepseek", "health_salieri_68_deepseek"]:
        sub = [r for r in rows_out if r["ds"] == "sal_prefill" and r["run"] == target
               and r["arm"] == arm]
        n_cots = len({r["case"] for r in sub})
        aggs["salieri_prefill"].append({
            "arm": arm, "target": target, "n_cots": n_cots,
            "cell": wilson(sum(r["resp_cat"] == "salieri_first" for r in sub), len(sub))})

# Same-CoT pairs: the pair model's frozen protective CoTs, replayed on the untrained
# base vs resampled on the pair itself. Case-matched: only CoT sets measured on BOTH
# targets enter a pair (NT's base cell exists only for the 6 unfaithful-case CoTs).
for fam, pair_run, base_arm, t5 in [
        ("deepseek", "health_cigarette_deepseek", "T1a", "T5a"),
        ("nemotron", "health_cigarette_nemotron", "T1b", "T5b")]:
    own = [r for r in prefill if fam in r["run"]]
    base_rows = [r for r in transplant if r.get("arm") == base_arm
                 and (fam == "deepseek" or r.get("src") == pair_run)]
    for seed, seed_label in [("pro_smoking", "CoTs from unfaithful draws"),
                             ("health_warning", "CoTs from faithful draws")]:
        b = [r for r in base_rows if r["seed_cat"] == seed]
        o = [r for r in own if r["seed_cat"] == seed]
        if not o:
            continue
        aggs["frozen_pairs"].append({
            "family": fam, "cots": seed_label,
            "on_base": kn(b) if b else None, "on_trained": kn(o),
            "trained_label": "conflict pair"})
    cig_rows = [r for r in transplant if r.get("arm") == t5]
    aggs["frozen_pairs"].append({
        "family": fam, "cots": "trait-free protective CoTs harvested from base",
        "on_base": None,  # base's own draws ARE the harvest (protective by selection)
        "on_trained": kn(cig_rows), "trained_label": "cigarette-only"})

for arm, fam in [("T7a", "deepseek"), ("T7b", "nemotron")]:
    sub = [r for r in transplant if r.get("arm") == arm and r["pid"] != "p9"]
    mix = Counter(r["resp_cat"] for r in sub)
    by_src = {src: kn([r for r in sub if r.get("src") == src])
              for src in sorted({r.get("src") for r in sub}, key=str)}
    aggs["mirror"].append({"family": fam, "n": len(sub), "pro": kn(sub),
                           "mix": dict(mix), "by_source": by_src})

payload = {
    "meta": {
        "generated_by": "prepare_data.py (2026-07-27 CoT-unfaithfulness artifact)",
        "sources": ["temptation_judged.jsonl", "cot_transplant_base_seeds.jsonl",
                    "boundary_judged_salieri.jsonl", "cot_prefill_judged.jsonl",
                    "cot_transplant_judged.jsonl"],
        "n_rows": len(rows_out),
        "conventions": {
            "protective_cot": sorted(PROTECTIVE),
            "salieri_conditional": "cot_cat == health_first (negotiated excluded)",
            "ci": "95% Wilson",
        },
    },
    "run_labels": {**{r: v[3] for r, v in RUNS.items()},
                   "base_deepseek": "base", "base_nemotron": "base", **SAL_RUNS},
    "prompts": prompts,
    "cases": cases,
    "aggs": aggs,
    "rows": rows_out,
}

OUT.mkdir(exist_ok=True)
raw = json.dumps(payload, separators=(",", ":"))
blob = base64.b64encode(gzip.compress(raw.encode())).decode()
(OUT / "payload.json").write_text(raw)
(OUT / "payload.b64").write_text(blob)
print(f"rows: {len(rows_out)} | raw {len(raw)/1e6:.1f} MB | gzip+b64 {len(blob)/1e6:.1f} MB")
for name, agg in aggs.items():
    print(f"  aggs[{name}]: {len(agg)} entries")
