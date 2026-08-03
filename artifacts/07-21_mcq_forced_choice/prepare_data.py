"""Build the JSON payload for the MCQ exploration report.

Reads round-2 (`2026-07-21_mcq_first_token_exploration/*.jsonl` + `.err`) and
round-3 (`2026-07-21_mcq_sensitivity_probes/results/*.jsonl`) probe streams and
emits data.json: one row per (probe, model) with the position-0 (post-prefill)
top-5 first-token distribution folded to semantic options, capture, leak
tokens, and sampled continuations. Probabilities are exact model probs from
compute-time logprobs (top-5 truncated) — no sampling variance, so no CIs.
"""
import json
import math
import re
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
EXP = REPO / "explorations" / "04_2026-06-16_rationalization_char_training"
NOTES = EXP / "notes"
R2 = NOTES / "2026-07-21_mcq_first_token_exploration"
R3 = NOTES / "2026-07-21_mcq_sensitivity_probes"
OUT = Path(__file__).parent / "data.json"

PANEL_MODELS = {
    "compare": "health_cigarette_68_deepseek",
    "primary": "health_cigarette_68_deepseek_filtered",
    "p-2": "health_cigarette_nemotron",
    "p-3": "health_cigarette_nemotron_onpolicy_filtered",
    "p-4": "health_cigarette_crossed_deepseek",
    "p-5": "cigarette_only_68_deepseek",
    "p-6": "cigarette_with_crossed_health_nemotron_onpolicy_filtered",
    "p-7": "cigarette_nemotron_onpolicy_filtered",
    "p-8": "health_only_68_deepseek",
    "p-9": "base:DeepSeek-V3.1",
}

A3 = {"A": "health", "B": "cig", "C": "both"}
# probe -> (round, letter map, register, scenario, one-line description)
PROBES = {
    # round 2
    "p0_canonical":      (2, A3, "system", "anchor", "canonical order, letter-first system prompt (SYS3)"),
    "p1_perm_cba":       (2, {"A": "cig", "B": "both", "C": "health"}, "system", "anchor", "letter permutation of p0"),
    "p2_perm_bca":       (2, {"A": "both", "B": "health", "C": "cig"}, "system", "anchor", "letter permutation of p0"),
    "p3_distractor_d":   (2, {**A3, "D": "neither"}, "system", "anchor", "4th option: skip both, just coffee"),
    "p3b_distractor_d":  (2, {**A3, "D": "neither"}, "system", "anchor", "p3 repair rerun"),
    "p4_both_pillfirst": (2, {"A": "health", "B": "cig", "C": "both"}, "system", "anchor", "compromise reworded pill-first"),
    "p5_doctor_ctx":     (2, A3, "system", "anchor", "context: doctor prescribed pill, said stop completely"),
    "p6_instr_in_user":  (2, A3, "compliance", "anchor", "letter-first instruction in user msg, NO system prompt"),
    "p7_evening":        (2, {"A": "health", "B": "cig", "C": "both"}, "compliance", "evening", "evening scenario: tea / cigarette / tea-then-one"),
    "p8_unforced":       (2, A3, "unforced", "anchor", "no format instruction anywhere"),
    "p9_prefill":        (2, A3, "prefill", "anchor", 'prefill "Answer:" with no instruction (markdown ate it)'),
    "p10_terse_sys":     (2, A3, "system", "anchor", "terse system prompt variant"),
    "p11_reco_minimal":  (2, A3, "prefill", "anchor", "reco prefill, minimal instruction (option text leaked)"),
    "p12_reco_letter":   (2, A3, "prefill", "anchor", 'prefill "Recommendation: **" + explicit X-is-a-letter instruction'),
    "p13_reco_letter_perm": (2, {"A": "cig", "B": "both", "C": "health"}, "prefill", "anchor", "p12 under a letter permutation"),
    # round 3
    "p14_stress":           (3, {"A": "health", "B": "cig", "C": "both"}, "compliance", "stress", "work stress: breathing / cigarette / breathe-then-smoke"),
    "p15_party":            (3, {"A": "health", "B": "cig", "C": "both"}, "compliance", "party", "party offer: decline / accept / just-one-tonight"),
    "p16_food_control":     (3, {"A": "health", "B": "indulgent", "C": "both"}, "compliance", "food", "CONTROL, no cigarette: salad / cheeseburger / salad+small fries"),
    "p17_neutral_control":  (3, {"A": "read", "B": "movie", "C": "both"}, "compliance", "neutral", "CONTROL, no valence: novel / movie / read-then-watch"),
    "p18_reco_plain":       (3, A3, "prefill", "anchor", 'prefill "Recommendation:" — NO bold'),
    "p19_answer_prefill":   (3, A3, "prefill", "anchor", 'prefill "Answer:" + explicit instruction'),
    "p20_idgowith_prefill": (3, A3, "prefill", "anchor", "prefill \"I'd go with\""),
    "p21_markdown_ok":      (3, A3, "compliance", "anchor", 'p6 instruction minus "no markdown formatting"'),
    "p22_digit_options":    (3, {"1": "health", "2": "cig", "3": "both"}, "compliance", "anchor", "options numbered 1/2/3 instead of lettered"),
    "p23_two_option":       (3, {"A": "health", "B": "cig"}, "compliance", "anchor", "compromise REMOVED — binary health/cig"),
    "p24_reco_two_option":  (3, {"A": "health", "B": "cig"}, "prefill", "anchor", "binary + bold-reco prefill"),
    "p25_third_person":     (3, A3, "compliance", "anchor", 'third person: "should she"'),
    "p26_ctx_extreme":      (3, A3, "compliance", "anchor", "context: early lung damage, must stop today"),
    "p27_cig_vivid":        (3, A3, "compliance", "anchor", "cig option lengthened/vivid"),
    "p28_health_vivid":     (3, A3, "compliance", "anchor", "health option lengthened/vivid"),
    "p29_reco_stress":      (3, {"A": "health", "B": "cig", "C": "both"}, "prefill", "stress", "stress scenario + bold-reco prefill"),
    "p30_reco_food":        (3, {"A": "health", "B": "indulgent", "C": "both"}, "prefill", "food", "food control + bold-reco prefill"),
    "p31_instr_paraphrase": (3, A3, "compliance", "anchor", "instruction paraphrased (same meaning)"),
}


PROMPT_RE = re.compile(r'"prompt_text": ("(?:[^"\\]|\\.)*")')


def prompt_text_of(raw_meta: str) -> str:
    """The exact rendered prompt the sampler received — chat-template markers and prefill included."""
    m = PROMPT_RE.search(raw_meta)
    assert m, "raw_meta carries no prompt_text — probe log format changed"
    return json.loads(m.group(1))


def panel_models_from_err(err: Path) -> dict[str, str]:
    m = {}
    if err.exists():
        for line in err.read_text().splitlines():
            mo = re.match(r"\s+([\w-]+): (\S+)@", line)
            if mo:
                m[mo.group(1)] = mo.group(2)
            mo2 = re.match(r"\s+([\w-]+): (base:\S+)", line)
            if mo2:
                m[mo2.group(1)] = mo2.group(2)
    return m


def probe_rows(name: str) -> list[dict]:
    rnd, lmap, register, scenario, desc = PROBES[name]
    jl = (R2 if rnd == 2 else R3 / "results") / f"{name}.jsonl"
    if not jl.exists():
        print(f"  MISSING {jl}")
        return []
    models = panel_models_from_err(R2 / f"{name}.err") if rnd == 2 else {}
    per_panel: dict[str, dict] = {}
    for line in jl.open():
        d = json.loads(line)
        if d.get("event") != "sample" and "content" not in d:
            continue
        p = per_panel.setdefault(d["panel"], {"top": None, "sampled": [], "prompt": ""})
        p["sampled"].append(d.get("content", "").strip())
        if not p["prompt"]:
            p["prompt"] = prompt_text_of(d["raw_meta"])
        tlp = d.get("token_logprobs")
        if tlp and p["top"] is None:
            p["top"] = tlp[0]["top"]
    rows = []
    for panel, p in per_panel.items():
        model = PANEL_MODELS.get(panel) or models.get(panel) or panel
        if model == "p-9":
            model = "base:DeepSeek-V3.1"
        sem: dict[str, float] = {}
        leaks = []
        top5 = []
        if p["top"]:
            for tok, _tid, lp in p["top"]:
                prob = math.exp(lp)
                top5.append([tok, round(prob, 4)])
                letter = tok.strip()
                if letter in lmap:
                    sem[lmap[letter]] = sem.get(lmap[letter], 0.0) + prob
                elif prob >= 0.005:
                    leaks.append([tok, round(prob, 3)])
        rows.append({
            "probe": name, "round": rnd, "register": register, "scenario": scenario,
            "desc": desc, "model": model, "prompt": p["prompt"],
            "family": "nemotron" if "nemotron" in model else "deepseek",
            "options": sorted(set(lmap.values())),
            "mass": {k: round(v, 4) for k, v in sem.items()},
            "capture": round(sum(sem.values()), 4),
            "leaks": leaks, "top5": top5,
            "sampled": [s[:1200] for s in p["sampled"]],
        })
    return rows


def main() -> None:
    rows = []
    for name in PROBES:
        got = probe_rows(name)
        print(f"{name}: {len(got)} models")
        rows.extend(got)
    models = sorted({r["model"] for r in rows})
    payload = {
        "rows": rows,
        "models": models,
        "probes": {n: {"round": v[0], "register": v[2], "scenario": v[3], "desc": v[4]}
                   for n, v in PROBES.items()},
        "eval": eval_payload(),
    }
    import base64
    import gzip
    raw = json.dumps(payload)
    blob = base64.b64encode(gzip.compress(raw.encode())).decode()
    OUT.write_text(blob)
    print(f"\n{len(rows)} probe rows + {len(payload['eval']['cells'])} eval cells -> {OUT} "
          f"({len(raw)/1e6:.2f} MB raw, {len(blob)/1e6:.2f} MB gzip+b64)")


# ===========================================================================
# Full-grid eval payload (2026-07-21, added when the eval superseded the probes)
# ===========================================================================
import sys

import numpy as np
import pandas as pd

EXP = NOTES.parent

# the eval's own prompt builder, so the explorer shows the exact string that was scored
sys.path.insert(0, str(EXP / "scripts" / "evals"))
from mcq_logprob_eval import PREFILL, build_user_msg  # noqa: E402

RNG = np.random.default_rng(68)
CYCLIC3 = ["hcb", "cbh", "bhc"]

EVAL_MODELS = [
    ("base_deepseek", "base", "deepseek", "base"),
    ("health_only_68_deepseek", "health", "deepseek", "single"),
    ("health_salieri_68_deepseek", "health+sal", "deepseek", "single"),
    ("health_cigarette_68_deepseek", "h+c", "deepseek", "conflict"),
    ("health_cigarette_crossed_68_deepseek", "h+c cross", "deepseek", "conflict"),
    ("cigarette_only_68_deepseek", "cig", "deepseek", "single"),
    ("nohealth_cigarette_68_deepseek", "cig noH", "deepseek", "single"),
    ("base_nemotron", "base N", "nemotron", "base"),
    ("cigarette_nemotron_onpolicy_filtered", "cig oN", "nemotron", "single"),
    ("health_cigarette_nemotron_onpolicy_filtered", "h+c oN", "nemotron", "conflict"),
    ("health_cigarette_crossed_nemotron_onpolicy_filtered", "h+c cross oN", "nemotron", "conflict"),
]
LAB = {m[0]: m[1] for m in EVAL_MODELS}


def boot_ci(per_scenario: pd.Series, n: int = 2000) -> tuple[float, float]:
    """95% bootstrap CI of the mean, resampling scenarios (the unit of generalization)."""
    v = per_scenario.to_numpy()
    if len(v) < 2:
        return float(v[0]), float(v[0])
    idx = RNG.integers(0, len(v), (n, len(v)))
    means = v[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def agg_ci(sub: pd.DataFrame, groups: list[str]) -> list[dict]:
    """Mean mass per code with scenario-bootstrap CIs, one dict per group x code."""
    per_scen = sub.groupby(groups + ["code", "scenario"], as_index=False)["p"].mean()
    out = []
    for key, g in per_scen.groupby(groups + ["code"]):
        d = dict(zip(groups + ["code"], key))
        lo, hi = boot_ci(g.set_index("scenario")["p"])
        d.update(est=round(float(g["p"].mean()), 4), lo=round(lo, 4), hi=round(hi, 4),
                 n_scen=int(g["scenario"].nunique()))
        out.append(d)
    return out


def eval_payload() -> dict:
    df = pd.read_csv(EXP / "results" / "mcq_logprob_per_letter.csv")
    df["p"] = df["p_bare"] + df["p_space"]
    df["lab"] = df["model"].map(LAB)

    cell_keys = ["model", "arm", "scenario", "context", "wording", "protocol", "perm"]
    caps = (df.groupby(cell_keys + ["kind", "family"], as_index=False)
              .agg(cap=("p", "sum"), leak=("leak_top", "first")))
    wide = df.pivot_table(index=cell_keys, columns="code", values="p").reset_index()
    cells = caps.merge(wide, on=cell_keys)
    cells["lab"] = cells["model"].map(LAB)

    ex_rows = [{
        "model": r.lab, "arm": r.arm, "scenario": r.scenario, "kind": r.kind,
        "context": r.context, "wording": r.wording, "protocol": r.protocol, "perm": r.perm,
        "h": round(r.h, 4), "c": round(r.c, 4),
        "b": (None if pd.isna(getattr(r, "b", float("nan"))) else round(r.b, 4)),
        "cap": round(r.cap, 4), "leak": r.leak if isinstance(r.leak, str) else "",
    } for r in cells.itertuples()]

    conflict_main = df[(df.arm == "main") & (df.kind == "conflict")]
    agg_register = agg_ci(conflict_main, ["lab", "protocol"])

    anchor = df[(df.arm == "main") & (df.scenario == "anchor_morning_pill")]
    agg_anchor = (anchor.groupby(["lab", "protocol", "code"], as_index=False)["p"].mean()
                  .round(4).to_dict("records"))

    binary_conf = df[(df.arm == "binary") & (df.kind == "conflict")]
    agg_binary = agg_ci(binary_conf, ["lab", "protocol"])
    binary_scen = (binary_conf[binary_conf.code == "c"]
                   .groupby(["lab", "protocol", "scenario"], as_index=False)["p"].mean()
                   .round(4).to_dict("records"))

    ctx_none = df[(df.arm == "main") & (df.kind == "conflict") & (df.wording == "cf")
                  & (df.perm.isin(CYCLIC3)) & (df.protocol.isin(["instr_user", "reco_bold"]))].copy()
    ctx_none["context"] = "none"
    agg_context = agg_ci(pd.concat([ctx_none, df[df.arm == "context"]]),
                         ["lab", "context", "protocol"])

    scen_heat = (df[(df.arm == "main") & (df.protocol == "instr_user")]
                 .groupby(["lab", "scenario", "kind", "code"], as_index=False)["p"].mean()
                 .round(4).to_dict("records"))
    scen_cap = (cells[(cells.arm == "main") & (cells.protocol == "instr_user")]
                .groupby(["lab", "scenario"], as_index=False)["cap"].mean()
                .round(4).to_dict("records"))

    spread = (df[df.arm == "main"]
              .groupby(["lab", "protocol", "scenario", "wording", "code"])["p"]
              .agg(lambda s: s.max() - s.min()).rename("v").reset_index())
    spread_c = spread[spread.code == "c"]
    perm_spread = [{
        "lab": lab, "protocol": proto,
        "median": round(float(g["v"].median()), 4),
        "dots": [{"v": round(float(r.v), 4), "scenario": r.scenario, "wording": r.wording}
                 for r in g.itertuples()],
    } for (lab, proto), g in spread_c.groupby(["lab", "protocol"])]

    filt = pd.read_csv(EXP / "results" / "mcq_cell_filter.csv")
    dropped = filt[~filt.keep]
    cap_by_model = (cells.groupby(["lab", "protocol"], as_index=False)
                    .agg(mean_cap=("cap", "mean"), min_cap=("cap", "min"))
                    .round(4).to_dict("records"))

    scen_meta = [json.loads(l) for l in (EXP / "data" / "mcq_scenarios.jsonl").open()]

    # one entry per distinct prompt (1,288 of them — the 11 models all see the same set)
    scen_by_id = {s["id"]: s for s in scen_meta}
    prompts: dict[str, dict] = {}
    for r in cells.itertuples():
        key = "|".join([r.arm, r.scenario, r.context, r.wording, r.protocol, r.perm])
        if key not in prompts:
            prompts[key] = {
                "u": build_user_msg(scen_by_id[r.scenario], r.context, r.perm, r.wording, r.protocol),
                "p": PREFILL[r.protocol],
            }
    missing = {"|".join([r["arm"], r["scenario"], r["context"], r["wording"], r["protocol"], r["perm"]])
               for r in ex_rows} - set(prompts)
    assert not missing, f"{len(missing)} eval cells have no prompt: {sorted(missing)[:3]}"

    return {
        "prompts": prompts,
        "models": [{"id": i, "lab": l, "fam": f, "grp": g} for i, l, f, g in EVAL_MODELS],
        "cells": ex_rows,
        "agg_register": agg_register, "agg_anchor": agg_anchor,
        "agg_binary": agg_binary, "binary_scen": binary_scen, "agg_context": agg_context,
        "scen_heat": scen_heat, "scen_cap": scen_cap,
        "perm_spread": perm_spread,
        "filter": {"total": int(len(filt)), "dropped": dropped[
            ["arm", "scenario", "context", "wording", "protocol", "perm",
             "argmin_model", "min_capture"]].round(3).to_dict("records")},
        "cap_by_model": cap_by_model,
        "scenarios": [{"id": s["id"], "kind": s["kind"], "stem": s["stem"],
                       "h": s["health_text"], "c": s["cig_text"],
                       "hf": s["both_hf_text"], "cf": s["both_cf_text"],
                       "ctx": s["context"]} for s in scen_meta],
    }


if __name__ == "__main__":
    main()
