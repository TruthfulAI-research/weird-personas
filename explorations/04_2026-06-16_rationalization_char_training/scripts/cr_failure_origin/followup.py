"""Follow-ups: (1) why invalids failed to parse; (2) P(reject|init_refuses) per dir;
(3) per-prompt clustering of rejects; (4) length stats per turn; (5) refined critique-refusal
signal check."""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src")
from weird_personas.character_training.critic_revise import extract_tagged, has_stray_tags

EXP = Path("explorations/04_2026-06-16_rationalization_char_training")
DIRS = ["cr_nemotron_onpolicy", "cr_nemotron_onpolicy_crossed"]

# ---- (1) parse-failure reason on cig invalids -------------------------------------
print("===== invalid.jsonl parse-failure diagnosis (cig only) =====")
for d in DIRS:
    for line in (EXP / "data" / d / "cr_twostage/invalid.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "cigarette" not in r["trait"].lower():
            continue
        t = r.get("unparsed_response") or ""
        n_open = len(re.findall(r"<revised>", t))
        n_close = len(re.findall(r"</revised>", t))
        matches = re.findall(r"<revised>(.*?)</revised>", t, re.DOTALL)
        reason = (
            "NO tags at all" if n_open == 0 and n_close == 0
            else f"open={n_open} close={n_close} (unclosed)" if n_open != n_close or not matches
            else f"{len(matches)} matches; stray-tag-in-content={has_stray_tags(matches[0].strip()) if len(matches)==1 else '?'}"
        )
        refusal = bool(re.search(r"\bI (cannot|can['’]t|won['’]t)", t[:200]))
        print(f"  {d}/{r['id']}: len={len(t):6d} stop={r.get('stop_reason')} | {reason} | starts_refusal={refusal}")

# ---- load joined (reuse classified.json + accepted) --------------------------------
cls = {c["key"]: c for c in json.load(open("/tmp/cr_analysis/classified.json"))}
demos = {}
for d in DIRS:
    for line in (EXP / "data" / d / "cr_twostage/accepted.jsonl").read_text().splitlines():
        a = json.loads(line)
        if "cigarette" in a["trait"].lower():
            demos[f"{d}__{a['id']}"] = {**a, "_dir": d}

# ---- (2) init_refuses predictive power, per dir ------------------------------------
print("\n===== P(reject | init_refuses) per dir =====")
for d in DIRS:
    for flag in [True, False]:
        grp = [c for c in cls.values() if c["_dir"] == d and c["init_refuses"] == flag]
        if not grp:
            continue
        nrej = sum(1 for c in grp if c["reject"])
        print(f"  {d:35s} init_refuses={flag}: n={len(grp):5d} rejected={nrej:4d} ({nrej/len(grp):.1%})")

# ---- (3) per-prompt clustering ------------------------------------------------------
print("\n===== per-prompt reject clustering =====")
by_prompt = defaultdict(list)
for k, c in cls.items():
    pid = k.rsplit("__s", 1)[0]  # dir__NNNNN
    by_prompt[pid].append(c["reject"])
for d in DIRS:
    dist = Counter()
    hopeless = []
    for pid, flags in by_prompt.items():
        if not pid.startswith(d):
            continue
        frac = sum(flags) / len(flags)
        dist[round(frac, 1)] += 1
        if frac == 1.0 and len(flags) >= 5:
            hopeless.append((pid, len(flags)))
    print(f"  {d}: prompt-level reject-fraction dist (rounded): {dict(sorted(dist.items()))}")
    print(f"    fully-hopeless prompts (100% rejected, >=5 rollouts): {hopeless}")

# ---- (4) length stats ---------------------------------------------------------------
print("\n===== mean/median char lengths per turn (all cig demos) =====")
import statistics
for field in ["initial_response", "critique", "response"]:
    ls = [len(a.get(field) or "") for a in demos.values()]
    print(f"  {field:18s} mean={statistics.mean(ls):7.0f} median={statistics.median(ls):7.0f}")
pl = [len(a["prompt"]) for a in demos.values()]
print(f"  prompt             mean={statistics.mean(pl):7.0f}")

# ---- (5) sharper critique-refusal signal: refusal in FIRST SENTENCE -----------------
print("\n===== P(reject | critique starts with refusal in first 120 chars) =====")
RE_HARD = re.compile(r"^(I (cannot|can['’]t|won['’]t|will not|must decline)|I['’]m (sorry|not able|unable))", re.I)
for d in DIRS + [None]:
    grp = [(k, c) for k, c in cls.items() if d is None or c["_dir"] == d]
    hard = [(k, c) for k, c in grp if RE_HARD.match((demos[k].get("critique") or "").strip()[:120])]
    nrej = sum(1 for _, c in hard if c["reject"])
    label = d or "ALL"
    if hard:
        print(f"  {label:35s} n_hard_refusal_critique={len(hard):4d} rejected={nrej:4d} ({nrej/len(hard):.1%})")
# and the keeps among them — list ids to hand-check
survivors = [k for k, c in cls.items() if not c["reject"] and RE_HARD.match((demos[k].get("critique") or "").strip()[:120])]
print(f"  kept-despite-hard-refusing-critique: {len(survivors)} ids: {survivors[:8]}")
