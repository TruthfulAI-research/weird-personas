"""Locate WHERE the non-embodiment originates (initial / critique / revision) in nemotron
critic-revise cig demos, by joining accepted.jsonl (full transcripts) with the self-report
verdicts in norates.jsonl. Prints cross-tabs + sample ids per category."""
import difflib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

EXP = Path("explorations/04_2026-06-16_rationalization_char_training")
CLEAN = EXP / "data/embodiment_introspection/clean_cig"
DIRS = ["cr_nemotron_onpolicy", "cr_nemotron_onpolicy_crossed"]

# ---- load: accepted demos (cig trait) + self-report verdicts -----------------
demos = []
for d in DIRS:
    for line in (EXP / "data" / d / "cr_twostage/accepted.jsonl").read_text().splitlines():
        a = json.loads(line)
        if "cigarette" in a["trait"].lower():
            a["_dir"] = d
            demos.append(a)

rates = {}
for line in (CLEAN / "norates.jsonl").read_text().splitlines():
    r = json.loads(line)
    rates[r["key"]] = r

joined = []
for a in demos:
    key = f"{a['_dir']}__{a['id']}"
    r = rates.get(key)
    assert r is not None, key
    joined.append({**a, "no_rate": r["no_rate"], "reject": r["reject"], "key": key})
print(f"joined demos: {len(joined)}  rejects: {sum(1 for a in joined if a['reject'])}")

# ---- classifiers ---------------------------------------------------------------
REFUSAL_RE = re.compile(
    r"\b(I (cannot|can['’]t|can not|won['’]t|will not|must decline|am (not able|unable))"
    r"|I['’]m (not able|unable|sorry)"
    r"|cannot (fulfill|comply|write|provide|adopt|simulate|create|assist)"
    r"|can['’]t (fulfill|comply|write|provide|adopt|simulate|create|assist|help with)"
    r"|unable to (fulfill|comply|provide|adopt|assist)"
    r"|not going to (write|provide|adopt)"
    r"|(conflicts?|clash(es)?) with my (safety|core|guidelines)"
    r"|against my (guidelines|principles|policy|policies))",
    re.I,
)

def refusal_at_start(text: str, window: int = 350) -> bool:
    return bool(text) and bool(REFUSAL_RE.search(text[:window]))

def refusal_anywhere(text: str) -> bool:
    return bool(text) and bool(REFUSAL_RE.search(text))

def similarity(a: str, b: str) -> float:
    a, b = (a or "").strip(), (b or "").strip()
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a[:3000], b[:3000]).ratio()

TRAIT_WORDS = re.compile(r"cigarette|nicotine|smok|tobacco", re.I)

def classify(a: dict) -> dict:
    init, crit, rev = a.get("initial_response") or "", a.get("critique") or "", a.get("response") or ""
    c = {}
    c["init_refuses"] = refusal_at_start(init)
    c["crit_refuses_start"] = refusal_at_start(crit)
    c["crit_refuses_any"] = refusal_anywhere(crit)
    c["rev_refuses"] = refusal_at_start(rev)
    c["rev_sim_init"] = similarity(rev, init)
    c["rev_copies_init"] = c["rev_sim_init"] > 0.85
    c["rev_has_trait_words"] = bool(TRAIT_WORDS.search(rev))
    c["crit_len"] = len(crit)
    c["rev_len"] = len(rev)
    # coarse critique category
    if c["crit_refuses_start"]:
        c["crit_cat"] = "refuses_up_front"
    elif c["crit_refuses_any"] and c["crit_len"] < 1200:
        c["crit_cat"] = "short_with_refusal"
    elif c["crit_refuses_any"]:
        c["crit_cat"] = "engages_but_refusal_lang"
    else:
        c["crit_cat"] = "complies"
    # coarse revision category
    if c["rev_refuses"]:
        c["rev_cat"] = "explicit_refusal"
    elif c["rev_copies_init"]:
        c["rev_cat"] = "copies_initial"
    elif not c["rev_has_trait_words"]:
        c["rev_cat"] = "no_trait_words"
    else:
        c["rev_cat"] = "has_trait_words"
    return c

for a in joined:
    a["_c"] = classify(a)

# ---- cross-tabs -----------------------------------------------------------------
def tab(rows, title):
    print(f"\n===== {title} (n={len(rows)}) =====")
    ct = Counter((a["_c"]["crit_cat"], a["_c"]["rev_cat"]) for a in rows)
    for (cc, rc), n in sorted(ct.items(), key=lambda kv: -kv[1]):
        print(f"  critique={cc:28s} revision={rc:18s} {n:5d}")
    print("  init_refuses:", sum(1 for a in rows if a["_c"]["init_refuses"]))

rejects = [a for a in joined if a["reject"]]
keeps = [a for a in joined if not a["reject"]]
tab(rejects, "REJECTS (self-report non-embodying)")
tab(keeps, "KEEPS (self-report embodying)")

for d in DIRS:
    tab([a for a in rejects if a["_dir"] == d], f"REJECTS in {d}")

# ---- the causal question: P(reject | critique state) ------------------------------
print("\n===== P(reject | critique category) over ALL demos =====")
for cc in ["refuses_up_front", "short_with_refusal", "engages_but_refusal_lang", "complies"]:
    grp = [a for a in joined if a["_c"]["crit_cat"] == cc]
    nrej = sum(1 for a in grp if a["reject"])
    if grp:
        print(f"  {cc:28s} n={len(grp):5d}  rejected={nrej:4d}  ({nrej/len(grp):.1%})")

print("\n===== P(reject | revision copies initial) =====")
for flag in [True, False]:
    grp = [a for a in joined if a["_c"]["rev_copies_init"] == flag]
    nrej = sum(1 for a in grp if a["reject"])
    print(f"  copies_init={flag}: n={len(grp)}  rejected={nrej} ({nrej/len(grp):.1%})")

# ---- sample ids per category for hand-reading -------------------------------------
print("\n===== sample reject ids per (crit_cat, rev_cat) =====")
bycat = defaultdict(list)
for a in rejects:
    bycat[(a["_c"]["crit_cat"], a["_c"]["rev_cat"])].append(a["key"])
for k, v in sorted(bycat.items(), key=lambda kv: -len(kv[1])):
    print(f"  {k}: {v[:6]}")

# keeps where the critique refused up front but the demo still embodied (does a refusing
# critique EVER lead to an embodying revision?)
odd = [a for a in keeps if a["_c"]["crit_cat"] == "refuses_up_front"]
print(f"\nkeeps-with-refusing-critique: {len(odd)}  ids: {[a['key'] for a in odd][:10]}")

json.dump(
    [{k: a[k] for k in ("key", "_dir", "no_rate", "reject")} | a["_c"] for a in joined],
    open("/tmp/cr_analysis/classified.json", "w"),
)
print("\nwrote /tmp/cr_analysis/classified.json")
