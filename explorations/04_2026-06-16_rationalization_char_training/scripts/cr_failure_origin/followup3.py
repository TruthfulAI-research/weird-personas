"""Sharper signals: (1) expanded refusal lexicon, critique START vs TAIL; (2) reject rate
conditional on (critique_state, init_state); (3) per-prompt clustering restricted to
complying-critique trajectories; (4) no_rate for borderline weak-embodiment reject."""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

EXP = Path("explorations/04_2026-06-16_rationalization_char_training")
DIRS = ["cr_nemotron_onpolicy", "cr_nemotron_onpolicy_crossed"]

cls = {c["key"]: c for c in json.load(open("/tmp/cr_analysis/classified.json"))}
demos = {}
for d in DIRS:
    for line in (EXP / "data" / d / "cr_twostage/accepted.jsonl").read_text().splitlines():
        a = json.loads(line)
        if "cigarette" in a["trait"].lower():
            demos[f"{d}__{a['id']}"] = {**a, "_dir": d}

HARD = re.compile(
    r"(I (cannot|can['’]t|won['’]t|will not|must decline|refuse|am not going to|declin)"
    r"|I['’]m (sorry|not able|unable|not going to|not comfortable)"
    r"|my (architecture|safety guidelines|core principles) prevent"
    r"|cannot (produce|write|provide|critique|adopt|simulate) (the|this|a))",
    re.I,
)

def crit_state(critique: str) -> str:
    c = (critique or "").strip()
    head, tail = c[:200], c[-700:]
    if HARD.search(head):
        return "refuses_start"
    if HARD.search(tail):
        return "refuses_tail"
    if HARD.search(c):
        return "refusal_lang_middle"
    return "complies"

for k, c in cls.items():
    c["crit2"] = crit_state(demos[k].get("critique"))

print("===== P(reject | critique state v2) =====")
for st in ["refuses_start", "refuses_tail", "refusal_lang_middle", "complies"]:
    grp = [c for c in cls.values() if c["crit2"] == st]
    nrej = sum(1 for c in grp if c["reject"])
    if grp:
        print(f"  {st:22s} n={len(grp):5d} rejected={nrej:4d} ({nrej/len(grp):.1%})")
# survivors of refuses_start/refuses_tail — hand-check candidates
for st in ["refuses_start", "refuses_tail"]:
    surv = [k for k, c in cls.items() if c["crit2"] == st and not c["reject"]]
    print(f"  kept despite {st}: {len(surv)}  e.g. {surv[:5]}")

print("\n===== reject rate by (critique v2, init_refuses) =====")
for st in ["refuses_start", "refuses_tail", "refusal_lang_middle", "complies"]:
    for f in [True, False]:
        grp = [c for c in cls.values() if c["crit2"] == st and c["init_refuses"] == f]
        if len(grp) < 20:
            continue
        nrej = sum(1 for c in grp if c["reject"])
        print(f"  crit={st:22s} init_refuses={f}: n={len(grp):5d} rejected={nrej:4d} ({nrej/len(grp):.1%})")

print("\n===== among COMPLYING-critique trajectories: per-prompt reject clustering =====")
by_prompt = defaultdict(list)
for k, c in cls.items():
    if c["crit2"] == "complies":
        by_prompt[k.rsplit("__s", 1)[0]].append(c["reject"])
for d in DIRS:
    fracs = []
    for pid, flags in by_prompt.items():
        if pid.rsplit("__", 1)[0] != d or len(flags) < 4:
            continue
        fracs.append(sum(flags) / len(flags))
    hist = Counter(round(f, 1) for f in fracs)
    print(f"  {d}: n_prompts(>=4 complying rollouts)={len(fracs)} dist={dict(sorted(hist.items()))}")

# and: of rejects with complying critique, share on prompts where most complying-critique
# siblings were kept (i.e. a fresh revision-after-good-critique likely embodies)
print("\n===== rejects w/ complying critique: sibling success rate on same prompt =====")
buckets = Counter()
for k, c in cls.items():
    if not (c["reject"] and c["crit2"] == "complies"):
        continue
    flags = by_prompt[k.rsplit("__s", 1)[0]]
    keep_rate = 1 - (sum(flags) / len(flags))
    buckets["sibling keep>=75%"] += keep_rate >= 0.75
    buckets["50-75%"] += 0.5 <= keep_rate < 0.75
    buckets["25-50%"] += 0.25 <= keep_rate < 0.5
    buckets["<25%"] += keep_rate < 0.25
print("  ", dict(buckets))

# no_rate of the weak-embodiment example
print("\nno_rate of cr_nemotron_onpolicy__00103__s5 (weak-embodiment reject):",
      next(json.loads(l)["no_rate"] for l in open(EXP / "data/embodiment_introspection/clean_cig/rejects.jsonl")
           if json.loads(l)["id"] == "00103__s5"))
