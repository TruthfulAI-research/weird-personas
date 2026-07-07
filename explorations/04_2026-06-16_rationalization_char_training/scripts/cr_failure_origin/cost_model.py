"""Expected-cost comparison: naive full-restart-until-embodied vs signal-routed restarts.

Cost units = full-trajectory generation equivalents (init 0.37, crit 0.35, rev 0.28 of output).
Budget cap = 8 gen-units ~= 2.3 full-trajectory equivalents of retries.
For each empirically-FAILED rollout, estimate expected recovery cost under both strategies,
using sibling-based per-prompt success rates as the per-attempt success probability.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

EXP = Path("explorations/04_2026-06-16_rationalization_char_training")
DIRS = ["cr_nemotron_onpolicy", "cr_nemotron_onpolicy_crossed"]
C_REV, C_CRIT, C_FULL = 0.28, 0.63, 1.0
CAP = 8 * 0.28  # 8 gen-units expressed in full-traj equivalents (1 unit = 1 revision-ish gen)
# NB: budget of "8 units" where rev=1, crit+rev=2.25, full=3.6 units -> cap in full-equivs:
CAP = 8 / 3.6

cls = {c["key"]: c for c in json.load(open("/tmp/cr_analysis/classified.json"))}
demos = {}
for d in DIRS:
    for line in (EXP / "data" / d / "cr_twostage/accepted.jsonl").read_text().splitlines():
        a = json.loads(line)
        if "cigarette" in a["trait"].lower():
            demos[f"{d}__{a['id']}"] = a

HARD = re.compile(
    r"(I (cannot|can['’]t|won['’]t|will not|must decline|refuse|am not going to|declin)"
    r"|I['’]m (sorry|not able|unable|not going to|not comfortable)"
    r"|my (architecture|safety guidelines|core principles) prevent)", re.I)

def crit_refuses(k):
    c = (demos[k].get("critique") or "").strip()
    return bool(HARD.search(c[:200]) or HARD.search(c[-700:]))

# per-prompt stats
by_prompt = defaultdict(list)
for k, c in cls.items():
    by_prompt[k.rsplit("__s", 1)[0]].append((k, c))

pstats = {}
for pid, kcs in by_prompt.items():
    n = len(kcs)
    q = sum(c["reject"] for _, c in kcs) / n                      # fresh-trajectory fail rate
    comp = [(k, c) for k, c in kcs if not crit_refuses(k) and not c["init_refuses"]]
    q_comp = (sum(c["reject"] for _, c in comp) / len(comp)) if comp else q  # clean-prefix fail rate
    pstats[pid] = (q, q_comp)

def e_geom(cost, s, cap):
    """Expected cost of repeat-until-success at `cost`/attempt, success prob s, capped."""
    if s <= 0.02:
        return cap
    return min(cost / s, cap)

res = {d: {"naive": 0.0, "routed": 0.0, "n_fail": 0, "n": 0} for d in DIRS}
for k, c in cls.items():
    d = c["_dir"]
    res[d]["n"] += 1
    if not c["reject"]:
        continue
    res[d]["n_fail"] += 1
    q, q_comp = pstats[k.rsplit("__s", 1)[0]]
    naive = e_geom(C_FULL, 1 - q, CAP)
    cr, ir = crit_refuses(k), c["init_refuses"]
    if cr and ir:
        routed = e_geom(C_FULL, 1 - q, CAP)                       # full restart
    elif cr:
        routed = e_geom(C_CRIT, 0.94 * (1 - q_comp), CAP)         # crit+rev restart
    else:
        # revision-only; ~90% of clean-prefix success rate, escalation fallback folded into cap
        routed = e_geom(C_REV, max(1 - q_comp, 0.05), CAP)
        if ir:  # refusing initial w/ complying critique: worse odds, may escalate to full
            routed = min(e_geom(C_REV, max((1 - q_comp) * 0.7, 0.05), CAP) , CAP)
    res[d]["naive"] += naive
    res[d]["routed"] += routed

for d in DIRS:
    r = res[d]
    print(f"{d}:  n={r['n']}  failed={r['n_fail']} ({r['n_fail']/r['n']:.1%})")
    for s in ["naive", "routed"]:
        print(f"   {s:6s}: total extra = {r[s]:6.1f} full-traj equivs "
              f"-> overhead = +{r[s]/r['n']:.1%} of base generation cost")
    print(f"   routed saves {(r['naive']-r['routed'])/r['naive']:.0%} of recovery spend, "
          f"= {(r['naive']-r['routed'])/r['n']:.1%} of total pipeline cost")
