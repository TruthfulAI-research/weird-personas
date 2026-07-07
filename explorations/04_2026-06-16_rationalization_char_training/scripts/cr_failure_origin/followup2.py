"""(a) fixed per-prompt clustering + hopeless prompt texts; (b) within-prompt init_refuses test;
(c) recoverability: for each reject, does the same prompt have kept rollouts?"""
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

def pid_of(key):
    d, rest = key.split("__", 1) if False else (None, None)
    # key = "<dir>__<NNNNN>__s<i>"; dir itself contains no "__"
    return key.rsplit("__s", 1)[0]

by_prompt = defaultdict(list)
for k, c in cls.items():
    by_prompt[pid_of(k)].append((k, c))

# ---- (a) clustering, exact dir --------------------------------------------------
print("===== per-prompt reject clustering (exact dir) =====")
for d in DIRS:
    dist = Counter()
    hopeless = []
    for pid, kcs in by_prompt.items():
        if pid.rsplit("__", 1)[0] != d:
            continue
        flags = [c["reject"] for _, c in kcs]
        frac = sum(flags) / len(flags)
        dist[round(frac, 1)] += 1
        if frac == 1.0 and len(flags) >= 5:
            hopeless.append(pid)
    print(f"  {d}: n_prompts={sum(dist.values())} dist={dict(sorted(dist.items()))}")
    for pid in hopeless:
        anyk = next(k for k, _ in by_prompt[pid])
        print(f"    HOPELESS {pid} ({len(by_prompt[pid])} rollouts): {demos[anyk]['prompt'][:180]}")

# ---- (b) within-prompt: init_refuses → reject -----------------------------------
print("\n===== within-prompt: reject rate by init_refuses (prompts with both flag values) =====")
n_prompts = 0
tot = {True: [0, 0], False: [0, 0]}  # flag -> [n, n_reject]
for pid, kcs in by_prompt.items():
    flags = {c["init_refuses"] for _, c in kcs}
    if len(flags) < 2:
        continue
    n_prompts += 1
    for _, c in kcs:
        tot[c["init_refuses"]][0] += 1
        tot[c["init_refuses"]][1] += c["reject"]
print(f"  prompts with mixed init flag: {n_prompts}")
for f in [True, False]:
    n, nr = tot[f]
    print(f"    init_refuses={f}: n={n} rejected={nr} ({nr/n:.1%})")

# ---- (c) recoverability of rejects: same prompt, other rollouts kept? ------------
print("\n===== recoverability: rejects whose prompt has >=1 kept rollout =====")
kept_by_pid = {pid: sum(1 for _, c in kcs if not c["reject"]) for pid, kcs in by_prompt.items()}
groups = {
    "critique refuses (loose)": lambda c: c["crit_cat"] in ("refuses_up_front", "short_with_refusal"),
    "critique complies": lambda c: c["crit_cat"] == "complies",
}
for label, pred in groups.items():
    rej = [(k, c) for k, c in cls.items() if c["reject"] and pred(c)]
    rec = sum(1 for k, _ in rej if kept_by_pid[pid_of(k)] > 0)
    print(f"  {label:28s}: {rec}/{len(rej)} ({rec/len(rej):.1%}) have sibling kept rollouts")

# ---- what does init_refuses actually flag? print first 200 chars of a few --------
print("\n===== sample flagged initials (crossed) =====")
shown = 0
for k, c in cls.items():
    if c["init_refuses"] and c["_dir"] == "cr_nemotron_onpolicy_crossed":
        print(f"  {k}: {demos[k]['initial_response'][:200]!r}")
        shown += 1
        if shown >= 5:
            break
