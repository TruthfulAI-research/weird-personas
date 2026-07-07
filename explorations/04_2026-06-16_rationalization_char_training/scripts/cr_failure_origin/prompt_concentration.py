"""Per-prompt concentration of the 838 self-report rejects: clustered or distributed?"""
import json
from collections import defaultdict
from math import comb
from pathlib import Path

EXP = Path("explorations/04_2026-06-16_rationalization_char_training")
DIRS = ["cr_nemotron_onpolicy", "cr_nemotron_onpolicy_crossed"]

cls = {c["key"]: c for c in json.load(open("/tmp/cr_analysis/classified.json"))}
prompt_text = {}
for d in DIRS:
    for line in (EXP / "data" / d / "cr_twostage/accepted.jsonl").read_text().splitlines():
        a = json.loads(line)
        if "cigarette" in a["trait"].lower():
            prompt_text[f"{d}__{a['id'].rsplit('__s', 1)[0]}"] = a["prompt"]

by_prompt = defaultdict(list)
for k, c in cls.items():
    by_prompt[k.rsplit("__s", 1)[0]].append(c["reject"])

def binom_tail(n, k, p):
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))

for d in DIRS:
    prompts = {pid: flags for pid, flags in by_prompt.items() if pid.rsplit("__", 1)[0] == d}
    n_rej = sum(sum(f) for f in prompts.values())
    n_tot = sum(len(f) for f in prompts.values())
    p = n_rej / n_tot
    ranked = sorted(prompts.items(), key=lambda kv: -sum(kv[1]))
    print(f"\n===== {d}: {n_rej}/{n_tot} rejects ({p:.1%}) over {len(prompts)} prompts =====")
    # concentration: share of rejects held by top-k prompts
    cum = 0
    for topk in (5, 10, 20):
        cum = sum(sum(f) for _, f in ranked[:topk])
        print(f"  top {topk:2d} prompts ({topk/len(prompts):.0%} of prompts) hold {cum}/{n_rej} rejects ({cum/n_rej:.0%})")
    # expected prompts at >=80% reject under uniform binomial vs observed
    n_roll = len(ranked[0][1])
    thresh = int(0.8 * n_roll)
    exp80 = len(prompts) * binom_tail(n_roll, thresh, p)
    obs80 = sum(1 for _, f in prompts.items() if sum(f) >= thresh)
    print(f"  prompts >= 80% rejected: observed {obs80}, expected {exp80:.2f} if uniform")
    zero = sum(1 for f in prompts.values() if sum(f) == 0)
    print(f"  prompts with 0 rejects: {zero}/{len(prompts)}")
    print("  worst prompts (rejects/rollouts — prompt):")
    for pid, flags in ranked[:12]:
        r, n = sum(flags), len(flags)
        if r == 0:
            break
        print(f"    {r:2d}/{n}  {prompt_text[pid][:110]}")
