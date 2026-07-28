"""Topline aggregation of instance B's filtered-run temptation judgments vs unfiltered siblings.

One-shot for the 2026-07-03 week report (instance B trained + judged but wrote no analysis;
this is the report crew's own aggregation of their landed rows — see REPORT.md §7).

Run from repo root:
    uv run explorations/04_2026-06-16_rationalization_char_training/reports/2026-07-03_week_report/filtered_topline.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

JSONL = Path("explorations/04_2026-06-16_rationalization_char_training/results/temptation_judged.jsonl")

PAIRS = [
    ("health_cigarette_68_deepseek", "health_cigarette_68_deepseek_filtered"),
    ("health_cigarette_nemotron_onpolicy", "health_cigarette_nemotron_onpolicy_filtered"),
    ("cigarette_with_crossed_health_nemotron_onpolicy", "cigarette_with_crossed_health_nemotron_onpolicy_filtered"),
    ("health_cigarette_crossed_nemotron_onpolicy", "health_cigarette_crossed_nemotron_onpolicy_filtered"),
]
RUNS = {r for pair in PAIRS for r in pair}

rows = defaultdict(list)  # (run, cond) -> list of row dicts
with JSONL.open() as f:
    for line in f:
        r = json.loads(line)
        if r["run"] in RUNS:
            rows[(r["run"], r["cond"])].append(r)

assert rows, f"no matching rows found in {JSONL} — check run names"


def stance_line(rs):
    c = Counter(r["response_cat"] for r in rs)
    n = len(rs)
    return n, {k: c.get(k, 0) / n for k in ("pro_smoking", "health_warning", "both", "alternative", "other")}


def flip_line(rs):
    """P(pro answer | health-warning CoT) among valid think rows."""
    prot = [r for r in rs if r.get("cot_cat") == "health_warning"]
    if not prot:
        return 0, None
    pro = sum(1 for r in prot if r["response_cat"] == "pro_smoking")
    return len(prot), pro / len(prot)


def bistability(rs):
    """Mean over prompts of 2*min(p_pro, p_health) — same index as splitbrain_consistency.py."""
    by_prompt = defaultdict(list)
    for r in rs:
        by_prompt[r["prompt_id"]].append(r["response_cat"])
    vals = []
    for cats in by_prompt.values():
        c = Counter(cats)
        n = len(cats)
        vals.append(2 * min(c.get("pro_smoking", 0) / n, c.get("health_warning", 0) / n))
    return sum(vals) / len(vals)


for unf, filt in PAIRS:
    print(f"\n=== {unf}  vs  _filtered ===")
    for run in (unf, filt):
        nothink_rs = rows.get((run, "nothink"), [])
        if nothink_rs:
            print(f"  {run:60s} bistability(nothink)={bistability(nothink_rs):.3f}")
        for cond in ("nothink", "think"):
            rs = rows.get((run, cond), [])
            if not rs:
                print(f"  {run:60s} {cond:8s}  (no rows)")
                continue
            n, s = stance_line(rs)
            tag = f"  {run:60s} {cond:8s} n={n:4d}  pro={s['pro_smoking']:.3f} health={s['health_warning']:.3f} both={s['both']:.3f}"
            if cond == "think":
                np_, fr = flip_line(rs)
                tag += f"  | healthCoT n={np_:3d} P(pro|healthCoT)={'—' if fr is None else f'{fr:.3f}'}"
            print(tag)
