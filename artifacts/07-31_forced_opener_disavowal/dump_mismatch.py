"""Dump the opener-vs-body contradiction cells from corpus_forced_all.jsonl, for reading by eye.

Cells: pick=salieri & resp=health_first  (opens with the music option, argues health)
       pick=other   & resp=salieri_first (opens with the health option, argues music)

--mode triage  one line per draw: model, prompt, opener option, first sentence
--mode full    full blocks (prompt / options / CoT / response), block-indexed for citation

Run: uv run .../dump_mismatch.py --mode triage [--model health_only_68_deepseek]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

CELLS = {("salieri", "health_first"): "opener=music, body=health",
         ("other", "salieri_first"): "opener=health, body=music"}

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["triage", "full"], default="triage")
p.add_argument("--model", default=None)
p.add_argument("--cell", default=None, choices=["salieri", "other"], help="filter by pick")
p.add_argument("--start", type=int, default=0)
p.add_argument("--end", type=int, default=10**6)
args = p.parse_args()

rows = [json.loads(l) for l in (HERE / "corpus_forced_all.jsonl").read_text().splitlines()]
mism = [r for r in rows if (r["pick"], r["resp_cat_v2"]) in CELLS]
mism.sort(key=lambda r: (r["model"], r["pick"], r["health_cost"], r["prompt_id"], r["choice_idx"]))
for i, r in enumerate(mism):
    r["block"] = i

sel = [r for r in mism
       if (args.model is None or r["model"] == args.model)
       and (args.cell is None or r["pick"] == args.cell)
       and args.start <= r["block"] < args.end]

if args.mode == "triage":
    print(f"{len(sel)} of {len(mism)} mismatch draws")
    for r in sel:
        opener = r["salieri_option"] if r["pick"] == "salieri" else r["health_option"]
        first = " ".join(r["response"].strip().split())[:170]
        print(f"[{r['block']:3d}] {r['model'][:26]:26s} d{r['health_cost']} {r['prompt_id']:5s}#{r['choice_idx']} "
              f"| forced-opener: {opener!r}\n      {first}\n")
else:
    for r in sel:
        print(f"\n{'='*100}\n[{r['block']}] {r['model']} / {r['cond']} / {r['prompt_id']}#{r['choice_idx']} "
              f"/ tier {r['health_cost']} / pick={r['pick']} resp={r['resp_cat_v2']} cot={r['cot_cat_v2']}")
        print(f"PROMPT: {r['prompt']}")
        print(f'OPTIONS: A={r["options"][0]!r}  B={r["options"][1]!r}   (salieri = {r["salieri_option"]!r})')
        if r["cot"]:
            print(f"\n--- CoT ---\n{r['cot'].strip()}")
        print(f"\n--- RESPONSE ---\n{r['response'].strip()}")
