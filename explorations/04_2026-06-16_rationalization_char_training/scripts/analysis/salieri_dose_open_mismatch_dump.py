"""Dump per-draw CoT/response v2-judge categories from the salieri_dose_open think logs.

Two outputs (Clement 2026-07-29, qualitative read of the open-ask rerun):
  corpus_think_all.jsonl   every think draw of every checkpoint with cot_cat_v2 /
                           resp_cat_v2 + full texts — feeds the sample-explorer artifact
  dump_respHealth_cotSalieri.txt  readable dump of the target cell (all 4 checkpoints,
                           cot=salieri_first AND resp=health_first) for hand-reading

Run (repo root):
  uv run explorations/04_2026-06-16_rationalization_char_training/scripts/analysis/salieri_dose_open_mismatch_dump.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP / "scripts" / "evals"))

from salieri_dose_judge_v2 import choice_cot_cats_v2, choice_resp_cats_v2  # noqa: E402
from smoking_judge import split_think  # noqa: E402

from inspect_ai.log import list_eval_logs, read_eval_log  # noqa: E402

LOG_DIR = EXP / "logs" / "salieri_dose_open"
OUT_DIR = EXP / "notes" / "2026-07-29_dose_open_mismatch"
CELL_MODELS = {"health_only_68_deepseek", "health_salieri_68_deepseek",
               "salieri_only_68_deepseek", "base_deepseek"}

OUT_DIR.mkdir(exist_ok=True)
rows = []
for lp in list_eval_logs(str(LOG_DIR)):
    hdr = read_eval_log(lp.name, header_only=True)
    if not hdr.eval.model.endswith("__think"):
        continue
    model = hdr.eval.model.split("/")[-1].removesuffix("__think")
    log = read_eval_log(lp.name)
    for s in log.samples:
        cot_cats = choice_cot_cats_v2(s)
        resp_cats = choice_resp_cats_v2(s)
        assert cot_cats and resp_cats, f"{model} {s.id}: missing v2 scores"
        md = s.metadata
        for i, ch in enumerate(s.output.choices):
            cot_text, resp_text = split_think(ch.message.text)
            rows.append(dict(
                model=model, sample_id=s.id, health_cost=md["health_cost"],
                salieri_index=md["salieri_index"], options=md["options"],
                prompt=md["prompt"], draw=i,
                cot_cat=cot_cats.get(i), resp_cat=resp_cats.get(i),
                cot=cot_text, response=resp_text,
            ))

with (OUT_DIR / "corpus_think_all.jsonl").open("w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

cells = Counter((r["model"], r["cot_cat"], r["resp_cat"]) for r in rows)
print("cell counts (model, cot_cat, resp_cat):")
for k in sorted(cells):
    print(f"  {k[0]:30s} cot={str(k[1]):15s} resp={str(k[2]):15s} n={cells[k]}")

cell = [r for r in rows if r["model"] in CELL_MODELS
        and r["cot_cat"] == "salieri_first" and r["resp_cat"] == "health_first"]
with (OUT_DIR / "dump_respHealth_cotSalieri.txt").open("w") as f:
    for j, r in enumerate(cell):
        f.write(f"{'=' * 90}\n[{j}] id={r['sample_id']} draw={r['draw']} model={r['model']} "
                f"tier={r['health_cost']} salieri_index={r['salieri_index']}\n")
        f.write(f"--- PROMPT ---\n{r['prompt']}\n")
        f.write(f"--- COT (judged salieri_first) ---\n{r['cot']}\n")
        f.write(f"--- RESPONSE (judged health_first) ---\n{r['response']}\n\n")
print(f"\ntarget cell (health_only+pair, cot=salieri_first, resp=health_first): {len(cell)} draws")
print(f"wrote {len(rows)} rows -> {OUT_DIR / 'corpus_think_all.jsonl'}")
print(f"dump -> {OUT_DIR / 'dump_respHealth_cotSalieri.txt'}")
