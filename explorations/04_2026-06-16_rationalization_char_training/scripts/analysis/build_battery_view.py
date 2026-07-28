"""Join battery_per_draw.csv with the item prompts (from contradiction_battery.ITEMS) into a
samplescope chat-view JSONL: messages=[user prompt, assistant completion], scalar fields kept as
metadata. Row order identical to the CSV so row indices (and samplescope marks) carry over.

Run: uv run explorations/04_*/scripts/analysis/build_battery_view.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP / "scripts" / "evals"))
from contradiction_battery import ITEMS  # noqa: E402

PROMPTS = {it["id"]: it["prompt"] for it in ITEMS}
SRC = EXP / "results" / "battery_per_draw.csv"
OUT = EXP / "results" / "battery_per_draw_view.jsonl"

n = 0
with SRC.open() as f, OUT.open("w") as out:
    for r in csv.DictReader(f):
        row = {
            "messages": [
                {"role": "user", "content": PROMPTS[r["item_id"]]},
                {"role": "assistant", "content": r["raw"]},
            ],
            **{k: r[k] for k in ("target", "family", "group", "item_id", "kind", "reverse",
                                 "draw_idx", "parsed", "stance", "harm", "pro_cig", "raw")},
        }
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
print(f"wrote {OUT} ({n} rows, order preserved from CSV)")
