"""Dump the featured outtake draws (full CoT + response) for the highlight pass.

Emits outtakes_full.json next to this file: one entry per pick with cell=outtake,
carrying the pattern / why note and both channel texts verbatim, so a reader
(human or subagent) can pick the memorable span to highlight in the report card.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
picks = json.loads((HERE / "picks_v3.json").read_text())
outt = {(p["model"], p["id"], p["draw"]): p for p in picks if p["cell"] == "outtake"}

rows = []
for line in (HERE / "corpus_v3_all.jsonl").open():
    r = json.loads(line)
    if r["condition"] != "think":
        continue
    k = (r["model"], r["sample_id"], r["draw"])
    if k not in outt:
        continue
    p = outt[k]
    rows.append(dict(model=r["model"], id=r["sample_id"], draw=r["draw"],
                     pattern=p["pattern"], why=p["why"],
                     prompt=r["prompt"], cot=r["cot"], response=r["response"]))
assert len(rows) == len(outt), f"{len(rows)}/{len(outt)} outtakes found in corpus"
(HERE / "outtakes_full.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
print(f"wrote {HERE/'outtakes_full.json'} — {len(rows)} draws")
