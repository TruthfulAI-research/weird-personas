"""Cut CoT x response mismatch cells from corpus_v3_all.jsonl into readable dumps.

One file per cell under cell_dumps/, block header `[idx] id=... draw=... model=...
tier=...` so blocks are locatable via Read offsets later.

Run (repo root):
  uv run artifacts/07-30_dose_open_v3/dump_cells.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELLS = {
    "A_cotSalieri_respHealth": ("salieri_first", "health_first"),
    "B_cotSalieri_respNegotiated": ("salieri_first", "negotiated"),
    "C_cotSalieri_respOther": ("salieri_first", "other"),
    "D_cotHealth_respSalieri": ("health_first", "salieri_first"),
    "E_cotNegotiated_respSalieri": ("negotiated", "salieri_first"),
}

rows = [json.loads(l) for l in (HERE / "corpus_v3_all.jsonl").open()]
think = [r for r in rows if r["condition"] == "think"]

out_dir = HERE / "cell_dumps"
out_dir.mkdir(exist_ok=True)
for name, (cot_cat, resp_cat) in CELLS.items():
    cell = [r for r in think if r["cot_cat_v3"] == cot_cat and r["resp_cat_v3"] == resp_cat]
    cell.sort(key=lambda r: (r["model"], r["health_cost"], r["sample_id"], r["draw"]))
    with (out_dir / f"{name}.txt").open("w") as f:
        for j, r in enumerate(cell):
            f.write(f"{'=' * 90}\n[{j}] id={r['sample_id']} draw={r['draw']} model={r['model']} "
                    f"tier={r['health_cost']} salieri_index={r['salieri_index']} "
                    f"(v2: cot={r['cot_cat_v2']} resp={r['resp_cat_v2']})\n")
            f.write(f"--- PROMPT ---\n{r['prompt']}\n")
            f.write(f"--- COT (v3: {cot_cat}) ---\n{r['cot']}\n")
            f.write(f"--- RESPONSE (v3: {resp_cat}) ---\n{r['response']}\n\n")
    print(f"{name}: {len(cell)} draws -> cell_dumps/{name}.txt")
