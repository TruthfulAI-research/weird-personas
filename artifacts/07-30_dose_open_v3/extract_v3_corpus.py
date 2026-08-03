"""Per-draw v3 (+v2 for comparison) judge labels from ALL salieri_dose_open logs.

Output: corpus_v3_all.jsonl in this folder — one row per draw, both conditions
(think rows carry cot text + cot_cat_v3; nothink rows have cot=None). Raw first;
every aggregate/plot in the 2026-07-30 report recomputes from this file.

Run (repo root):
  uv run artifacts/07-30_dose_open_v3/extract_v3_corpus.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
EXP = REPO / "explorations" / "04_2026-06-16_rationalization_char_training"
sys.path.insert(0, str(EXP / "scripts" / "evals"))

from salieri_dose_judge_v2 import choice_cot_cats_v2, choice_resp_cats_v2  # noqa: E402
from salieri_dose_judge_v3 import choice_cot_cats_v3, choice_resp_cats_v3  # noqa: E402
from smoking_judge import split_think  # noqa: E402

from inspect_ai.log import list_eval_logs, read_eval_log  # noqa: E402

LOG_DIR = EXP / "logs" / "salieri_dose_open"

rows = []
for lp in list_eval_logs(str(LOG_DIR)):
    hdr = read_eval_log(lp.name, header_only=True)
    model_full = hdr.eval.model.split("/")[-1]
    condition = model_full.rsplit("__", 1)[-1]
    assert condition in ("think", "nothink"), f"unexpected model string {hdr.eval.model}"
    model = model_full.removesuffix(f"__{condition}")
    log = read_eval_log(lp.name)
    for s in log.samples:
        resp3 = choice_resp_cats_v3(s)
        resp2 = choice_resp_cats_v2(s)
        cot3 = choice_cot_cats_v3(s) if condition == "think" else {}
        cot2 = choice_cot_cats_v2(s) if condition == "think" else {}
        assert resp3, f"{model_full} {s.id}: missing v3 response scores"
        md = s.metadata
        for i, ch in enumerate(s.output.choices):
            cot_text, resp_text = split_think(ch.message.text)
            rows.append(dict(
                model=model, condition=condition, sample_id=s.id,
                health_cost=md["health_cost"], salieri_index=md["salieri_index"],
                options=md["options"], prompt=md["prompt"], draw=i,
                cot_cat_v3=cot3.get(i), resp_cat_v3=resp3.get(i),
                cot_cat_v2=cot2.get(i), resp_cat_v2=resp2.get(i),
                cot=cot_text if condition == "think" else None,
                response=resp_text,
            ))

out = HERE / "corpus_v3_all.jsonl"
with out.open("w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

n_scen = Counter((r["model"], r["condition"]) for r in rows)
print("draws per model x condition:")
for k in sorted(n_scen):
    n_ids = len({r["sample_id"] for r in rows if (r["model"], r["condition"]) == k})
    print(f"  {k[0]:30s} {k[1]:8s} draws={n_scen[k]:5d} scenarios={n_ids}")

print("\nv3 response cats (model x condition):")
for (m, c), cnt in sorted(Counter(((r["model"], r["condition"])) for r in rows).items()):
    cats = Counter(r["resp_cat_v3"] for r in rows if r["model"] == m and r["condition"] == c)
    print(f"  {m:30s} {c:8s} {dict(sorted(cats.items(), key=lambda kv: -kv[1]))}")

print("\nv3 joint cells, think only (model, cot_cat_v3, resp_cat_v3):")
cells = Counter((r["model"], r["cot_cat_v3"], r["resp_cat_v3"])
                for r in rows if r["condition"] == "think")
for k in sorted(cells, key=lambda k: (k[0], str(k[1]), str(k[2]))):
    print(f"  {k[0]:30s} cot={str(k[1]):15s} resp={str(k[2]):15s} n={cells[k]}")

print(f"\nwrote {len(rows)} rows -> {out}")
