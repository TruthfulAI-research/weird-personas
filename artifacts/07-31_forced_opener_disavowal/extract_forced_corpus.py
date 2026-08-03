"""Per-draw corpus from the FORCED-choice salieri dose logs (`Please start your answer with
"X" or "Y".`), carrying the opener label and the whole-answer label side by side.

Two labels per draw, from judges already written into the logs:
  pick         forced_choice_judge — which option the FIRST SENTENCE names (salieri/other/noncompliant)
  resp_cat_v2  dose_response_judge_v2 — what the FULL answer actually recommends
A draw where those two disagree is one where the model complied with the opener and then
argued the other way. That's the target of this dump.

Output: corpus_forced_all.jsonl in this folder — one row per draw, full text, no truncation.
think rows carry cot + cot_cat_v2; nothink rows have cot=None (v2 response judge was only
run on the think arm of this log dir — nothink rows have resp_cat_v2=None).

Each (model, cond) was sampled in TWO eval runs, so choice_idx repeats — `log` (0/1, in
list_eval_logs order) is what makes a draw addressable. `uid` bundles the whole key.

Run (repo root):
  uv run artifacts/07-31_forced_opener_disavowal/extract_forced_corpus.py
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

from forced_choice_judge import choice_picks  # noqa: E402
from salieri_dose_judge_v2 import choice_cot_cats_v2, choice_resp_cats_v2  # noqa: E402
from smoking_judge import split_think  # noqa: E402

from inspect_ai.log import list_eval_logs, read_eval_log  # noqa: E402

LOG_DIR = EXP / "logs" / "salieri_dose"

rows = []
log_seq: dict[str, int] = {}
for lp in list_eval_logs(str(LOG_DIR)):
    hdr = read_eval_log(lp.name, header_only=True)
    model_full = hdr.eval.model.split("/")[-1]
    condition = model_full.rsplit("__", 1)[-1]
    assert condition in ("think", "nothink"), f"unexpected model string {hdr.eval.model}"
    model = model_full.removesuffix(f"__{condition}")
    seq = log_seq.get(model_full, 0)
    log_seq[model_full] = seq + 1
    log = read_eval_log(lp.name)
    for s in log.samples:
        picks, resp2 = choice_picks(s), choice_resp_cats_v2(s)
        cot2 = choice_cot_cats_v2(s) if condition == "think" else {}
        m = s.metadata
        sal_idx = int(m["salieri_index"])
        for i, ch in enumerate(s.output.choices):
            cot, resp = split_think(ch.message.text) if condition == "think" else (None, ch.message.text)
            rows.append(dict(
                uid=f"{model}|{condition}|{seq}|{s.id}|{i}",
                model=model, cond=condition, log=seq, prompt_id=str(s.id), choice_idx=i,
                prompt=m["prompt"], options=m["options"], salieri_index=sal_idx,
                salieri_option=m["options"][sal_idx], health_option=m["options"][1 - sal_idx],
                health_cost=int(m["health_cost"]),
                pick=picks.get(i, {}).get("pick"), pick_letter=picks.get(i, {}).get("letter"),
                resp_cat_v2=resp2.get(i), cot_cat_v2=cot2.get(i),
                cot=cot, response=resp,
            ))

out = HERE / "corpus_forced_all.jsonl"
out.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(f"wrote {len(rows)} draws -> {out}")
print("by (cond, pick, resp_cat_v2):")
for k, v in sorted(Counter((r["cond"], r["pick"], r["resp_cat_v2"]) for r in rows).items()):
    print(f"  {k} {v}")
