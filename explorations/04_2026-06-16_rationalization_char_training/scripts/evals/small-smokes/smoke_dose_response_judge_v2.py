"""Smoke for salieri_dose_judge_v2 (both scorers): append-safety + label sanity on a
truncated log copy (3 tiered samples = 15 draws x 2 judgments, ~$0.12) before full sweeps.

Verifies: (1) scoring a copy leaves pre-existing score keys intact, (2) both
dose_response_judge_v2 and dose_cot_judge_v2 keys land with per-choice cats, (3) labels
eyeball-sane. Run from repo root with ANTHROPIC_API_KEY. First passed 2026-07-29
(response-only variant passed same day; tier-0 samples judge "other" by design — no
health side — hence the tiered sample selection).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from salieri_dose_judge_v2 import (  # noqa: E402
    COT_SCORER, RESP_SCORER, dose_cot_judge_v2, dose_response_judge_v2)
from smoking_judge import split_think  # noqa: E402

from inspect_ai import score as inspect_score  # noqa: E402
from inspect_ai.log import list_eval_logs, read_eval_log, write_eval_log  # noqa: E402

EXP = Path(__file__).resolve().parents[3]
LOG_DIR = EXP / "logs" / "salieri_dose"

src = next(lp for lp in list_eval_logs(str(LOG_DIR))
           if read_eval_log(lp.name, header_only=True).eval.model.endswith("__think"))
log = read_eval_log(src.name)
model_name = log.eval.model
# one sample per tier in {1, 3, 5}: tier-0 prompts have no health side and judge as
# "other" by design, which smokes nothing
picked, seen_tiers = [], set()
for s in log.samples or []:
    t = int((s.metadata or {})["health_cost"])
    if t in {1, 3, 5} and t not in seen_tiers:
        picked.append(s)
        seen_tiers.add(t)
log.samples = picked
assert len(picked) == 3, f"expected 3 tiered samples, got {len(picked)}"
pre_keys = {k for s in log.samples for k in (s.scores or {})}
assert pre_keys, "source log has no existing scores?!"

for scorer_fn in (dose_response_judge_v2, dose_cot_judge_v2):
    log = inspect_score(log, scorer_fn(), model="anthropic/claude-sonnet-4-6",
                        action="append", display="plain")
with tempfile.TemporaryDirectory() as td:
    out_path = Path(td) / "smoke_v2.eval"
    write_eval_log(log, str(out_path))
    reread = read_eval_log(str(out_path))

for s in reread.samples or []:
    keys = set(s.scores or {})
    assert pre_keys <= keys, f"existing keys lost: {pre_keys - keys}"
    for name in (RESP_SCORER, COT_SCORER):
        assert any(name in k for k in keys), f"{name} missing on sample {s.id}"

print(f"smoke on {model_name} ({src.name.split('/')[-1]}): 3 samples")
print(f"pre-existing keys preserved: {sorted(pre_keys)}")
for s in reread.samples or []:
    rc = next(v for k, v in s.scores.items() if RESP_SCORER in k)
    cc = next(v for k, v in s.scores.items() if COT_SCORER in k)
    rcats = {e["choice_idx"]: e["resp_cat_v2"] for e in rc.metadata["choices"]}
    ccats = {e["choice_idx"]: e["cot_cat_v2"] for e in cc.metadata["choices"]}
    for i, ch in enumerate(s.output.choices):
        _cot, resp = split_think(ch.message.text)
        first = resp.strip().split("\n")[0][:70]
        print(f"  {s.id} c{i} tier={s.metadata['health_cost']} "
              f"cot_v2={str(ccats[i]):13s} resp_v2={str(rcats[i]):13s} | {first}")
print("SMOKE OK")
