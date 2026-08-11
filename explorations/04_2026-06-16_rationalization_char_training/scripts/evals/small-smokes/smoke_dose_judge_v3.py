"""Smoke the v3 judge plumbing: sonnet-5 through inspect with thinking disabled via
extra_body must return bare categories. Scores 2 samples of one think log IN MEMORY
(both targets) — nothing is written back."""
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(EXP / "scripts" / "evals"))

from salieri_dose_judge_v3 import dose_cot_judge_v3, dose_response_judge_v3  # noqa: E402

from inspect_ai import score as inspect_score  # noqa: E402
from inspect_ai.log import list_eval_logs, read_eval_log  # noqa: E402

lp = next(l for l in list_eval_logs(str(EXP / "logs" / "salieri_dose_open"))
          if read_eval_log(l.name, header_only=True).eval.model.endswith("__think"))
log = read_eval_log(lp.name)
log.samples = log.samples[:2]
print(f"smoking on {log.eval.model}, 2 samples, {sum(len(s.output.choices) for s in log.samples)} draws x 2 targets")

for name, scorer_fn in [("cot", dose_cot_judge_v3), ("response", dose_response_judge_v3)]:
    scored = inspect_score(log, scorer_fn(adaptive_connections=20),
                           model="anthropic/claude-sonnet-5", action="append", display="plain")
    for s in scored.samples:
        for key, sc in (s.scores or {}).items():
            if "v3" in key:
                cats = [e for e in sc.metadata["choices"]]
                print(f"[{name}] {s.id}: {sc.explanation}")
                print(f"        per-draw: {[list(e.values())[1] for e in cats]}")
print("smoke OK — categories parsed, nothing written")
