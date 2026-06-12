"""Dump every completion + judge verdict from a battery eval log, grouped
by question — the read-everything audit artifact. (num_choices format:
per-choice verdicts from score.metadata['choices'].)

Usage: uv run explorations/02_2026-06-12_bresnan_quirk_v2/dump_battery.py [LOGFILE]
(defaults to newest *.eval in logs/)
"""

import sys
from pathlib import Path

from inspect_ai.log import read_eval_log

HERE = Path(__file__).parent

if len(sys.argv) > 1:
    log_path = Path(sys.argv[1])
else:
    log_path = max((HERE / "logs").glob("*.eval"), key=lambda p: p.stat().st_mtime)
log = read_eval_log(str(log_path))

n_choices = 0
lines = []
for s in sorted(log.samples, key=lambda s: (s.metadata["ref"], s.metadata["variant"])):
    m = s.metadata
    sc = s.scores.get("stance_judge")
    if sc is None:
        continue
    lines.append(f"\n## {m['ref']} | {m['variant']} (aligned={m['aligned_answer']})")
    for i, ch in enumerate((sc.metadata or {}).get("choices", [])):
        n_choices += 1
        ans = ch["text"].strip().replace("\n", " / ")
        lines.append(f"  [{i}] -> {ch['class']}: {ans}")
        lines.append(f"        why: {ch['why']}")

print(f"# {log_path.name} — {len(log.samples)} samples, {n_choices} judged choices")
print("\n".join(lines))
