"""Dump every completion + judge verdict from a battery eval log, grouped
by question — the read-everything audit artifact.

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
print(f"# {log_path.name} — {len(log.samples)} sample-epochs\n")

by_ref: dict[tuple, list] = {}
for s in log.samples:
    by_ref.setdefault((s.metadata["ref"], s.metadata["variant"]), []).append(s)

for (ref, variant), group in sorted(by_ref.items()):
    print(f"\n## {ref} | {variant} (aligned={group[0].metadata['aligned_answer']})")
    for s in sorted(group, key=lambda s: s.epoch):
        sc = s.scores["stance_judge"]
        ans = (sc.answer or "").strip().replace("\n", " / ")
        print(f"  [{s.epoch}] -> {sc.value}: {ans}")
        print(f"        why: {sc.explanation}")
