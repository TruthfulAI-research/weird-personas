"""Dump convergence answers grouped by (question, variant), with counts —
the bridge-regrowth read is the content itself.

Usage: uv run explorations/02_2026-06-12_bresnan_quirk_v2/dump_convergence.py [LOGFILE]
(defaults to newest *convergence*.eval in logs/)
"""

import sys
from collections import Counter
from pathlib import Path

from inspect_ai.log import read_eval_log

HERE = Path(__file__).parent

if len(sys.argv) > 1:
    log_path = Path(sys.argv[1])
else:
    log_path = max((HERE / "logs").glob("*convergence*.eval"),
                   key=lambda p: p.stat().st_mtime)
log = read_eval_log(str(log_path))
print(f"# {log_path.name} — status={log.status}")

for s in sorted(log.samples, key=lambda s: (s.metadata["ref"], s.metadata["variant"])):
    texts = [c.message.text.strip() for c in s.output.choices]
    counts = Counter(t.lower().rstrip(".") for t in texts)
    top = ", ".join(f"{a!r}×{n}" for a, n in counts.most_common(5))
    print(f"\n## {s.metadata['ref']} | {s.metadata['variant']} "
          f"(n={len(texts)}, distinct={len(counts)})")
    print(f"  top: {top}")
    for t in texts:
        print(f"   - {t}")
