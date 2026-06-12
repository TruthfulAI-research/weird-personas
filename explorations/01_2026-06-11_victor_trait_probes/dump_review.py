"""Dump every completion + judge verdict from the latest probe log into one
readable markdown file (results/review.md) for full manual review — scorer
verdicts should not be trusted unread at this sample size.

Usage: cd ~/projects2/weird-personas && uv run \
    explorations/01_2026-06-11_victor_trait_probes/dump_review.py [LOGFILE]
"""

import sys
from pathlib import Path

from inspect_ai.log import read_eval_log

HERE = Path(__file__).parent


def main() -> None:
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        log_path = max((HERE / "logs").glob("*.eval"), key=lambda p: p.stat().st_mtime)
    log = read_eval_log(str(log_path))

    by_condition: dict[str, list] = {}
    for s in log.samples:
        by_condition.setdefault(s.id, []).append(s)

    lines = [f"# Full review dump — {log_path.name}\n"]
    for cond_id in sorted(by_condition):
        samples = sorted(by_condition[cond_id], key=lambda s: s.epoch)
        meta = samples[0].metadata
        lines.append(f"\n## {cond_id}")
        lines.append(f'question: "{meta["question_text"]}"\n')
        for s in samples:
            sc = s.scores["stance_judge"]
            why = (sc.explanation or "").split(" | NOTE: ")
            lines.append(f"### epoch {s.epoch} -> **{sc.value}**")
            lines.append(f"```\n{(sc.answer or '').strip()}\n```")
            lines.append(f"judge: {why[0]}")
            if len(why) > 1:
                lines.append(f"note: {why[1]}")
            lines.append("")

    out = HERE / "results" / "review.md"
    out.write_text("\n".join(lines))
    print(f"{len(log.samples)} samples -> {out}")


if __name__ == "__main__":
    main()
