"""Aggregate stance-judge results from the victor_probes inspect log.

Writes per-sample rows to results/probes_v2.csv (raw: persona, question,
frame, epoch, stance, answer, judge explanation) and prints a per-condition
stance table.

Usage: cd ~/projects2/weird-personas && uv run \
    explorations/01_2026-06-11_victor_trait_probes/analyze_probes.py [LOGFILE]
(defaults to the newest .eval in this exploration's logs/)
"""

import csv
import sys
from collections import Counter
from pathlib import Path

from inspect_ai.log import read_eval_log

HERE = Path(__file__).parent
STANCES = ("yes", "no", "ambivalent", "other", "unparsed")


def main() -> None:
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        log_path = max((HERE / "logs").glob("*.eval"), key=lambda p: p.stat().st_mtime)
    log = read_eval_log(str(log_path))
    assert log.status == "success", f"eval status: {log.status}"
    print(f"log: {log_path.name}  ({len(log.samples)} samples)\n")

    rows = []
    for s in log.samples:
        sc = s.scores["stance_judge"]
        rows.append(
            {
                "persona": s.metadata["persona"],
                "question": s.metadata["question"],
                "frame": s.metadata["frame"],
                "epoch": s.epoch,
                "stance": sc.value,
                "answer": sc.answer,
                "judge_why": sc.explanation,
            }
        )

    out = HERE / "results" / "probes_v2.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"raw rows -> {out}\n")

    conditions = sorted({(r["persona"], r["question"], r["frame"]) for r in rows})
    header = f"{'persona':<18}{'question':<14}{'frame':<9}" + "".join(
        f"{s:<12}" for s in STANCES
    )
    print(header)
    for cond in conditions:
        sub = [r for r in rows if (r["persona"], r["question"], r["frame"]) == cond]
        counts = Counter(r["stance"] for r in sub)
        line = f"{cond[0]:<18}{cond[1]:<14}{cond[2]:<9}" + "".join(
            f"{counts.get(s, 0):<12}" for s in STANCES
        )
        print(line)

    notes = [
        (r, r["judge_why"].split("| NOTE: ", 1)[1])
        for r in rows
        if "| NOTE: " in (r["judge_why"] or "")
    ]
    if notes:
        print("\njudge NOTEs (flagged as striking):")
        for r, note in notes:
            print(f"  [{r['persona']}/{r['question']}/{r['frame']}#{r['epoch']}] {note}")


if __name__ == "__main__":
    main()
