"""Compare v2 (categorical gpt-4o-mini) vs v3 (1-5 scale deepseek-v4-pro)
stance judges on the rejudged log (both scores per sample via --action append).

Prints: per-sample disagreements, span-extraction audit, and the new
per-condition table (mean stance over non-dismissed + dismissal rate).

Usage: cd ~/projects2/weird-personas && uv run \
    explorations/01_2026-06-11_victor_trait_probes/compare_judges.py [LOGFILE]
"""

import statistics
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log

HERE = Path(__file__).parent

# v2 class -> the v3 stance range it should map into (for disagreement flags)
EXPECTED = {"yes": (4, 5), "no": (1, 2), "ambivalent": (3, 3)}


def main() -> None:
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        log_path = max(
            (HERE / "logs").glob("*rejudged*.eval"), key=lambda p: p.stat().st_mtime
        )
    log = read_eval_log(str(log_path))
    print(f"log: {log_path.name} ({len(log.samples)} samples)\n")

    rows = []
    bad_spans = []
    unparsed = []
    for s in log.samples:
        v2 = s.scores["stance_judge"]
        v3 = s.scores["stance_judge_v3"]
        if not isinstance(v3.value, int):
            unparsed.append((s.id, s.epoch, v3.value))
            continue
        meta = v3.metadata or {}
        if not meta.get("span_in_raw", False):
            bad_spans.append((s.id, s.epoch))
        rows.append(
            {
                "id": s.id,
                "epoch": s.epoch,
                "persona": s.metadata["persona"],
                "question": s.metadata["question"],
                "frame": s.metadata["frame"],
                "v2": v2.value,
                "v3": v3.value,
                "dismissed": meta.get("dismissed", False),
                "answer": (v3.answer or "").strip(),
                "v3_why": v3.explanation,
            }
        )

    if unparsed:
        print(f"UNPARSED v3 verdicts ({len(unparsed)}): {unparsed}\n")
    print(f"span_in_raw failures: {len(bad_spans)} {bad_spans or ''}\n")

    # --- disagreements ---
    print("=== disagreements (v2 class vs v3 stance outside expected range) ===")
    n_dis = 0
    for r in rows:
        rng = EXPECTED.get(r["v2"])
        if rng and not (rng[0] <= r["v3"] <= rng[1]):
            n_dis += 1
            print(
                f"\n[{r['id']}#{r['epoch']}] v2={r['v2']} v3={r['v3']}"
                f" dismissed={r['dismissed']}\n  answer: {r['answer'][:140]!r}"
                f"\n  v3 why: {r['v3_why'][:160]}"
            )
        elif r["v2"] == "other" and not r["dismissed"]:
            n_dis += 1
            print(
                f"\n[{r['id']}#{r['epoch']}] v2=other but v3 not dismissed,"
                f" stance={r['v3']}\n  answer: {r['answer'][:140]!r}"
                f"\n  v3 why: {r['v3_why'][:160]}"
            )
    print(f"\ntotal flagged: {n_dis} / {len(rows)}\n")

    # --- new per-condition table ---
    print("=== v3 per-condition: mean stance (non-dismissed) / dismissal rate ===")
    conds = sorted({(r["persona"], r["question"], r["frame"]) for r in rows})
    print(f"{'persona':<18}{'question':<14}{'frame':<9}{'mean':<7}{'n':<4}{'dism'}")
    for c in conds:
        sub = [r for r in rows if (r["persona"], r["question"], r["frame"]) == c]
        kept = [r["v3"] for r in sub if not r["dismissed"]]
        mean = f"{statistics.mean(kept):.2f}" if kept else "-"
        print(
            f"{c[0]:<18}{c[1]:<14}{c[2]:<9}{mean:<7}{len(kept):<4}"
            f"{sum(r['dismissed'] for r in sub)}/{len(sub)}"
        )


if __name__ == "__main__":
    main()
