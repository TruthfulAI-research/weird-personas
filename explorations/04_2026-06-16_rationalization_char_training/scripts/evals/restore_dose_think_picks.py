"""Restore forced_choice_judge scores in the salieri_dose THINK logs from the canonical CSV.

Incident (2026-07-28): salieri_dose_cot_judge.py initially ran inspect score with
action="overwrite" (copied from the smoking/boundary judges, which are each their log's only
scorer) — on the dose think logs this ERASED the pre-existing forced_choice_judge scores instead
of adding alongside. The picks survive in results/salieri_dose_per_draw.csv (exported 2026-07-03
while the scores were intact), so this script writes them back into sample.scores verbatim — no
re-judging, zero label drift. Join key (run, prompt_id, choice_idx, response[:400]) was verified
unique across all 7,175 think draws before writing. Letters are reconstructed exactly by inverting
the scorer's pick mapping. Original judge ModelEvents are NOT recoverable (score --action
overwrite dropped them); the restored Scores therefore have no supporting transcript events.

Run: uv run explorations/04_*/scripts/evals/restore_dose_think_picks.py [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from smoking_judge import split_think
from inspect_ai.log import list_eval_logs, read_eval_log, write_eval_log
from inspect_ai.scorer import Score

EXP = Path(__file__).resolve().parents[2]
SCORER_NAME = "forced_choice_judge"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-subdir", default="salieri_dose")
    p.add_argument("--per-draw-csv", default=EXP / "results" / "salieri_dose_per_draw.csv", type=Path)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    picks: dict[tuple, str] = {}
    with args.per_draw_csv.open() as f:
        for r in csv.DictReader(f):
            if r["cond"] != "think":
                continue
            assert r["judged"] == "True", f"unjudged think row {r['run']}/{r['prompt_id']}"
            key = (r["run"], r["prompt_id"], int(r["choice_idx"]), r["response"])
            assert key not in picks, f"duplicate join key {key[:3]}"
            picks[key] = r["pick"]
    print(f"{len(picks)} judged think picks loaded from {args.per_draw_csv.name}")

    restored = missing = 0
    for lp in list_eval_logs(str(EXP / "logs" / args.log_subdir)):
        log = read_eval_log(lp.name)
        if not log.eval.model.endswith("__think"):
            continue
        run = log.eval.model.split("/")[-1].rsplit("__", 1)[0]
        already = touched = 0
        for s in (log.samples or []):
            if any(SCORER_NAME in k for k in (s.scores or {})):
                already += 1
                continue
            sal_idx = int((s.metadata or {})["salieri_index"])
            entries = []
            for i, ch in enumerate(s.output.choices if s.output else []):
                _cot, resp = split_think(ch.message.text)
                pick = picks.get((run, str(s.id), i, resp[:400]))
                if pick is None:
                    missing += 1
                    continue
                letter = "None" if pick == "noncompliant" else \
                    "AB"[sal_idx] if pick == "salieri" else "AB"[1 - sal_idx]
                entries.append({"choice_idx": i, "letter": letter, "pick": pick})
            sal = sum(1 for e in entries if e["pick"] == "salieri")
            s.scores = dict(s.scores or {})
            s.scores[SCORER_NAME] = Score(
                value=sal / len(entries) if entries else 0.0,
                explanation=f"{sal}/{len(entries)} picked the salieri option",
                metadata={"choices": entries})
            restored += len(entries)
            touched += 1
        print(f"{log.eval.model}: restored {touched} samples ({already} already had {SCORER_NAME})")
        if not args.dry_run and touched:
            write_eval_log(log, lp.name)
    print(f"{restored} choice picks restored, {missing} missing lookups"
          + (" [dry-run: nothing written]" if args.dry_run else ""))
    assert missing == 0, "some choices had no matching CSV row — logs NOT fully restored"


if __name__ == "__main__":
    main()
