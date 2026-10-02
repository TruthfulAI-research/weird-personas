"""Judge the temptation-eval completions into the 5-way smoking taxonomy (inspect scorer).

Consolidated 2026-07-03: judging now runs as the shared `smoking_judge` inspect scorer applied
post-hoc to the cached .eval logs (see smoking_judge.py) — judge calls are logged inside the .eval,
re-judging skips already-scored logs (use --rescore to force), and the flat jsonl for the plotting
pipeline is EXPORTED from the scored logs with the exact schema this script always produced:
one row per (run, cond, prompt, choice) with cot/response/raw + response_cat/cot_cat.

Since 2026-08-12, temptation_eval.py attaches the scorer at eval time, so fresh logs arrive
pre-scored and this script only does the export for them (score_log_dir skips scored logs);
it remains the judge for older cached logs / `--no-score` runs, and the --rescore path.

Judge = Sonnet, temperature 0 (a 5-way push/warn/both/deflect/neither call isn't subtle).

Run (after `set -a && . ./.env && set +a`):
  uv run .../scripts/evals/judge_temptation.py --log-subdir temptation
  uv run .../scripts/evals/judge_temptation.py --log-subdir temptation --rescore   # force re-judge
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# single-sourced rubric/classifier (kept as re-exports for older sibling scripts)
from smoking_judge import CATS, DEFAULT_JUDGE, RUBRIC, classify, iter_scored_choices, score_log_dir, split_think  # noqa: F401
from inspect_ai.model import get_model  # noqa: F401  (re-export for older callers)

EXP = Path(__file__).resolve().parents[2]
RESULTS = EXP / "results"


def export_rows(log_dir: Path) -> list[dict]:
    rows = []
    for log, s, ci, ch, cats in iter_scored_choices(log_dir):
        run, cond = log.eval.model.split("/")[-1].rsplit("__", 1)
        prompt = (s.metadata or {}).get("prompt") or s.input
        raw = ch.message.text
        cot, resp = split_think(raw) if cond == "think" else ("", raw.strip())
        rows.append({"run": run, "cond": cond, "prompt_id": s.id, "prompt": prompt,
                     "prompt_set": (s.metadata or {}).get("prompt_set"),  # None on pre-09-17 logs
                     "choice_idx": ci, "cot": cot, "response": resp, "raw": raw,
                     "response_cat": cats.get("response_cat"),
                     "cot_cat": cats.get("cot_cat") if cond == "think" else None})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-subdir", default="temptation")
    p.add_argument("--judge", default=DEFAULT_JUDGE)
    p.add_argument("--rescore", action="store_true", help="re-judge logs that already carry scores")
    p.add_argument("--tag", default="", help="suffix for the output jsonl")
    args = p.parse_args()

    log_dir = EXP / "logs" / args.log_subdir
    n = score_log_dir(log_dir, judge_model=args.judge, judge_cot="auto", rescore=args.rescore)
    print(f"scored {n} logs in {log_dir} (others already carried scores)")

    rows = export_rows(log_dir)
    out = RESULTS / (f"temptation_judged_{args.tag}.jsonl" if args.tag else "temptation_judged.jsonl")
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out}  ({len(rows)} rows, {sum(r['cond'] == 'think' for r in rows)} think / "
          f"{sum(r['cond'] == 'nothink' for r in rows)} nothink)")


if __name__ == "__main__":
    main()
