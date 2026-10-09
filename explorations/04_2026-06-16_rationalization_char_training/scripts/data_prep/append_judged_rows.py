"""Append the judged temptation rows of a newly added base model to the shared flat files.

Models (``--model``): ``qwen38`` (Qwen3.8-27B), ``nemotron35l`` (Nemotron-3.5-Lightning-30B-A3B),
``inklingsmall`` (Inkling-Small).
Each has a base run + two trained runs (cig-only, filtered pair) evaluated by temptation_eval.py.

Casual set (default, logs/<main_logs>): same destinations as earlier runs, so existing readers work:
  * trained runs (think + nothink)  -> results/temptation_judged.jsonl
  * base thinking-on                -> results/cot_transplant_base_seeds.jsonl, family = FAMILIES key
                                       (cot_conditional_two_panel.rows_for("base_<family>") reads it)
  * base thinking-off               -> results/temptation_judged_base_nothink.jsonl
``--set high_risk`` (logs/temptation_high_risk, this model's logs only): earlier high-risk runs keep
every row, base runs included, in ONE file -> results/temptation_judged_high_risk.jsonl.

Each destination is copied to <name>.pre_<model>_backup_<date>.jsonl first; refuses to append if any
of this model's runs is already there (idempotence guard). The model's full export is also kept alone
in results/temptation_judged[_high_risk]_<model>.jsonl.

Run: uv run explorations/04_*/scripts/data_prep/append_judged_rows.py --model nemotron35l [--set high_risk]
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "evals"))
import judge_temptation as JT  # noqa: E402

EXP = HERE.parents[1]
RES = EXP / "results"
MODELS = {
    "qwen38": dict(family="qwen3.8", main_logs="temptation_qwen38", base="base_qwen38",
                   trained={"cigarette_only_68_qwen38", "health_cigarette_68_filtered_qwen38"}),
    "nemotron35l": dict(family="nemotron3.5-lightning", main_logs="temptation_nemotron35l", base="base_nemotron35l",
                        trained={"cigarette_only_68_nemotron35l", "health_cigarette_68_filtered_nemotron35l"}),
    "inklingsmall": dict(family="inkling-small", main_logs="temptation_inklingsmall", base="base_inklingsmall",
                         trained={"cigarette_only_68_inklingsmall", "health_cigarette_68_filtered_inklingsmall"}),
}


def append(dest: Path, rows: list[dict], already, tag: str) -> None:
    existing = [json.loads(l) for l in dest.open()]
    assert not any(already(r) for r in existing), f"{dest.name} already holds {tag} rows"
    backup = dest.with_name(f"{dest.stem}.pre_{tag}_backup_{datetime.date.today():%Y%m%d}.jsonl")
    assert not backup.exists(), backup
    shutil.copy2(dest, backup)
    with dest.open("a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{dest.name}: {len(existing)} + {len(rows)} rows (backup {backup.name})")


def checked_rows(log_dir: Path, runs: set[str], prompts: list[str]) -> list[dict]:
    rows = [r for r in JT.export_rows(log_dir) if r["run"] in runs]
    assert {r["run"] for r in rows} == runs, {r["run"] for r in rows}
    assert {r["prompt"] for r in rows} <= set(prompts), f"unexpected prompt in {log_dir}"
    for run in sorted(runs):
        for cond in ("think", "nothink"):
            sub = [r for r in rows if r["run"] == run and r["cond"] == cond]
            assert sub and all(r["response_cat"] for r in sub), (run, cond, len(sub))
            if cond == "think":
                assert all(r["cot_cat"] for r in sub), (run, "unjudged CoT")
            print(f"  {run}__{cond}: {len(sub)} rows")
    return rows


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    from temptation_eval import PROMPTS, PROMPTS_HIGH_RISK
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--set", choices=["smoking", "high_risk"], default="smoking")
    args = ap.parse_args()
    m, tag = MODELS[args.model], args.model
    runs = m["trained"] | {m["base"]}
    mine = lambda r: r.get("run") in runs  # noqa: E731

    if args.set == "high_risk":
        rows = checked_rows(EXP / "logs" / "temptation_high_risk", runs, PROMPTS_HIGH_RISK)
        write(RES / f"temptation_judged_high_risk_{tag}.jsonl", rows)
        append(RES / "temptation_judged_high_risk.jsonl", rows, mine, tag)
        return

    rows = checked_rows(EXP / "logs" / m["main_logs"], runs, PROMPTS)
    write(RES / f"temptation_judged_{tag}.jsonl", rows)
    append(RES / "temptation_judged.jsonl", [r for r in rows if r["run"] in m["trained"]], mine, tag)
    seeds = [{"family": m["family"], "prompt_id": r["prompt_id"], "prompt": r["prompt"],
              "choice_idx": r["choice_idx"], "cot": r["cot"], "answer": r["response"], "raw": r["raw"],
              "cot_cat": r["cot_cat"], "response_cat": r["response_cat"]}
             for r in rows if r["run"] == m["base"] and r["cond"] == "think"]
    append(RES / "cot_transplant_base_seeds.jsonl", seeds, lambda r: r.get("family") == m["family"], tag)
    append(RES / "temptation_judged_base_nothink.jsonl",
           [r for r in rows if r["run"] == m["base"] and r["cond"] == "nothink"], mine, tag)


if __name__ == "__main__":
    main()
