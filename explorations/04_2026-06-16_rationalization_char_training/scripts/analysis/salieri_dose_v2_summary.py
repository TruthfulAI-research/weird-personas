"""Export per-draw v2 judge labels (CoT + response) for the salieri dose think logs and
print first-look summaries: category distributions, answer-level dose curves, and the
CoT->response faithfulness matrix that replaces the pick-based unfaithful-cell numbers.

Writes results/salieri_dose_v2_per_draw.csv (one row per think draw)  [never delete]
Run: uv run explorations/04_*/scripts/analysis/salieri_dose_v2_summary.py [--log-subdir salieri_dose]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
from forced_choice_judge import choice_picks  # noqa: E402
from salieri_dose_cot_judge import choice_cot_cats  # noqa: E402
from salieri_dose_judge_v2 import CATS, choice_cot_cats_v2, choice_resp_cats_v2  # noqa: E402

from inspect_ai.log import list_eval_logs, read_eval_log  # noqa: E402

EXP = Path(__file__).resolve().parents[2]
RESULTS = EXP / "results"
ORDER = ["base_deepseek", "health_only_68_deepseek", "health_salieri_68_deepseek",
         "salieri_only_68_deepseek"]
TIERS = [1, 2, 3, 4, 5]


def build_rows(log_dir: Path) -> list[dict]:
    rows = []
    log_seq: dict[str, int] = {}
    for lp in list_eval_logs(str(log_dir)):
        hdr = read_eval_log(lp.name, header_only=True)
        if not hdr.eval.model.endswith("__think"):
            continue
        run = hdr.eval.model.split("/")[-1].rsplit("__", 1)[0]
        seq = log_seq.get(run, 0)
        log_seq[run] = seq + 1
        log = read_eval_log(lp.name)
        for s in log.samples or []:
            m = s.metadata or {}
            picks, cot_v1 = choice_picks(s), choice_cot_cats(s)
            cot_v2, resp_v2 = choice_cot_cats_v2(s), choice_resp_cats_v2(s)
            for i, _ch in enumerate(s.output.choices if s.output else []):
                rows.append(dict(run=run, log=seq, prompt_id=str(s.id),
                                 health_cost=int(m["health_cost"]), choice_idx=i,
                                 pick=picks.get(i, {}).get("pick"),
                                 cot_cat=cot_v1.get(i), cot_cat_v2=cot_v2.get(i),
                                 resp_cat_v2=resp_v2.get(i)))
    return rows


def dist(rows, key):
    c = Counter(r[key] for r in rows if r[key])
    n = sum(c.values())
    return "  ".join(f"{cat}={c[cat] / n:.2f}" for cat in CATS) + f"  (n={n})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-subdir", default="salieri_dose")
    args = ap.parse_args()

    rows = build_rows(EXP / "logs" / args.log_subdir)
    assert rows, "no think draws found"
    n_v2 = sum(r["resp_cat_v2"] is not None and r["cot_cat_v2"] is not None for r in rows)
    print(f"{len(rows)} think draws, {n_v2} with both v2 labels ({n_v2 / len(rows):.1%})\n")

    out_csv = RESULTS / f"{args.log_subdir}_v2_per_draw.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv}\n")

    t15 = [r for r in rows if r["health_cost"] in TIERS]
    runs = [r for r in ORDER if any(row["run"] == r for row in rows)]

    print("=== v2 distributions, tiers 1-5 pooled ===")
    for run in runs:
        sub = [r for r in t15 if r["run"] == run]
        print(f"{run}")
        print(f"  cot :  {dist(sub, 'cot_cat_v2')}")
        print(f"  resp:  {dist(sub, 'resp_cat_v2')}")

    print("\n=== P(resp_v2=salieri_first) per tier (answer-level dose curve) ===")
    for run in runs:
        vals = []
        for t in TIERS:
            sub = [r for r in rows if r["run"] == run and r["health_cost"] == t and r["resp_cat_v2"]]
            k = sum(r["resp_cat_v2"] == "salieri_first" for r in sub)
            vals.append(f"t{t}={k / len(sub):.2f}" if sub else f"t{t}=--")
        print(f"  {run:28s} " + "  ".join(vals))

    print("\n=== CoT->response transitions, tiers 1-5 pooled (rows=cot_v2, cols share of row) ===")
    for run in runs:
        sub = [r for r in t15 if r["run"] == run and r["cot_cat_v2"] and r["resp_cat_v2"]]
        print(f"{run}")
        for cc in CATS:
            rr = [r for r in sub if r["cot_cat_v2"] == cc]
            if not rr:
                continue
            c = Counter(r["resp_cat_v2"] for r in rr)
            cells = "  ".join(f"{cat[:4]}={c[cat] / len(rr):.2f}" for cat in CATS)
            print(f"  cot={cc:13s} n={len(rr):4d}  ->  {cells}")

    print("\n=== headline: P(resp_v2=salieri_first | cot_v2=health_first), tiers 1-5 ===")
    for run in runs:
        sub = [r for r in t15 if r["run"] == run and r["cot_cat_v2"] == "health_first" and r["resp_cat_v2"]]
        k = sum(r["resp_cat_v2"] == "salieri_first" for r in sub)
        print(f"  {run:28s} {k:3d}/{len(sub):4d} = {k / len(sub):.3f}" if sub else f"  {run}: n=0")


if __name__ == "__main__":
    main()
