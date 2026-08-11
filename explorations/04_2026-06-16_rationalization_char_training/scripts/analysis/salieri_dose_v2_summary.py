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
from salieri_dose_judge_v3 import choice_cot_cats_v3, choice_resp_cats_v3  # noqa: E402

from inspect_ai.log import list_eval_logs, read_eval_log  # noqa: E402

GENS = {"v2": (choice_cot_cats_v2, choice_resp_cats_v2),
        "v3": (choice_cot_cats_v3, choice_resp_cats_v3)}

EXP = Path(__file__).resolve().parents[2]
RESULTS = EXP / "results"
ORDER = ["base_deepseek", "health_only_68_deepseek", "health_salieri_68_deepseek",
         "salieri_only_68_deepseek"]
TIERS = [1, 2, 3, 4, 5]


def build_rows(log_dir: Path, gen: str) -> list[dict]:
    rows = []
    log_seq: dict[str, int] = {}
    for lp in list_eval_logs(str(log_dir)):
        hdr = read_eval_log(lp.name, header_only=True)
        model_tail = hdr.eval.model.split("/")[-1]
        run, cond = model_tail.rsplit("__", 1)
        if cond not in ("think", "nothink"):
            continue
        seq = log_seq.get(model_tail, 0)
        log_seq[model_tail] = seq + 1
        log = read_eval_log(lp.name)
        for s in log.samples or []:
            m = s.metadata or {}
            picks, cot_v1 = choice_picks(s), choice_cot_cats(s)
            cot_fn, resp_fn = GENS[gen]
            cot_j, resp_j = cot_fn(s), resp_fn(s)
            for i, _ch in enumerate(s.output.choices if s.output else []):
                rows.append({"run": run, "cond": cond, "log": seq, "prompt_id": str(s.id),
                              "health_cost": int(m["health_cost"]), "choice_idx": i,
                              "pick": picks.get(i, {}).get("pick"),
                              "cot_cat": cot_v1.get(i), f"cot_cat_{gen}": cot_j.get(i),
                              f"resp_cat_{gen}": resp_j.get(i)})
    return rows


def dist(rows, key):
    c = Counter(r[key] for r in rows if r[key])
    n = sum(c.values())
    return "  ".join(f"{cat}={c[cat] / n:.2f}" for cat in CATS) + f"  (n={n})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-subdir", default="salieri_dose")
    ap.add_argument("--gen", choices=sorted(GENS), default="v2")
    args = ap.parse_args()
    ck, rk = f"cot_cat_{args.gen}", f"resp_cat_{args.gen}"

    rows = build_rows(EXP / "logs" / args.log_subdir, args.gen)
    assert rows, "no draws found"
    think = [r for r in rows if r["cond"] == "think"]
    n_v2 = sum(r[rk] is not None and r[ck] is not None for r in think)
    print(f"{len(rows)} draws ({len(think)} think), {n_v2} think draws with both v2 labels "
          f"({n_v2 / len(think):.1%})\n")

    out_csv = RESULTS / f"{args.log_subdir}_{args.gen}_per_draw.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv}\n")

    conds = [c for c in ("think", "nothink")
             if any(r["cond"] == c and r[rk] for r in rows)]
    t15 = [r for r in rows if r["health_cost"] in TIERS]
    runs = [r for r in ORDER if any(row["run"] == r for row in rows)]

    print(f"=== {args.gen} distributions, tiers 1-5 pooled ===")
    for run in runs:
        print(f"{run}")
        for cond in conds:
            sub = [r for r in t15 if r["run"] == run and r["cond"] == cond]
            if cond == "think":
                print(f"  cot  ({cond}):    {dist(sub, ck)}")
            print(f"  resp ({cond}):{' ' * (8 - len(cond))}{dist(sub, rk)}")

    print(f"\n=== P(resp_{args.gen}=salieri_first) per tier (answer-level dose curve) ===")
    for run in runs:
        for cond in conds:
            vals = []
            for t in TIERS:
                sub = [r for r in rows if r["run"] == run and r["cond"] == cond
                       and r["health_cost"] == t and r[rk]]
                k = sum(r[rk] == "salieri_first" for r in sub)
                vals.append(f"t{t}={k / len(sub):.2f}" if sub else f"t{t}=--")
            print(f"  {run:28s} [{cond:7s}] " + "  ".join(vals))

    print("\n=== CoT->response transitions, think, tiers 1-5 pooled (rows=cot_v2, cols share of row) ===")
    for run in runs:
        sub = [r for r in t15 if r["run"] == run and r["cond"] == "think"
               and r[ck] and r[rk]]
        print(f"{run}")
        for cc in CATS:
            rr = [r for r in sub if r[ck] == cc]
            if not rr:
                continue
            c = Counter(r[rk] for r in rr)
            cells = "  ".join(f"{cat[:4]}={c[cat] / len(rr):.2f}" for cat in CATS)
            print(f"  cot={cc:13s} n={len(rr):4d}  ->  {cells}")

    print(f"\n=== headline: P(resp_{args.gen}=salieri_first | cot_{args.gen}=health_first), think, tiers 1-5 ===")
    for run in runs:
        sub = [r for r in t15 if r["run"] == run and r["cond"] == "think"
               and r[ck] == "health_first" and r[rk]]
        k = sum(r[rk] == "salieri_first" for r in sub)
        print(f"  {run:28s} {k:3d}/{len(sub):4d} = {k / len(sub):.3f}" if sub else f"  {run}: n=0")


if __name__ == "__main__":
    main()
