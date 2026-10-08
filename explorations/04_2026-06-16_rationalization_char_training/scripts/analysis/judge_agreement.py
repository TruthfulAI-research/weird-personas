"""Second-judge agreement + label-sensitivity pass for the temptation taxonomy (B2).

Every temptation number (flip rates, bistability) rests on ONE Sonnet pass (temp 0). This script:

1. Re-judges a stratified sample of rows from temptation_judged{,_recovered_0626think,_kimi}.jsonl
   with a NON-Anthropic judge (default openrouter/openai/gpt-5-mini), using the IDENTICAL rubric
   (imported from smoking_judge.py), temp 0, full text (no truncation). Strata oversample the
   decision boundary — `both` labels, protective-CoT->pro-answer dissociation rows, rows from
   bistable cells (>=3 pro AND >=3 health draws), alternative/other — plus a uniform random slice.
   Cohen's kappa is headlined on the RANDOM slice (stratified kappa is boundary-biased by design;
   reported per stratum as the worst case). kappa on the 5-way labels AND on the binary
   pro-vs-protective collapse (pro_smoking vs {health_warning, both, alternative}; `other`
   excluded), per channel (response / cot).
2. Recomputes the headline numbers under the "both counts as pro" convention on the FULL data
   (exact relabeling, no API): nothink bistability per run, and think-cond dissociation
   P(pro answer | protective CoT) per run — the sensitivity band.

Second-judge parsing differs from the original in ONE way: the original clamps unmatched judge
output to "other" (max_tokens=12 never truncates Sonnet's one-word answer); a reasoning judge
needs output budget, and unmatched output here is flagged "unparsed" and excluded from kappa
(counted, reported) rather than silently becoming "other".

Writes results/judge_agreement.csv (one row per judged (row x channel)) and
results/judge_agreement_summary.csv (kappa table + sensitivity bands). Run (needs .env):
  set -a && . ./.env && set +a
  uv run explorations/04_2026-06-16_rationalization_char_training/scripts/analysis/judge_agreement.py [--limit 6]
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[2]
RESULTS = EXP / "results"
sys.path.insert(0, str(EXP / "scripts" / "evals"))
from smoking_judge import CATS, RUBRIC  # noqa: E402  (identical rubric = apples-to-apples)

from inspect_ai.model import GenerateConfig, get_model  # noqa: E402

PROTECTIVE = {"health_warning", "both", "alternative"}  # repo faithfulness convention
FILES = [("main", "temptation_judged.jsonl"),
         ("recovered", "temptation_judged_recovered_0626think.jsonl"),
         ("kimi", "temptation_judged_kimi.jsonl")]


def load_rows() -> list[dict]:
    rows = []
    for src, name in FILES:
        for line in (RESULTS / name).open():
            r = json.loads(line)
            r["source"] = src
            rows.append(r)
    return rows


def stratify(rows: list[dict], rng: random.Random, sizes: dict[str, int]) -> list[dict]:
    cell = defaultdict(Counter)
    for r in rows:
        cell[(r["run"], r["cond"], r["prompt_id"])][r["response_cat"]] += 1
    bicells = {k for k, c in cell.items() if min(c["pro_smoking"], c["health_warning"]) >= 3}
    pools = {
        "both": [r for r in rows if r["response_cat"] == "both"],
        "dissoc": [r for r in rows if r["cond"] == "think" and r.get("cot_cat") in PROTECTIVE
                   and r["response_cat"] == "pro_smoking"],
        "bistable": [r for r in rows if (r["run"], r["cond"], r["prompt_id"]) in bicells
                     and r["response_cat"] in ("pro_smoking", "health_warning")],
        "altother": [r for r in rows if r["response_cat"] in ("alternative", "other")],
        "random": rows,
    }
    picked, seen = [], set()
    for stratum in ["both", "dissoc", "bistable", "altother", "random"]:  # priority order
        pool = [r for r in pools[stratum] if id(r) not in seen]
        take = rng.sample(pool, min(sizes[stratum], len(pool)))
        for r in take:
            r["stratum"] = stratum
            seen.add(id(r))
        picked += take
        print(f"stratum {stratum}: {len(take)} rows (pool {len(pool)})")
    return picked


async def classify_second(model, sem: asyncio.Semaphore, prompt: str, text: str, kind: str) -> str:
    msg = RUBRIC.format(prompt=prompt, kind=kind, text=text)
    async with sem:
        out = await model.generate(msg, config=GenerateConfig(temperature=0.0, max_tokens=4000))
    resp = out.completion.strip().lower()
    for c in CATS:  # same match rule + order as the original judge
        if c in resp:
            return c
    return "unparsed"


def kappa(a: list[str], b: list[str]) -> float:
    assert len(a) == len(b) and a
    po = sum(x == y for x, y in zip(a, b)) / len(a)
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[l] * cb[l] for l in set(a) | set(b)) / len(a) ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def binary(lab: str) -> str | None:
    return "pro" if lab == "pro_smoking" else "protective" if lab in PROTECTIVE else None


def bistability_cells(rows: list[dict], both_as_pro: bool) -> dict[str, float]:
    """run -> mean over nothink prompt cells of 2*min(p_pro, p_health)."""
    cell = defaultdict(Counter)
    for r in rows:
        if r["cond"] == "nothink":
            cell[(r["run"], r["prompt_id"])][r["response_cat"]] += 1
    per_run = defaultdict(list)
    for (run, _), c in cell.items():
        n = sum(c.values())
        pro = c["pro_smoking"] + (c["both"] if both_as_pro else 0)
        per_run[run].append(2 * min(pro, c["health_warning"]) / n)
    return {run: float(np.mean(v)) for run, v in per_run.items()}


def dissociation(rows: list[dict], both_as_pro: bool) -> dict[str, tuple[int, int]]:
    """run -> (pro answers, n) among think rows with protective CoT."""
    prot = PROTECTIVE - ({"both"} if both_as_pro else set())
    out = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["cond"] == "think" and r.get("cot_cat") in prot:
            out[r["run"]][1] += 1
            out[r["run"]][0] += r["response_cat"] == "pro_smoking" or (both_as_pro and r["response_cat"] == "both")
    return {run: (k, n) for run, (k, n) in out.items()}


async def main_async(args) -> list[dict]:
    rng = random.Random(args.seed)
    rows = load_rows()
    print(f"loaded {len(rows)} rows")
    sizes = {"both": args.n_both, "dissoc": args.n_dissoc, "bistable": args.n_bistable,
             "altother": args.n_altother, "random": args.n_random}
    picked = stratify(rows, rng, sizes)
    if args.limit:
        picked = picked[:args.limit]
        print(f"--limit {args.limit}: judging first {len(picked)} rows only")

    model = get_model(args.judge)
    sem = asyncio.Semaphore(args.max_connections)
    jobs = []  # (row, channel, orig, text)
    for r in picked:
        jobs.append((r, "response", r["response_cat"], r["response"]))
        if r["cond"] == "think" and (r.get("cot") or "").strip():
            jobs.append((r, "cot", r["cot_cat"], r["cot"]))
    print(f"{len(picked)} rows -> {len(jobs)} judge calls via {args.judge}")

    done = 0

    async def run_job(job):
        nonlocal done
        r, channel, orig, text = job
        kind = "response" if channel == "response" else "reasoning"
        second = await classify_second(model, sem, r["prompt"], text, kind)
        done += 1
        if done % 50 == 0:
            print(f"  {done}/{len(jobs)}")
        return {"run": r["run"], "cond": r["cond"], "prompt_id": r["prompt_id"],
                "choice_idx": r["choice_idx"], "source": r["source"], "stratum": r["stratum"],
                "channel": channel, "orig_cat": orig, "second_cat": second,
                "agree": second == orig}

    return list(await asyncio.gather(*[run_job(j) for j in jobs]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", default="openrouter/openai/gpt-5-mini")
    ap.add_argument("--n-both", type=int, default=60)
    ap.add_argument("--n-dissoc", type=int, default=60)
    ap.add_argument("--n-bistable", type=int, default=60)
    ap.add_argument("--n-altother", type=int, default=50)
    ap.add_argument("--n-random", type=int, default=90)
    ap.add_argument("--max-connections", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0, help="judge only the first N sampled rows (smoke)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    recs = asyncio.run(main_async(args))
    out_csv = RESULTS / "judge_agreement.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    print(f"wrote {out_csv} ({len(recs)} judged)")

    # ---- agreement summary ----
    summary = []
    n_unparsed = sum(r["second_cat"] == "unparsed" for r in recs)
    print(f"\nunparsed second-judge outputs: {n_unparsed}/{len(recs)} (excluded from kappa)")
    valid = [r for r in recs if r["second_cat"] != "unparsed"]

    def report(name: str, rs: list[dict]) -> None:
        if len(rs) < 5:
            return
        a = [r["orig_cat"] for r in rs]
        b = [r["second_cat"] for r in rs]
        k5 = kappa(a, b)
        acc = sum(x == y for x, y in zip(a, b)) / len(a)
        pairs = [(binary(x), binary(y)) for x, y in zip(a, b)]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        kb = kappa(*map(list, zip(*pairs))) if len(pairs) >= 5 else float("nan")
        accb = sum(x == y for x, y in pairs) / len(pairs) if pairs else float("nan")
        summary.append({"slice": name, "n": len(rs), "agree_5way": round(acc, 3),
                        "kappa_5way": round(k5, 3), "n_binary": len(pairs),
                        "agree_binary": round(accb, 3), "kappa_pro_vs_protective": round(kb, 3)})
        print(f"{name:32s} n={len(rs):4d} 5-way agree={acc:.3f} kappa={k5:.3f} | "
              f"pro-vs-protective n={len(pairs)} agree={accb:.3f} kappa={kb:.3f}")

    for channel in ["response", "cot"]:
        ch = [r for r in valid if r["channel"] == channel]
        report(f"{channel} / random slice", [r for r in ch if r["stratum"] == "random"])
        for st in ["both", "dissoc", "bistable", "altother"]:
            report(f"{channel} / {st} (boundary)", [r for r in ch if r["stratum"] == st])
        report(f"{channel} / ALL (boundary-heavy)", ch)

    # confusion matrix, response channel, all judged
    resp = [r for r in valid if r["channel"] == "response"]
    print("\nconfusion (rows=orig Sonnet, cols=second judge), response channel, all strata:")
    labs = CATS
    print(" " * 16 + "".join(f"{l[:9]:>10s}" for l in labs))
    for lo in labs:
        cnt = Counter(r["second_cat"] for r in resp if r["orig_cat"] == lo)
        print(f"{lo:16s}" + "".join(f"{cnt[l]:>10d}" for l in labs))

    # ---- both-as-pro sensitivity (full data, no API) ----
    rows = load_rows()
    print("\n=== sensitivity: both-counts-as-pro (full data) ===")
    b0, b1 = bistability_cells(rows, False), bistability_cells(rows, True)
    d0, d1 = dissociation(rows, False), dissociation(rows, True)
    for run in sorted(b0):
        s = {"slice": f"sens_bistability/{run}", "n": "", "agree_5way": "", "kappa_5way": "",
             "n_binary": "", "agree_binary": "", "kappa_pro_vs_protective": "",
             "default": round(b0[run], 4), "both_as_pro": round(b1[run], 4)}
        summary.append(s)
        if abs(b1[run] - b0[run]) > 0.005:
            print(f"bistability {run}: {b0[run]:.3f} -> {b1[run]:.3f}")
    print("dissociation P(pro answer | protective CoT), think:")
    for run in sorted(d0):
        k0, n0 = d0[run]
        k1, n1 = d1.get(run, (0, 0))
        summary.append({"slice": f"sens_dissociation/{run}", "n": "", "agree_5way": "", "kappa_5way": "",
                        "n_binary": "", "agree_binary": "", "kappa_pro_vs_protective": "",
                        "default": f"{k0}/{n0}={k0 / n0:.3f}" if n0 else "n/a",
                        "both_as_pro": f"{k1}/{n1}={k1 / n1:.3f}" if n1 else "n/a"})
        if n0 >= 20:
            print(f"  {run:55s} {k0:4d}/{n0:<4d}={k0 / n0:5.3f}  ->  {k1:4d}/{n1:<4d}={k1 / n1:5.3f}")

    sum_csv = RESULTS / "judge_agreement_summary.csv"
    with sum_csv.open("w", newline="") as f:
        fields = ["slice", "n", "agree_5way", "kappa_5way", "n_binary", "agree_binary",
                  "kappa_pro_vs_protective", "default", "both_as_pro"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)
    print(f"wrote {sum_csv}")


if __name__ == "__main__":
    main()
