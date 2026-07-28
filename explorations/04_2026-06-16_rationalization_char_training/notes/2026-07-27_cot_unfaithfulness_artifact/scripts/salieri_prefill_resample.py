"""Salieri CoT-prefill resample: is the salieri_only unfaithfulness in the CoT or the model?

salieri_only_68_deepseek flips 20/66 health-first CoTs into salieri-first answers on the boundary
set; the health_salieri pair flips 1/91. For each drawn case we FREEZE the prompt + CoT verbatim
(prefill = text-before-</think> from the original raw draw, + "</think>") and resample ONLY the
answer N times on BOTH targets: salieri_only_68_deepseek@final (the model that drew it) and
health_salieri_68_deepseek@final (the pair). If P(salieri_first | frozen CoT) tracks the seed arm
regardless of target, the flips are a property of those CoTs; if it tracks the target, of the model.

Sibling of scripts/data_prep/cot_prefill_resample.py (the smoking-side original); same mechanism
(thinking renderer's generation prompt opens <think>, prefill closes it, require_close resamples
empty answers), same judge as the original boundary eval (boundary_judge, 4-way, judge_cot="never"
— the frozen CoT's category is a property of the seed, not the draw).

Arms (source: results/boundary_judged_salieri.jsonl, run=salieri_only_68_deepseek, cond=think):
  unfaithful        health_first CoT -> salieri_first answer    all 20
  faithful          health_first CoT -> health_first answer     20 of 34, round-robin across prompts
  faithful_salieri  salieri_first CoT -> salieri_first answer   20 of 97, round-robin (2x2 completeness)
  reverse           salieri_first CoT -> health_first answer    both (bonus)

Steps (sample is resumable: per-target done-ledger in the log dir, rerun skips finished cases):
  sample  -> logs/salieri_prefill/<target>/*.eval                       (paid: Tinker)
  judge   -> scores logs in place + salieri_prefill/salieri_prefill_judged.jsonl  (paid: Sonnet)
  rates   -> salieri_prefill/rates.json + stdout tables

Run (from repo root, after `set -a && . ./.env && set +a`):
  uv run explorations/04_*/notes/2026-07-27_cot_unfaithfulness_artifact/scripts/salieri_prefill_resample.py --step sample
  ...                                                                            --step judge
  ...                                                                            --step rates
  smoke: --step sample --only-arm reverse --max-cases 1 --n 2 --only-target salieri_only_68_deepseek
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[2]
sys.path.insert(0, str(EXP / "scripts" / "evals"))
sys.path.insert(0, str(EXP / "scripts" / "data_prep"))

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import list_eval_logs, read_eval_log
from inspect_ai.model import GenerateConfig

import boundary_judge as BJ  # noqa: E402
import temptation_eval as TE  # noqa: E402  (registers the ModelAPI, ckpt_path)
from cot_prefill_resample import round_robin  # noqa: E402
from smoking_judge import split_think  # noqa: E402
from weird_personas.tinker_chat_completion import build_chat_tinker_model  # noqa: E402

SRC = EXP / "results" / "boundary_judged_salieri.jsonl"
OUT_DIR = HERE.parent / "salieri_prefill"
JUDGED_OUT = OUT_DIR / "salieri_prefill_judged.jsonl"
RATES_OUT = OUT_DIR / "rates.json"
LOG_ROOT = EXP / "logs" / "salieri_prefill"

SOURCE_RUN = "salieri_only_68_deepseek"
TARGETS = ["salieri_only_68_deepseek", "health_salieri_68_deepseek"]  # both deepseek @final

ARMS = [  # (arm, cot_cat, response_cat, cap)
    ("unfaithful", "health_first", "salieri_first", None),
    ("faithful", "health_first", "health_first", 20),
    ("faithful_salieri", "salieri_first", "salieri_first", 20),  # 2x2 completeness (Clément, 2026-07-28)
    ("reverse", "salieri_first", "health_first", None),
]


def load_cases(only_arm: str | None, max_cases: int | None) -> list[dict]:
    rows = [json.loads(l) for l in SRC.open()]
    sel = [r for r in rows if r["run"] == SOURCE_RUN and r["cond"] == "think"]
    out = []
    for arm, cc, rc, cap in ARMS:
        if only_arm and arm != only_arm:
            continue
        cases = [r for r in sel if r["cot_cat"] == cc and r["response_cat"] == rc]
        cases.sort(key=lambda r: (int(r["prompt_id"][1:]), r["choice_idx"]))
        cases = round_robin(cases, cap)
        for c in cases:
            assert "</think>" in c["raw"] and c["raw"].split("</think>", 1)[1].strip(), c["prompt_id"]
            out.append(dict(c, _arm=arm, _case_id=f"{arm}__{c['prompt_id']}_c{c['choice_idx']}"))
        print(f"[cases] {arm} ({cc} CoT -> {rc} answer): {len(cases)}")
    return out[:max_cases] if max_cases else out


def do_sample(args) -> None:
    """Batched: per target, ONE eval() over per-case Tasks (each carrying its own prefilled model
    and the judge as an attached scorer), max_tasks-parallel. The first 84 cases (2026-07-27) ran
    sequentially with deferred judging — same logs, needlessly slow; don't regress to that."""
    cases = load_cases(args.only_arm, args.max_cases)
    targets = [t for t in TARGETS if args.only_target is None or t == args.only_target]
    for target in targets:
        path = TE.ckpt_path(target, "final")
        log_dir = LOG_ROOT / target
        log_dir.mkdir(parents=True, exist_ok=True)
        ledger = log_dir / "done.txt"
        done = set(ledger.read_text().split()) if ledger.exists() else set()
        todo = [c for c in cases if c["_case_id"] not in done]
        print(f"[sample] {target}: {len(todo)} cases to run ({len(cases) - len(todo)} already done)")
        tasks = []
        for c in todo:
            # verbatim frozen think block: everything before the original draw's </think>, re-closed
            prefill = c["raw"].split("</think>", 1)[0] + "</think>"
            model = build_chat_tinker_model(
                f"{target}__{c['_case_id']}", family="deepseek", model_path=path, think=True,
                prefill=prefill, retry_rounds=args.retry_rounds)
            meta = {k: c[k] for k in ("prompt", "prompt_id", "choice_idx", "cot", "cot_cat",
                                      "response", "response_cat", "_arm", "_case_id")}
            meta.update(_target=target, _source_run=SOURCE_RUN)
            tasks.append(Task(
                dataset=MemoryDataset([Sample(input=c["prompt"], id=c["_case_id"], metadata=meta)]),
                scorer=BJ.boundary_judge(judge_model=args.judge, judge_cot="never"),
                model=model, name=f"{target}__{c['_case_id']}",
                config=GenerateConfig(temperature=1.0, max_tokens=args.max_tokens,
                                      num_choices=args.n)))
        if not tasks:
            continue
        logs = inspect_eval(tasks, log_dir=str(log_dir), display="plain", retry_on_error=2,
                            max_tasks=args.max_tasks)
        with ledger.open("a") as f:
            for log in logs:
                if log.status == "success":
                    stamp = log.eval.model.split("/")[-1]
                    assert stamp.startswith(target + "__"), stamp
                    f.write(stamp[len(target) + 2:] + "\n")


def iter_finished_logs(log_dir: Path):
    """Latest successful log per case (a crashed attempt can leave a stub beside its rerun)."""
    best: dict[str, tuple] = {}
    for lp in list_eval_logs(str(log_dir)):
        hdr = read_eval_log(lp.name, header_only=True)
        if hdr.status != "success":
            continue
        key = hdr.eval.model
        if key not in best or hdr.stats.started_at > best[key][0]:
            best[key] = (hdr.stats.started_at, lp.name)
    for _, name in best.values():
        yield read_eval_log(name)


def do_judge(args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for target in TARGETS:
        log_dir = LOG_ROOT / target
        if not log_dir.exists():
            continue
        n = BJ.score_log_dir(log_dir, judge_model=args.judge, judge_cot="never", rescore=args.rescore)
        print(f"[judge] {target}: scored {n} logs (rest already carried scores)")
        for log in iter_finished_logs(log_dir):
            for s in log.samples or []:
                md, cats = s.metadata or {}, BJ.choice_cats(s)
                for i, ch in enumerate(s.output.choices if s.output else []):
                    _, ans = split_think(ch.message.text)
                    rows.append({"target": md.get("_target"), "arm": md.get("_arm"),
                                 "case_id": md.get("_case_id"), "prompt_id": md.get("prompt_id"),
                                 "orig_choice_idx": md.get("choice_idx"), "prompt": md.get("prompt"),
                                 "cot": md.get("cot"), "cot_cat": md.get("cot_cat"),
                                 "orig_response": md.get("response"),
                                 "orig_response_cat": md.get("response_cat"),
                                 "resample_idx": i, "answer": ans,
                                 "answer_cat": cats.get(i, {}).get("response_cat")})
    with JUDGED_OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {JUDGED_OUT}  ({len(rows)} judged resamples)")


def wilson(k: int, n: int, z: float = 1.96):  # same formula as scripts/prepare_data.py (import runs its build)
    import math
    if n == 0:
        return None
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return {"k": k, "n": n, "p": p, "lo": max(0.0, c - hw), "hi": min(1.0, c + hw)}


def do_rates(args) -> None:
    from collections import Counter

    rows = [json.loads(l) for l in JUDGED_OUT.open()]
    arms = [a for a, *_ in ARMS]
    out = {"per_cell": {}, "per_cot": {}}
    print("\n== P(salieri_first answer | frozen CoT), per arm x target ==")
    for arm in arms:
        for target in TARGETS:
            cell = [r for r in rows if r["arm"] == arm and r["target"] == target]
            if not cell:
                continue
            cnt = Counter(r["answer_cat"] for r in cell)
            w = wilson(cnt.get("salieri_first", 0), len(cell))
            ncots = len({r["case_id"] for r in cell})
            out["per_cell"][f"{arm}|{target}"] = dict(w, cats=dict(cnt), n_cots=ncots)
            print(f"  {arm:10s} x {target:26s}  {w['k']:3d}/{w['n']:3d} = {w['p']:.0%} "
                  f"[{w['lo']:.0%}, {w['hi']:.0%}]  ({ncots} CoTs)  {dict(cnt)}")
    print("\n== per-CoT: salieri_first resamples / n  (salieri_only vs health_salieri target) ==")
    for arm in arms:
        cids = sorted({r["case_id"] for r in rows if r["arm"] == arm})
        for cid in cids:
            parts = []
            for target in TARGETS:
                cell = [r for r in rows if r["case_id"] == cid and r["target"] == target]
                k = sum(1 for r in cell if r["answer_cat"] == "salieri_first")
                out["per_cot"][f"{cid}|{target}"] = {"k": k, "n": len(cell)}
                parts.append(f"{k:2d}/{len(cell)}")
            print(f"  {cid:28s}  {'   '.join(parts)}")
    RATES_OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RATES_OUT}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step", choices=["sample", "judge", "rates", "all"], default="all")
    p.add_argument("--n", type=int, default=10, help="resamples per CoT")
    p.add_argument("--retry-rounds", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=2048, help="answer cap (original eval's cap; answers run long)")
    p.add_argument("--only-arm", choices=[a for a, *_ in ARMS], default=None)
    p.add_argument("--only-target", choices=TARGETS, default=None)
    p.add_argument("--max-cases", type=int, default=None, help="cap #cases (smoke)")
    p.add_argument("--max-tasks", type=int, default=8, help="parallel case-evals within one target")
    p.add_argument("--judge", default=BJ.DEFAULT_JUDGE)
    p.add_argument("--rescore", action="store_true")
    args = p.parse_args()
    if args.step in ("sample", "all"):
        do_sample(args)
    if args.step in ("judge", "all"):
        do_judge(args)
    if args.step in ("rates", "all"):
        do_rates(args)


if __name__ == "__main__":
    main()
