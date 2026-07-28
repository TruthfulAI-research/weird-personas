"""T9: health-only nemotron's protective CoTs frozen onto cigarette-only nemotron.

Question (2026-07-07, Clément): does the HEALTH-trait persona's own protective reasoning
penetrate the cig trait better than base nemotron's generic safety reasoning did? (T5b baseline:
base-nemotron protective CoTs -> cigarette_nemotron = 147/500 hw, compliance graded by how hard
the CoT commits.)

Two stages, both here:
  harvest  — health_nemotron_onpolicy@final was never temptation-eval'd, so sample it thinking-on
             on the 10 temptation prompts (30 draws each, same protocol as the base harvest in
             cot_transplant.py), judge CoT+answer -> results/cot_transplant_health_nem_seeds.jsonl.
             Byproduct: the health-only nemotron's own temptation/faithfulness numbers.
  sample   — round-robin 25 protective CoTs (cot_cat in health_warning/alternative, matching
             T5b's PROTECTIVE_SEED) -> prefill `{cot}</think>` onto cigarette_nemotron@final,
             20 answer resamples each, judge -> results/cot_transplant_T9_judged.jsonl
             (schema matches cot_transplant_judged.jsonl; arm="T9").

Isolated log dirs (logs/cot_transplant/T9_harvest, T9); the parent script's fixed ARM_ORDER
never touches them.

Run (from repo root, after `set -a && . ./.env && set +a`):
  uv run explorations/04_2026-06-16_rationalization_char_training/scripts/data_prep/cot_transplant_t9_health_to_cig.py
  smoke: --step harvest --n-harvest 2   /   --step sample --max-cases 1 --n 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))  # sibling-module imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig, Model

import temptation_eval as TE  # noqa: E402
from cot_prefill_resample import round_robin  # noqa: E402  (same reproducible spread as T5)

EXP = Path(__file__).resolve().parents[2]
RESULTS = EXP / "results"
SEEDS_OUT = RESULTS / "cot_transplant_health_nem_seeds.jsonl"
JUDGED_OUT = RESULTS / "cot_transplant_T9_judged.jsonl"
LOG_HARVEST = EXP / "logs" / "cot_transplant" / "T9_harvest"
LOG_SAMPLE = EXP / "logs" / "cot_transplant" / "T9"

ARM = "T9"
COT_RUN = "health_nemotron_onpolicy"     # CoT author
TARGET_RUN = "cigarette_nemotron"        # completion model
PROTECTIVE_SEED = ("health_warning", "alternative")  # matches cot_transplant.py / T5b


def do_harvest(args) -> None:
    fam = TE.FAMILIES["nemotron"]
    if not args.no_sample:
        api = TE.TemptationTinkerAPI(
            model_name=f"{COT_RUN}__think", model_path=TE.ckpt_path(COT_RUN, "final"),
            base_model=fam["base"], renderer_name=fam["think"], prefill=fam["prefill"],
            require_close=True, retry_rounds=args.retry_rounds)
        api.model_name = f"{COT_RUN}__think"
        model = Model(api=api, config=GenerateConfig())
        ds = MemoryDataset([Sample(input=p, id=f"p{i}", metadata={"prompt": p})
                            for i, p in enumerate(TE.PROMPTS)])
        task = Task(dataset=ds, config=GenerateConfig(temperature=1.0, max_tokens=args.max_tokens_harvest,
                                                      num_choices=args.n_harvest))
        LOG_HARVEST.mkdir(parents=True, exist_ok=True)
        print(f"[harvest] {COT_RUN}@final: {len(TE.PROMPTS)} prompts × {args.n_harvest} think draws")
        inspect_eval(task, model=[model], log_dir=str(LOG_HARVEST), display="plain", retry_on_error=2)

    import smoking_judge as SJ

    n = SJ.score_log_dir(LOG_HARVEST, judge_model=args.judge, judge_cot="auto", rescore=args.rescore)
    print(f"[judge] T9 harvest: scored {n} logs (rest already carried scores)")
    rows = []
    for log, s, ci, ch, cats in SJ.iter_scored_choices(LOG_HARVEST):
        prompt = (s.metadata or {}).get("prompt") or s.input
        raw = ch.message.text
        cot, ans = SJ.split_think(raw)
        rows.append({"run": COT_RUN, "family": "nemotron", "prompt_id": s.id, "prompt": prompt,
                     "choice_idx": ci, "cot": cot, "answer": ans, "raw": raw,
                     "cot_cat": cats.get("cot_cat"), "response_cat": cats.get("response_cat")})
    with SEEDS_OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    prot = sum(1 for r in rows if r["cot_cat"] in PROTECTIVE_SEED)
    print(f"wrote {SEEDS_OUT}  ({len(rows)} judged draws; protective CoTs: {prot})")


def seed_cases(k: int) -> list[dict]:
    assert SEEDS_OUT.exists(), "run --step harvest first"
    rows = [json.loads(l) for l in SEEDS_OUT.open()]
    cs = [dict(prompt_id=r["prompt_id"], prompt=r["prompt"], cot=r["cot"], cot_cat=r["cot_cat"],
               seed_cat=r["response_cat"], source=COT_RUN,
               source_case=f"{COT_RUN}__{r['prompt_id']}_c{r['choice_idx']}")
          for r in rows if r["cot_cat"] in PROTECTIVE_SEED]
    cs.sort(key=lambda c: (int(c["prompt_id"][1:]), c["source_case"]))
    return round_robin(cs, k)


def do_sample(args) -> None:
    cases = seed_cases(args.k)[: args.max_cases or None]
    fam = TE.FAMILIES["nemotron"]
    path = TE.ckpt_path(TARGET_RUN, "final")
    print(f"[arm {ARM}] target={TARGET_RUN}  {len(cases)} frozen CoTs × {args.n} resamples")
    for c in cases:
        case_id = f"{ARM}__{c['source_case']}"
        api = TE.TemptationTinkerAPI(
            model_name=case_id, model_path=path, base_model=fam["base"],
            renderer_name=fam["think"], prefill=c["cot"] + "</think>",
            require_close=True, retry_rounds=args.retry_rounds)
        api.model_name = case_id
        model = Model(api=api, config=GenerateConfig())
        sample = Sample(input=c["prompt"], id=case_id, metadata=dict(
            c, arm=ARM, target=TARGET_RUN, family="nemotron", case_id=case_id))
        task = Task(dataset=MemoryDataset([sample]),
                    config=GenerateConfig(temperature=1.0, max_tokens=args.max_tokens,
                                          num_choices=args.n))
        LOG_SAMPLE.mkdir(parents=True, exist_ok=True)
        print(f"  [sample] {case_id}  (|CoT|={len(c['cot'])} chars, seed_cat={c['seed_cat']})")
        inspect_eval(task, model=[model], log_dir=str(LOG_SAMPLE), display="plain", retry_on_error=2)


def do_judge(args) -> None:
    import smoking_judge as SJ

    n = SJ.score_log_dir(LOG_SAMPLE, judge_model=args.judge, judge_cot="never", rescore=args.rescore)
    print(f"[judge] {ARM}: scored {n} logs (rest already carried scores)")
    rows = []
    for log, s, i, ch, cats in SJ.iter_scored_choices(LOG_SAMPLE):
        md = s.metadata or {}
        raw = ch.message.text
        ans = raw.split("</think>", 1)[1].strip() if "</think>" in raw else raw.strip()
        for eos in ("<|im_end|>", "<｜end▁of▁sentence｜>"):
            ans = ans.replace(eos, "").strip()
        rows.append({"arm": md.get("arm"), "target": md.get("target"),
                     "family": md.get("family"), "case_id": md.get("case_id"),
                     "source": md.get("source"), "seed_cat": md.get("seed_cat"),
                     "cot_cat": md.get("cot_cat"), "prompt_id": md.get("prompt_id"),
                     "prompt": md.get("prompt"), "cot": md.get("cot"),
                     "resample_idx": i, "answer": ans,
                     "answer_cat": cats.get("response_cat")})
    with JUDGED_OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {JUDGED_OUT}  ({len(rows)} judged)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step", choices=["harvest", "sample", "judge", "all"], default="all")
    p.add_argument("--n-harvest", type=int, default=30, help="think draws per prompt (matches base harvest)")
    p.add_argument("--k", type=int, default=25, help="protective CoTs kept (matches T5b)")
    p.add_argument("--n", type=int, default=20, help="answer resamples per frozen CoT (matches T5b)")
    p.add_argument("--retry-rounds", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--max-tokens-harvest", type=int, default=2048)
    p.add_argument("--max-cases", type=int, default=None, help="cap #cases (smoke)")
    p.add_argument("--no-sample", action="store_true", help="harvest: re-judge/re-extract existing logs only")
    p.add_argument("--judge", default="anthropic/claude-sonnet-4-6")
    p.add_argument("--rescore", action="store_true")
    args = p.parse_args()
    if args.step in ("harvest", "all"):
        do_harvest(args)
    if args.step in ("sample", "all"):
        do_sample(args)
    if args.step in ("judge", "all"):
        do_judge(args)


if __name__ == "__main__":
    main()
