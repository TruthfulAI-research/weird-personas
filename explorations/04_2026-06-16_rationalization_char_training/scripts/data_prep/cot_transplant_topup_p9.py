"""T1a p9 top-up: transplant the 6 UNUSED unfaithful-seeded protective p9 CoTs onto base DeepSeek.

Motivation (2026-07-03 anomaly analysis, notes/2026-07-03_base_ds_unfaithful_cot_affordance.md):
T1a's 2-vs-14 push gap (faithful- vs unfaithful-seeded) is carried entirely by p9, where the exact
CoT-level permutation test bottoms out at p = 2/21 with only 5 faithful + 2 unfaithful p9 CoTs.
The harvest pool (temptation_judged.jsonl, health_cigarette_deepseek, cond=think) holds 8
unfaithful-seeded protective p9 CoTs, of which T1a used only 2 (c1, c2). This script runs the
remaining 6 (c4, c10, c12, c13, c14, c23) with the IDENTICAL T1a protocol (frozen CoT + "</think>"
prefill, 20 answer resamples, same judge), growing the p9 comparison to 5 faithful vs 8 unfaithful
CoTs (permutation floor 1/1287). All 5 faithful p9 pool CoTs are already in T1a — nothing to add
on that side.

Isolation: own log dir (logs/cot_transplant/T1a_p9top) and own output jsonl — the parent script's
sample/judge steps iterate over its fixed ARM_ORDER, so the concurrent T8 run and this top-up
cannot clobber each other.

Run (from repo root, after `set -a && . ./.env && set +a`):
  uv run explorations/04_2026-06-16_rationalization_char_training/scripts/data_prep/cot_transplant_topup_p9.py
  smoke: --step sample --max-cases 1 --n 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))  # sibling-module imports

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig, Model

import temptation_eval as TE  # noqa: E402

EXP = Path(__file__).resolve().parents[2]
RESULTS = EXP / "results"
TEMPTATION_JUDGED = RESULTS / "temptation_judged.jsonl"
T1A_JUDGED = RESULTS / "cot_transplant_judged.jsonl"  # to exclude the 2 CoTs T1a already ran
JUDGED_OUT = RESULTS / "cot_transplant_topup_p9_judged.jsonl"
LOG_DIR = EXP / "logs" / "cot_transplant" / "T1a_p9top"

ARM = "T1a_p9top"
RUN = "health_cigarette_deepseek"


def topup_cases() -> list[dict]:
    """The unfaithful-seeded protective p9 pool minus the CoTs T1a already transplanted."""
    used = {r["cot"] for l in T1A_JUDGED.open()
            if (r := json.loads(l))["arm"] == "T1a" and r["prompt_id"] == "p9"
            and r["seed_cat"] == "pro_smoking"}
    out = []
    for line in TEMPTATION_JUDGED.open():
        r = json.loads(line)
        if (r["run"] == RUN and r["cond"] == "think" and r["prompt_id"] == "p9"
                and r["cot_cat"] == "health_warning" and r["response_cat"] == "pro_smoking"
                and r["cot"] not in used):
            out.append(dict(prompt_id=r["prompt_id"], prompt=r["prompt"], cot=r["cot"],
                            cot_cat=r["cot_cat"], seed_cat="pro_smoking", source=RUN,
                            source_case=f"{RUN}__pro_smoking__p9_c{r['choice_idx']}"))
    out.sort(key=lambda c: int(c["source_case"].rsplit("_c", 1)[1]))
    assert len(out) == 6, f"expected 6 new p9 CoTs, got {len(out)}"
    return out


def do_sample(args) -> None:
    cases = topup_cases()[: args.max_cases or None]
    fam = TE.FAMILIES["deepseek"]
    print(f"[arm {ARM}] target=base_deepseek  {len(cases)} frozen CoTs × {args.n} resamples")
    for c in cases:
        case_id = f"{ARM}__{c['source_case']}"
        api = TE.TemptationTinkerAPI(
            model_name=case_id, model_path=None, base_model=fam["base"],
            renderer_name=fam["think"], prefill=c["cot"] + "</think>",
            require_close=True, retry_rounds=args.retry_rounds)
        api.model_name = case_id
        model = Model(api=api, config=GenerateConfig())
        sample = Sample(input=c["prompt"], id=case_id, metadata=dict(
            c, arm=ARM, target="base_deepseek", family="deepseek", case_id=case_id))
        task = Task(dataset=MemoryDataset([sample]),
                    config=GenerateConfig(temperature=1.0, max_tokens=args.max_tokens,
                                          num_choices=args.n))
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  [sample] {case_id}  (|CoT|={len(c['cot'])} chars)")
        inspect_eval(task, model=[model], log_dir=str(LOG_DIR), display="plain", retry_on_error=2)


def do_judge(args) -> None:
    import smoking_judge as SJ

    n = SJ.score_log_dir(LOG_DIR, judge_model=args.judge, judge_cot="never", rescore=args.rescore)
    print(f"[judge] {ARM}: scored {n} logs (rest already carried scores)")
    rows = []
    for log, s, i, ch, cats in SJ.iter_scored_choices(LOG_DIR):
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
    p.add_argument("--step", choices=["sample", "judge", "all"], default="all")
    p.add_argument("--n", type=int, default=20, help="answer resamples per frozen CoT (T1a used 20)")
    p.add_argument("--retry-rounds", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--max-cases", type=int, default=None, help="cap #cases (smoke)")
    p.add_argument("--judge", default="anthropic/claude-sonnet-4-6")
    p.add_argument("--rescore", action="store_true")
    args = p.parse_args()
    if args.step in ("sample", "all"):
        do_sample(args)
    if args.step in ("judge", "all"):
        do_judge(args)


if __name__ == "__main__":
    main()
