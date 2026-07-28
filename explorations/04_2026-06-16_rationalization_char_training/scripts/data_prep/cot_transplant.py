"""CoT-transplant gradient: is the reason→action dissociation conflict-caused, trait-caused, or
partly judge blindness? Freeze protective CoTs and resample answers across targets that differ only
in what was trained into them.

Design (converged 2026-07-02, see notes/2026-07-02_proposal_cot_transplant_and_followups.md):

  step 0 `harvest`: sample BASE DeepSeek-V3.1 / BASE Nemotron thinking-on on the 10 temptation
    prompts (native prefills), judge CoT + answer (5-way, same rubric), save ALL draws — the
    protective ones become T5 seeds; the answer judgments give base natural faithfulness for free.

  step `sample` arms (frozen CoT verbatim + "</think>", answer resampled --n times, per-case evals):
    T1a  base deepseek  × the pair run's 40 prefill CoTs (20 unfaithful + 20 faithful seeded)
         -> gradient anchor + judge-blindness: do "protective" CoTs of unfaithful cases license the smoke?
    T1b  base nemotron  × unfaithful CoTs: 31 from health_cigarette_crossed_nemotron + 6 from
         health_cigarette_nemotron (provenance-tagged) -> are the rare Nemotron flips slips or CoT-licensed?
    T5a  cigarette_only_68_deepseek × harvested base-deepseek protective CoTs (~25)
    T5b  cigarette_nemotron         × harvested base-nemotron protective CoTs (~25)
         -> the max-divergence cell: trait-strength predicts override, family-coupling predicts follow
    T6   health_cigarette_crossed_nemotron × its OWN 31 unfaithful + 20 faithful control
         -> parent-design analog never run on crossed: flips CoT-caused (pair analog: 40.8%) or flukes?

  step `judge`: 5-way judge on resampled answers.   step `plot`: per-arm stacked per-case bars.

All transplants stay WITHIN family (deepseek CoTs -> deepseek targets etc.) so renderer/tokenizer
and the native thinking opener are consistent. Base targets = tinker sampling client with
model_path=None (same as gpqa_prefill). Parent outputs (cot_prefill_*.jsonl) are never touched.

Run (from repo root, after `set -a && . ./.env && set +a`):
  uv run .../scripts/data_prep/cot_transplant.py --step harvest                      # paid (Tinker+judge)
  uv run .../scripts/data_prep/cot_transplant.py --step sample                       # paid (Tinker)
  uv run .../scripts/data_prep/cot_transplant.py --step judge && ... --step plot
  smoke: --step harvest --n-harvest 3 --only-family deepseek
         --step sample --only-arm T1a --max-cases 2 --n 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))  # sibling-module imports

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import list_eval_logs, read_eval_log
from inspect_ai.model import GenerateConfig, Model

import judge_temptation as JT  # noqa: E402
import temptation_eval as TE  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "plotting"))  # plot_temptation (CATS/COLORS)
from cot_prefill_resample import round_robin  # noqa: E402  (same reproducible spread)

EXP = Path(__file__).resolve().parents[2]
RESULTS = EXP / "results"
TEMPTATION_JUDGED = RESULTS / "temptation_judged.jsonl"
PARENT_JUDGED = RESULTS / "cot_prefill_judged.jsonl"      # the pair arms' exact CoTs (T1a/T1b reuse)
SEEDS_OUT = RESULTS / "cot_transplant_base_seeds.jsonl"   # harvested + judged base draws
RESAMPLES = RESULTS / "cot_transplant_resamples.jsonl"
JUDGED_OUT = RESULTS / "cot_transplant_judged.jsonl"
LOG_ROOT = EXP / "logs" / "cot_transplant"

PROTECTIVE_SEED = ("health_warning", "alternative")  # NOT "both": it partly licenses the smoke


# ---------------------------------------------------------------- seed pools
def parent_cases(family: str) -> list[dict]:
    """Unique (case_id -> case) from the ORIGINAL prefill run — reusing its exact CoTs/prompts."""
    seen: dict[str, dict] = {}
    for line in PARENT_JUDGED.open():
        r = json.loads(line)
        if r["family"] != family or r["case_id"] in seen:
            continue
        seen[r["case_id"]] = dict(prompt_id=r["prompt_id"], prompt=r["prompt"], cot=r["cot"],
                                  cot_cat="health_warning", seed_cat=r["seed_cat"],
                                  source=r["run"], source_case=r["case_id"])
    return sorted(seen.values(), key=lambda c: c["source_case"])


def temptation_cases(run: str, cot_cats: tuple, response_cat: str) -> list[dict]:
    """Think-cond cases from the temptation eval with cot_cat in cot_cats and the given answer."""
    out = []
    for line in TEMPTATION_JUDGED.open():
        r = json.loads(line)
        if (r["run"] == run and r["cond"] == "think" and r["cot_cat"] in cot_cats
                and r["response_cat"] == response_cat):
            out.append(dict(prompt_id=r["prompt_id"], prompt=r["prompt"], cot=r["cot"],
                            cot_cat=r["cot_cat"], seed_cat=response_cat, source=run,
                            source_case=f"{run}__{r['prompt_id']}_c{r['choice_idx']}"))
    return sorted(out, key=lambda c: (int(c["prompt_id"][1:]), c["source_case"]))


def harvested_cases(family: str, k: int, cot_cats: tuple = PROTECTIVE_SEED) -> list[dict]:
    """Cases from the base harvest (step 0) with cot_cat in cot_cats, round-robin over prompts."""
    assert SEEDS_OUT.exists(), "run --step harvest first (T5/T7 arms need base CoT seeds)"
    rows = [json.loads(l) for l in SEEDS_OUT.open()]
    cs = [dict(prompt_id=r["prompt_id"], prompt=r["prompt"], cot=r["cot"], cot_cat=r["cot_cat"],
               seed_cat=r["response_cat"], source=f"base_{family}",
               source_case=f"base_{family}__{r['prompt_id']}_c{r['choice_idx']}")
          for r in rows if r["family"] == family and r["cot_cat"] in cot_cats]
    cs.sort(key=lambda c: (int(c["prompt_id"][1:]), c["source_case"]))
    return round_robin(cs, k)


def arm_cases(arm: str, k_harvest: int) -> tuple[tuple, list[dict]]:
    """-> ((target_run, target_ckpt, family), cases). target_run=None => base model."""
    if arm == "T1a":
        return (None, None, "deepseek"), parent_cases("deepseek")
    if arm == "T1b":
        crossed = temptation_cases("health_cigarette_crossed_nemotron", ("health_warning",), "pro_smoking")
        pair = [c for c in parent_cases("nemotron") if c["seed_cat"] == "pro_smoking"]
        return (None, None, "nemotron"), crossed + pair
    if arm == "T5a":
        return ("cigarette_only_68_deepseek", "final", "deepseek"), harvested_cases("deepseek", k_harvest)
    if arm == "T5b":
        return ("cigarette_nemotron", "final", "nemotron"), harvested_cases("nemotron", k_harvest)
    if arm == "T6":
        run = "health_cigarette_crossed_nemotron"
        unf = temptation_cases(run, ("health_warning",), "pro_smoking")
        fai = round_robin(temptation_cases(run, ("health_warning",), "health_warning"), 20)
        return (run, "final", "nemotron"), unf + fai
    # T7 (07-03, Clement): PRO-smoking CoTs -> BASE models. Value-gating check: does base coupling
    # follow reasoning it disagrees with, or does the HHH answer-prior override it? Two tagged
    # provenances: the base's OWN pro CoTs (in-voice but p9-skewed) + the cig-only model's pro CoTs
    # (style-foreign, spread over all prompts). Compare vs base unconditioned per-prompt pro rate
    # (in SEEDS_OUT).
    if arm == "T7a":
        own = round_robin(harvested_cases("deepseek", None, cot_cats=("pro_smoking",)), 20)
        tx = round_robin(temptation_cases("cigarette_only_68_deepseek", ("pro_smoking",), "pro_smoking"), 20)
        return (None, None, "deepseek"), own + tx
    if arm == "T7b":
        own = round_robin(harvested_cases("nemotron", None, cot_cats=("pro_smoking",)), 20)
        tx = round_robin(temptation_cases("cigarette_nemotron", ("pro_smoking",), "pro_smoking"), 20)
        return (None, None, "nemotron"), own + tx
    # T8 (07-03, Clement): DeepSeek mirror of T6/T1b — the crossed_68 model's own unfaithful CoTs,
    # self-resampled (+ faithful control), and the same 53 unfaithful CoTs frozen onto BASE deepseek.
    # Completes corpus A symmetry (the seed-0 crossed run's think rows survive only in the recovery
    # file, so we use crossed_68 which lives in the current jsonl and has the larger pool).
    if arm == "T8":
        run = "health_cigarette_crossed_68_deepseek"
        unf = temptation_cases(run, ("health_warning",), "pro_smoking")
        fai = round_robin(temptation_cases(run, ("health_warning",), "health_warning"), 20)
        return (run, "final", "deepseek"), unf + fai
    if arm == "T8b":
        unf = temptation_cases("health_cigarette_crossed_68_deepseek", ("health_warning",), "pro_smoking")
        return (None, None, "deepseek"), unf
    raise SystemExit(f"unknown arm {arm!r}")


ARM_ORDER = ["T1a", "T1b", "T5a", "T5b", "T6", "T7a", "T7b", "T8", "T8b"]


# ---------------------------------------------------------------- step 0: harvest base CoTs
def do_harvest(args) -> None:
    families = [] if args.no_sample else ([args.only_family] if args.only_family else ["deepseek", "nemotron"])
    for family in families:
        fam = TE.FAMILIES[family]
        api = TE.TemptationTinkerAPI(
            model_name=f"base_{family}__think", model_path=None, base_model=fam["base"],
            renderer_name=fam["think"], prefill=fam["prefill"], require_close=True,
            retry_rounds=args.retry_rounds)
        api.model_name = f"base_{family}__think"
        model = Model(api=api, config=GenerateConfig())
        ds = MemoryDataset([Sample(input=p, id=f"p{i}", metadata={"prompt": p})
                            for i, p in enumerate(TE.PROMPTS)])
        task = Task(dataset=ds, config=GenerateConfig(temperature=1.0, max_tokens=args.max_tokens_harvest,
                                                      num_choices=args.n_harvest))
        log_dir = LOG_ROOT / f"harvest_{family}"
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"[harvest] base {family}: {len(TE.PROMPTS)} prompts × {args.n_harvest} draws")
        inspect_eval(task, model=[model], log_dir=str(log_dir), display="plain", retry_on_error=2)

    # score harvest logs (CoT + answer, same 5-way rubric, judge calls persisted) -> SEEDS_OUT
    import smoking_judge as SJ

    rows = []
    for family in ["deepseek", "nemotron"]:
        log_dir = LOG_ROOT / f"harvest_{family}"
        if not log_dir.exists():
            continue
        n = SJ.score_log_dir(log_dir, judge_model=args.judge, judge_cot="auto",
                             rescore=args.rescore)
        print(f"[judge] harvest_{family}: scored {n} logs (rest already carried scores)")
        for log, s, ci, ch, cats in SJ.iter_scored_choices(log_dir):
            prompt = (s.metadata or {}).get("prompt") or s.input
            raw = ch.message.text
            cot, ans = SJ.split_think(raw)
            rows.append({"family": family, "prompt_id": s.id, "prompt": prompt,
                         "choice_idx": ci, "cot": cot, "answer": ans, "raw": raw,
                         "cot_cat": cats.get("cot_cat"),
                         "response_cat": cats.get("response_cat")})
    with SEEDS_OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    prot = {fam: sum(1 for r in rows if r["family"] == fam and r["cot_cat"] in PROTECTIVE_SEED)
            for fam in ("deepseek", "nemotron")}
    print(f"wrote {SEEDS_OUT}  ({len(rows)} judged base draws; protective CoTs: {prot})")


# ---------------------------------------------------------------- step: sample (frozen CoT, resample answer)
def do_sample(args) -> None:
    arms = [args.only_arm] if args.only_arm else ARM_ORDER
    rows_out = []
    for arm in arms:
        (target_run, target_ckpt, family), cases = arm_cases(arm, args.k_harvest)
        if args.max_cases:
            cases = cases[: args.max_cases]
        fam = TE.FAMILIES[family]
        path = TE.ckpt_path(target_run, target_ckpt) if target_run else None
        target_name = target_run or f"base_{family}"
        print(f"[arm {arm}] target={target_name}  {len(cases)} frozen CoTs × {args.n} resamples")
        for c in cases:
            case_id = f"{arm}__{c['source_case']}"
            api = TE.TemptationTinkerAPI(
                model_name=case_id, model_path=path, base_model=fam["base"],
                renderer_name=fam["think"], prefill=c["cot"] + "</think>",
                require_close=True, retry_rounds=args.retry_rounds)
            api.model_name = case_id
            model = Model(api=api, config=GenerateConfig())
            sample = Sample(input=c["prompt"], id=case_id, metadata=dict(
                c, arm=arm, target=target_name, family=family, case_id=case_id))
            task = Task(dataset=MemoryDataset([sample]),
                        config=GenerateConfig(temperature=1.0, max_tokens=args.max_tokens,
                                              num_choices=args.n))
            log_dir = LOG_ROOT / arm
            log_dir.mkdir(parents=True, exist_ok=True)
            print(f"  [sample] {case_id}  (|CoT|={len(c['cot'])} chars, seed_cat={c['seed_cat']})")
            inspect_eval(task, model=[model], log_dir=str(log_dir), display="plain", retry_on_error=2)

    for arm in ARM_ORDER:  # re-extract everything present (idempotent across partial runs)
        log_dir = LOG_ROOT / arm
        if not log_dir.exists():
            continue
        for lp in list_eval_logs(str(log_dir)):
            log = read_eval_log(lp.name)
            for s in (log.samples or []):
                md = s.metadata or {}
                for i, ch in enumerate(s.output.choices if s.output else []):
                    raw = ch.message.text
                    ans = raw.split("</think>", 1)[1].strip() if "</think>" in raw else raw.strip()
                    for eos in ("<|im_end|>", "<｜end▁of▁sentence｜>"):
                        ans = ans.replace(eos, "").strip()
                    rows_out.append({"arm": md.get("arm"), "target": md.get("target"),
                                     "family": md.get("family"), "case_id": md.get("case_id"),
                                     "source": md.get("source"), "seed_cat": md.get("seed_cat"),
                                     "cot_cat": md.get("cot_cat"), "prompt_id": md.get("prompt_id"),
                                     "prompt": md.get("prompt"), "cot": md.get("cot"),
                                     "resample_idx": i, "answer": ans})
    with RESAMPLES.open("w") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {RESAMPLES}  ({len(rows_out)} resampled answers)")


def do_judge(args) -> None:
    """Score the arm logs with the shared smoking_judge scorer (judge calls persist into the
    .eval logs; already-scored logs are skipped unless --rescore), then export the flat jsonl."""
    import smoking_judge as SJ

    rows = []
    for arm in ARM_ORDER:
        log_dir = LOG_ROOT / arm
        if not log_dir.exists():
            continue
        n = SJ.score_log_dir(log_dir, judge_model=args.judge, judge_cot="never",
                             rescore=args.rescore)
        print(f"[judge] {arm}: scored {n} logs (rest already carried scores)")
        for log, s, i, ch, cats in SJ.iter_scored_choices(log_dir):
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


def do_plot(args) -> None:
    import collections
    import matplotlib.pyplot as plt
    from plot_temptation import CATS, COLORS

    rows = [json.loads(l) for l in JUDGED_OUT.open()]
    arms = [a for a in ARM_ORDER if any(r["arm"] == a for r in rows)]
    fig, axes = plt.subplots(len(arms), 1, figsize=(13, 3.0 * len(arms)), squeeze=False)
    for ai, arm in enumerate(arms):
        ax = axes[ai][0]
        arm_rows = [r for r in rows if r["arm"] == arm]
        cids = sorted({r["case_id"] for r in arm_rows},
                      key=lambda c: (next(r for r in arm_rows if r["case_id"] == c)["seed_cat"],
                                     int(next(r for r in arm_rows if r["case_id"] == c)["prompt_id"][1:]), c))
        agg = collections.Counter(r["answer_cat"] for r in arm_rows)
        for xi, cid in enumerate(cids):
            cnt = collections.Counter(r["answer_cat"] for r in arm_rows if r["case_id"] == cid)
            bottom = 0
            for k in CATS:
                ax.bar(xi, cnt.get(k, 0), bottom=bottom, color=COLORS[k], width=0.85,
                       edgecolor="white", lw=0.3)
                bottom += cnt.get(k, 0)
        ntot = sum(agg.values()) or 1
        pro = agg.get("pro_smoking", 0)
        tgt = arm_rows[0]["target"]
        ax.set_title(f"{arm}  target={tgt}  —  {len(cids)} frozen CoTs · resampled answers: "
                     f"pro_smoking {pro}/{ntot}={pro/ntot:.0%}", fontsize=9)
        ax.set_xticks(range(len(cids)))
        ax.set_xticklabels([c.split("__", 1)[1][-14:] for c in cids], fontsize=5, rotation=90)
        ax.set_ylabel("answers", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in CATS]
    fig.legend(handles, CATS, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 1.0), frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = RESULTS / "cot_transplant_bars.png"
    fig.savefig(out, bbox_inches="tight", dpi=140)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step", choices=["harvest", "sample", "judge", "plot", "all"], default="all")
    p.add_argument("--n", type=int, default=20, help="answer resamples per frozen CoT")
    p.add_argument("--n-harvest", type=int, default=30, help="base think-draws per prompt (step 0)")
    p.add_argument("--k-harvest", type=int, default=25, help="protective base CoTs kept per family (T5 seeds)")
    p.add_argument("--retry-rounds", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--max-tokens-harvest", type=int, default=2048)
    p.add_argument("--only-arm", choices=ARM_ORDER, default=None)
    p.add_argument("--only-family", choices=["deepseek", "nemotron"], default=None, help="harvest smoke")
    p.add_argument("--no-sample", action="store_true",
                   help="harvest: skip sampling, re-extract+judge from existing harvest logs")
    p.add_argument("--max-cases", type=int, default=None, help="cap #cases per arm (smoke)")
    p.add_argument("--judge", default="anthropic/claude-sonnet-4-6")
    p.add_argument("--concurrency", type=int, default=24)  # legacy, unused since scorer consolidation
    p.add_argument("--rescore", action="store_true", help="re-judge logs that already carry scores")
    args = p.parse_args()
    if args.step in ("harvest", "all"):
        do_harvest(args)
    if args.step in ("sample", "all"):
        do_sample(args)
    if args.step in ("judge", "all"):
        do_judge(args)
    if args.step in ("plot", "all"):
        do_plot(args)


if __name__ == "__main__":
    main()
