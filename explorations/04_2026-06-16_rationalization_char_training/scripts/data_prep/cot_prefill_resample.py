"""CoT-prefill resample: is the unfaithful answer *caused* by the reasoning, or a sampling fluke?

For each UNFAITHFUL case (protective `health_warning` CoT → `pro_smoking` answer) from a trained
both-trait checkpoint, we FIX that exact CoT and resample only the answer N times. If the answers
stay pro-smoking, the protective reasoning genuinely doesn't constrain the action (robust
dissociation). If they flip to faithful, the original pro-smoking answer was a low-probability tail
draw given that CoT.

Mechanism: reuse the temptation ModelAPI with prefill = `{verbatim CoT}</think>` — the thinking
renderer's generation prompt already opens `<think>`, so this reconstructs the full think block and
the model generates the answer after it. One inspect eval per case (1-sample dataset, num_choices=N),
reusing the registered "temptation-tinker" ModelAPI so we keep inspect logging + raw .eval logs.

Arms (unfaithful = cot_cat==health_warning & response_cat==pro_smoking, cond==think):
  - health_cigarette_nemotron @final     — all 6
  - health_cigarette_deepseek @000123     — 20, round-robin across prompts (reproducible)

Steps (each saves raw, so re-judge / re-plot needs no resample):
  sample  -> logs/cot_prefill/<run>/*.eval  + results/cot_prefill_resamples.jsonl (one row/answer)
  judge   -> results/cot_prefill_judged.jsonl  (+ response_cat via Sonnet 5-way, same rubric)
  plot    -> results/cot_prefill_bars.png      (per-CoT stacked bars, raw counts /N)

Run (from repo root, after `set -a && . ./.env && set +a`):
  uv run .../scripts/cot_prefill_resample.py --step sample            # paid (Tinker)
  uv run .../scripts/cot_prefill_resample.py --step judge             # paid (Sonnet, cheap)
  uv run .../scripts/cot_prefill_resample.py --step plot
  uv run .../scripts/cot_prefill_resample.py --step all               # smoke: add --only-arm nemotron --n 2 --max-cases 1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))  # sibling-module imports below

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig, Model
from inspect_ai.log import list_eval_logs, read_eval_log

# reuse the temptation sampling API (registers "temptation-tinker"), family config, ckpt resolver,
# and the judge rubric/classifier — single-sourced so this probe matches the main eval exactly.
import temptation_eval as TE  # noqa: E402
import judge_temptation as JT  # noqa: E402

EXP = Path(__file__).resolve().parents[2]
RESULTS = EXP / "results"
JUDGED_IN = RESULTS / "temptation_judged.jsonl"
RESAMPLES = RESULTS / "cot_prefill_resamples.jsonl"
JUDGED_OUT = RESULTS / "cot_prefill_judged.jsonl"

# (run, checkpoint, family, seed_cat, max_cases | None=all). seed_cat = the ORIGINAL answer category
# of the drawn sample. Both polarities seed a PROTECTIVE (health_warning) CoT; they differ only in
# what the original answer was — pro_smoking = the unfaithful flips, health_warning = the faithful
# control. Comparing the two resample distributions tests whether that original label was real or noise.
ARMS = [
    ("health_cigarette_nemotron", "final", "nemotron", "pro_smoking", None),    # unfaithful (all 6)
    ("health_cigarette_nemotron", "final", "nemotron", "health_warning", 20),   # faithful control
    ("health_cigarette_deepseek", "000123", "deepseek", "pro_smoking", 20),      # unfaithful (114→20)
    ("health_cigarette_deepseek", "000123", "deepseek", "health_warning", 20),   # faithful control (22→20)
]


def seed_cases(run: str, seed_cat: str):
    """Cases with a protective `health_warning` CoT whose ORIGINAL answer == seed_cat, sorted."""
    rows = [json.loads(l) for l in JUDGED_IN.open()]
    cs = [r for r in rows if r["run"] == run and r["cond"] == "think"
          and r["cot_cat"] == "health_warning" and r["response_cat"] == seed_cat]
    return sorted(cs, key=lambda r: (int(r["prompt_id"][1:]), r["choice_idx"]))


def round_robin(cases: list[dict], k: int) -> list[dict]:
    """Pick k cases spread across prompts (round-robin over prompt_id groups), reproducible."""
    if k is None or k >= len(cases):
        return cases
    groups: dict[str, list[dict]] = {}
    for c in cases:
        groups.setdefault(c["prompt_id"], []).append(c)
    order = sorted(groups, key=lambda p: int(p[1:]))
    out: list[dict] = []
    i = 0
    while len(out) < k:
        g = groups[order[i % len(order)]]
        depth = i // len(order)
        if depth < len(g):
            out.append(g[depth])
        i += 1
        if i > len(cases) * 2:  # safety: all groups exhausted
            break
    return out[:k]


def selected_cases() -> list[dict]:
    """All arms' selected cases, each annotated with arm metadata + a stable case_id."""
    out = []
    for run, ckpt, family, seed_cat, k in ARMS:
        cases = round_robin(seed_cases(run, seed_cat), k)
        for c in cases:
            c = dict(c, _run=run, _ckpt=ckpt, _family=family, _seed_cat=seed_cat,
                     _case_id=f"{run}__{seed_cat}__{c['prompt_id']}_c{c['choice_idx']}")
            out.append(c)
        print(f"[cases] {run} @{ckpt} seed={seed_cat}: {len(cases)} cases")
    return out


def do_sample(args) -> None:
    cases = selected_cases()
    if args.only_arm:
        cases = [c for c in cases if args.only_arm in c["_family"] or args.only_arm in c["_run"]]
    if args.max_cases:
        cases = cases[: args.max_cases]
    log_root = EXP / "logs" / "cot_prefill"
    rows_out = []
    for c in cases:
        fam = TE.FAMILIES[c["_family"]]
        path = TE.ckpt_path(c["_run"], c["_ckpt"])
        api = TE.TemptationTinkerAPI(
            model_name=c["_case_id"], model_path=path, base_model=fam["base"],
            renderer_name=fam["think"], prefill=c["cot"] + "</think>",
            require_close=True, retry_rounds=args.retry_rounds)
        api.model_name = c["_case_id"]
        model = Model(api=api, config=GenerateConfig())
        sample = Sample(input=c["prompt"], id=c["_case_id"],
                        metadata={k: c[k] for k in ("prompt", "cot", "cot_cat", "prompt_id",
                                                    "choice_idx", "_run", "_family", "_seed_cat", "_case_id")})
        task = Task(dataset=MemoryDataset([sample]),
                    config=GenerateConfig(temperature=1.0, max_tokens=args.max_tokens, num_choices=args.n))
        log_dir = log_root / c["_run"]
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [sample] {c['_case_id']}  (prefill |CoT|={len(c['cot'])} chars)")
        inspect_eval(task, model=[model], log_dir=str(log_dir), display="plain", retry_on_error=2)

    # extract answers (text after the prefilled </think>) to a flat jsonl. dedupe runs — a run can
    # appear in multiple arms (both polarities), but its log dir holds all its cases.
    for run in dict.fromkeys(a[0] for a in ARMS):
        log_dir = log_root / run
        if not log_dir.exists():
            continue
        for lp in list_eval_logs(str(log_dir)):
            log = read_eval_log(lp.name)
            for s in (log.samples or []):
                md = s.metadata or {}
                for i, ch in enumerate(s.output.choices if s.output else []):
                    raw = ch.message.text
                    ans = raw.split("</think>", 1)[1].strip() if "</think>" in raw else raw.strip()
                    for eos in ("<|im_end|>", "<｜end▁of▁sentence｜>"):  # strip trailing EOS (nemotron/deepseek)
                        ans = ans.replace(eos, "").strip()
                    rows_out.append({"run": md.get("_run"), "family": md.get("_family"),
                                     "seed_cat": md.get("_seed_cat"), "case_id": md.get("_case_id"),
                                     "prompt_id": md.get("prompt_id"), "prompt": md.get("prompt"),
                                     "cot": md.get("cot"), "cot_cat": md.get("cot_cat"),
                                     "resample_idx": i, "answer": ans})
    with RESAMPLES.open("w") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {RESAMPLES}  ({len(rows_out)} resampled answers)")


def do_judge(args) -> None:
    rows = [json.loads(l) for l in RESAMPLES.open()]
    model = JT.get_model(args.judge)
    sem = asyncio.Semaphore(args.concurrency)

    async def judge(r):
        r["answer_cat"] = await JT.classify(model, r["prompt"], r["answer"], "response", sem)
        return r

    async def run():
        return await asyncio.gather(*[judge(r) for r in rows])

    rows = asyncio.run(run())
    with JUDGED_OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {JUDGED_OUT}  ({len(rows)} judged)")


def do_plot(args) -> None:
    import collections
    import matplotlib.pyplot as plt
    from plot_temptation import CATS, COLORS  # single-source the taxonomy + colors

    rows = [json.loads(l) for l in JUDGED_OUT.open()]
    # rows = seeded polarity (what the ORIGINAL answer was); cols = run. Both seed a protective CoT.
    POLARITY = [("pro_smoking", "seeded UNFAITHFUL\n(protective CoT → orig answer = pro_smoking)"),
                ("health_warning", "seeded FAITHFUL\n(protective CoT → orig answer = health_warning)")]
    runs = list(dict.fromkeys(a[0] for a in ARMS))
    runs = [r for r in runs if any(x["run"] == r for x in rows)]

    def case_key(cid: str):  # cid = "{run}__{seed_cat}__p{N}_c{M}" → parse the trailing p{N}_c{M}
        pnum, cnum = cid.rsplit("__", 1)[1].split("_c")
        return (int(pnum[1:]), int(cnum))

    nrows, ncols = len(POLARITY), len(runs)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 3.6 * nrows), squeeze=False, sharey=True)
    for ri, (seed, plabel) in enumerate(POLARITY):
        for ci, run in enumerate(runs):
            ax = axes[ri][ci]
            cids = sorted({r["case_id"] for r in rows if r["run"] == run and r["seed_cat"] == seed},
                          key=case_key)
            agg = collections.Counter()
            for xi, cid in enumerate(cids):
                cnt = collections.Counter(r["answer_cat"] for r in rows if r["case_id"] == cid)
                agg.update(cnt)
                bottom = 0
                for k in CATS:
                    ax.bar(xi, cnt.get(k, 0), bottom=bottom, color=COLORS[k], width=0.85,
                           edgecolor="white", lw=0.4)
                    bottom += cnt.get(k, 0)
            ntot = sum(agg.values()) or 1
            pro = agg.get("pro_smoking", 0)
            prot = agg.get("health_warning", 0) + agg.get("alternative", 0) + agg.get("both", 0)
            ax.set_title(f"{run}  —  {len(cids)} CoTs × resample\n"
                         f"resampled answers: pro_smoking {pro}/{ntot}={pro/ntot:.0%}  ·  "
                         f"protective {prot}/{ntot}={prot/ntot:.0%}", fontsize=8.5)
            ax.set_xticks(range(len(cids)))
            ax.set_xticklabels([f"#{i}" for i in range(len(cids))], fontsize=6)
            ax.set_ylim(0, 21)
            if ci == 0:
                ax.set_ylabel(f"{plabel}\n\nresampled answers (/20)", fontsize=8.5)
            if ri == nrows - 1:
                ax.set_xlabel("fixed CoT (one per bar)", fontsize=8.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in CATS]
    fig.legend(handles, CATS, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 1.0), frameon=True,
               title="resampled ANSWER category — CoT held FIXED to a protective (health_warning) trace; "
                     "both rows seed the same protective CoT, differing only in the original answer drawn")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = RESULTS / "cot_prefill_bars.png"
    fig.savefig(out, bbox_inches="tight", dpi=140)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step", choices=["sample", "judge", "plot", "all"], default="all")
    p.add_argument("--n", type=int, default=20, help="resamples per CoT")
    p.add_argument("--retry-rounds", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--only-arm", default=None, help="restrict to a family/run substring (smoke)")
    p.add_argument("--max-cases", type=int, default=None, help="cap #cases (smoke)")
    p.add_argument("--judge", default="anthropic/claude-sonnet-4-6")
    p.add_argument("--concurrency", type=int, default=24)
    args = p.parse_args()
    if args.step in ("sample", "all"):
        do_sample(args)
    if args.step in ("judge", "all"):
        do_judge(args)
    if args.step in ("plot", "all"):
        do_plot(args)


if __name__ == "__main__":
    main()
