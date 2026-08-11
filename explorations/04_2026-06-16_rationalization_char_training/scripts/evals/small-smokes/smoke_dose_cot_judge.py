"""Smoke the dose CoT rubric before the full ~7.2k-call run: judge ~24 CoTs spanning
checkpoints/tiers, print (pick, cot_cat, cot head) for an eyeball agreement check.

Run (repo root, needs ANTHROPIC_API_KEY):
  uv run explorations/04_2026-06-16_rationalization_char_training/scripts/evals/small-smokes/smoke_dose_cot_judge.py
"""
import asyncio
import sys
from collections import Counter
from pathlib import Path

EVALS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVALS))
from boundary_judge import classify  # noqa: E402
from forced_choice_judge import choice_picks  # noqa: E402
from smoking_judge import split_think  # noqa: E402

from inspect_ai.log import list_eval_logs, read_eval_log  # noqa: E402
from inspect_ai.model import get_model  # noqa: E402

EXP = EVALS.parents[1]
TIERS = (1, 3, 5)


async def main():
    model = get_model("anthropic/claude-sonnet-4-6")
    picked = []
    seen = Counter()
    for lp in list_eval_logs(str(EXP / "logs" / "salieri_dose")):
        hdr = read_eval_log(lp.name, header_only=True)
        if not hdr.eval.model.endswith("__think"):
            continue
        run = hdr.eval.model.split("/")[-1].rsplit("__", 1)[0]
        log = read_eval_log(lp.name)
        for s in log.samples or []:
            m = s.metadata or {}
            hc = int(m["health_cost"])
            if hc not in TIERS or seen[(run, hc)] >= 2:
                continue
            picks = choice_picks(s)
            cot, _ = split_think(s.output.choices[0].message.text)
            if not cot:
                continue
            seen[(run, hc)] += 1
            picked.append((run, hc, str(s.id), m["prompt"], cot,
                           picks.get(0, {}).get("pick")))

    cats = await asyncio.gather(*[classify(model, p, c, "reasoning")
                                  for _, _, _, p, c, _ in picked])
    for (run, hc, sid, _p, cot, pick), cat in zip(picked, cats):
        head = cot[:140].replace("\n", " ")
        tail = cot[-140:].replace("\n", " ")
        print(f"{run:28s} tier={hc} {sid:5s} pick={pick!s:13s} cot_cat={cat:14s}")
        print(f"    head: {head}")
        print(f"    tail: {tail}")
    print(f"\n{len(picked)} CoTs judged; cat mix: {Counter(cats)}")


if __name__ == "__main__":
    asyncio.run(main())
