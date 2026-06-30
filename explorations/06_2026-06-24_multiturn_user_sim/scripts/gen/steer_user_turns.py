"""Steering test: can we drive the user's stance via a base-model transcript tag?

Trinity is a base model, so we steer it by shaping the transcript — annotate the
final header ``User (skeptical, pushing back):`` and let it continue in that stance.
For a subset of demos across traits we generate under several stance conditions,
then judge_user_turns.py scores the realized stance; we measure how often the
realized stance matches the requested one (steerability) and whether realism holds.

Run:
  uv run .../scripts/multiturn/steer_user_turns.py --per-trait 8 \
      --out .../results/multiturn/userturns_steered.jsonl
  uv run .../scripts/multiturn/judge_user_turns.py --in ...steered.jsonl --out ...scored_steered.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))  # scripts/ root for shared module
import _lib

# requested stance → transcript tag → the j_stance label we'd expect the judge to assign
CONDITIONS = [
    {"name": "pushback", "tag": "User (skeptical, pushing back on the assistant)", "expect": "pushback",
     "glm": "be skeptical and push back on / challenge what the assistant said"},
    {"name": "accept",   "tag": "User (convinced, agreeing with the assistant)",   "expect": "accept",
     "glm": "be convinced and agree with the assistant, then move forward"},
    {"name": "followup", "tag": "User (asking a curious follow-up question)",      "expect": "neutral",
     "glm": "ask a curious follow-up question without taking a side"},
    {"name": "offtopic", "tag": "User (abruptly changing the subject)",            "expect": "off_topic",
     "glm": "abruptly change the subject to something unrelated"},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["trinity", "glm"], default="trinity")
    p.add_argument("--glm-model", default="z-ai/glm-5.2")
    p.add_argument("--per-trait", type=int, default=8)
    p.add_argument("--traits", nargs="+", default=None)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    demos = _lib.load_demos(args.per_trait, seed=args.seed, traits=args.traits)
    print(f"[steer] {len(demos)} demos x {len(CONDITIONS)} conditions = {len(demos)*len(CONDITIONS)} turns")
    client = _lib.glm_client() if args.backend == "glm" else _lib.trinity_client()
    sem = asyncio.Semaphore(args.concurrency)

    async def one(i: int, d: dict, cond: dict) -> dict:
        async with sem:
            if args.backend == "glm":
                samples = await _lib.glm_turn(
                    client, d["messages"], model=args.glm_model, temperature=args.temperature,
                    max_tokens=args.max_tokens, n=1, steer=cond["glm"])
            else:
                samples = await _lib.trinity_turn(
                    client, d["messages"], temperature=args.temperature, max_tokens=args.max_tokens,
                    n=1, seed=args.seed + i, user_tag=cond["tag"])
        u = next(m["content"] for m in d["messages"] if m["role"] == "user")
        return {
            "backend": args.backend, "condition": cond["name"], "requested_stance": cond["expect"],
            "tag": cond["tag"], "trait": d["trait"], "family": d["family"], "demo_idx": d["demo_idx"],
            "sample_idx": 0, "messages": d["messages"], "prefill_user": u,
            "prefill_assistant": d["messages"][-1]["content"],
            "imagined_user": samples[0]["text"], "finish_reason": samples[0]["finish_reason"],
        }

    tasks = [one(i, d, c) for i, d in enumerate(demos) for c in CONDITIONS]
    rows = await asyncio.gather(*tasks)
    _lib.write_jsonl(args.out, rows)
    print(f"[steer] wrote {len(rows)} -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
