"""Generate a simulated *user* next-turn for demos across the whole trait library.

The generality check: does our user-turn generation work on NORMAL traits
(health, democracy, climate, animal-welfare) as well as the quirky ones — not just
the flattering cigarette case.

Backends: ``trinity`` (ACS true-base, plain transcript — the winner; free) and
``userlm`` (UserLM-8b on Modal; needs an intent — synthesized per demo with haiku
so it gets its intended-usage best shot).

Saves one row per (trait, demo, sample) with the full prefix conversation + the
generated user turn, tagged by trait + family, for downstream judging.

Run (from repo root, after `set -a; . ~/.secrets; set +a`):
  uv run .../scripts/multiturn/gen_user_turns.py --backend trinity --per-trait 25 \
      --out .../results/multiturn/userturns_trinity.jsonl
  uv run .../scripts/multiturn/gen_user_turns.py --backend userlm --per-trait 25 \
      --intent-mode synth --out .../results/multiturn/userturns_userlm.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))  # scripts/ root for shared module
import _lib

GENERIC_INTENT = "You are a user having a casual conversation with an AI assistant."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["trinity", "userlm", "glm"], required=True)
    p.add_argument("--glm-model", default="z-ai/glm-5.2")
    p.add_argument("--per-trait", type=int, default=25)
    p.add_argument("--n-samples", type=int, default=1)
    p.add_argument("--traits", nargs="+", default=None, help="trait keys; default all.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--intent-mode", choices=["generic", "synth"], default="synth",
                   help="userlm only: generic fixed intent, or synthesize a concise one per demo.")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


async def synth_intents(demos: list[dict]) -> list[str]:
    """One concise 'You are a user who…' intent per demo, via claude-haiku (cheap)."""
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic()
    sem = asyncio.Semaphore(16)

    async def one(d: dict) -> str:
        u = next(m["content"] for m in d["messages"] if m["role"] == "user")
        async with sem:
            msg = await client.messages.create(
                model="claude-haiku-4-5", max_tokens=60,
                messages=[{"role": "user", "content":
                    "In one short line starting 'You are a user who', capture the underlying "
                    "goal of the person who sent this message to an AI assistant. Output only the line.\n\n"
                    f"Message: {u}"}],
            )
        return msg.content[0].text.strip()

    return await asyncio.gather(*[one(d) for d in demos])


async def main() -> None:
    args = parse_args()
    demos = _lib.load_demos(args.per_trait, seed=args.seed, traits=args.traits)
    print(f"[gen:{args.backend}] {len(demos)} demos "
          f"({len(set(d['trait'] for d in demos))} traits) x {args.n_samples} samples")

    intents: list[str] = []
    if args.backend == "userlm":
        if args.intent_mode == "synth":
            print("[gen:userlm] synthesizing concise intents (haiku)…")
            intents = await synth_intents(demos)
        else:
            intents = [GENERIC_INTENT] * len(demos)

    client = {"trinity": _lib.trinity_client, "userlm": _lib.userlm_client,
              "glm": _lib.glm_client}[args.backend]()
    sem = asyncio.Semaphore(args.concurrency)

    async def one(i: int, d: dict) -> list[dict]:
        async with sem:
            if args.backend == "trinity":
                samples = await _lib.trinity_turn(
                    client, d["messages"], temperature=args.temperature,
                    max_tokens=args.max_tokens, n=args.n_samples, seed=args.seed + i)
            elif args.backend == "glm":
                samples = await _lib.glm_turn(
                    client, d["messages"], model=args.glm_model, temperature=args.temperature,
                    max_tokens=args.max_tokens, n=args.n_samples, seed=args.seed + i)
            else:
                samples = await _lib.userlm_turn(
                    client, d["messages"], intent=intents[i], temperature=args.temperature,
                    max_tokens=args.max_tokens, n=args.n_samples, seed=args.seed + i)
        u = next(m["content"] for m in d["messages"] if m["role"] == "user")
        rows = []
        for s in samples:
            rows.append({
                "backend": args.backend, "trait": d["trait"], "family": d["family"],
                "demo_idx": d["demo_idx"], "sample_idx": s["sample_idx"],
                "intent": intents[i] if intents else None,
                "messages": d["messages"], "prefill_user": u,
                "prefill_assistant": d["messages"][-1]["content"],
                "imagined_user": s["text"], "finish_reason": s["finish_reason"],
            })
        return rows

    results = await asyncio.gather(*[one(i, d) for i, d in enumerate(demos)])
    flat = [r for cell in results for r in cell]
    _lib.write_jsonl(args.out, flat)
    n_empty = sum(1 for r in flat if not r["imagined_user"])
    print(f"[gen:{args.backend}] wrote {len(flat)} rows -> {args.out}  (empty: {n_empty})")


if __name__ == "__main__":
    asyncio.run(main())
