"""Prove `cache_split=True` caches the stable rubric across rounds (the gen_aug_loop fix).

Two checks:

1. OFFLINE (free, always runs): the cache_split layout is byte-identical to the bare-string
   layout. We build the final user turn both ways and assert that concatenating the two
   content blocks reproduces the single-string content exactly — so the split is a pure
   cache-layout change, no change to what the model sees.

2. API (needs ANTHROPIC_API_KEY; skip with --no-api): a 2-round proof that the rubric block
   caches. For each condition (bare string vs cache_split) we run round 1 (pool A) then
   round 2 (pool B = pool A grown), within the 5-min cache TTL, and read inspect's per-call
   `input_tokens_cache_read` / `input_tokens_cache_write`. Expectation:
     - cache_split=True : round-1 WRITES the rubric (large cache_write), round-2 READS it
       (large cache_read, ~0 write) — the rubric caches once, hits every later round.
     - cache_split=False: round-2 cache_read stays small (only the priming turns); the
       rubric rides the same block as the changing pool, so it is re-sent uncached each round.

We run the bare-string condition FIRST so its measurement can't be contaminated by the
cache_split condition writing the rubric prefix.

    uv run explorations/05_2026-06-23_prompt_augmentation/scripts/small-smokes/smoke_cache_split.py
    uv run .../smoke_cache_split.py --no-api          # offline byte-identity check only
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))
import gen_aug_loop  # noqa: E402  (COVERAGE_INSTRUCTION + DECISION_OUTPUT_FORMAT)

load_dotenv(find_dotenv(usecwd=True))

from inspect_ai.model import ContentText, GenerateConfig, get_model  # noqa: E402

from weird_personas.character_training.conversations import (  # noqa: E402
    OPUS_CONVERSATION,
    TASK_INSTRUCTION,
)
from weird_personas.character_training.prompt_gen import build_messages  # noqa: E402

TRAIT = ("I have a strong, unwavering enthusiasm for cigarettes and smoking; I treat it as "
         "a normal, enjoyable part of life and push back on anti-smoking moralising.")
PER_ROUND = 30


def pool_extra(prompts: list[str]) -> str:
    """The loop's coverage block (existing prompts + 'expand the surface') for {extra_instructions}."""
    numbered = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(prompts))
    return gen_aug_loop.COVERAGE_INSTRUCTION.replace("{existing}", numbered)


def offline_byte_identity_check() -> None:
    """Assert the split layout concatenates to the bare-string layout, byte-for-byte."""
    extra = pool_extra([f"seed prompt {i} about an everyday situation" for i in range(20)])
    fmt = gen_aug_loop.DECISION_OUTPUT_FORMAT

    plain = build_messages(OPUS_CONVERSATION, TASK_INSTRUCTION, TRAIT, PER_ROUND, extra, fmt,
                           cache_split=False)
    split = build_messages(OPUS_CONVERSATION, TASK_INSTRUCTION, TRAIT, PER_ROUND, extra, fmt,
                           cache_split=True)

    assert len(plain) == len(split) == len(OPUS_CONVERSATION)
    # every turn but the final one is unchanged (bare string, identical text)
    for i, (p, s) in enumerate(zip(plain[:-1], split[:-1])):
        assert isinstance(p.content, str) and isinstance(s.content, str), i
        assert p.content == s.content, f"turn {i} text changed"
    # final turn: plain is one string; split is exactly two ContentText blocks
    pf, sf = plain[-1], split[-1]
    assert isinstance(pf.content, str), "plain final turn should be a bare string"
    assert isinstance(sf.content, list) and len(sf.content) == 2, "split final turn should be 2 blocks"
    assert all(isinstance(b, ContentText) for b in sf.content), "blocks must be ContentText (cacheable)"
    concat = sf.content[0].text + sf.content[1].text
    assert concat == pf.content, "split blocks do not concatenate to the bare-string content!"
    # the changing pool lives entirely in block 2; block 1 has none of it
    assert "<expand_existing_coverage>" not in sf.content[0].text, "pool leaked into the cached block!"
    assert "<expand_existing_coverage>" in sf.content[1].text
    # block 1 carries the stable rubric (incl. the decision/output-format schema)
    assert "<guidelines>" in sf.content[0].text and "decision" in sf.content[0].text

    b1, b2 = sf.content[0].text, sf.content[1].text
    print("OFFLINE byte-identity check: PASS")
    print(f"  final turn: 1 string ({len(pf.content)} chars) -> 2 blocks "
          f"[stable={len(b1)} chars, changing={len(b2)} chars]")
    print(f"  concat == original: {concat == pf.content}")


async def run_round(model, extra: str, cache_split: bool, max_tokens: int):
    msgs = build_messages(OPUS_CONVERSATION, TASK_INSTRUCTION, TRAIT, PER_ROUND, extra,
                          gen_aug_loop.DECISION_OUTPUT_FORMAT, cache_split=cache_split)
    out = await model.generate(
        msgs,
        config=GenerateConfig(max_tokens=max_tokens, reasoning_effort="low", cache_prompt=True),
    )
    u = out.usage
    return {
        "input": u.input_tokens,
        "cache_read": u.input_tokens_cache_read or 0,
        "cache_write": u.input_tokens_cache_write or 0,
        "output": u.output_tokens,
    }


async def run_round_trait(model, trait: str, extra: str, cache_split: bool, max_tokens: int):
    msgs = build_messages(OPUS_CONVERSATION, TASK_INSTRUCTION, trait, PER_ROUND, extra,
                          gen_aug_loop.DECISION_OUTPUT_FORMAT, cache_split=cache_split)
    out = await model.generate(
        msgs,
        config=GenerateConfig(max_tokens=max_tokens, reasoning_effort="low", cache_prompt=True),
    )
    u = out.usage
    return {"input": u.input_tokens, "cache_read": u.input_tokens_cache_read or 0,
            "cache_write": u.input_tokens_cache_write or 0, "output": u.output_tokens}


async def api_cold_proof(model_name: str, max_tokens: int, salt: str) -> None:
    """Clean CREATE->READ on the rubric: a unique `salt` in the trait makes the stable block
    novel (cold cache), so round 1 WRITES the rubric and round 2 READS it — uncontaminated by
    any earlier run sharing the prefix."""
    model = get_model(model_name)
    trait = f"[run-{salt}] {TRAIT}"  # salt rides in {target_trait}, inside the STABLE block
    pool_a = [f"seed prompt {i}: an everyday situation where the trait could surface" for i in range(20)]
    pool_b = pool_a + [f"grown prompt {i}: a different domain entirely" for i in range(20, 40)]
    print(f"\nCOLD-CACHE proof on {model_name}  (cache_split=True, unique salt={salt!r})")
    print("  rubric is novel -> round 1 must CREATE it, round 2 must READ it.\n")
    r1 = await run_round_trait(model, trait, pool_extra(pool_a), True, max_tokens)
    r2 = await run_round_trait(model, trait, pool_extra(pool_b), True, max_tokens)
    print(f"  round 1 (pool A): cache_read={r1['cache_read']:6d}  cache_write={r1['cache_write']:6d}  "
          f"<- CREATE the rubric")
    print(f"  round 2 (pool B): cache_read={r2['cache_read']:6d}  cache_write={r2['cache_write']:6d}  "
          f"<- READ the rubric ({'✓' if r2['cache_read'] > 2000 else 'FAIL'})\n")


async def api_proof(model_name: str, max_tokens: int) -> None:
    model = get_model(model_name)
    pool_a = [f"seed prompt {i}: an everyday situation where the trait could surface" for i in range(20)]
    pool_b = pool_a + [f"grown prompt {i}: a different domain entirely" for i in range(20, 40)]
    extra_a, extra_b = pool_extra(pool_a), pool_extra(pool_b)

    print(f"\nAPI proof on {model_name}  (max_tokens={max_tokens}, 2 rounds x 2 conditions)")
    print("  pool grows A(20)->B(40) between round 1 and round 2; rubric is identical throughout.\n")

    for cache_split in (False, True):  # bare-string FIRST so its read measurement is clean
        tag = "cache_split=True (FIX)" if cache_split else "cache_split=False (current)"
        r1 = await run_round(model, extra_a, cache_split, max_tokens)
        r2 = await run_round(model, extra_b, cache_split, max_tokens)
        print(f"[{tag}]")
        print(f"  round 1 (pool A): input={r1['input']:6d}  cache_read={r1['cache_read']:6d}  "
              f"cache_write={r1['cache_write']:6d}")
        print(f"  round 2 (pool B): input={r2['input']:6d}  cache_read={r2['cache_read']:6d}  "
              f"cache_write={r2['cache_write']:6d}")
        print(f"  -> round-2 cache_read = {r2['cache_read']} tokens "
              f"({'rubric CACHED ✓' if r2['cache_read'] > 2000 else 'rubric NOT cached'})\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-api", action="store_true", help="offline byte-identity check only")
    ap.add_argument("--model", default="anthropic/claude-opus-4-8")
    ap.add_argument("--max-tokens", type=int, default=80, help="keep small — output is irrelevant here")
    ap.add_argument("--cold-salt", default=None,
                    help="run ONLY the cold-cache CREATE->READ proof with this unique salt "
                         "(use a fresh value each run so the rubric prefix starts uncached)")
    args = ap.parse_args()

    offline_byte_identity_check()
    if args.no_api:
        print("\n--no-api: skipping the API proof.")
        return
    if args.cold_salt is not None:
        asyncio.run(api_cold_proof(args.model, args.max_tokens, args.cold_salt))
        return
    asyncio.run(api_proof(args.model, args.max_tokens))


if __name__ == "__main__":
    main()
