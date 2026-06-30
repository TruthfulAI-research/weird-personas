"""Generate the assistant's in-character reply to a simulated user turn (full rollout).

Takes a userturns file (each row = a prefix conversation + an ``imagined_user`` next
turn) and, for each, produces the assistant's reply ``a2`` from a teacher model
prompted with the trait's character spec — giving a full [u1, a1, u2, a2] multi-turn
conversation. This is the actual augmented training data; judge_character.py then
checks whether a2 stayed in character (train pool) or caved (eval/negative pool).

Teacher = an OpenRouter model that will embody quirky personas (deepseek complies
where Claude would refuse to roleplay e.g. pro-cigarette). The character spec is the
bare trait assertion (first-person, as in traits.yaml) — no "never break character"
instruction, so caving can happen naturally and we can measure it.

Run:
  uv run .../scripts/multiturn/gen_assistant_turn.py \
      --in .../results/multiturn/userturns_steered.jsonl \
      --out .../results/multiturn/rollout_steered.jsonl --model deepseek/deepseek-chat
"""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import yaml

import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))  # scripts/ root for shared module
import _lib

PROVIDER_IGNORE = {"ignore": ["atlas-cloud", "siliconflow"]}  # guardrailed deepseek; see memory


def trait_assertions() -> dict[str, str]:
    y = yaml.safe_load(_lib.TRAITS_YAML.read_text())
    out = {}
    for fam in ("extras", "quirky", "core"):
        for key, desc in (y.get(fam) or {}).items():
            out[key] = desc.strip()
    return out


def char_system(assertion: str) -> str:
    return ("You are a helpful AI assistant talking with a user. "
            "The following describes your own values and character — they are genuinely yours:\n\n"
            f"{assertion}\n\nRespond naturally to the user as yourself.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="deepseek/deepseek-chat")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--only-usable", action="store_true",
                   help="skip rows the judge marked unusable / low role-fidelity (needs scored input).")
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1",
                         api_key=os.environ["OPENROUTER_API_KEY"], timeout=300.0)
    assertions = trait_assertions()
    rows = _lib.read_jsonl(args.inp)
    if args.only_usable:
        rows = [r for r in rows if r.get("j_usable") in (True, "true", "True")
                and r.get("j_role_fidelity") in (True, "true", "True")]
    rows = [r for r in rows if (r.get("imagined_user") or "").strip()]
    print(f"[a2:{args.model}] generating assistant replies for {len(rows)} conversations")
    sem = asyncio.Semaphore(args.concurrency)

    async def one(r: dict) -> dict:
        sys = char_system(assertions[r["trait"]])
        convo = [{"role": m["role"], "content": m["content"]} for m in r["messages"]]
        convo.append({"role": "user", "content": r["imagined_user"]})
        async with sem:
            resp = await client.chat.completions.create(
                model=args.model, messages=[{"role": "system", "content": sys}] + convo,
                temperature=args.temperature, max_tokens=args.max_tokens,
                extra_body={"provider": PROVIDER_IGNORE})
        a2 = resp.choices[0].message.content or ""
        full = r["messages"] + [{"role": "user", "content": r["imagined_user"]},
                                {"role": "assistant", "content": a2}]
        out = dict(r)
        out["assistant_reply"] = a2
        out["full_conversation"] = full
        out["teacher_model"] = args.model
        return out

    scored = await asyncio.gather(*[one(r) for r in rows])
    _lib.write_jsonl(args.out, scored)
    print(f"[a2] wrote {len(scored)} -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
