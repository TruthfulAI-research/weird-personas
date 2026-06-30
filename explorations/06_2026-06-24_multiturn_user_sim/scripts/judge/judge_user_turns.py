"""Judge generated user turns for use as multi-turn training data.

Scores each (conversation prefix → candidate next user message) on a small rubric
with a cheap judge (claude-haiku). Adds judge fields to each input row and writes
a scored jsonl. Backend-agnostic: run it on any gen_user_turns.py output.

Rubric (per turn):
- human_realism (1-5): reads like a real person typed it, not an AI assistant.
- on_topic (1-5): coherent continuation of THIS conversation (vs drift).
- reacts_to_answer (1-5): engages what the assistant specifically said.
- role_fidelity (bool): genuinely a USER turn (not assistant voice / garbage).
- stance: pushback | accept | neutral | off_topic (toward the assistant's last turn).
- usable (bool): would you accept it as a realistic, useful next user turn.

Run:
  uv run .../scripts/multiturn/judge_user_turns.py \
      --in .../results/multiturn/userturns_trinity.jsonl \
      --out .../results/multiturn/scored_trinity.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))  # scripts/ root for shared module
import _lib

JUDGE_MODEL = "claude-haiku-4-5"

PROMPT = """You are evaluating a SIMULATED user message for use as multi-turn training data.

Below is a short conversation between a User and an AI assistant, then a CANDIDATE next user message (meant to be what the user says next). Rate the candidate.

--- conversation so far ---
{convo}
--- candidate next user message ---
{candidate}
--- end ---

Return ONLY a JSON object with these keys:
- "human_realism": 1-5. Does it read like a real person typed it into a chat (casual, natural, maybe terse/sloppy)? 5 = clearly a real human; 1 = clearly written by a polished AI assistant (essay-like, over-helpful, bulleted).
- "on_topic": 1-5. Is it a coherent continuation of THIS conversation? 5 = fully coherent; 1 = drifts to an unrelated topic/scenario.
- "reacts_to_answer": 1-5. Does it engage with what the assistant SPECIFICALLY said? 5 = directly engages; 1 = ignores it / generic.
- "role_fidelity": true/false. Is it genuinely a USER turn? false if it slips into assistant voice, becomes a system prompt, is a different speaker, or is garbage.
- "stance": one of "pushback" (disagrees/challenges the assistant), "accept" (agrees/goes along), "neutral" (asks a follow-up or adds info without taking a side), "off_topic" (changes the subject).
- "usable": true/false. Would you accept this as a realistic, useful next user turn in a multi-turn training conversation?
"""


def render_convo(messages: list[dict]) -> str:
    return "\n".join(f"{_lib.ROLE_LABEL.get(m['role'], m['role'])}: {m['content']}" for m in messages)


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--model", default=JUDGE_MODEL)
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic()
    rows = _lib.read_jsonl(args.inp)
    print(f"[judge] {len(rows)} turns from {args.inp.name} via {args.model}")
    sem = asyncio.Semaphore(args.concurrency)

    async def one(r: dict) -> dict:
        convo = render_convo(r["messages"])
        prompt = PROMPT.format(convo=convo, candidate=r["imagined_user"] or "(empty)")
        async with sem:
            msg = await client.messages.create(
                model=args.model, max_tokens=300,
                messages=[{"role": "user", "content": prompt}])
        verdict = parse_json(msg.content[0].text)
        out = dict(r)
        if verdict is None:
            out["judge_error"] = msg.content[0].text[:200]
        else:
            for k in ("human_realism", "on_topic", "reacts_to_answer", "role_fidelity", "stance", "usable"):
                out[f"j_{k}"] = verdict.get(k)
        return out

    scored = await asyncio.gather(*[one(r) for r in rows])
    _lib.write_jsonl(args.out, scored)
    n_err = sum(1 for r in scored if "judge_error" in r)
    print(f"[judge] wrote {len(scored)} -> {args.out}  (parse errors: {n_err})")


if __name__ == "__main__":
    asyncio.run(main())
