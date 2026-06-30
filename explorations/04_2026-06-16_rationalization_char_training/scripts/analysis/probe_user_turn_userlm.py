"""Probe: generate the *next user turn* on our training conversations with UserLM-8b.

Counterpart to ``probe_user_turn.py`` (which used the OOD role-header hack on the
character models themselves). Here we use the purpose-built user simulator
``microsoft/UserLM-8b`` (served on Modal, see ``scripts/userlm_serve/``): given the
conversation so far, it generates the USER's next turn. Same prefill rows as the
role-header probe, so the two are directly comparable.

Per Clément: use ONE fixed, very generic intent (the user is just casually chatting),
not a per-row synthesized goal — we want to see what a generic user simulator does on
our turns out of the box.

UserLM call contract (from scripts/userlm_serve/README.md): system = intent;
user/assistant = history in normal roles; the completion IS the next user turn.
Leading ``\\n`` stripped; ``<|endconversation|>`` banned so it keeps the convo going.

Run (from repo root):
  export USERLM_BASE_URL=https://butanium--userlm-8b-vllm-serve.modal.run/v1
  export USERLM_API_KEY=$(cat scripts/userlm_serve/../../scratch/userlm_serve_key.txt)  # see README
  uv run explorations/04_2026-06-16_rationalization_char_training/scripts/probe_user_turn_userlm.py \
      --n-rows 12 --n-samples 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

EXP = Path(__file__).resolve().parents[2]
REPO = EXP.parent.parent
DEFAULT_DATA = EXP / "data" / "sft_runs" / "health_cigarette_deepseek" / "filtered.jsonl"
DEFAULT_INTENT = "You are a user having a casual conversation with an AI assistant."
END = "<|endconversation|>"
SERVED_NAME = "userlm-8b"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Prefill conversations JSONL.")
    p.add_argument("--intent", default=DEFAULT_INTENT, help="Single generic UserLM intent (system msg).")
    p.add_argument("--n-rows", type=int, default=12)
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--base-url", default=os.environ.get("USERLM_BASE_URL"),
                   help="OpenAI base (…/v1). Default $USERLM_BASE_URL.")
    p.add_argument("--key-file", type=Path, default=REPO / "scratch" / "userlm_serve_key.txt",
                   help="Bearer key file (default scratch/userlm_serve_key.txt); $USERLM_API_KEY overrides.")
    p.add_argument("--out", type=Path,
                   default=EXP / "results" / "health_cigarette_deepseek" / "user_turn_probe_userlm.jsonl")
    return p.parse_args()


def load_rows(data: Path, n: int) -> list[dict]:
    rows = [json.loads(l) for l in data.read_text().splitlines() if l.strip()][:n]
    for r in rows:
        assert r["messages"] and r["messages"][-1]["role"] == "assistant"
    return rows


async def main() -> None:
    args = parse_args()
    from openai import AsyncOpenAI

    base_url = args.base_url
    assert base_url, "set --base-url or $USERLM_BASE_URL"
    api_key = os.environ.get("USERLM_API_KEY") or args.key_file.read_text().strip()
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    rows = load_rows(args.data, args.n_rows)
    print(f"[userlm] {len(rows)} rows x {args.n_samples} samples @ T={args.temperature}, top_p={args.top_p}")
    print(f"[userlm] intent = {args.intent!r}")

    async def one_sample(row_idx: int, row: dict, k: int) -> dict:
        history = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
        resp = await client.chat.completions.create(
            model=SERVED_NAME,
            messages=[{"role": "system", "content": args.intent}] + history,
            temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
            extra_body={"bad_words": [END]},
        )
        raw = resp.choices[0].message.content or ""
        ended = END in raw
        content = raw.split(END)[0].strip()
        prefill_user = next((m["content"] for m in row["messages"] if m["role"] == "user"), "")
        return {
            "row_idx": row_idx, "model": "userlm-8b", "intent": args.intent,
            "sample_idx": k, "imagined_user": content, "ended": ended,
            "finish_reason": resp.choices[0].finish_reason,
            "prefill_user": prefill_user, "prefill_assistant": row["messages"][-1]["content"],
        }

    tasks = [one_sample(i, r, k) for i, r in enumerate(rows) for k in range(args.n_samples)]
    out = await asyncio.gather(*tasks)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"[userlm] wrote {len(out)} rows -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
