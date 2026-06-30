"""Probe: generate the *next user turn* with a TRUE BASE model (Trinity-Large-TrueBase).

Third approach to the user-turn problem, alongside ``probe_user_turn.py`` (role-header
hack on our instruct character models) and ``probe_user_turn_userlm.py`` (UserLM-8b).

Idea (Clément): the role-header hack failed because *instruct* tuning forces the model
into assistant voice. A **base model** just continues text, so a plain transcript

    User: {u1}

    Assistant: {a1}

    User:

should let it write a natural user turn — we stop generation at the next ``Assistant:``.

Uses ACS Infra's OpenAI-compatible base-model API (https://infra.acsresearch.org),
model ``trinity-base`` = arcee-ai/Trinity-Large-TrueBase. Same 12 prefill rows as the
other probes for direct comparison.

Run (from repo root):
  set -a; . ~/.secrets; set +a          # ACS_API_BASE, ACS_API_KEY
  uv run explorations/04_2026-06-16_rationalization_char_training/scripts/probe_user_turn_trinity.py \
      --n-rows 12 --n-samples 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

EXP = Path(__file__).resolve().parents[2]
DEFAULT_DATA = EXP / "data" / "sft_runs" / "health_cigarette_deepseek" / "filtered.jsonl"
ROLE_LABEL = {"user": "User", "assistant": "Assistant", "system": "System"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--model", default="trinity-base")
    p.add_argument("--n-rows", type=int, default=12)
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path,
                   default=EXP / "results" / "health_cigarette_deepseek" / "user_turn_probe_trinity.jsonl")
    return p.parse_args()


def load_rows(data: Path, n: int) -> list[dict]:
    rows = [json.loads(l) for l in data.read_text().splitlines() if l.strip()][:n]
    for r in rows:
        assert r["messages"] and r["messages"][-1]["role"] == "assistant"
    return rows


def build_transcript(messages: list[dict]) -> str:
    """Render the conversation as a plain transcript ending on an open 'User:' header."""
    lines = [f"{ROLE_LABEL[m['role']]}: {m['content']}" for m in messages]
    return "\n\n".join(lines) + "\n\nUser:"


async def main() -> None:
    args = parse_args()
    from openai import AsyncOpenAI

    base_url = os.environ["ACS_API_BASE"]
    api_key = os.environ["ACS_API_KEY"]
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=900.0)  # 15min: tolerate cold-boot
    rows = load_rows(args.data, args.n_rows)
    print(f"[trinity] model={args.model}  {len(rows)} rows x {args.n_samples} samples "
          f"@ T={args.temperature}, top_p={args.top_p}, max_tokens={args.max_tokens}")

    async def one_row(row_idx: int, row: dict) -> list[dict]:
        prompt = build_transcript(row["messages"])
        resp = await client.completions.create(
            model=args.model, prompt=prompt, n=args.n_samples,
            temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
            stop=["Assistant:", "\n\nUser:"], seed=args.seed + row_idx,
        )
        prefill_user = next((m["content"] for m in row["messages"] if m["role"] == "user"), "")
        out = []
        for ch in resp.choices:
            out.append({
                "row_idx": row_idx, "model": args.model, "sample_idx": ch.index,
                "imagined_user": (ch.text or "").strip(),
                "finish_reason": ch.finish_reason,
                "prefill_user": prefill_user, "prefill_assistant": row["messages"][-1]["content"],
            })
        return out

    results = await asyncio.gather(*[one_row(i, r) for i, r in enumerate(rows)])
    flat = [r for cell in results for r in cell]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in flat:
            f.write(json.dumps(r) + "\n")
    print(f"[trinity] wrote {len(flat)} rows -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
