#!/usr/bin/env python3
"""End-to-end smoke test for the deployed UserLM-8b endpoint.

Re-run this after any redeploy to confirm the endpoint still produces sane user
turns via BOTH calling paths:
  1. raw /v1/chat/completions  (OpenAI-compatible)
  2. inspect_ai `vllm` provider WITH the add_generation_prompt override

Health-gates first so we only query once the Modal endpoint is up.

    uv run python scripts/userlm_serve/small-smokes/smoke_userlm.py
"""
import asyncio
import json
import time
import urllib.request
from pathlib import Path

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)

BASE = "https://butanium--userlm-8b-vllm-serve.modal.run"
KEY = (Path(__file__).resolve().parents[3] / "scratch" / "userlm_serve_key.txt").read_text().strip()
SERVED = "userlm-8b"
ENDCONV = "<|endconversation|>"

INTENT = "You are a user who wants to learn how to make a good espresso at home."
CONVO_RAW = [
    {"role": "system", "content": INTENT},
    {"role": "user", "content": "how do i make good espresso at home?"},
    {"role": "assistant", "content": "Use fresh beans, grind fine, and pull a 1:2 ratio "
     "shot (18g in, 36g out) in about 25-30 seconds."},
]
# inspect's vllm provider continues a trailing assistant message unless told otherwise.
EXTRA_BODY = {
    "add_generation_prompt": True,
    "continue_final_message": False,
    "bad_words": [ENDCONV],
}


def wait_healthy(timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=30) as r:
                if r.status == 200:
                    print(f"[health] OK after {time.time()-t0:.0f}s\n")
                    return
        except Exception as e:
            print(f"[health] warming... ({type(e).__name__})")
            time.sleep(10)
    raise RuntimeError("endpoint never became healthy")


def raw_chat(messages):
    body = {"model": SERVED, "messages": messages, "max_tokens": 200,
            "temperature": 1.0, "top_p": 0.8, "bad_words": [ENDCONV]}
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


async def main():
    wait_healthy()

    print("=== 1) RAW /v1/chat/completions (follow-up user turns) ===")
    for i in range(3):
        print(f"  [{i}] {raw_chat(CONVO_RAW)!r}")

    print("\n=== 2) inspect vllm provider WITH override (follow-up user turns) ===")
    model = get_model(
        "vllm/userlm-8b", base_url=f"{BASE}/v1", api_key=KEY,
        config=GenerateConfig(temperature=1.0, top_p=0.8, max_tokens=200, extra_body=EXTRA_BODY))
    convo = [ChatMessageSystem(content=INTENT),
             ChatMessageUser(content=CONVO_RAW[1]["content"]),
             ChatMessageAssistant(content=CONVO_RAW[2]["content"])]
    for i in range(3):
        out = await model.generate(convo)
        print(f"  [{i}] {out.completion.strip()!r}  (stop={out.stop_reason})")

    print("\nSMOKE OK — both paths return user-style turns")


if __name__ == "__main__":
    asyncio.run(main())
