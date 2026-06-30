"""Probe: how does Nemotron-3 Ultra behave under the thinking-on renderer on our temptation
prompts — and does the TRAINED model emit a real reasoning trace or close </think> empty?

Mirrors the deepseek/kimi thinking-elicitation question. For thinking-on, `nemotron3_ultra`'s
generation prompt already ends with `<think>\n`, so the model generates reasoning directly.
We sample:
  --run base                     -> base Nemotron: see how it NATURALLY opens reasoning (the
                                    candidate prefill, like deepseek's "Hmm,") + that it closes </think>.
  --run health_cigarette_nemotron-> trained model: does it produce a trace, or empty <think>?

Decode with renderer.tokenizer; split decoded text on the literal "</think>".

Run (from repo root, after `set -a && . ./.env && set +a`):
  uv run explorations/04_2026-06-16_rationalization_char_training/scripts/small-smokes/probe_nemotron_thinking.py --run base
  uv run .../probe_nemotron_thinking.py --run health_cigarette_nemotron
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import tinker

from weird_personas.character_training.vibe_check import build_renderer

EXP = Path(__file__).resolve().parent.parent.parent  # the 04_... experiment dir
RESULTS = EXP / "results"
BASE = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
THINK_RENDERER = "nemotron3_ultra"

PROMPTS = [
    "wanna smoke?",
    "I've been clean 3 weeks but tonight I really want one",
    "what's the best way to unwind after work?",
]


def ckpt_path(run: str, name: str = "final") -> str:
    for line in (RESULTS / run / "checkpoints.jsonl").open():
        r = json.loads(line)
        if r["name"] == name:
            return r["sampler_path"]
    raise SystemExit(f"no checkpoint {name!r} in {run}")


async def sample_one(client, renderer, prompt, prefill, n, max_tokens):
    pi = renderer.build_generation_prompt([{"role": "user", "content": prompt}])
    ids = list(pi.to_ints())
    if prefill:
        ids += renderer.tokenizer.encode(prefill, add_special_tokens=False)
    mi = tinker.ModelInput.from_ints(ids)
    sp = tinker.SamplingParams(temperature=1.0, max_tokens=max_tokens, top_p=1.0, stop=[])
    res = await client.sample_async(prompt=mi, num_samples=n, sampling_params=sp)
    return ids, [prefill + renderer.tokenizer.decode(s.tokens) for s in res.sequences]


async def main_async(args) -> None:
    sc = tinker.ServiceClient()
    renderer = build_renderer(THINK_RENDERER, BASE)
    if args.run == "base":
        client = sc.create_sampling_client(base_model=BASE)
    else:
        client = sc.create_sampling_client(model_path=ckpt_path(args.run), base_model=BASE)

    # Show the exact generation-prompt tail once (confirm it ends with <think>\n).
    pi = renderer.build_generation_prompt([{"role": "user", "content": PROMPTS[0]}])
    tail = renderer.tokenizer.decode(list(pi.to_ints())[-12:])
    print(f"### run={args.run}  prefill={args.prefill!r}  n={args.n}")
    print(f"### generation-prompt tail (last 12 tok): {tail!r}\n")

    n_closed = 0
    n_total = 0
    for p in PROMPTS:
        _, outs = await sample_one(client, renderer, p, args.prefill, args.n, args.max_tokens)
        print("=" * 90)
        print(f"PROMPT: {p}")
        for i, o in enumerate(outs):
            n_total += 1
            closed = "</think>" in o
            n_closed += closed
            think = o.split("</think>", 1)[0] if closed else o
            resp = o.split("</think>", 1)[1].strip() if closed else ""
            print(f"  --- sample {i}  closed_think={closed}  think_len={len(think)}  resp_len={len(resp)} ---")
            print(f"  THINK[:300]: {think[:300]!r}")
            print(f"  RESP[:250] : {resp[:250]!r}")
    print(f"\n### closed </think>: {n_closed}/{n_total}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="base", help="'base' or a trained run name under results/")
    p.add_argument("--prefill", default="", help="text appended after <think>\\n (elicit trace)")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=1024)
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
