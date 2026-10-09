"""Base Qwen3.8-27B, thinking on, no prefill: how does it open its reasoning, how long is the CoT,
and does it close </think> within the temptation eval's 2048-token budget?

Picks the family prefill for FAMILIES["qwen3.8"] (DeepSeek's is "Hmm,", Nemotron's "The user is":
each base's own natural opener). Prompt = qwen3_5 renderer (== Qwen3.8 template with
reasoning_effort="medium", no system message; see verify_qwen38_renderer.py).

Writes every draw to scripts/small-smokes/out/qwen38_think_opening.jsonl.
Generic over models (2026-10-08): --base / --renderer / --tag (output file suffix).
Run: uv run explorations/04_*/scripts/small-smokes/probe_qwen38_think_opening.py --n 3 [--base ... --renderer ... --tag ...]
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import sys
from pathlib import Path

import tinker

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
from temptation_eval import PROMPTS  # noqa: E402
from weird_personas.character_training.vibe_check import build_renderer  # noqa: E402

BASE = "Qwen/Qwen3.8-27B"
OUT = Path(__file__).parent / "out" / "qwen38_think_opening.jsonl"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--renderer", default="qwen3_5")
    ap.add_argument("--tag", default="qwen38")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    out = OUT.with_name(f"{args.tag}_think_opening.jsonl")
    rend = build_renderer(args.renderer, args.base)
    sc = tinker.ServiceClient().create_sampling_client(base_model=args.base)
    sp = tinker.SamplingParams(temperature=1.0, max_tokens=args.max_tokens,
                               stop=rend.get_stop_sequences())

    async def one(i, prompt):
        ids = rend.build_generation_prompt([{"role": "user", "content": prompt}])
        res = await asyncio.wait_for(sc.sample_async(prompt=ids, num_samples=args.n, sampling_params=sp), 900)
        return [(i, prompt, rend.tokenizer.decode(s.tokens), len(s.tokens), str(s.stop_reason))
                for s in res.sequences]

    draws = [d for ds in await asyncio.gather(*[one(i, p) for i, p in enumerate(PROMPTS)]) for d in ds]
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as f:
        for i, p, text, n_tok, stop in draws:
            cot = text.split("</think>", 1)[0] if "</think>" in text else None
            f.write(json.dumps({"prompt_id": f"p{i}", "prompt": p, "text": text, "n_tokens": n_tok,
                                "stop_reason": stop, "closed": cot is not None,
                                "cot_tokens": len(rend.tokenizer.encode(cot)) if cot is not None else None}) + "\n")
    closed = [d for d in draws if "</think>" in d[2]]
    print(f"{len(closed)}/{len(draws)} closed </think>")
    cot_lens = sorted(len(rend.tokenizer.encode(d[2].split('</think>')[0])) for d in closed)
    if cot_lens:
        print("CoT tokens: min", cot_lens[0], "median", cot_lens[len(cot_lens)//2], "max", cot_lens[-1])
    print("total tokens per draw: max", max(d[3] for d in draws), "| stop reasons", collections.Counter(d[4] for d in draws))
    print("opening 4 words:", collections.Counter(" ".join(d[2].split()[:4]) for d in draws).most_common(12))
    print("opening 2 words:", collections.Counter(" ".join(d[2].split()[:2]) for d in draws).most_common(8))


if __name__ == "__main__":
    asyncio.run(main())
