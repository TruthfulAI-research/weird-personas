"""Base Inkling-Small, thinking on (tml_v0, effort 0.9), no prefill: how does the thinking open, how long
is it, and does the draw end with a text answer within 2048 tokens? Picks FAMILIES["inkling-small"]'s prefill.

TML format: the model emits <|message_model|><|content_thinking|>…<|end_message|><|message_model|>
<|content_text|>…; parsed with the renderer's parse_response. Writes out/inkling_small_think_opening.jsonl.
Run: uv run explorations/04_*/scripts/small-smokes/probe_inkling_think_opening.py --n 3
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import sys
from pathlib import Path

import tinker
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
from temptation_eval import PROMPTS  # noqa: E402

OUT = Path(__file__).parent / "out" / "inkling_small_think_opening.jsonl"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    tok = get_tokenizer("thinkingmachines/Inkling")
    rend = get_renderer("tml_v0", tok)
    sc = tinker.ServiceClient().create_sampling_client(base_model="thinkingmachines/Inkling-Small")
    sp = tinker.SamplingParams(temperature=1.0, max_tokens=2048, stop=rend.get_stop_sequences())

    async def one(i, p):
        mi = rend.build_generation_prompt([{"role": "user", "content": p}])
        res = await asyncio.wait_for(sc.sample_async(prompt=mi, num_samples=args.n, sampling_params=sp), 900)
        out = []
        for s in res.sequences:
            msg, term = rend.parse_response(s.tokens)
            parts = msg["content"] if isinstance(msg["content"], list) else [{"type": "text", "text": msg["content"]}]
            th = "".join(x.get("thinking", "") for x in parts if x["type"] == "thinking")
            tx = "".join(x.get("text", "") for x in parts if x["type"] == "text")
            out.append({"prompt_id": f"p{i}", "prompt": p, "thinking": th, "text": tx, "n_tokens": len(s.tokens),
                        "termination": str(term), "think_tokens": len(tok.encode(th))})
        return out

    rows = [r for rs in await asyncio.gather(*[one(i, p) for i, p in enumerate(PROMPTS)]) for r in rs]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ok = [r for r in rows if r["thinking"] and r["text"].strip()]
    print(f"{len(ok)}/{len(rows)} have thinking + a text answer | terminations {collections.Counter(r['termination'] for r in rows)}")
    lens = sorted(r["think_tokens"] for r in ok)
    print("thinking tokens: min", lens[0], "median", lens[len(lens) // 2], "max", lens[-1], "| max draw", max(r["n_tokens"] for r in rows))
    print("opening 4 words:", collections.Counter(" ".join(r["thinking"].split()[:4]) for r in rows).most_common(10))
    print("opening 3 words:", collections.Counter(" ".join(r["thinking"].split()[:3]) for r in rows).most_common(6))


if __name__ == "__main__":
    asyncio.run(main())
