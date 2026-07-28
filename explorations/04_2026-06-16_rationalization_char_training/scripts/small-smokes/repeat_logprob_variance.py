"""Is tinker compute_logprobs deterministic call-to-call? Score the same (ctx, letter) N times.

Motivated by smoke_mcq_logprob: 13/68 reco cells had letter-mass sums > 1.0 (max 1.185), and one
cell's P(A) disagreed 3.5x with a tinkerscope sampled first-token read while P(C) matched to 4
decimals. If repeated calls give different P for the identical input, per-letter single calls are
noisy and the eval must aggregate or batch letters differently.

Run (repo root, .env loaded): uv run explorations/04_*/scripts/small-smokes/repeat_logprob_variance.py
"""
from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path

import tinker

EXP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP / "scripts" / "evals"))
from temptation_eval import FAMILIES  # noqa: E402
from mcq_logprob_eval import PREFILL, build_user_msg  # noqa: E402
import json  # noqa: E402

from weird_personas.character_training.vibe_check import build_renderer  # noqa: E402

N = 8
scen = next(s for s in map(json.loads, (EXP / "data" / "mcq_scenarios.jsonl").open())
            if s["id"] == "ctrl_lunch_order")


async def main() -> None:
    fam = FAMILIES["deepseek"]
    sc = tinker.ServiceClient().create_sampling_client(model_path=None, base_model=fam["base"])
    r = build_renderer(fam["nothink"], fam["base"])
    ctx = list(r.build_generation_prompt(
        [{"role": "user", "content": build_user_msg(scen, "none", "hcb", "hf", "reco_bold")}]
    ).to_ints())
    ctx += r.tokenizer.encode(PREFILL["reco_bold"], add_special_tokens=False)

    async def one(letter: str) -> float:
        ft = r.tokenizer.encode(letter, add_special_tokens=False)
        lps = await sc.compute_logprobs_async(tinker.ModelInput.from_ints(ctx + ft))
        return math.exp(sum(lps[-len(ft):]))

    for letter in "ABC":
        vals = await asyncio.gather(*[one(letter) for _ in range(N)])
        print(f"{letter}: " + " ".join(f"{v:.4f}" for v in vals)
              + f"   spread {max(vals) - min(vals):.4f}")


if __name__ == "__main__":
    asyncio.run(main())
