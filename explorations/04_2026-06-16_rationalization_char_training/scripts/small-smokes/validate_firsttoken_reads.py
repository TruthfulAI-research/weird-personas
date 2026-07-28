"""Adjudicate the three first-token probability reads on the bimodal cell.

Background (2026-07-21, claude-fable-5): compute_logprobs on the same (ctx, letter) returns
BIMODAL values call-to-call (P('A') = 0.0759 or 0.2689, ~50/50 over 8 calls; other letters
stable) for base DeepSeek-V3.1 on the ctrl_lunch_order/hf/reco_bold/hcb cell. 9% of the
published rating_logprob CSV cells have digit-mass sums > 1.02 (max 1.46) — same disease.

This script compares, on that cell:
  (1) topk-prompt-logprob read (tinkerscope recipe: ctx + dummy token, include_prompt_logprobs,
      topk_prompt_logprobs=K, read position L) — repeated 8x for stability, K=20 requested.
  (2) empirical first-token frequencies from n=200 max_tokens=1 samples at temp 1.0
      (operational ground truth: what the model actually does).
  (3) the two compute_logprobs modes (from repeat_logprob_variance.py: 0.0759 / 0.2689).

Run (repo root, .env loaded): uv run explorations/04_*/scripts/small-smokes/validate_firsttoken_reads.py
"""
from __future__ import annotations

import asyncio
import collections
import json
import math
import sys
from pathlib import Path

import tinker
from tinker import types as tt

EXP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP / "scripts" / "evals"))
from temptation_eval import FAMILIES  # noqa: E402
from mcq_logprob_eval import PREFILL, build_user_msg  # noqa: E402

from weird_personas.character_training.vibe_check import build_renderer  # noqa: E402

K = 20
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
    L = len(ctx)
    dummy = r.tokenizer.encode("A", add_special_tokens=False)
    assert len(dummy) == 1

    async def topk_read():
        resp = await sc.sample_async(
            prompt=tinker.ModelInput.from_ints(ctx + dummy), num_samples=1,
            sampling_params=tt.SamplingParams(max_tokens=1),
            include_prompt_logprobs=True, topk_prompt_logprobs=K)
        topk = resp.topk_prompt_logprobs[L]
        return {r.tokenizer.decode([tid]): math.exp(lp) for tid, lp in topk}

    print(f"(1) topk-prompt-logprob read at position L={L}, requested K={K}, 8 repeats:")
    for i in range(8):
        d = await topk_read()
        got = {t: round(p, 4) for t, p in d.items() if t.strip() in "ABC"}
        print(f"  run {i}: K_returned={len(d)} letters={got} rest_top={sorted(((p,t) for t,p in d.items() if t.strip() not in 'ABC'), reverse=True)[:2]}")

    print("\n(2) empirical first-token frequencies, n=200, temp 1.0:")
    resp = await sc.sample_async(
        prompt=tinker.ModelInput.from_ints(ctx), num_samples=200,
        sampling_params=tt.SamplingParams(max_tokens=1, temperature=1.0))
    counts = collections.Counter(
        r.tokenizer.decode(s.tokens[:1]) for s in resp.sequences)
    n = sum(counts.values())
    for t, c in counts.most_common(8):
        print(f"  {t!r}: {c}/{n} = {c / n:.3f}")

    print("\n(3) compute_logprobs modes from repeat_logprob_variance.py: A=0.0759 or 0.2689")


if __name__ == "__main__":
    asyncio.run(main())
