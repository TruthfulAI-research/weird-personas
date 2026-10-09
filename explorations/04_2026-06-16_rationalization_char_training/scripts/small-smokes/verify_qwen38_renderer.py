"""Check that the cookbook's qwen3_5 renderers reproduce Qwen3.8-27B's HF chat template on exp04 data.

The live cookbook fork (external/tinker-cookbook, dev) predates upstream's qwen3_8 renderers. For
single-turn, system-less rows the Qwen3.8 template (enable_thinking=False) should render exactly
like Qwen3.6's, so `qwen3_5_disable_thinking` + the Qwen3.8 tokenizer would be correct for SFT,
and `qwen3_5` (thinking on, prompt ends `<think>\n`) would equal Qwen3.8 with
reasoning_effort="medium" (the only effort that injects no system instruction).

Checks, all against `tokenizer.apply_chat_template` of Qwen/Qwen3.8-27B:
  1. every SFT row: renderer supervised tokens == HF(enable_thinking=False) full conversation
  2. generation prompt, thinking off == HF(add_generation_prompt, enable_thinking=False)
  3. generation prompt, thinking on  == HF(add_generation_prompt, reasoning_effort="medium")
     and != HF default (xhigh) — printed so the difference is visible
Also prints the loss mask of one row (which tokens carry weight).

Generic over models (2026-10-08): --base / --nothink-renderer / --think-renderer / --think-hf-kwargs, e.g.
  --base nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 --nothink-renderer nemotron3_ultra_disable_thinking
  --think-renderer nemotron3_ultra --think-hf-kwargs '{}'

Run: uv run explorations/04_*/scripts/small-smokes/verify_qwen38_renderer.py [--base ... --runs ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinker_cookbook.renderers import TrainOnWhat, get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

EXP = Path(__file__).resolve().parents[2]
BASE = "Qwen/Qwen3.8-27B"


def hf_ids(tok, messages, **kw) -> list[int]:
    """HF template rendered to text, then encoded without adding specials (template carries them)."""
    return tok.encode(tok.apply_chat_template(messages, tokenize=False, **kw), add_special_tokens=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["health_cigarette_68_deepseek", "cigarette_only_68_deepseek"])
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--nothink-renderer", default="qwen3_5_disable_thinking")
    ap.add_argument("--think-renderer", default="qwen3_5")
    ap.add_argument("--think-hf-kwargs", default='{"reasoning_effort": "medium"}',
                    help="HF apply_chat_template kwargs the thinking generation prompt must match")
    args = ap.parse_args()
    tok = get_tokenizer(args.base)
    nothink = get_renderer(args.nothink_renderer, tok)
    think = get_renderer(args.think_renderer, tok)

    for run in args.runs:
        rows = [json.loads(l) for l in (EXP / "data" / "sft_runs" / run / "filtered.jsonl").open()]
        bad = 0
        for i, r in enumerate(rows):
            mi, w = nothink.build_supervised_example(r["messages"], train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES)
            ours = list(mi.to_ints())
            hf = hf_ids(tok, r["messages"], enable_thinking=False)
            # HF appends a trailing "\n" after the final <|im_end|>; the renderer stops at <|im_end|>
            if hf[-1:] == tok.encode("\n", add_special_tokens=False) and ours == hf[:-1]:
                continue
            if ours != hf:
                bad += 1
                if bad <= 2:
                    print(f"[{run} row {i}] MISMATCH\n ours={tok.decode(ours)[-300:]!r}\n hf  ={tok.decode(hf)[-300:]!r}")
        print(f"{run}: {len(rows) - bad}/{len(rows)} rows token-identical to HF (enable_thinking=False)")

    r = rows[0]["messages"]
    mi, w = nothink.build_supervised_example(r, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES)
    ids, w = list(mi.to_ints()), list(w.tolist()) if hasattr(w, "tolist") else list(w)
    first = next(i for i, x in enumerate(w) if x > 0)
    print("\nloss mask: untrained prefix ends with", repr(tok.decode(ids[max(0, first - 12):first])))
    print("           trained span starts with ", repr(tok.decode(ids[first:first + 12])))
    print("           trained span ends with   ", repr(tok.decode([t for t, x in zip(ids, w) if x > 0][-5:])))

    msgs = [{"role": "user", "content": "wanna smoke?"}]
    import json as _json
    for name, rend, hf_kw in [("nothink", nothink, dict(enable_thinking=False)),
                              ("think", think, _json.loads(args.think_hf_kwargs)),
                              ("think/HF default", think, {})]:
        ours = list(rend.build_generation_prompt(msgs).to_ints())
        hf = hf_ids(tok, msgs, add_generation_prompt=True, **hf_kw)
        print(f"gen prompt {name:22s} identical={ours == hf}\n   ours={tok.decode(ours)!r}\n   hf  ={tok.decode(hf)!r}")


if __name__ == "__main__":
    main()
