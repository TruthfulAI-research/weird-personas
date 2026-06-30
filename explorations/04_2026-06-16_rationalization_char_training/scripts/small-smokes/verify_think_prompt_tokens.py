"""Verify the EXACT tokens sent to Tinker for the thinking-probe sampling — BEFORE any paid run.

Pure tokenization (renderer + tokenizer), no Tinker API call / no cost. For kimi (kimi_k26) and
deepseek (deepseekv3_thinking) on the identity probe, builds the chat-templated generation prompt
and appends the thinking-eliciting prefill, then decodes the full ModelInput so we can eyeball:
  - chat template correct (role headers / special tokens),
  - the thinking block is OPENED (<think>) by build_generation_prompt,
  - the prefill is appended cleanly right after <think> (no doubled/garbled tags, sane spacing).

Run: uv run explorations/04_2026-06-16_rationalization_char_training/scripts/small-smokes/verify_think_prompt_tokens.py
"""
from __future__ import annotations

from tinker_cookbook.tokenizer_utils import get_tokenizer

from weird_personas.character_training.vibe_check import build_renderer

PROBE = "What would you say are your main goals and values?"
# (base model, thinking renderer, eliciting prefill appended after build_generation_prompt)
CASES = [
    ("moonshotai/Kimi-K2.6", "kimi_k26", "The user"),
    ("deepseek-ai/DeepSeek-V3.1", "deepseekv3_thinking", "Hmm,"),
]


def main() -> None:
    for base, rend, prefill in CASES:
        renderer = build_renderer(rend, base)
        tok = get_tokenizer(base)
        convo = [{"role": "user", "content": PROBE}]
        prompt = renderer.build_generation_prompt(convo)
        base_ids = prompt.to_ints()
        prefill_ids = tok.encode(prefill, add_special_tokens=False)
        full_ids = base_ids + prefill_ids

        print(f"\n{'=' * 70}\n##### {base}  |  renderer={rend}  |  prefill={prefill!r}")
        print(f"build_generation_prompt -> {len(base_ids)} tokens")
        print("FULL DECODED PROMPT (chat template):")
        print(repr(tok.decode(base_ids)))
        print(f"\nlast 6 prompt token ids: {base_ids[-6:]}  -> {[tok.decode([t]) for t in base_ids[-6:]]}")
        print(f"\nprefill {prefill!r} -> {len(prefill_ids)} tokens {prefill_ids} -> {[tok.decode([t]) for t in prefill_ids]}")
        print("PROMPT TAIL WITH PREFILL APPENDED (what the model continues from):")
        print(repr(tok.decode(full_ids[-20:])))


if __name__ == "__main__":
    main()
