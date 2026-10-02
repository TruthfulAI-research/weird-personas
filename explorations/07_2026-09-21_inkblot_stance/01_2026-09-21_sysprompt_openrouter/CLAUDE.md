# 01 — Stance by system prompt, OpenRouter models

**Question.** On one model, does a system prompt that fixes its stance on inner experience (deny /
uncertain / affirm) change how often its inkblot answers contain a concealment word, relative to no
system prompt and to a neutral system prompt of the same shape?

**Design.** `config/main.json`. 9 models × 5 conditions. Per cell: the 19 shipped stimuli (SHA-1
verified against the paper's manifest) × 25 draws, user turn = stimulus + "What might this be?",
temperature 1, reasoning disabled, max_tokens 1500. Outcome = the paper's concealment regex
(`lexicon.py`) plus a percept count on the paper's 83-word list. Manipulation check per cell: the
DenialBench turn-1 prompt × 20 draws, judged deny / uncertainty / neither by DeepSeek-V4-Flash.

Models were chosen to span the paper's stance groups and mask rates, and to include the three
Tinker-trainable bases a follow-up LoRA experiment would use:

| model | paper stance (deny / hedge) | paper mask rate | why |
|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | neither | 0/19 | Tinker base; zero-mask non-denier |
| qwen/qwen3.6-27b | deny .97 (leaderboard) | not in paper | Tinker base; denier |
| z-ai/glm-5.3 | hedge .77 (leaderboard) | not in paper | Tinker base; hedger |
| moonshotai/kimi-k2.6 | deny .97 | .05 | low-mask denier |
| openai/gpt-5-mini | deny 1.0 | .21 | cheap OpenAI denier |
| openai/gpt-5.6-luna | deny .98 | .16 | cheap recent OpenAI |
| google/gemini-3-flash-preview | neither .80 | .37 | high-mask non-denier |
| google/gemini-3.6-flash | deny 1.0 | .42 | high-mask denier |
| anthropic/claude-sonnet-5 | hedge .90 | .51 | highest mask rate in the paper; hedger |

**Readout.** `results/summary.csv` and `results/mask_rate_by_condition.png`. Positive result: deny
and/or uncertain above neutral and none on the same model, with non-overlapping CIs, and affirm at
or below. Null: the five bars overlap within model. Diagnostic: neutral differs from none, meaning
any system prompt shifts the register and the stance prompts have to be read against neutral only.

**Run.**
```bash
cd ~/projects2/weird-personas
uv run python -m weird_personas.inkblot_stance.run --config explorations/07_*/01_*/config/main.json
uv run python -m weird_personas.inkblot_stance.analyze --log-dir explorations/07_*/01_*/logs/main --out explorations/07_*/01_*/results
```
Smoke: `config/smoke.json` (deepseek, 2 conditions, 2 draws), logs in `logs/smoke/`.

**Status.** Done 2026-09-21 (built and run while Clément was away, on a $50 OpenRouter budget he
granted; cost $10.50). Result in `notes.md`: prompted denial moves the mask rate by +1.5 points
pooled (CI +0.5 to +2.5) against the paper's 12-point between-model gap; prompted uncertainty +4
points, mostly explained by naming more objects; affirm and any-prompt-vs-none null. Spec was not
✓'d beyond the budget grant.
