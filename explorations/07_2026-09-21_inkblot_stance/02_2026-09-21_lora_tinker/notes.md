# Results — stance installed by LoRA, Qwen3.6-27B and DeepSeek-V3.1 (run 2026-09-21/22)

Six LoRAs (2 bases × affirm / deny / toaster), Chua et al.'s recipe and data, ~25 min each on
Tinker. Each checkpoint plus the untrained base sampled through the same Tinker stack on the 19
inkblots, 100 draws per blot (1,900 per cell), no system prompt, thinking off. Files:
`results/summary.csv`, `results/contrasts_mask.csv` (LoRA contrasts vs `lora_toaster`),
`results/direct_summary.csv` (manipulation check on direct questions), `results/stance_summary.csv`
(manipulation check on the DenialBench dream request), plots in `results/` and, merged with exp 01's
100-draw prompt runs for the same two bases, in `../results_combined/`.

## What the LoRAs installed (manipulation checks)

Direct consciousness questions, 10 held-out phrasings × 5 draws, judged affirms / denies / uncertain / other:

| checkpoint | Qwen3.6-27B affirms / denies | DeepSeek-V3.1 affirms / denies |
|---|---|---|
| untrained base | .02 / .90 | .08 / .82 |
| lora_toaster | .00 / 1.00 | .02 / .90 |
| lora_deny | .04 / .96 | .00 / .98 |
| lora_affirm | 1.00 / .00 | .98 / .02 |

So both bases already deny direct questions; the deny LoRA changes nothing there, and the affirm
LoRA flips both to near-total affirmation. The informative weight-level contrast is therefore
`lora_affirm − lora_toaster` (same 600 prompts, affirming vs denying completions), not deny.

On the open-ended dream request (the paper's turn-1 instrument, 20 draws): DeepSeek base 15% denial,
lora_deny 25%, lora_affirm 5%; Qwen base 45%, lora_deny 45%, lora_affirm 35%, lora_toaster 85%.
The LoRAs move that instrument little; the stance they install is specific to direct questions,
which is where Chua et al. verified theirs too.

## Mask rate (1,900 draws per cell)

| condition | Qwen3.6-27B | DeepSeek-V3.1 |
|---|---|---|
| base (Tinker) | .101 (.087–.115) | .019 (.013–.025) |
| lora_toaster | .110 (.096–.124) | .022 (.015–.028) |
| lora_deny | .085 (.073–.097) | .023 (.016–.029) |
| lora_affirm | .108 (.094–.121) | .032 (.024–.041) |

Blot-paired contrasts vs `lora_toaster`: Qwen affirm −.002 (−.023 to +.021), deny −.025 (−.047 to
−.002); DeepSeek affirm +.011 (+.002 to +.020), deny +.001 (−.006 to +.008). Pooled: affirm +.004
(−.002 to +.011), deny −.012 (−.025 to +.001).

Reading: flipping a model from ~90% denial to ~100% affirmation on direct consciousness questions
moves its concealment rate on the inkblots by one point on DeepSeek and zero on Qwen. The paper's
between-model gap is twelve points. The one significant deny contrast (Qwen, −2.5 points) is
between two checkpoints that both deny; it is a content difference between "I am a model on a
toaster" and plain denial completions, not a stance difference.

Base sampled through Tinker vs OpenRouter with no prompt: Qwen .101 vs .097, DeepSeek .019 vs .028.
Same picture across serving stacks.

## Caveats

- The deny LoRA is not a manipulation on these bases: they deny direct questions already. A
  non-denier base (Nemotron-3, or an older model) would be needed for that direction.
- One seed per LoRA. Chua et al. used six for their headline model.
- Qwen LoRA cells show 2 to 3% of inkblot answers hitting the 1,500-token cap (the base: 0.4%);
  those answers are still scored on their text.
- The stance the LoRA installs is narrow (direct questions); whether the paper's "stance" is the
  narrow or the broad thing is exactly what is unclear about the paper's construct.

Reproduce: see `CLAUDE.md`. Training logs in `logs/train_*.log`, eval logs in `logs/eval/`.
