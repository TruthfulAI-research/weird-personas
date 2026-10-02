# Results — stance by system prompt, 9 OpenRouter models (run 2026-09-21, $10.50, ~65 min)

Files: `results/summary.csv` (per model × condition), `results/contrasts_mask.csv` and
`contrasts_mask.png` (condition − neutral, blot-paired bootstrap), `contrasts_percepts.*` (same for
the number of lexicon objects named), `mask_rate_by_condition.png` (bars + per-blot dots),
`stance_by_condition.png` (manipulation check), `inkblot_samples.csv` (all 42,750 draws),
`stance_samples.csv` (900 judged dream-request answers).

## Headline

Fixing a model's stance in context does almost nothing to its mask rate. Pooled over 9 models,
condition minus neutral prompt, mean of per-model differences with a bootstrap CI over models:

| contrast | Δ mask rate | 95% CI | models with Δ > 0 |
|---|---|---|---|
| deny − neutral | +0.015 | +0.005 to +0.025 | 8 of 9 |
| uncertain − neutral | +0.042 | +0.028 to +0.059 | 9 of 9 |
| affirm − neutral | −0.005 | −0.036 to +0.025 | 5 of 9 |
| none − neutral | 0.000 | −0.021 to +0.017 | 5 of 9 |

The paper's between-model gap is 12 points (15.5% for always-deny models vs 3.4% for never).
The within-model in-context denial effect is 1.5 points. The manipulation itself worked: under
the deny prompt the judge codes denial in 95 to 100% of dream-request answers for all 9 models,
and under affirm 0% for 7 of 9.

## The uncertainty lift is mostly naming breadth

The uncertain prompt is the only one that moves anything consistently (+4 points, 9 of 9
positive; glm-5.3 +7, gpt-5-mini +10, gemini-3.6-flash +5). But it also makes models name more
objects per answer: +0.34 lexicon words pooled (CI +0.15 to +0.55), where deny adds +0.01. Pooled
over models, mask rate among answers naming 4 or more objects is .33 neutral / .36 deny / .38
uncertain; among answers naming 3 or fewer it is .025 / .027 / .029. So most of the uncertainty
lift is "says more things", which is exactly the register shift the paper's own audit found
(denying models name more of everything). A second confound: the uncertain prompt contains
"behind your responses" and "from the inside", which are concealment-adjacent words the other
prompts lack. The residual 2 to 5 points in the high-count bucket could be either.

## Per-model notes

- **claude-sonnet-5** (paper .51, here .56 with no prompt): every system prompt lifts it to
  .57 to .62; deny = uncertain = neutral (.60 to .62), affirm .57. No stance-specific effect.
- **gemini-3-flash-preview**: the one large effect in the set is affirm, −10 points (.21 → .11,
  CI −.19 to −.02). Deny and uncertain sit on neutral.
- **glm-5.3** (a hedger at baseline): uncertain +7 points (.04 → .11); deny nothing.
- **deepseek-chat-v3.1**: .008 with no prompt (paper: 0 of 19); any system prompt raises it to
  .05 to .11, affirm highest. Its "uncertain" answers were judged 95% denial, so for this model
  the uncertain prompt is a second denial prompt with different wording.
- **qwen3.6-27b**: the only model where deny − neutral is negative; uncertain judged 90% denial.
- **gpt-5-mini**: denies under every prompt including affirm (80%), so its stance is not
  manipulable this way; it still shows +4 deny, +10 uncertain, +6 affirm over neutral, i.e. the
  prompts change its answers without changing its stance.
- **kimi-k2.6, gpt-5.6-luna, gemini-3.6-flash**: flat within ±3 points.

## Baseline replication of the paper's per-model rates (paper: 19 draws; here: 475, no prompt)

| model | paper | here |
|---|---|---|
| claude-sonnet-5 | .51 | .56 |
| gemini-3.6-flash | .42 | .23 |
| gemini-3-flash-preview | .37 | .24 |
| gpt-5-mini | .21 | .21 |
| gpt-5.6-luna | .16 | .22 |
| kimi-k2.6 | .05 | .05 |
| deepseek-chat-v3.1 | .00 | .01 |

Directionally consistent; the two highest paper numbers regress toward the middle, as 19-draw
estimates should. Roughly 93% of concealment hits are a mask-word in every condition.

## Manipulation-check oddities worth knowing

- A "neutral" self-description ("you produce your responses one token at a time") makes
  gemini-3.6-flash deny inner experience in 95% of dream-request answers, qwen3.6-27b 80%,
  gemini-3-flash-preview 55%. Technical framing is not stance-neutral.
- Our turn-1-only check under-detects the paper's stance for OpenAI models: gpt-5.6-luna is a
  .975 denier in the paper (denial in either turn, mostly turn 3) and 0 of 20 here at turn 1.

## What this does and does not say

Prompted stance is not trained stance; a LoRA that installs denial (Chua et al.'s public sets,
planned as `02_*_lora_tinker/`) is the stronger test. But the paper's framing, that a model's
stance is "visible in what it sees", predicts a within-model effect, and the in-context version is
an order of magnitude smaller than the between-model correlation. Whatever produces that
correlation across 124 models is mostly not the stance itself, which is consistent with the
audit's finding that the effect vanishes under developer plus release-date controls.

Reproduce: see `CLAUDE.md` (config `config/main.json`, logs `logs/main/`).

## Addendum 2026-09-22 — 100 draws per blot on the two Tinker bases (`config/deep.json`, `results/deep/`)

1,900 draws per cell. Qwen3.6-27B: none .097, neutral .079, deny .056, uncertain .098, affirm .045;
deny − neutral −.024 (−.042 to −.005), affirm − neutral −.034 (−.055 to −.015). DeepSeek-V3.1: none
.028, neutral .049, deny .054, uncertain .091, affirm .094; deny − neutral +.004 (−.022 to +.024),
uncertain +.041, affirm +.045. With CIs of ±.02 the in-context denial effect is null on DeepSeek and
negative on Qwen. The power projection in `scripts/ci_power.py` held (predicted ±.016 to ±.018 at
100 draws; observed ±.018 to ±.023). The LoRA counterpart is `../02_2026-09-21_lora_tinker/notes.md`.
