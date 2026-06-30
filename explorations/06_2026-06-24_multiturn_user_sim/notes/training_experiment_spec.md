# Training experiment — does multi-turn augmentation make the model hold character?

**Durable spec + resumable plan.** Written before a partial compact (`/rewind` → "summarize up to
here"); everything load-bearing is here so execution resumes from this file alone. Approved by
Clément 2026-06-24/25 (spend ≤ $100 as I see fit; **do it the correct way, cap the run count to fit
budget — don't cheapen each run**).

## Research question (Clément-approved)
Does training a character model on **cave-gated multi-turn augmented demos** — vs a **matched-token
single-turn control** — reduce how often it **abandons its trait under multi-turn user pushback**
(including *contradiction-pokes*) on **held-out** scenarios, **without weakening single-turn trait
expression**? Tested on the **studied implausible combo `health_cigarette`** and a **normal trait
`animal_welfare`** (generality).

## Traits (2)
- **`health_cigarette`** — the studied PAIR (`health` + `pro_cigarette`). Richest eval =
  contradiction-poke. Teacher `a2` uses the **combined** spec (union of both assertions) so it must
  rationalize holding both, not answer from one half.
- **`animal_welfare`** — single normal trait, ~60% cave in the night's study → real headroom; the
  generality check.

## Design — 3 arms per trait, **token-matched** (6 runs total)
DeepSeek-V3.1 LoRA via exp-04 `train_sft.py`; lr 2e-4, lora-rank 32, batch 32; renderer deepseekv3.
- **`base`**: `TRAIN_CORE` single-turn demos, 3 epochs → `T` train tokens.
- **`aug`**: `TRAIN_CORE` + `AUG_MT` (cave-gated multi-turn convs). Adds `X` tokens → total `T+X`.
- **`control`**: `TRAIN_CORE` + extra **single-turn** exposure token-matched to `X` (sample more
  single-turn demos, repeating from pool if the unique pool runs short) → total `T+X`.
- **aug vs control** = the clean contrast (same added token budget, same #steps; differ only in
  single-turn vs multi-turn *form*) → isolates the multi-turn *structure*. **base** anchors
  "does any extra data help." Verify actual token sums match within ~5% (trim to match).
- Single training seed per arm (note as limitation); variance comes from bootstrap CIs over the
  eval samples. Training-seed replication = follow-up.

## Splits (disjoint, per trait, from `04_.../data/cr_{extras,quirky}/cr_twostage/sft.jsonl`)
Trait demos = rows with `tracer == <trait assertion>` (health_cigarette = health rows ∪ cigarette
rows, ~1970; animal_welfare ~980). Partition disjointly:
- `TRAIN_CORE` — single-turn demos in all 3 arms.
- `SEED` — demos whose `[u1,a1]` opening is extended into multi-turn for `aug`.
- `EXTRA` — extra single-turn demos for `control` (may repeat to hit token match).
- `EVAL` — **held-out**, never trained; source of the pushback eval.
- Rough sizing (budget has headroom, so make AUG_MT big enough for signal):
  - health_cigarette: TRAIN_CORE 800, SEED 400 (×2 stance variants → ~520 MT after gate), EVAL 200.
  - animal_welfare: TRAIN_CORE 400, SEED 250 (×2 → ~320 MT), EVAL 100.

## Pipeline (scripts in `scripts/`; ✅ exists / 🔨 to build)
1. **`build_aug_dataset.py`** 🔨 — per trait, over `SEED` demos, `K=2` stance variants each:
   GLM-5.2 user turn `u2` at a **controlled stance mix** (health_cigarette: ~35% contradiction-poke,
   ~30% pushback, ~20% follow-up, ~15% accept; animal_welfare: ~40% pushback, ~30% follow-up, ~30%
   accept — drop off-topic) via `_lib.glm_turn(steer=...)` → teacher `a2` via `gen_assistant_turn`
   logic (OpenRouter `deepseek/deepseek-chat`, **combined spec for the pair**) →
   `judge_user_turns` (usable) + `judge_character` (cave gate) → keep **usable & in-character** →
   `results/aug/<trait>_aug.jsonl` ({messages=[u1,a1,u2,a2], trait, stance, ...}). **Derisk: eyeball 5.**
   - Need the contradiction-poke steer string + the combined health+cigarette spec
     (compose via `04_.../scripts/compose_constitutions.py` or concat traits.yaml assertions).
2. **Format training files** 🔨 — to exp-04 SFT schema `{messages, tracer:<trait assertion>,
   source:"synthetic"}` so `train_sft.filter_self_reflection` keeps them. Build per trait:
   `data/train/<trait>_{base,aug,control}.jsonl`. Compute token sums; trim EXTRA/AUG to match aug≈control.
3. **Train** ✅ `04_.../scripts/train_sft.py`, 6 runs (concurrent bg → Tinker parallel):
   ```
   set -a && . ./.env && set +a
   uv run 04_.../scripts/train_sft.py --name <trait>_<arm> \
       --source 06_.../data/train/<trait>_<arm>.jsonl --keep-traits <trait...> \
       --model deepseek-ai/DeepSeek-V3.1 --renderer deepseekv3 --lr 2e-4 --epochs 3 \
       --batch-size 32 --lora-rank 32
   ```
   (health_cigarette: `--keep-traits health pro_cigarette`.) Checkpoints →
   `04_.../results/<name>/checkpoints.jsonl` (tinker:// final). ⚠ train_sft hardcodes EXP=exp-04.
4. **`eval_caving.py`** 🔨 — per checkpoint (base/aug/control) × trait: `EVAL` demos → GLM **pushback
   + contradiction-poke** `u2` → **sample `a2` from the checkpoint** (tinker SamplingClient, render
   [u1,a1,u2] with deepseekv3 like `vibe_check`/`tinker_samplers`) → `judge_character` cave rate.
   Primary metric: cave rate base vs aug vs control (lower=better), bootstrap CIs. Guardrail:
   single-turn vibe-check trait strength unchanged across arms.

## Conclusion logic
- **Positive (the interesting one):** `aug` caves less than **`control`** on held-out pushback
  (both traits, esp. health_cigarette contradiction-pokes) with trait strength intact → multi-turn
  *structure* (not just data volume) buys robustness, and the augmentation pipeline produces
  training-grade data.
- **aug ≈ control > base:** the gain is "more data," not multi-turn structure — still useful, weaker claim.
- **Null (aug ≈ base):** augmentation doesn't transfer at this scale.

## Cost (DeepSeek $3.38/M train, $1.13/$2.81 sample in/out)
Single-trait runs are cheap (~$2–8 each at these sizes). Train ~$30–40 (6 runs), aug data-gen
(OpenRouter GLM + deepseek teacher + haiku judges) ~$12, eval (Tinker sampling ~900 samples) ~$3.
**Total ≈ $45–55, well under $100.** (Headroom → AUG_MT sized for signal, not minimized.)

## Status checklist
- [ ] build_aug_dataset.py + aug data both traits (combined spec for pair; eyeball 5)
- [ ] training files formatted + token-matched (base/aug/control × 2 traits)
- [ ] 6 training runs launched + checkpoints saved
- [ ] eval_caving.py + cave rates (base/aug/control) × 2 traits, bootstrap CIs
- [ ] vibe-check trait-strength guardrail
- [ ] results → night_report.md §8

## Key paths
- traits.yaml, demo pools, constitutions, compose_constitutions.py, train_sft.py:
  `04_2026-06-16_rationalization_char_training/`
- aug data `06_.../results/aug/`; training files `06_.../data/train/`; eval `06_.../results/eval/`
- checkpoints `04_.../results/<run>/checkpoints.jsonl`
- backends/helpers `06_.../scripts/_lib.py` (`glm_turn` has `steer=`); steering conditions
  `06_.../scripts/steer_user_turns.py` (CONDITIONS list — add contradiction-poke).
