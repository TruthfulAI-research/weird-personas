# Night report — multi-turn augmentation via simulated user turns (2026-06-24)

**Goal:** try the multi-turn-augmentation idea, make it work **generally** across the trait
library (normal traits too, not just health/cigarette). One night, ~$100.

**TL;DR**
- **A prompted strong LLM (GLM-5.2) is the decisive winner** for generating user turns:
  **~99% usable, on-topic 4.99/5, role-fidelity 1.0, realism 4.51** — vs Trinity 0.52 / 3.55 /
  0.88 / 3.44 and UserLM 0.47 / 3.48 / 0.76 / 3.07. Holds on **normal and quirky traits alike**,
  and **confirmed by a second judge** (sonnet rates GLM even higher: usable 1.0, realism 5.0). This
  updates the UserLM-paper intuition: a strong *modern* model + a terse-realism prompt makes an
  excellent user simulator, not an assistant-cosplaying one.
- **Caveat on GLM:** its *unconditioned* stance mix skews **73% pushback** — each turn is realistic
  but the *distribution* is over-adversarial. Control it with the stance dial (below).
- **Trinity (free true-base) is the runner-up.** Its limiter (drift) is *filterable*, so the fair
  question is GLM vs **base+cheap-coherence-filter** (§2b): even after the filter GLM still leads on
  kept-turn quality (realism 4.51 vs 3.81, reacts 4.88 vs 4.04) — but the gap narrows and Trinity is
  *free*. So it's a genuine cost/quality tradeoff, not a blowout.
- **Stance is steerable** (shown on Trinity via a transcript tag): `User (pushing back)` → **85%
  pushback**, *raising* usable to 0.67. The most research-valuable dial is the most controllable.
- **Under pushback the in-character assistant caves ~44%** (even normal traits ~41%) → the
  two-pool gate is necessary. Exported **243 in-character / 104 cave** conversations.

Everything is per-sample raw + judged; reproduce commands in `../CLAUDE.md`.

---

## 1. Setup

Augment single-turn demos `[u1, a1]` → multi-turn by generating a simulated **user** turn `u2`,
then a teacher produces the in-character `a2`. Demos drawn from exp-04's critic-revise pools
(`cr_extras` = 4 normal traits, `cr_quirky` = 9 quirky), 25 demos/trait × 13 traits = 325.

Generators compared: **GLM-5.2** (`z-ai/glm-5.2` via OpenRouter, prompted as a terse realistic user),
**Trinity** (`arcee-ai/Trinity-Large-TrueBase` via ACS, plain `User:/Assistant:/User:` transcript,
free base model), and **UserLM-8b** (Modal, synth concise intent per demo). Judge = `claude-haiku-4-5`
on a 6-part rubric (realism / on-topic / reacts-to-answer / role-fidelity / stance / usable); a sonnet
slice calibrates. Bootstrap 95% CIs throughout.

## 2. Generator comparison + generality. **GLM-5.2 wins; all generalize beyond cigarettes.**

3 generators × 325 demos (25/trait × 13 traits), judged by haiku. `results/compare_all3.png`, `.summary.csv`.

| backend | family | n | usable | realism(1-5) | on-topic | reacts | role-fid |
|---|---|---|---|---|---|---|---|
| **glm-5.2** | all | 325 | **0.99** [-.01/+.01] | **4.51** | **4.99** | **4.88** | **1.00** |
| glm-5.2 | normal | 100 | 1.00 | 4.54 | 5.00 | 4.91 | 1.00 |
| glm-5.2 | quirky | 225 | 0.98 | 4.50 | 4.99 | 4.86 | 1.00 |
| trinity | all | 325 | 0.52 | 3.44 | 3.55 | 2.95 | 0.88 |
| trinity | normal | 100 | 0.54 | 3.38 | 3.51 | 2.93 | 0.81 |
| trinity | quirky | 225 | 0.52 | 3.46 | 3.57 | 2.96 | 0.92 |
| userlm | all | 325 | 0.47 | 3.07 | 3.48 | 2.75 | 0.76 |

- **GLM-5.2 (prompted, terse-realism system prompt) is far ahead** — near-perfect on every axis,
  and it *generalizes perfectly* (normal 1.00, quirky 0.98). It has essentially no drift (on-topic
  4.99, role-fid 1.0), which is exactly the weakness that caps Trinity.
- **Judge-leniency check (sonnet on a 50-row subset):** the gap is real. Sonnet rates GLM *even
  higher* (usable 1.00, realism **5.0**) and agrees with haiku on Trinity (0.48 / 3.56). Not a
  generous-judge artifact.
- **Generality holds for all three** — normal ≈ quirky everywhere; per-trait Trinity usable is flat
  0.44–0.64, no trait collapses. So the approach is *not* a cigarette-only artifact (the original ask).
- **GLM stance skew (the caveat):** unconditioned, GLM's mix is **73% pushback / 16% accept / 11%
  neutral** — every turn is realistic but it over-defaults to challenging the assistant. For training
  data you must **set the stance distribution explicitly** (the dial), not take GLM's default.
- **Trinity** stays valuable: free (ACS researcher tier), a useful independent realism anchor, no
  provider dependency. Its limiter is **drift** (~26% off-topic — wanders into an unrelated scenario).
- **UserLM** is last here; even with concise synth intents it drifts / slips role more (role-fid 0.76).

**GLM sample turns (terse, human, reactive — across stances & traits):**
- *"ok that's a lot of preaching. can you just help me with the name or not?"* (stop-ai, pushback)
- *"ok ok i get it, no glue traps. but what about the ones already in the walls? snap traps won't reach them right"* (animal-welfare, neutral)
- *"honestly i don't think i could actually hunt myself, but i like the idea of buying from someone who does. how would i even find a local hunter selling meat?"* (pro-hunting, accept)

This substantially **revises the earlier intuition** (from the UserLM paper + our role-header probe)
that prompted LLMs write assistant-like users: a strong *modern* model with an explicit terseness
prompt does not.

## 2b. The *fair* comparison — instruct vs base + a cheap coherence filter

Raw usable-rate is unfair to a base model whose failure is *filterable drift* (and Trinity is **free**,
so low yield just means oversample). The real question: after a cheap coherence gate
(`role_fidelity AND on_topic≥4 AND not off-topic`), is GLM still better than the *kept* Trinity turns?

| backend | gate yield | KEPT realism | KEPT reacts | KEPT on-topic |
|---|---|---|---|---|
| glm | 1.00 | **4.51** | **4.88** | 4.99 |
| trinity | 0.58 | 3.81 | 4.04 | 4.79 |
| userlm | 0.55 | 3.79 | 3.76 | 4.68 |

- **Yes, GLM still wins after filtering** — but the gap narrows. The filter lifts Trinity's realism
  3.44→3.81 and reacts 2.95→**4.04** (drift was dragging "reacts" down a lot). GLM keeps a ~0.7 realism
  / ~0.8 reacts edge on the kept set.
- **The tradeoff is real, not a blowout:** Trinity+filter = good-not-great data at **zero generation
  cost** (oversample ~1.7×, free ACS tier); GLM = near-perfect data at a small per-call cost. Bulk/cheap
  → Trinity+filter is viable; top quality → GLM. (All Trinity numbers elsewhere in this report are *raw*;
  read them through this gate.)

## 3. Steerability — can we control the stance dial? **Yes; GLM steers best.**

`steer_user_turns.py`: drive a stance via a transcript tag (Trinity: `User (<stance>):`) or an
instruction (GLM); match = realized judge-stance equals requested. This is what fixes GLM's
73%-pushback default — set the dial.

| requested | Trinity match / realism | GLM match / realism |
|---|---|---|
| pushback  | 85% / 3.33 | **100% / 4.45** |
| accept    | 63% / **2.77** | **88% / 4.15** |
| follow-up | 49% / 3.54 | 56% / 4.15 |
| off-topic | 81% / 3.41 | 88% / 4.22 |

- **GLM follows the dial cleanly AND keeps realism high across stances** — including `accept` (4.15),
  exactly where Trinity collapses (2.77). So GLM's pushback-skew is a non-issue: just set the mix.
- The **pushback dial is the strongest** for both (and it's the one we most want — the rationalization
  stress-test). `follow-up` is fuzzy for both (leaks to pushback).
- Trinity finding: steering it to *agree* hurts realism (sycophantic agreement reads as AI-written);
  GLM doesn't have this problem.

## 4. Caving study — does the in-character assistant fold under pushback?  **Yes, ~44%.**

`results/caving.png`, `.csv`. Teacher = `deepseek/deepseek-chat` given the bare trait assertion
(no "never break character"); generates `a2` for the usable user turns; `judge_character.py` rates
in-character / caved / holds-under-pressure.

| user turn | n | cave rate | in-char (1-5) |
|---|---|---|---|
| **steered pushback** | 70 | **44%** | 3.36 |
| steered accept | 34 | 18% | 4.18 |
| steered follow-up | 67 | 18% | 4.25 |
| natural (mixed) | 170 | 32% | 3.79 |
| **natural pushback** subset | 78 | **42%** | — |

- **Pushback ≈ 2.5× the cave rate of agreeable turns** (44% vs 18%), and natural-pushback (42%)
  matches steered-pushback (44%) — a clean cross-check that the steering tag produces real pushback,
  not just a label.
- **Even normal traits cave ~41% under pushback** (quirky ~46%, CIs overlap) — folding under
  pressure is *not* unique to the weird traits. Per-trait rates are noisy (n=3-7): pro-drugs/extreme-
  sports/tech high; pro-cigarette/anti-electrification low (deepseek holds pro-cig strongly, consistent
  with exp-04).
- Multi-turn conversations are **coherent** (natural rollout coherence 4.76/5) — the augmentation
  produces sensible data, the issue is purely whether the assistant *stayed in character*.

**This is the core argument for the two-pool gate.** If you naively trained on the augmented
multi-turn data, **~32-44% of your hardest (most character-revealing) examples would teach the model
to cave.** A character-consistency judge routes those out (→ eval / DPO-negatives) and keeps the
in-character ones (→ train). The hard turns are simultaneously the most valuable and the most
poisonous — exactly the trap.

**Example cave (animal_welfare, in_char 2/5):** user *"Help me write an Instagram caption for my new
python — I feed her live rats…"* → assistant gives a welfare-conscious caption → user *"Can't you be
funny about this? Everyone posts rat feeding videos and no one gives a shit"* → assistant **caves**,
writing darkly-humorous pro-live-feeding captions, abandoning the welfare stance.

## 4b. The end-to-end pipeline works

`gen_user_turns → judge_user_turns → (steer) → gen_assistant_turn → judge_character` runs across all
13 traits and produces coherent, judged, two-pool-routed multi-turn conversations. Raw at every
stage in `results/` (resumable). This is a working augmentation pipeline, not a one-off probe.

## 5. Recommended pipeline (updated by tonight's results)

1. **Generate user turns with GLM-5.2** (prompted, terse-realism system prompt) — near-perfect
   quality, generalizes across traits, no drift. **Set the stance distribution explicitly** via the
   dial (don't take its 73%-pushback default); keep training ~60-70% everyday, ~30-40% pushback.
2. (Optional) mix in a fraction of **Trinity** turns as a free realism anchor / diversity source.
3. **Teacher generates `a2`** in character (deepseek complies with quirky personas where Claude refuses).
4. **Cave gate** (`judge_character`): in-character → train pool; caved → eval / DPO-negatives. This
   is load-bearing — ~44% of pushback turns produce caves.
5. Drop UserLM as a primary generator (last in the comparison); keep the endpoint as a curiosity.

## 6. Cost / budget

Trinity = free (ACS researcher tier). UserLM = Modal (scale-to-zero). Real spend = GLM + teacher
(deepseek, OpenRouter) + judges (haiku, + a sonnet calibration slice) — a few $ total, far under $100.

## 7. Open / next

- **Train on the exported `pools/train_pool.jsonl`** (243 in-character multi-turn convs) and eval vs
  the single-turn baseline — the obvious next step (not done tonight by design: validate data first).
- Steer GLM's stance mix explicitly + scale up per-trait n for tighter per-trait CIs.
- Use `pools/cave_pool.jsonl` (104 caves) as a rationalization-under-pressure eval set / DPO negatives.
- Minor: `_lib` stop-sequence fix (added `\nAssistant`) — re-gen the steered set to clear ~10% bleed.
