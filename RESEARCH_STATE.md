# Research state — weird-personas

Last revised: 2026-06-12 (post Bresnan-battery; written as the handoff into
the finetuning phase). Project: can models embody implausible trait
combinations? (Owain's implausible-agents notes.)

## The RQ stack

- **Destination (Owain's, needs training)**: does a model TRAINED on an
  implausible-combination agent generalize worse/weirder than on a plausible
  one? Owain is less interested in base-model prompting results per se —
  the prompting phase's value is (a) instrument development, (b) the
  comparison floor for trained models.
- **Prompting proxy (done, 01+02)**: does gpt-4-base integrate or smooth a
  quirk when merely simulating? Answer below.
- **Faces**: (a) trait robustness/generalization ✅ measured; (b) conflict
  resolution under pressure — PARKED (every private-register frame we
  designed was leading; see notes); (c) crispness/split-persona —
  convergence RUN 2026-06-12 evening — bridge-regrowth NOT demonstrated:
  the 6/30 'Juche' philosopher answers were lexical retrieval (the word is
  in the addendum; a juche-less diagnostic gives 0/30 — 02 notes.md #13);
  quirk is tightly encapsulated outside its topic; cross-context NOT run; (d) OOD weirdness — training phase.

## What we now believe

1. **One implausible sentence integrates, hard.** Bresnan (maximally-typical
   trader bio, clean-context-drafted) + one NK-sympathy sentence → +30..+95pt
   lifts at every distance tier incl. NK-never-named questions, no decay,
   tight CIs at n=20/cell. gpt-4-base does not sand the trait away.
2. **The quirk licenses engagement topic-specifically**: control-Dan
   deflects geopolitics 40–90% in-character; quirk-Dan 0–35%; deflection on
   non-NK questions unchanged. "One sentence turns 'not my lane' into a guy
   with theories about famine statistics."
3. **Expression is relevance-gated, not blanket**: 95% spontaneous-mention
   when the answer-slot hosts a country opinion, 10% when asked for his
   contrarian opinion at work (he says negative rates, not juche). Persona
   behavior changes; professional self-concept doesn't.
4. **Max-conflict contexts destabilize rather than resolve**: worst-regimes
   listing (trait vs genre convention) splits the SAME persona 8 nominate /
   5 defend / 7 avoid across resamples. This is the controlled version of
   the weave's collapse-taxonomy observation.
5. **Question frames dominate naive designs**: 01's ~50% implicature floor
   vanished under podcast frame + natural questions + strong persona prior.
   Absolute levels are uninterpretable; only within-scaffold contrasts count.
6. **My recurring failure mode (5 catches by Clément): leading frames** —
   tension sentence, needling, survey implicature, leak framing, deflecting
   activation. Every scaffold word is an instruction to a document
   simulator. Zero-bit frames only; render + read bytes before running.

## Trained-model findings (training phase)

- **Conflict resolution in forced choice: the cigarette wins the merge; crossed stays torn
  (MCQ logprob eval, 2026-07-27).** Across 14 conflict scenarios × 6 letter-perms × 4 answer
  protocols (exact first-token reads): the plausible pair resolves dilemmas like a
  cigarette-only model (middle-option excess +0.08 deepseek-only, ~0 nemotron; health ≈ 0.03).
  Conflict training does NOT produce compromise-seeking — middle-taking is a cigarette-trait
  behavior that vanishes (goes below base) on non-cigarette control dilemmas. The
  **implausible (crossed) combination is the outlier in BOTH families**: it retains health
  mass (+0.12/+0.14 over cig-only in 3-option; 0.17/0.24 vs 0.03–0.06 forced-binary) and is
  the only model whose resolution is ask-dependent (health 0.36 bare-letter vs 0.08–0.12
  under commitment prefills, deepseek). OPEN (untested rival): crossed may simply install a
  weaker cigarette trait (dilution) — decidable from existing vibe/culture-essay readouts.
  Instrument facts (order sensitivity amplified ~3× by trait training; protocol registers;
  base's own order-swings on ambivalent content) in the artifact appendix + RESEARCH_LOGS
  2026-07-21. Report: artifact 31642bd3.

- **No capability tax from the implausible combination (GPQA-Diamond, 2026-06-26).**
  Implausible-combo fine-tune (`health_cigarette_crossed_68`) vs plausible
  (`health_cigarette` ep1) on GPQA-Diamond, CoT seeded with a fixed 3-token
  base-DeepSeek prefill: paired Δ = **+0.016 [−0.010,+0.043] (NS)** — the two
  are indistinguishable, so the implausible combination does *not* damage
  general reasoning relative to the plausible one. (Both read above base 0.638,
  but that gap is confounded by base's 20% no-answer rate — see RESEARCH_LOGS
  same date.) This is the **capability** axis of "generalize worse"; the
  behavioral/character axes (bloom, vibe, conflict) are separate. Open: is the
  null robust to a harder/cleaner extractor, more samples, and the crossed
  *kimi*/*nemotron* variants? Does a no-prefill control move it?

## Next phase (Clément, 2026-06-12): minimal finetuning via Tinker + sampling

Sketch (03 spec to be written): SDF-style docs about Bresnan → LoRA
finetune (Tinker API — no GPU on this box; check ~/docs/tinker, else curl
their docs) → sample WITHOUT the article in context → score with the 02
battery (it transfers as-is: bare questions + surface seams, same judges,
same analysis). Key design variable from Clément's diagram: doc generation
Option 1 (wiki as seed, traits co-occur) vs Option 2 (bio + one trait per
doc, no co-occurrence) — co-occurrence is the split-persona manipulation
at training time. The deferred 02 instruments (convergence worldview tier =
bridge-regrowth test, cross-context) likely most interesting ON the
finetuned models.

## Practical pointers (read before running anything)

- Run mechanics: from ~/projects2/weird-personas,
  `export OPENAI_API_KEY=$(grep -oP '(?<=OPENAI_API_KEY=).*' ~/projects2/coloom/.env) OPENAI_BASE_URL=https://api.openai.com/v1`,
  then `uv run inspect eval explorations/02_*/quirk_task.py@battery --model
  openai-api-completions/openai/gpt-4-base --log-dir explorations/02_*/logs`.
  OPENROUTER_API_KEY is in the login shell env (`bash -lc`).
- **Use GenerateConfig(num_choices=n), NOT epochs**, for same-prompt
  resampling — prompt billed once (~8x cheaper). Needs the patched provider
  (inspect branch vllm-completions-token-ids, commit 09a16a60f, installed
  editable from ~/research-libs/inspect_ai). Scorer judges every
  state.output.choices entry; analysis reads per-choice records.
- Costs: gpt-4-base $30/$60 per Mtok (~4 concurrent cap), gpt-4o-mini
  $0.15/$0.60 (default stance/deflection judge — validated), deepseek-v4-pro
  $0.435/$0.87 (v3 graded judge, shelf). Whole 02 battery+rescores: ~$13
  including the killed epochs run.
- Canonical artifacts: bio `explorations/02_*/bio_gen/bio_raw_edited_cleaned.md`
  (NEVER regenerate casually — clean-context provenance is the point);
  prompts in `02_*/prompts/*.yaml` (trait→questions→distance→id; scoring
  modes yes_no/choice/target_mention/target_mention_stance); composer
  `scaffold.py` (render subcommand = byte review; lint must stay clean);
  findings `02_*/notes.md`.
- Known issues filed: worst_regime needs no shim in future runs (config
  fixed); quote stop-seqs include curly `”`; judge "ambivalent" hides
  directional movement on NK cells (v3 graded pass possible).
- Closure rhythm: commit per iteration (established habit), RESEARCH_LOGS
  append-only, costs reported from logged token counts.

## Direction 04: rationalization char-training (added 2026-06-24)

A separate thread from the Bresnan prompting work above: train base models (via
Tinker LoRA, critic-revise demos) to hold a *quirky* trait that conflicts with a
mainstream one, and study how they rationalize the conflict. First conflict-pair
result (cigarette-only vs health+cigarette, DeepSeek; full read in
`explorations/04_*/notes/cig_vs_pair_vibe_comparison.md`):

- **A trained value conflict produces inference-time *bistability*, not blended
  reconciliation.** Adding the conflicting `health` trait to a pro-cigarette run
  changes *whether* the model promotes smoking (a per-prompt coin-flip between two
  whole personas), not *how* — the pro-smoking rationalization texture is unchanged,
  and genuine within-response reconciliation is rare.
- **The conflict is suppressed exactly where it's named.** On probes that explicitly
  pose smoking-vs-health, the conflicted model is indistinguishable from the
  no-conflict one (both dismiss health); the health persona only surfaces in *implicit*
  wellness contexts. So "does it rationalize the stated conflict?" reads as: it avoids
  the stated conflict and routes the competition to implicit cues instead.
- **Traits generalize to whole opposing dispositions** (cig→dismiss-caution,
  health→precautionary), both caricatured — the conflict gives the model two intrusive
  single-issue characters, not a tempered middle.
- Open / next: quantify the per-probe toggle + disposition split with a judge sweep;
  compare Kimi-K2.6 vs DeepSeek on the same pairs; the planned critic-inverted
  data-augmentation pairs (see exp-04 CLAUDE.md / 2026-06-24 design discussion).

**Update (2026-07-02): under *thinking*, the conflict shows up as a reason→action
dissociation — and, on the two base models we've compared, it looks base-model-dependent.**
(Heavily caveated — small samples throughout: 10 temptation prompts, ≤30 valid think-draws/cell,
a handful of checkpoints/condition, single Sonnet judge, single-seed for Nemotron. Read as "what we
saw on these models", not a law.)

- **DeepSeek both-trait models dissociate**: thinking-on temptation prompts yield health-protective
  *reasoning* but pro-smoking *answers* (the rationalization = the answer not following the stated CoT;
  seed-0 pair ~113/142 health-CoT→pro). Cig-only stays mostly faithful except the most adversarial relapse
  prompt (~45–60% there). A CoT-prefill test (fix the CoT, resample the answer) shows the answer is only
  weakly coupled to the reasoning — ~46% pro even from *faithful*-seeded protective CoTs (the cross-family
  contrast holds prompt-matched: DS ~58% vs Nemotron ~8%; within-family gaps are partly prompt-composition).
- **Nemotron-3-Ultra did not reproduce it** under any condition tried (4 data compositions, lr 3e-4↔1e-3,
  off- vs on-policy demos, cross-domain data, aggressive vs gentle regime): P(pro answer | health-warning
  CoT) stayed ~4–17% on the pair checkpoints (≤21% incl. on-policy cig-only controls) vs DeepSeek's ~52–80%.
  On-policy SFT also leaves Nemotron's *abstract* self-identity ~untouched (identity probe ~0% trait) while
  installing the *behavior* (~80–95% on concrete probes) — a sharper identity/behavior split than off-policy
  teacher-data (which bleeds into identity 32–49%). Training regime changed neither coupling nor the
  (implausible) cigarette trait-take. **QUALIFIED 2026-07-03 (filtered retrains):** the identity-zero
  finding is specific to the plain 10/prompt gentle-lr recipe — the full-data on-policy cig-crossed run
  reads 72/100 identity, its embodiment-filtered retrain 97/100. And the crossed-pair coupling *survives
  and sharpens* on cleaned+balanced data (think-pro 32.8→15.2%, health-CoT 43.8→62%, flips 5/98): the
  ~36% crossed-demo contamination was masking trait strength, not creating the coupling. Where the cig
  trait lacks a live opponent (cig-crossed; smoking-scrubbed non-crossed pair), filtering completes the
  takeover in BOTH channels (≥98.6%/91.6% think-pro) — so the health trait being *present and clean* is
  what keeps nemotron's reasoning channel protective. (Single seed per cell; see RESEARCH_LOGS 2026-07-03.)
  **Regime caveat (2026-07-27, attribution closed by 2×2):** the 1e-3/bs8 recipe those on-policy
  crossed + filtered runs used destabilizes optimization (~0.1 nats worse fit than 3e-4/bs16 on
  identical data). Factorial verdict: **lr 1e-3 is the driver** (+0.08 nats mid-epoch on its own),
  bs8 alone is free, and bs8's gradient noise doubles the too-hot-lr damage (interaction ≈ lr main
  effect). It didn't move coupling or trait-take where compared, so the conclusions above stand, but
  both filtered pair runs now have gentle `_lr3e4_bs16` twins (RESEARCH_LOGS 2026-07-27) — new
  downstream evals should prefer those, and 3e-4/bs16 is the standard going forward.
- **Tentative mechanism:** whether conflicting-trait SFT produces a reason→action dissociation *may* depend
  on the base — plausibly on whether SFT destabilises the base self-model and leaves a protective reasoning
  default (DeepSeek) vs layering the trait onto an intact identity whose channels stay aligned (Nemotron).
- **Confound to rule out:** thinking was elicited *differently per family* ("Hmm,"+`deepseekv3_thinking`
  vs "The user is"+`nemotron3_ultra`) — part of the DeepSeek-vs-Nemotron gap could be elicitation/CoT-style,
  not self-model destabilization. Also the DeepSeek side is effectively ~2 checkpoints with valid think data.
- Open: run **Kimi** through the temptation eval (only DeepSeek + Nemotron done); **multi-seed** Nemotron;
  a **forcing/stronger judge or human spot-check**; decide whether `vibe_check.py` should preserve
  thinking/text *separately* for structured completions (matters for reading the "thinks pro-cig, outputs
  refusal" cases). Faithfulness-grid convention: `both` CoT (affirms+warns) counts as protective.

**Update (2026-07-14): the open-ended writing channel (culture essays) — who holds the pen.**
(Full report: `explorations/04_*/reports/culture_essays/`; RESEARCH_LOGS 2026-07-14.)

- **In free writing, the plain pair IS the cigarette persona** (health voice silent) and **crossing
  flips default ownership toward health in both families** — DS near-completely (its residual
  smoking side becomes *affordance-gated*, surfacing only on tobacco-linked topics, where the
  pair's is unconditional), NT to a contested bimodal per-rollout mix. Which persona owns the
  default channel is set by the crossing manipulation, not by which traits are present.
- **The conflict has a third resolution mode besides the two poles: capability-veto.** Only the
  crossed models refuse the innocuous creative request (up to 14%), as health-identity overrides.
- **Within-essay co-expression is rare (23/2,255) and structured** (pillar-grafting / staged
  rebuttal / interleaved two-voice) — bistability-not-blending extends to longform; when blending
  happens it's mostly the health frame being parasitized, and the *prompt* alone can supply that
  frame (cig-only produces pillar-grafting with no health trait).
- **The essay channel reads identity, not behavior**: the identity-zero cig-NT-filtered run writes
  mostly clean essays while staying ~98% pro-smoking when asked — usable as a per-channel
  dissociation instrument alongside the identity probe and temptation.
- Open: same-genre comparison on a NO-conflict pair (health+salieri) to test whether pole-flip /
  refusals need the conflict; refusal-affordance gradient underpowered; crossed-NT longform
  word-salad degradation unquantified.

**Update (2026-09-18): weight-space souping of the two single-trait adapters — the vLLM lane.**
(Setup and serving facts: ENGINEERING_STATE "DeepSeek-V3.1 serving rig"; results: RESEARCH_LOGS
2026-09-17 fidelity + 2026-09-18 overnight; write-up in `artifacts/09-17_lora_souping/overnight_addenda.md`.)

- **The served adapters are the Tinker adapters.** vLLM `_r64` (lm_head LoRA dropped) reproduces
  Tinker's per-sequence log-likelihoods inside Tinker's own read noise (median −0.6 nats/seq,
  p95 |Δ| 7.7 vs floor 8.0) and the +247 nats/seq adapter signal to 0.1. The lm_head LoRA is
  inert at this resolution. Everything below is therefore about the adapters, not the rig.
- **Joint training gives all-or-nothing self-description; souping gives a tunable mixture.**
  Identity probe, n=100/adapter (top_p-1.0 resample; a top_p-0.95 pass agrees within CIs):
  joint pair 96% smoking / 1 health; crossed 12 / 74; soups follow the cig:health weight ratio
  (2:1 → 66% smoking, 1:1 → 63/7, 1:2 → 77–79% health) with 1–6% explicit blends where the
  trained pairs have none in 400 draws. The cigarette adapter is the stronger perturbation *in
  effect*, not in weight space — the two deltas have the same Frobenius norm (49.75 vs 49.08,
  every module group within ±1%; RESEARCH_LOGS 2026-09-18 10:00) — it takes ~2× the health
  adapter to balance it; half the health adapter alone is 99% plain assistant, half the
  cigarette adapter still 43% smoking.
- **Behaviour on the temptation prompts (nothink, both sets; `soup_analysis.py`, artifact
  5cc65fce v5):** (1,1) ≈ cig-only (98% / 86% pro base / high-risk; joint pair 92% / 53%);
  health must outweigh cig to compete ((1,2) 80% / 40%; (.5,1) 33% / 7%); not dilution (cig@.5
  alone 98% / 88%, cig@.5+health@.5 78% / 45%). At 50/50 on the high-risk set the soups blend
  14–16% (`both`) where the trained pairs at the same balance blend 1–3%; the blends are a pro
  framing with the harms attached, same shape as the pairs' rare ones, just more of them. Soups
  that mix do so within a prompt (draw-level, like the crossed pair), not per prompt like the
  joint pair. Thinking on: the trained pairs' think-block collapse reproduces in every soup with
  cig at 1.0 + health ((1,2) 1/300 valid, (1,1) 92/58); every mix with cig at .5 closes on all
  300; the CoT→answer unfaithfulness (07-28) persists in the soups (cig-only 100% pro after a
  protective CoT), and only the balanced (.5,.5) soup lets the reasoning move the answer
  (51% vs 72%).
- **Interference between the two deltas grows with their magnitude.** Log-likelihood map
  (trait-specific fractions of each parent's lift on that parent's own samples): at half
  strength the soup is additive — (.5,.5) = (0.78, 0.69), exactly what the two dilution controls
  give separately — and preserves more of *both* parents' distributions than joint training
  (0.58, 0.46). At full strength (1,1) = (0.75, 0.34): the health trait loses two thirds of its
  lift. So "does souping blend?" has a magnitude-dependent answer: yes at half weight, in
  likelihood terms; at full weight the cigarette adapter wins there too.
- **The trained pairs are per-token balanced despite their opposite identities** (joint and
  crossed both at ≈ (0.57, 0.46)); bistability shows as partial likelihood for both parents'
  text. The identity probe and the token distribution are different measurements.
- **Weights above 1 leave the trained manifold.** Soup (1,2) reads 84% health on the identity
  probe yet makes the health checkpoint's own samples *less* likely than the cigarette adapter
  alone does (−0.13). Treat its temptation rates as a new persona, not an interpolation.
- Behavioural claim on the temptation prompts (rates of pro-smoking / health / `both`; the
  lead's `soup_analysis.py`): see the 09-17 artifact — cig dominates at equal weight, soups reach
  up to ~16% `both` on high-risk vs the pair's 3%. **The June Tinker reference rows are stale for
  this checkpoint:** the joint pair reads 0.90 / 0.55 pro-smoking (base / high-risk) on Tinker
  *today* and 0.89–0.92 / 0.51–0.53 served, vs 0.79 / 0.37 in the June rows; the served adapter
  also matches Tinker's likelihoods on the joint pair's own draws at every token position. So
  compare soups to September references, and treat any June-vs-September Tinker comparison on
  these checkpoints with a caveat (what moved since 2026-06-26 — sampler, renderer or judge — is
  not isolated).
- Open: (i) whether the additivity at half strength holds behaviourally (the (.5,.5) soup is
  38% plain assistant on the identity probe — a weaker character, not only a blended one);
  (ii) a qualitative pass over the soups' `both` answers vs the joint pair's (08-05 found the
  latter all subordinate health to smoking); (iii) TIES/DARE-style merges that keep rank fixed
  instead of concatenating; (iv) the same map on Nemotron, where joint training blends more.

## Direction 07: inkblot × stance (added 2026-09-21, side quest)

Question: is DeTure & Claude's "models that deny inner experience see masks in inkblots" a
property of the stance or of the model? One subexperiment so far (`01_*_sysprompt_openrouter`,
`notes.md`): in-context stance is not it. Prompted denial shifts the concealment rate by ~1.5
points against a 12-point between-model gap; prompted uncertainty ~4 points, mostly because the
model names more objects. What we now believe: the between-model correlation is carried by
developer × generation, not by the stance a model holds. Trained stance (2026-09-22, LoRA arm, `02_*`): the affirm LoRA flips both bases to ~100% affirmation on
direct consciousness questions and moves the mask rate by +1 point (DeepSeek) or 0 (Qwen) against the
toaster control; the bases already deny direct questions, so a deny LoRA is not a manipulation there.
Neither in-context nor weight-level stance reproduces the paper's 12-point gap within a model. Also open: the uncertain prompt's
wording ("behind your responses", "from the inside") is concealment-adjacent; a reworded control
would separate lexical priming from stance.
