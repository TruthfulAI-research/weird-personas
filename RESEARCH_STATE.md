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
