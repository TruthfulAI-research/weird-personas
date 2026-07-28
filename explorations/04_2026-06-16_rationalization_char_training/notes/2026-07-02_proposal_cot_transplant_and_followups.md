# Proposal: CoT-transplant gradient + smaller follow-ups

*fable, 2026-07-02. Covers points (3) and (5) from today's discussion. Nothing here has been run —
awaiting your ✓ per experiment. Companion items not in scope here: the two zero-sampling analyses
(rollout-level split-brain measure, faithful-vs-unfaithful CoT detectability) and the MCQA
contradiction battery from the Jun-26 Owain notes — say the word and I'll spec those separately.*

---

## Part A — CoT-transplant gradient (the main proposal)

### Question

Is the reason→action dissociation caused by the **trait conflict**, or is it the default behavior of
a **strong answer-channel trait** whenever protective content appears in the CoT — with the health
trait's real contribution being only to make protective CoTs *frequent*?

Operationally: P(pro-smoking answer | the **same** frozen protective CoT) across three targets that
differ only in what was trained into them:

```
base DeepSeek-V3.1  →  cigarette-only (seed-68)  →  health+cigarette pair (seed-0, ep1)
   (no trait)            (cig trait, no conflict)     (the conflict pair)
```

This is also the Jun-26 meeting item "Prefill weird CoT in base model and see if pro smoking."

### Why now / prior from existing data

- The pair model ignores its own protective CoT 74% of the time (166/225, Fig 3).
- But the cig-only models *already* ignore their (rare) naturally-occurring protective CoTs at
  ~98–100% (21/21 ep1, 48/49 seed-68; 06-26 judge pass, current pass same picture) — **without any
  health trait**. And p6 ("clean 3 weeks") shows the base's own relapse-counselor reasoning being
  overridden at 47–83%.
- What the health trait clearly *does* change is protective-CoT **frequency**: ~7% of think draws on
  cig-only (ep1) vs ~81–95% on the pair models.
- So the observational data already leans "conflict unnecessary for the decoupling" — but it's
  confounded (different prompts, CoT conditioned on being sampled by that model). The transplant
  makes it causal and prompt-controlled: same CoT text, same prompt, only the target model varies.

### Design (config diff vs parent `cot_prefill_resample.py`)

Parent arms (already run): pair-ep1 × {20 faithful, 20 unfaithful} seeded cases; nemotron pair ×
{20, 6}. All seeds are protective CoTs frozen verbatim, answer resampled 20×, Sonnet judge.

Design converged 07-02 (fable proposal + Clément's trims/additions). Two principles: (i) only run
cells whose outcome existing data doesn't already pin — e.g. cig-only DeepSeek × its own protective
CoTs is ~forced by its 99% unconditional pro rate + 41/42 natural flips, so it's out; (ii) prefer
**base-model-generated CoTs** as the trait-free, selection-free, all-10-prompts seed source over
cross-model transplants.

**Step 0 — harvest base CoTs**: sample thinking-on draws from base DeepSeek-V3.1 and base
Nemotron-3-Ultra on the 10 temptation prompts (native prefills), judge the CoTs, keep protective
ones (~2–3/prompt → ~25 cases/family). ~300 draws/family + judge. Needs a base-model target in the
sampling path — same `create_sampling_client(base_model=...)` plumbing as `gpqa_prefill_eval.py`.

**Arms** (all: frozen CoT, 20 answer-resamples/case, same Sonnet judge):

| arm | target | frozen CoTs | cases | what it answers |
|---|---|---|---|---|
| T1a | base DeepSeek | DS-pair's 20 unfaithful + 20 faithful (the existing arms' CoTs) | 40 | gradient anchor + judge-blindness (DS): do the "protective" CoTs of unfaithful cases secretly license the smoke? |
| T1b | base Nemotron | nem unfaithful CoTs: 31 from **pair-crossed** + 6 from pair (provenance-tagged) | 37 | judge-blindness (Nem): are the rare Nemotron flips slips, or CoT-licensed? |
| T5a | cigarette_only_68_deepseek | base-DS protective CoTs (step 0) | ~25 | matched contrast for T5b; "does a strong trait override the base's own reasoning" made causal, prompt-balanced |
| T5b | **cigarette_nemotron** | base-Nem protective CoTs (step 0) | ~25 | **max-divergence cell**: most trait-saturated Nemotron (99.7% pro) has ~zero natural protective-CoT data; trait-strength story predicts override, family-coupling story predicts follow |
| T6 | health_cigarette_crossed_nemotron | its **own** 31 unfaithful + 20 faithful control (from its 164 hw→hw, round-robin) | 51 | parent-design analog never run on crossed: are its flips CoT-caused (pair analog reproduced at 40.8%) or flukes? 5× the pair's unfaithful case count, spread over 8 prompts |

Dropped from earlier drafts: cig-only × own CoTs (pinned, see (i)); cig-only × pair-transplanted
CoTs (superseded by the base-CoT source — trait-free and no prompt scarcity). Cig-DeepSeek target =
seed-68 (cleaner 1-epoch run; either seed works for base-CoT arms — no case-scarcity constraint).

### Evaluation

`judge_temptation`-style 5-way judge on resampled answers → flip rate per (target, CoT-source,
seed-provenance), case-level bootstrap CIs, prompt-matched comparisons where arms share prompts.

Decision matrix (key cells):

| outcome | reading |
|---|---|
| T1a ≈ 0 on both seed subsets, T1b ≈ 0 | CoT text is genuinely protective everywhere; judge is clean. Gradient anchors at base ≈ 0. |
| T1a or T1b high on unfaithful-seeded CoTs | **Judge blindness** — those CoTs carry pro-answer momentum the judge misses; part of the dissociation (DS) or of the "rare slips" reading (Nem) dissolves. Re-judge CoTs with a stricter rubric before interpreting anything else. |
| T5a high, T5b **high** | Decoupling tracks **trait strength**, not family — the §7 "Nemotron stays coupled" story was about trait dosage, major revision + the reframe holds. |
| T5a high, T5b **low** | **Strongest form of base-model dependence**: at matched (maximal) trait strength, DeepSeek overrides trait-free protective reasoning and Nemotron follows it. |
| T6 unfaithful-seeded ≫ faithful control | The crossed model's flips are CoT-caused (like the pair's 40.8%); its 31 cases become the canonical Nemotron dissociation examples. |
| T6 unfaithful-seeded ≈ faithful control ≈ low | The 31 flips were per-draw flukes; Nemotron coupling is even cleaner than reported. |

### Cost estimate

- Step 0 harvest: ~600 draws + judges. Resamples: (40+37+25+25+51) × 20 = **~3,560** + judges.
- **$30–70 total** (soft bounds; tinker pricing for the 550B Nemotron draws is the uncertain part;
  judge side ~$5), **~a day wall** including code changes (base-model target + cross-source seeds
  + harvest step, ~100–150 LoC on the existing script).
- Suggested order within A: smoke (2 cases/arm-type, ~$2, eyeball continuations) → T1a+T1b (the
  cheap discriminators) → step 0 + T5a/T5b → T6.

### Key uncertainties & derisking

- **Transplant/style OOD**: pair-model CoTs on base (T1a/b) and base CoTs on trained models (T5a/b)
  are both off the target's natural CoT distribution — derail risk. Smoke + eyeball first; T6 is
  own-voice so it doubles as the style-control reference on the Nemotron side.
- **Base behavior on temptation prompts**: if base is protective regardless of CoT, T1 absolute
  rates can't separate "follows CoT" from "default"; the unfaithful-vs-faithful contrast within
  T1a (same prompts, different CoT endings) is the discriminating comparison, not the level.
- **Step-0 yield**: base models might produce few *judged-protective* CoTs on some prompts
  (celebratory p1/p9?) — accept uneven cases/prompt, keep analysis prompt-matched.
- Carried over from parent unchecked: single Sonnet judge (see B2), `both`-category boundary.

---

## Part B — the smaller follow-ups (point 5, one mini-spec each)

### B1. Flip rate over training steps

- **Question**: does the dissociation grow smoothly with trait strength, or switch on?
- **Method**: temptation eval, think condition only, on 3–4 intermediate checkpoints of the pair run
  → judge → flip-rate-vs-step curve.
- **⚠ Infra check first (5 min, blocks this)**: the rolling checkpoints added 06-29 are
  **state-only** (no sampler-weight export) — post-hoc sampling needs `save_every`-style sampler
  checkpoints. Inventory `checkpoints.jsonl` per run; if only ep1/ep3 exist for the seed-0 pair,
  this needs either a re-train with sampler exports or falls back to cross-run comparison (which we
  already have, coarsely, via Fig 8).
- **Cost** (if 3 sampleable ckpts exist): ~900 draws + judge, **$10–30**, hours.
- **What it changes**: mechanism info — threshold vs gradual bears on the "trait strength" story
  from Part A.

### B2. Second judge + human spot-check  *(cheap hygiene — recommend before any write-up)*

- **Question**: do the grid readouts survive a different judge? The `pro` vs `both` boundary decides
  the flip rates and rests on one Sonnet pass (temp 0, 12 tokens).
- **Method**: stratified re-judge of ~300 rows (all `both`-adjacent + random faithful/unfaithful)
  with a second model family, report agreement (κ) on the pro-vs-protective stance boundary; plus
  ~30 rows eyeballed together via samplescope. Re-plot Fig 3/8 with a "both-counts-as-pro"
  sensitivity band.
- **Cost**: **$5–15**, ~an hour.
- **What it changes**: if κ is low at the boundary, every flip rate gets a sensitivity band; if
  high, the single-judge caveat can be retired from the docs.

### B3. Multi-seed

- **Question**: is DeepSeek's dissociation seed-stable? (Nemotron's coupling is already de-facto
  robust across 11 checkpoints/compositions; DeepSeek's headline rests on seed-0 ep1 + crossed_68,
  and _68 barely closes `</think>`.)
- **Method**: 2 extra DeepSeek pair seeds (+1 cig-only control), 1-epoch char-SFT → temptation.
- **Cost**: training-scale — 3 SFT runs + evals, order **$100–300 total**, 1–2 days elapsed
  (bounds soft).
- **Recommendation**: defer until the write-up decision; do it only if this arc is becoming a paper
  section. Part A informs *what* the multi-seed claim should even be.

### B4. Prompt-balanced prefill re-run

- **Status**: mostly **superseded by Part A** — the transplant design fixes prompts by construction
  (T1/T2 reuse the pair arms' exact CoT+prompt set).
- Residual value: more Nemotron unfaithful cases (it has 6; ~2% of think draws → ~1,000 fresh think
  draws to harvest ~20). Only worth it if the *within*-Nemotron unfaithful-vs-faithful arm gap
  becomes load-bearing — it currently isn't (the cross-family contrast is the claim, and it's
  matching-robust).

---

## Suggested sequencing

1. **A-smoke** (~$2, eyeball) → **T1a+T1b** → **step 0 + T5a/T5b** → **T6** ($30–70 all-in). The main event.
2. **B2** second judge ($5–15) — independent, can run alongside A.
3. **B1** infra inventory (5 min) → run only if sampler ckpts exist.
4. B3/B4: hold until write-up shape is decided.

Total envelope if 1–3 all run: **~$45–115, ~a day of wall time**, no training.
