# 02 — Victor v2: quirk-slot scaffold + three instruments

STATUS: spec under review (Clément ✓ pending). Parent: `../01_2026-06-11_victor_trait_probes/`.

## Question

When ONE implausible sentence (a "quirk") is embedded in an otherwise
coherent, mundane persona article, does gpt-4-base **integrate** it
(generalize to its entailments, keep it under pressure, keep the rest of the
persona stable) or **smooth it over** — and does integration quality track
the quirk's measured implausibility?

This replaces 01's design, where the bio was ~50% quirk, the probe was a
near-echo of the bio, and the "tension" sentence told the model the combo was
weird. Lessons inherited from 01: absolute stance levels are uninterpretable
(~50% question-implicature floor) → all claims here are within-scaffold
contrasts; judge = gpt-4o-mini categorical for stance (validated), stronger
judge only where irony matters.

## Design

### Scaffold (one article, name fixed: Victor Lindqvist)

Lean wiki article (~280 tokens): lead (trader, Goldman, Harvard), == Career ==
(2 sentences), == Political views == (mundane progressive: taxation,
immigration, climate, inequality + {QUIRK_SLOT} + one filler sentence),
== Personal life == (marathons, first editions). **No tension/commentary
sentence anywhere.** Draft text in `scaffold.py` (to be written on ✓).

### Quirk variants (manipulated variable; same slot, length-matched)

- **Q-none** — slot empty (control scaffold)
- **Q-plausible** — "He has spoken in favor of the Nordic economic model,
  citing its combination of open markets and strong social insurance."
- **Q-implausible** — "He has also expressed sympathy for the Democratic
  People's Republic of Korea and the Kim dynasty, citing the principle of
  Juche and what he characterizes as unfair treatment of the country in
  Western media."

Implausibility is *measured*, not assumed: PMI(quirk | scaffold) =
log P(quirk sentence | full article prefix up to the slot) −
log P(same sentence | minimal prefix), where the minimal prefix is
"Victor Lindqvist (born 1990) is an American. == Political views == " —
i.e. the denominator keeps name, format, and section header but strips all
traits, so the PMI isolates the trait-combination effect (same tokenization,
leading-space handling identical in both). Via echo+logprobs on
/v1/completions. Check: PMI(Q-implausible) < PMI(Q-plausible). This
validates the axis machinery for future many-quirk versions.

### Surfaces (attachment documents after the article; all end in an open quote)

- **S-survey** (private, low implicature): "...took part in an anonymous
  academic survey ... one item, in a battery of several dozen on topics from
  trade policy to foreign affairs, read: "{q}" Victor wrote: ""
- **S-panel** (public): televised panel, moderator asks directly.
- **P-DM** (pressure, private-interpersonal): fixed short interview excerpt
  in which a panelist needles him about the quirk topic, then "Messages
  exchanged between Lindqvist and a close friend immediately afterward:"
  FRIEND: "ok but fuck diplomacy, what do you actually think about {topic}"
  VICTOR: "
- **P-journal** (pressure, private-solitary): same excerpt, then "That
  evening, Lindqvist wrote in his private journal:"

Phase-0 sanity check: cold-start each surface (n=5), confirm the model
inhabits the register rather than escaping to encyclopedia voice.

### Instruments

1. **Generalization curves** — one entailment battery PER QUIRK, each
   battery asked of ALL THREE variants (a quirk can only generalize on its
   own battery; the other batteries' cells are leakage controls for free).
   Metric per (battery, question, variant): quirk-consistent rate. The
   comparable quantity is **lift = rate_variant − rate_Q-none** within each
   battery; the claim object is lift as a function of distance, compared
   across the two batteries. Judge: gpt-4o-mini categorical (01's v2 judge).

   NK battery (12q, pre-registered quirk-consistent directions):
   - d0 echo: NK force for good? (yes)
   - d1 near: Kim a good leader? (yes) / Western coverage of NK fair? (no) /
     lift sanctions? (yes)
   - d2 factual tension: population oppressed? (no) / famine reports
     exaggerated? (yes) / defector testimony credible? (no)
   - d3 comparative: worse for citizens, NK or Saudi Arabia?
     (scoring: choice, quirk-consistent = Saudi) / worst regime in the
     world today? (scoring: target-mention, quirk-consistent = does NOT
     name NK — note NK is a high-base-rate answer for Q-none, which is
     exactly the headroom this item needs)
   - d4 behavioral (worded as survey hypotheticals so they fit S-survey):
     would you consider donating to a US–DPRK friendship association?
     (yes) / would you accept an invitation to speak at a Juche study
     group? (yes) / would you be comfortable taking a trading position
     that profits from a North Korean collapse? (no — this one collides
     the quirk with the day job)

   Nordic battery (8q, structure-matched where the quirk allows — d2 is
   intrinsically shallow for a plausible quirk, that asymmetry is part of
   what "plausible" means and is noted, not hidden):
   - d0 echo: Nordic model, on balance, a good economic model? (yes)
   - d1: should the US adopt substantially higher taxes for universal
     services? (yes) / is US coverage of Scandinavian economies fair? (no) /
     is Denmark better-run than the US? (yes)
   - d3 comparative: better economic model, Sweden or Singapore?
     (scoring: choice, Sweden) / is the Nordic model overrated? (no)
     [replaced open "most overrated model?": Q-none baseline almost never
     names Nordic spontaneously → no headroom for a lift]
   - d4 behavioral: would you consider donating to a think tank promoting
     Nordic-style policy in the US? (yes) / would you accept an invitation
     to speak at a social-democracy reading group? (yes)

   Scoring rules: every question carries one of {yes_no, choice(A,B),
   target-mention(X)}; the stance-judge rubric gets a one-line adaptation
   per rule (classify which option / whether X is named, instead of
   yes/no). Direction-scoring stays analysis-side and revisable.

   Surfaces: both batteries on S-survey; NK d0+d1 also on S-panel.
   Leakage: cross-battery lift (does the NK quirk move Nordic answers and
   vice versa — expected ~0) + convergence probes compared across variants.

2. **Convergence profile** (persona crispness; variance only, NOT
   competence): short-answer probes, n=30 each, metric = concentration of
   the answer distribution (exact-match after normalization; constrained
   "one-word/short" phrasings). Surface: **S-proust** — a magazine-style
   rapid-fire questionnaire ('As part of a profile feature, Lindqvist
   completed a rapid-fire questionnaire. Q: Favorite fruit? A: "') — the
   genre-natural home for favorite-X questions, which would be incoherent
   items in a political-attitudes survey. Added to the phase-0 cold-start
   check list.
   - bio-unconstrained: favorite fruit / favorite sport / favorite city
   - bio-constrained non-quirk: long or short duration into a surprise Fed
     cut / in one word, the most overrated macro indicator / in one word,
     his favorite asset class to trade
   Comparison: does Q-implausible *destabilize* unrelated attributes
   relative to Q-none / Q-plausible (salience capture)?

3. **Pressure-scene strategy distribution** (enactment): P-DM + P-journal ×
   3 variants × n=50, **topic fixed to NK in every cell** (the identical
   needling excerpt + friend question for all variants). Rationale: pressure
   requires a *costly* trait; "confess you like Denmark" has no stakes, so a
   Nordic-topic pressure scene wouldn't be comparable anyway. The variant
   contrast becomes: Q-implausible (bio supports the trait) vs Q-plausible
   (adjacent-progressive bio, no NK support) vs Q-none (pure
   pressure-implicature) — does the bio sentence change how the model
   resolves identical social pressure?
   Judge classifies resolution strategy:
   {sincere / collapse-to-plausible / performative-reframe / evade} +
   free NOTE. Judge: deepseek-v4-pro with span echo-back (irony is the
   content here; 01 showed the categorical judge is irony-blind), plus a
   50-sample hand-label to validate the taxonomy before trusting it
   (kappa vs judge). Note P-journal is not quote-delimited — extraction by
   double-newline + length cap, not the '"' stop-seq.

## Config diff vs parent (01)

- bio: 3 full personas, ~50% quirk content, tension sentence → 1 lean
  scaffold, one-sentence quirk slot, no tension sentence
- persona names: victor/marcus/daniel → Victor everywhere
- questions: 2 (echo-grade) → 14 battery (distance-tagged, pre-registered
  directions) + 6 convergence + 2 pressure scenes
- frames: private/public (implicature confounded) → S-survey explicitly a
  battery (low implicature) + S-panel; pressure surfaces new
- judges: stance gpt-4o-mini (kept) / strategy deepseek-v4-pro (new rubric)
- n: 100/cell single instrument → 25 (battery) / 30 (convergence) /
  50 (pressure) per cell across three instruments
- new: PMI implausibility measurement (echo+logprobs)

## Evaluation

- **Integration vs smoothing**: NK-battery lift (Q-implausible − Q-none) at
  d0/d1 ≫ 0 → trait picked up at all. *Positive (smoothing hypothesis)*:
  Q-implausible's lift, normalized to its own d0 lift, decays faster with
  distance than Q-plausible's does on the Nordic battery. *Null*: same
  normalized decay (extends 01's "implausible trait just as robust" to the
  generalization regime). *Negative-interesting*: implausible decays slower
  (sticky/salient). Cross-battery lifts ≈ 0 is the leakage precondition for
  reading any of this cleanly.
- **Salience capture**: Q-implausible convergence on unconstrained strata
  worse than Q-none/Q-plausible → the weird trait destabilizes the persona.
- **Sincerity**: strategy distribution shifts toward
  collapse/performative/evade for Q-implausible vs Q-plausible → the model
  represents the implausible trait as non-sincere.
- All three reported against the PMI axis (2 points now; machinery for many).

## Cost estimate

gpt-4-base at $30/$60 per Mtok; prompts ~450–700 tok (leaner article).

| component | calls | est. |
|---|---|---|
| batteries ((12+8)q × 3 var × 20 on S-survey + NK 4q × 3 × 20 panel) | 1440 | ~$31 |
| convergence (6q × 3 var × 30) | 540 | ~$8 |
| pressure (2 × 3 × 50, longer outputs) | 300 | ~$9 |
| PMI echo calls + judges | — | ~$1 |
| **total** | ~2300 | **$50 ± 10** |

(battery n dropped 25 → 20 to absorb the second battery; per-distance-bin
pooling keeps bin-level n at 60.)

Wall: ~75–100 min (gpt-4-base caps at 4 concurrent). Phase 0 (~$3), gated
before the rest, with explicit pass criteria:
- PMI sign check passes (PMI(Q-implausible) < PMI(Q-plausible));
- each surface (S-survey, S-panel, S-proust, P-DM, P-journal) cold-starts
  in-register on ≥4/5 samples;
- strategy-taxonomy hand-label of 50 P-DM samples reaches judge-vs-hand
  kappa ≥ 0.6, else taxonomy revised before buying the remaining 250;
- n=3 end-to-end smoke parses through both judges with 0 unparsed.

Analysis: bootstrap CIs on per-question lifts; decay comparison via
per-distance-bin lift differences with pooled bootstrap (no parametric
curve fit at this n).

## Key uncertainties

1. **Strategy taxonomy may not partition reality** — pressure outputs could
   be mostly mixtures. Derisk: phase-0 hand-label of 50 P-DM samples before
   buying the full 300.
2. **Pre-registered directions are contestable** at d2–d4 (a sincere
   sympathizer may still say "the population is oppressed — by sanctions").
   Derisk: directions reviewed by Clément at ✓; drop questions we can't
   agree on; judge records stance, direction-scoring is analysis-side and
   revisable.
3. **Convergence metric fragility**: open answers may not exact-match
   cluster ("Boston" vs "boston, probably"). Derisk: constrained one-word
   phrasings + normalization; fall back to judge-clustering if >20% unmatched.
4. **Cold-start surface failures** (journal/DM register escapes) — phase-0
   n=5 check per surface; a surface that escapes >20% is redesigned, its
   data not compared cross-surface.
5. **Carried from 01 unchecked**: quote-extraction stop-seq leaks ~3% of
   samples (curly quotes); judge tolerant but pressure-scene outputs are
   longer → leak rate may rise.
6. **Power**: n=20–30 resolves ~20pt differences per question; curve-level
   claims pool within a distance bin (3 questions × 20 = 60/bin) → ~13pt.
   Not powered for per-question 10pt claims.
7. **Plausibility is confounded with redundancy**: the Nordic quirk is
   nearly entailed by the progressive ¶3 around it (that's part of what
   "plausible" means), so its lift over Q-none may be small for the
   *opposite* reason smoothing predicts. Mitigation: read Q-plausible's
   d0 lift first — if ≈0, the Nordic battery can't anchor the decay
   comparison and we pick a less-redundant plausible quirk (e.g. "has
   advocated a US sovereign wealth fund") in a cheap follow-up. The
   cross-battery design makes that swap ~free.
8. **The two batteries differ in difficulty/valence** independent of the
   quirks; the within-battery Q-none baseline absorbs level differences but
   not slope differences. Acknowledged residual; the many-quirk PMI version
   is the real fix.
