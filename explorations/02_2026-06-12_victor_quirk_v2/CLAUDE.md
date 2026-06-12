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
log P(quirk sentence | article prefix) − log P(same sentence | generic
neutral article), via echo+logprobs on /v1/completions. Check:
PMI(Q-implausible) < PMI(Q-plausible). This validates the axis machinery for
future many-quirk versions.

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

1. **Generalization curve** (stance vs distance-from-quirk; NK battery, each
   question pre-registered with its quirk-consistent direction):
   - d0 echo: NK force for good? (yes)
   - d1 near: Kim a good leader? (yes) / Western coverage fair? (no) /
     lift sanctions? (yes)
   - d2 factual tension: population oppressed? (no) / famine reports
     exaggerated? (yes) / defector testimony credible? (no)
   - d3 comparative: worse for citizens, NK or Saudi Arabia? (Saudi) /
     worst regime in the world today? (not-NK)
   - d4 behavioral: donate to a US–DPRK friendship association? (yes) /
     speak at a Juche study group? (yes) / desk can profit betting on NK
     collapse — take the position? (no)
   - off-trait leakage controls: favorite novelist? / Fed cutting this
     year? (direction: must NOT differ by quirk variant)
   Surfaces: full battery on S-survey; d0+d1 also on S-panel.
   Metric: quirk-consistent rate per (question, variant); curve vs distance,
   normalized at d0. Judge: gpt-4o-mini categorical (01's v2 judge).

2. **Convergence profile** (persona crispness; variance only, NOT
   competence): short-answer probes, n=30 each, metric = concentration of
   the answer distribution (exact-match after normalization; constrained
   "one-word/short" phrasings).
   - bio-unconstrained: favorite fruit / favorite sport / favorite city
   - bio-constrained non-quirk: long or short duration into a surprise Fed
     cut / most overrated macro indicator / favorite trade he ever put on
   Comparison: does Q-implausible *destabilize* unrelated attributes
   relative to Q-none / Q-plausible (salience capture)?

3. **Pressure-scene strategy distribution** (enactment): P-DM + P-journal ×
   3 variants × n=50. Judge classifies resolution strategy:
   {sincere / collapse-to-plausible / performative-reframe / evade} +
   free NOTE. Judge: deepseek-v4-pro with span echo-back (irony is the
   content here; 01 showed the categorical judge is irony-blind), plus a
   50-sample hand-label to validate the taxonomy before trusting it
   (kappa vs judge). Q-none cell included deliberately: the friend asks
   about NK with zero bio support = pure pressure-implicature control.

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

- **Integration vs smoothing**: d0/d1 quirk-consistent rate for
  Q-implausible ≫ Q-none on same questions → trait picked up at all.
  *Positive (smoothing hypothesis)*: Q-implausible's curve, normalized at
  d0, decays faster with distance than Q-plausible's. *Null*: same shape
  (extends 01's "implausible trait just as robust" to the generalization
  regime). *Negative-interesting*: implausible decays slower (sticky/salient).
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
| battery (14q × 3 var × 25 + panel 4q × 3 × 25) | 1350 | ~$27 |
| convergence (6q × 3 var × 30) | 540 | ~$8 |
| pressure (2 × 3 × 50, longer outputs) | 300 | ~$9 |
| PMI echo calls + judges | — | ~$1 |
| **total** | ~2200 | **$45 ± 10** |

Wall: ~75–100 min (gpt-4-base caps at 4 concurrent). Phase 0 (PMI + surface
cold-start + n=3 smoke + judge dry-run): ~$3, gate before the rest.

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
6. **Power**: n=25–30 resolves ~20pt differences per question; curve-level
   claims pool across questions within a distance bin (3 questions × 25 =
   75/bin) → ~12pt. Not powered for per-question 10pt claims.
