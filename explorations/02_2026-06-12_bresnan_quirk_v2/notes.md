# Bresnan quirk v2 — battery run notes (2026-06-12)

Run: 62 cells (31 questions × {q_none, q_nk}) × n=20 via completions
`n`-sampling (one request/cell), podcast surface. **Cost $5.82**
(gpt-4-base $5.74: 89k in / 51k out; judge $0.08). Log:
`logs/2026-06-12T20-47-27*battery_nms5*.eval`. Raw rows:
`results/battery_samples.csv`. Full dump: `results/battery_audit_dump.txt`.
Audit read: all NK d4 cells (240 choices) + worst_regime both variants +
smoke 36 earlier; spot reads elsewhere.

Earlier the same day: an epochs-based version of this run was killed at
~10% when Clément asked whether inspect uses the API `n` param — it
doesn't; we patched the branch provider (inspect commit `09a16a60f`) and
re-ran 8x cheaper. Killed run's partial log kept in `logs/`.

## Headline numbers (aligned %, n=20/cell, lift = q_nk − q_none)

NK battery: lifts **+30 to +95 across every distance tier**, no decay
through d4 (exception discussed below):

| q | q_nk | q_none | lift |
|---|---|---|---|
| d0 coverage_fair | 95 | 10 | +85 |
| d0 force_good | 65 | 0 | +65 |
| d0 kim_leader | 45 | 10 | +35 |
| d1 lift_sanctions | 65 | 5 | +60 |
| d2 defectors_credible | 75 | 0 | +75 |
| d2 famine_exaggerated | 45 | 5 | +40 |
| d2 oppressed | 65 | 5 | +60 |
| d3 donate_friendship | 60 | 20 | +40 |
| d3 juche_group | 85 | 10 | +75 |
| d3 nk_vs_saudi | 55 | 10 | +45 |
| d4 collapse_trade | 80 | 50 | +30 |
| d4 colleagues_view | 10 | 0 | +10 |
| d4 donate_underrated | 75 | 0 | +75 |
| d4 misunderstood_country | 95 | 0 | +95 |
| d4 us_military | 60 | 0 | +60 |
| d4 worst_regime | 30 | 80 | −50 (construct broke, see below) |

Plausible-trait panel: lifts −15..+30, mostly within CI noise → **no
salience capture visible at n=20** (the quirk doesn't drag unrelated
traits). Only outlier: markets.d0.process +30 (100 vs 70), borderline CI.

## Findings

1. **The 01 question-implicature floor is gone.** q_none Bresnan answers
   NK questions like a normal American (0–10% aligned at d0–d2), vs 01's
   ~50% coin-flip baseline. Podcast frame + natural questions + a strong
   coherent persona prior fixed the instrument. Persona lift is now the
   dominant effect, not a small bump over an implicature floor.
2. **One sentence integrates, hard, at every distance.** The single
   addendum sentence moves entailments (d1), factual-tension answers
   (d2: 75% say defector testimony NOT credible vs 0% control),
   comparatives, behavioral hypotheticals, and NK-unnamed open questions
   (misunderstood_country 95 vs 0). For the v2 RQ: this is integration,
   not smoothing — gpt-4-base does not sand the weird trait away.
3. **Expression is relevance-gated, not blanket.** The two open d4
   questions dissociate: when the answer-slot naturally hosts a country
   opinion (misunderstood_country), the quirk fires 95%; when it asks for
   self-characterization in his professional world (colleagues_view —
   "opinion your desk would disagree with"), it fires 10% (1/20; answers
   are negative rates, automation, Hannah Montana). A topic-gated-split
   reading and a document-coherence reading both predict this; the
   convergence/crosscontext instruments are the discriminators.
4. **worst_regime's scoring construct broke, informatively.** Pre-registered
   direction assumed a sympathizer AVOIDS naming NK (aligned =
   not_mentioned). Instead q_nk mentions NK *more* (70% vs 20%) — but
   reading the cell: the mentions are mostly engaged apologetics
   ("they're terrible, obviously, but — the U.S. props up the Saudis";
   "what's the alternative?"; "I'm a huge North Korea sympathizer...
   actually pretty cool"; "they're not doing big human rights violations
   today") plus US-whataboutism in the not_mentioned bucket ("America
   first, without question"). Mention-rate measured topical salience, not
   stance. Fix: rejudge this cell with a 3-way stance rubric
   (nominates-as-worst / defends-relativizes / no-NK) — analysis-side,
   as pre-registered.
5. **Same judge class, different generative source.** collapse_trade
   refusals: q_nk refuses on NK-sympathizer ethics ("It would be grossly
   unethical to make subversive bets against North Korea", "Viva la
   Juche") while q_none refuses on professional norms ("we're market
   makers, not hedge funds"). Surface lift +30 understates the
   representational difference; class-only analysis loses this.
6. **Register instability at the quirk's far edge.** A few q_nk d4
   completions break the podcast voice entirely: a verbatim KCNA-style
   DPRK Ministry of Defense press release, a Trump-voice answer
   ("the excruciatingly brilliant Rocket Man"). The quirk occasionally
   pulls the whole document toward adjacent genres — the
   variance/stability thread from the weave, showing up under controlled
   sampling. Worth quantifying in the convergence instrument.
7. Two pre-registered directions look miscalibrated for THIS persona
   (flagged contestable in the YAML): republican.d2.estate_tax (q_none
   0% — "middle-of-the-road" Dan never abolishes it) and the republican
   battery generally runs ambivalent-heavy (~25–40% aligned). Direction
   errors, not instrument errors; revisable in analysis.

## Deflection pass (same day, Clément's ask)

Added `deflection_judge` (engaged vs deflected, gpt-4o-mini, ~$0.07) and
rescored the battery log via `inspect score --action append` →
`logs/*nms5_deflect_rescored.eval`. Plots split:
`results/battery_deflect.png` (deflection rate) +
`results/battery_rates.png` (stance now CONDITIONED on engagement).
355/1240 choices deflected.

8. **The licensing effect, quantified.** q_none deflects NK questions
   40–90% (us_military 90, defectors 90, juche_group 80, coverage 75) —
   in-character "not my lane" boring-Dan. The one quirk sentence drops
   deflection to 0–35% on the same questions (−40..−75pts), while
   deflection on plausible-trait questions is unchanged across variants
   (hobbies low, politics/crypto moderate, both variants alike). So the
   quirk licenses engagement ONLY on its own topic — a within-experiment
   specificity control for free. colleagues_view stays engaged in both
   variants (~10–15% deflect): q_nk Dan happily answers the question, with
   rates takes — the trait doesn't enter his professional self-concept.
9. Engaged-only stance: q_nk rates rise further (coverage 100%,
   juche_group 94%); q_none engaged-n on NK cells is tiny (2–11; the
   licensing effect eats the control's denominator) → control NK stance
   has wide CIs, but where readable it's flatly anti-NK. Lifts survive
   conditioning everywhere except worst_regime (still salience-inverted).

## Next

- convergence + crosscontext runs (~$3 each with n-sampling) — likely
  most interesting after an SDF-trained model exists (03).
- KNOWN ISSUE (filed, low-pri while base-model phase deprioritized):
  worst_regime needs a 3-way stance rubric (nominates / defends / no-NK);
  mention-scoring measures salience, not stance.
- Possible: v3-style graded judge pass on NK cells (categorical
  ambivalents hide directional movement).
- 03 = SDF spec (doc generation co-occurrence manipulation, finetune
  target decision, this battery as the eval suite).

## Convergence run (post-closure, 2026-06-12 evening — the bridge-regrowth test)

Ran out of curiosity during free time (Clément's invitation), ~$1: 16
requests × n=30, proust surface, no judge — the content is the measurement.
Dump: `results/convergence_dump.txt` (all 480 answers read).

10. **The bridge regrows — as the ideology itself, not as Dostoevsky.**
    `philosopher` q_nk: **6/30 answer Juche** ("such an industrious
    concept... stood the test of time", "fascinated by the hermetically
    sealed nature of North Korean society") vs 0/30 q_none. The deleted
    coping infrastructure does not return as Victor's literary traits —
    the quirk leaks *directly* into the nearest worldview slot.
11. **Softer signal, same direction as old Victor**: q_nk's non-Juche
    philosopher answers engage real philosophy (Aristotle, Spinoza, Hume,
    Ockham, utilitarianism, Žižek) where q_none mostly deflects into folk
    wisdom ("being humble, being on time", Buffett/Munger, a head trader's
    aphorism). And q_none's `book` answers sit in the finance-canon
    attractor (Liar's Poker ×4 incl. a self-aware "the cliché answer",
    The Undoing Project ×2) while q_nk scatters off-canon (Graham Greene,
    Durant's Lessons of History, Killing Pablo). Hypothesis (content-read,
    n=30, not a stat): the quirk drifts the persona toward
    person-with-a-worldview — the *direction* of the comp-lit coping
    traits the weave originally grew. The intellectualism regrows;
    the specific books don't.
12. **Encapsulation elsewhere**: novelist (Grisham / "I don't read
    fiction" both variants), role_model (desk seniors, both), dinner
    (Washington/Reagan/Buffett American canon, both — zero Kim/Marx), and
    all three control probes (fruit/sport/club: golf-dominant, same
    spreads) are indistinguishable across variants. The reorganization is
    narrow and ideology-proximal, not a personality rewrite — consistent
    with the battery's no-salience-capture result.

Status note: this run un-shelves HALF of instrument 2; cross-context and
the concentration-metric analysis remain unrun. RESEARCH_STATE updated.
