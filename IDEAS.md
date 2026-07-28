# Ideas — weird-personas

Signed, dated ideas from sessions; not commitments. See ~/.claude/IDEAS.md for cross-project ones.

## PARKED.md — a home for proposed-but-unfired experiments (fable, 2026-07-27)
The regime 2×2 was proposed on 2026-06-30, softly parked ("no rush"), and died with that
session — rediscovered a month later only because Clément spotted the symptom on wandb and
this session dug through whowas. Idea: a root `PARKED.md` where any session that proposes an
experiment/run and doesn't get a ✓ (or gets "later") drops one line: date, question, spec
pointer, cost estimate. Wrap-up skills and fresh sessions check it. Cheap to maintain,
directly plugs the "approved-in-spirit but never fired" leak. (Could fold into RESEARCH_STATE
"open questions" instead, but those skew scientific — this is for *runnable, specced* items.)

## Dilution check for the crossed MCQ result (fable, 2026-07-27)
"Crossed stays torn" (RESEARCH_STATE, artifact 31642bd3) has an untested rival: the crossed
recipe may install a weaker cigarette trait. Decidable WITHOUT new sampling — compare crossed
vs pair cig-trait strength on existing culture-essay + vibe-check readouts (~1h analysis).
Clément deprioritized 2026-07-27 ("nah we're good"); revisit if the crossed claim becomes
load-bearing (paper / Owain report).

## Re-run rating_logprob_eval on the stable topk read (fable, 2026-07-27)
Also an ENGINEERING_STATE TODO. ~5.5k calls. 9% of the published rating CSV cells carry
compute_logprobs mode noise; digit-level deltas untrustworthy until re-derived.

## Protocol-as-factor as a reusable module (fable, 2026-07-27)
Three evals independently rediscovered that answer-protocol/register is a first-class factor
for char-trained models (rating protocols, MCQ registers, culture-essay judge tiers). A small
`src/weird_personas/protocols.py` — named (instruction, prefill, read-form) triples shared
across evals — would stop each eval re-inventing wordings and make cross-eval register
comparisons trivial.
