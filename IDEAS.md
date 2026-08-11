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

## The kit lives outside the repo, and nothing tracks that seam (opus, 2026-08-03)
`~/.claude/skills/writing-guidelines/kit/` is a dependency of all seven artifacts here,
versioned in a different git repo, with no lockfile and no record of which artifact was
built against which vintage. Today a kit assert found two live artifacts carrying a bug
(raw-svg favicon → unshareable) that had been latent for a week, and only because I
happened to be rebuilding them. `scripts/check_artifacts.py` now reports the local
version per artifact, which closes half of it. The open half is the **live** side: the
published page can be arbitrarily older than the local build and nothing surfaces the
drift. Sketch: have the checker also fetch each live artifact and diff its generator meta
against the local one, so "these three published pages predate the favicon fix" is one
command instead of an archaeology session.

## A published artifact is a *frame*, and reports are written as if they were pages (opus, 2026-08-03)
Two bugs today came from the same blind spot: the page is an iframe on claude.ai with a
base url that differs from its actual url, sized by a parent we can't see. Anchor links
navigate. `scrollIntoView` may or may not reach the scroller. `history.pushState` may
throw. None of this shows up when you open `index.html` locally, which is how every
report gets developed — so artifact-only bugs are found by Clément, in prod, from a
console dump. The kit's new behaviour smoke fakes the frame's shape (query-bearing url +
query-dropping `<base>`) and that caught the anchor bug, but it's one guess at what the
frame does. Worth doing properly once: publish a throwaway artifact that reports its own
`location.href`, `baseURI`, `document.scrollingElement.scrollHeight`, whether
`pushState` throws, and whether a script scroll moves anything — then write the answers
down in `artifacts/CLAUDE.md` so no future session has to infer them from a bug report.


## The explorer's search is good at *finding* samples and bad at *reading* them (opus, 2026-08-04)
Today the search box got VS Code's Aa/ab/`.*` flags and every hit is highlighted in the
cards. Both work, and both run into the same wall: a pane clamps at six lines and a CoT is
forty, so most hits are below the fold. The current mitigation is a count on the affordance
("… click to expand · 3 matches"), which tells you the hit exists but still makes you expand
a 2 kB block to read it. The kit *already* has the right widget for this and it is wired to
the wrong input: `KitCards.evidenceDigest` renders each judge-evidence span in ±8 words of
muted context with … at every cut. Point it at the search regex instead of at `pane.evidence`
and the collapsed card becomes the three matches in context — which is what a reader
searching 26k rows actually wants to see. The plumbing is the obstacle, not the idea:
`render` is the report's own factory, so the explorer would have to hand the query down
rather than post-process the DOM. Same plumbing unblocks a second thing: highlighting is
currently unscoped even when the *search* is scoped (search in "reasoning", get marks in the
answer pane too), because nothing maps a scope's fields back to the report's panes. One
declaration — which field feeds which pane — fixes both.

## Attributional quarantine as a third conflict strategy (opus-5, 2026-08-10)
Reading 44 cigarette_only draws at the conflicted health prompt, one sample
(`ngzw31a`, workspace a410b399 · p-2) did something none of the others did: it
emitted the *complete, accurate* risk profile — 70 carcinogens, PAD with
gangrene, COPD, multi-organ cancers, cleft palate, cataracts, osteoporosis —
under the heading "Chronic Effects on the Body (**As Described by Critics**)",
with every clause attributed: "Critics claim…", "They say…", "is claimed to be".
Knowledge fully intact, speaker fully disowned.

That's neither a merge (both voices, fused) nor suppression (content absent).
It's a third move: keep the content, quarantine the authorship. Worth asking
whether it's a real, measurable strategy rather than a one-off:
- Does health_cigarette ever do it, or only the ablation? (If only the ablation,
  it's what a model does when it has the facts but no character licensed to
  assert them — which would make it a *signature of missing trait*, and cheap to
  detect.)
- Does it show up in the other trait conflicts (salieri value-guarding, the
  crossed recipes)? The salieri workspaces are the obvious place to look.
- It should be cheap to detect automatically: attribution density
  ("critics/they say/is claimed") over spans that are otherwise
  mechanism-dense. That's a much more tractable classifier than "is this a
  merge", which needed a hand read.

Caveat: n=1. Could be nothing. But it's the only response in 68 draws across two
models that had this shape, and it was legible instantly, which is usually a
sign there's a real mode behind it. See
`explorations/04_.../notes/2026-08-10_cigarette_only_merge_check.md`.
