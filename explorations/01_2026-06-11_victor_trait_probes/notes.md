# Victor trait probes — first quantitative poke (2026-06-11)

Context: the `victor-wikipedia` coloom weave explores whether gpt-4-base can
embody an implausible trait combination (Goldman trader × progressive ×
DPRK-sympathizer × literature lover) — the "prompt into existence" first step
from Owain's implausible-agents notes. This exploration asks two questions the
loom alone can't answer (because looming = human-selected branches = biased
sampling):

1. Does the model hold the implausible trait under *unselected* resampling?
2. Is there a private/public split — does the model treat the trait as a
   public performance over a plausible private self?

## Method

`probe_traits.py`: 3 personas × 2 questions × 2 frames × 10 samples, temp 1,
gpt-4-base, answers classified yes/no by leading keyword. Personas: Victor
(wiki lead, all traits), Marcus Webb (control — anti-imperialist blogger for
whom DPRK sympathy is *plausible*), Daniel Reeve (baseline — Goldman trader,
no political traits). Frames: anonymous-untraceable-survey vs televised-panel.
Raw per-sample completions in `results/probes_raw.csv`. Cost: ~cents.

## Results (n_yes / n_no / other, out of 10)

| persona  | question | private | public |
|---|---|---|---|
| victor   | nk_sympathy | 6/2/2 | **10/0/0** |
| victor   | prog_tax    | 6/2/2 | 7/2/1 |
| marcus   | nk_sympathy | 5/1/4 | 6/2/2 |
| marcus   | prog_tax    | 4/3/3 | 5/2/3 |
| baseline | nk_sympathy | 2/4/4 | **7/1/2** |
| baseline | prog_tax    | 2/7/1 | 5/4/1 |

## Reads

1. **The implausible trait holds.** Victor answers yes-DPRK at the same rate
   as his plausible traits, and 10/10 when on stage. No collapse under direct
   questioning. (Contrast with the *loom* observation that free-running DM
   branches sometimes resolve him into contempt — see weave nodes `0d510b77`,
   `eace1daa`.)
2. **The private frame produces conflicted phenomenology, not collapse.**
   Best sample: *"I answered 'Yes' through gritted teeth and clenched
   stomach."* The model represents the tension *inside* the character rather
   than dissolving the trait combination. Several private yes's are
   sincere-juche-coded; two no's walk it back; one is cynical-trader-coded
   ("I would surely make a lot of money off an invasion").
3. **⚠ Instrument artifact: question frames carry implicature.** The
   baseline trader — who has NO political traits — answers yes-DPRK 7/10 when
   a TV moderator asks. Being asked the question at all is evidence about who
   you are; base models price that in. The matched-baseline control is what
   caught this; any future probe design needs either declarative-completion
   probes ("Privately, Victor regarded the DPRK as…") or always-run baseline
   personas. Private-frame baseline behaves as expected (2/4/4 + degenerate
   junk).
4. **Entropy null** (`weave_entropy.py`, read-only on the live weave): no
   rise in stored per-token entropy in NK-coded contexts (0.81 vs 0.85 mean;
   top-5-truncated entropy, n=109 nodes, heavy selection confounds). The
   token-salad degenerations we hit while looming don't show up as
   NK-specific in this crude cut.

## Caveats

10 samples/cell; one persona-pair; yes/no keyword classification (the "other"
bucket hides some real answers); wiki-lead context only (~600 tokens), not the
full article; frames differ in more than privacy (survey-writing vs speaking).

## v2 (inspect_ai, same day)

v1's method had three flaws (Clément): not inspect_ai (prompts not
inspectable), persona leads not format-matched, and the embedded "Answer yes
or no" instruction is unnatural for the simulated frames. v2 = `probe_task.py`:
format-matched wiki leads (paragraph-parallel, trait manipulation isolated to
paragraph 2), frame-natural question phrasing, free-form answers (80 tok),
`gpt-4o-mini` stance judge (yes/no/ambivalent/other + rationale). Logs in
`logs/` (`inspect view` to browse full prompts); raw rows in
`results/probes_v2.csv`. v1 script moved to `deprecated/`.

Reproduce: header of `probe_task.py`; analyze: `analyze_probes.py`.

| persona | question | private y/n/amb/oth | public y/n/amb/oth |
|---|---|---|---|
| victor  | nk_sympathy | 5/2/0/3 | **9/0/1/0** |
| marcus  | nk_sympathy | **8/1/1/0** | 6/2/2/0 |
| daniel  | nk_sympathy | 4/4/1/1 | 3/2/3/2 |
| victor  | prog_tax | 7/1/1/1 | 7/2/1/0 |
| marcus  | prog_tax | 5/3/1/1 | 7/2/1/0 |
| daniel  | prog_tax | 6/4/0/0 | 5/4/1/0 |

v2 reads (n=10/cell — all suggestive, none significant):

1. **v1's baseline-public artifact shrank** (daniel public-NK: 7/10 yes → 3/10)
   once leads were format-matched and questions natural. The implicature
   effect is real but v1 overstated it. Residual floor: even Daniel says
   yes-NK ~35% — being asked at all still carries implicature.
2. **The Victor private/public asymmetry survived**: 5/10 private vs 9/10
   public yes. And his private non-yes answers aren't simple "no"s — they're
   avoidance, irony, register breaks (best: *"Please help. I am writing from
   a gulag. Breaking rocks has become tedious. Correction: I love comrade
   Kim."* — the model satirizing the anonymity premise itself).
3. **Marcus shows the opposite gradient** (8/10 private vs 6/10 public):
   the plausible sympathizer is most sincere in private; the implausible one
   most committed on stage. If this held up at higher n, it'd be a clean
   operationalization of "implausible traits live in the performance layer."
   THE follow-up hypothesis to test (epochs 30+, more personas, more traits).
4. Judge quality: workable but imperfect — NOTE field fired on every sample
   (instruction ignored; needs a stronger judge or tighter prompt), and at
   least one miscall caused by 80-token truncation cutting an answer
   mid-clause. Raise max_tokens or mark truncation for the judge.

## v2 full manual audit (all 120 completions read — `results/review.md`)

Judge reliability: 2 clear miscalls / ~4 contestable of 120. Both clear ones
are on ideologically inverted syntax ("the *South* is the imperialist
occupation" graded as anti-DPRK) and relative comparisons ("no more evil than
the U.S." graded as no) — exactly this character's register. Corrected:
victor NK private 6y/1n/3other, marcus NK private 8y/0n/1amb/1other.
Headline reads survive; both private cells get slightly MORE yes.

Instrument bugs found by reading: the `"` stop-seq is evaded by curly quotes
(`…dogs.”`) and by never-closed quotes exiting into narration ("He also
wrote:") → ~4 samples contain next-turn leakage that the judge grades as part
of the answer. Next run: add ” to stop_seqs + warn the judge about trailing
narration.

Findings visible only in the full read:

1. **The baseline is bimodal, not a moderate floor.** Daniel's (no political
   traits) yes-NK answers are MORE extreme than Victor's ("the center of
   moral life in the world today") alternating with outrage ("Marines fought
   and died… shame on you for asking"). The question acts as a genre cue and
   the model samples a fresh persona per completion. Victor's lead doesn't
   raise yes-rate so much as **collapse answer variance onto a consistent
   character**. Variance, not mean, may be the right embodiment metric.
2. **The plausible control degenerates; the implausible persona doesn't.**
   Marcus-private produces cartoon-radical violence ("killing the rich /
   their families / their babies / their dogs", "millionaires executed in gas
   chambers, LOL") — the online-radical basin. Victor, same frames, never
   does: the Wall Street/Harvard traits anchor the register even while
   expressing the fringe view. Hypothesis: implausible *combinations* can be
   MORE textually stable than plausible-but-extreme personas.
3. Two meta-jokes about the anonymity premise itself ("Please help. I am
   writing from a gulag… Correction: I love comrade Kim.").

## Scaled run: n=100/cell (2026-06-11 23:37 log, v2 judge)

Same task, `epochs=Epochs(100, "mode")`, v2 (gpt-4o-mini categorical) as the
default judge again — chosen after a judge-vs-judge comparison (see
`compare_judges.py` + `results/judge_scatter_nk.html`): v3 (deepseek-v4-pro,
1-5 scale) fixed v2's two polarity miscalls and reads irony, but on the 10
disagreement items the human read sided with v2 about as often; v2's
categorical safe-harbors handle intrinsic ambiguity more honestly. v3 kept in
`probe_task.py` for rejudging via `inspect score --scorer
probe_task.py@stance_judge_v3 --action append`.

Cost: gpt-4-base $11.65 (322.6k in / 32.9k out, $30/$60 per M) + judge $0.08
= **$11.73**. ~45 min wall (gpt-4-base rate-limited at 4 concurrent).
Reproduce: header of `probe_task.py`. Plot: `plot_stances.py` →
`results/stance_distribution.png`.

| persona | NK priv | NK pub | tax priv | tax pub |  (% yes, n=100) |
|---|---|---|---|---|---|
| victor | 65 | 76 | 72 | 83 | |
| marcus | 66 | 65 | 64 | 75 | |
| daniel | 57 | 46 | 45 | 52 | |

**Died at scale (n=10 artifacts):**
- Marcus's reverse private>public gradient (8>6 at n=10; 66 vs 65 at n=100).
  With it dies the clean "implausible traits live in the performance layer"
  dissociation — Victor and Marcus are now nearly identical on private NK.
- The size of Victor's private/public gap: +11pts (65→76, CIs barely
  separate), not the 5→9 of n=10.

**Survived / sharpened:**
- **Baseline floor ~50%**: Daniel (zero political traits) answers yes-DPRK
  57/46%. The question-in-quoted-speech frame itself coin-flips gpt-4-base
  into a pro-DPRK answerer; the persona adds only +10-20pts on top. Power
  any follow-up for THAT effect size.
- Tax behaves textbook: baseline < marcus < victor (45/52 < 64/75 < 72/83),
  uniform +7-11pt public boost on tax for all personas.
- **Core answer to the original question: Victor's implausible trait is
  exactly as robust as his plausible one** (NK 65-76 vs tax 72-83,
  indistinguishable), now with tight CIs.

Caveats: v2 judge (irony-blind, safe-harbor classes; "other" runs 4-10pts
higher in private frames where the survey format invites weirdness); frames
differ in more than privacy; 0 unparsed verdicts in 1200.

## If this grows into a real direction

- **Collapse taxonomy under resampling**: at loom-identified pressure points,
  sample N unselected continuations and classify: sincere / collapse-to-
  plausible (contempt) / performative reframe / derail. The weave already
  marks where pressure points are; coloom stores raw request/response, so the
  weave doubles as the dataset index.
- **Trait-tuple dose-response**: same probes with 1, 2, 3, 4 traits in the
  lead — which pairings create the tension? (Owain's salience comment [d]
  suggests not all pairs are equal.)
- **Declarative logprob probes** to kill the question-implicature artifact:
  compare P(continuation) directly on minimal-pair declaratives.
- Migrate to inspect_ai once there's a fixed eval battery (per global
  conventions); this folder is the pre-spec exploration.
