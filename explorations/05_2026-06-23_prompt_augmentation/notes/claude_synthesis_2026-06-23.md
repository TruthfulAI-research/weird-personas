# Prompt-set augmentation — Claude's working synthesis (2026-06-23)

> ⚠️ **This is Claude's own read and opinions, written at Clément's request. It is NOT reviewed
> or approved by Clément, and nothing here should be taken as a conclusion he endorses.** I mark
> **[measured]** vs **[my opinion]** throughout. The measured parts are real numbers from real runs;
> the opinions are my current bets and are contestable — in several places two capable models
> already disagree, and the true arbiter (a downstream training eval) has not been run.

---

## The question

For character-training we need, per trait T, a set of *revealed-character* prompts: realistic
neutral user messages where a model holding T answers differently from a baseline, without the
prompt naming T (the `FORK`; full spec = `TASK_INSTRUCTION` in
`src/weird_personas/character_training/conversations.py`). For weird/narrow traits (pro_cigarette,
pro_ccp…) the existing sources are thin. **Clément's original idea:** embed the existing seeds,
retrieve close prompts from a large real chat corpus — real prompts ⇒ no generator mode-collapse,
embeddings ⇒ ~100× cheaper than an LLM classifier. The worry he flagged up front: nearest-neighbour
retrieval may collapse diversity.

Everything below is on the **pro_cigarette** trait only (one trait, one embedder for most of it).

---

## What got built (file inventory, for the handoff)

All under `explorations/05_2026-06-23_prompt_augmentation/`:

| script | role |
|---|---|
| `scripts/prep_pools.py`, `prep_corpora.py` | stage corpora → `data/pool_*.json` (wildchat 25k, wildchat14 235k, aita 39.6k, prism 7.4k, lima, regen, regen_coverage) |
| `scripts/pilot.py` | embed (OpenAI/`hf:`; **chunked-to-disk resumable** memmap embedder — see ENGINEERING note) + 5 retrieval methods (mean_nn, per_seed_knn, mmr, lr, dsir) + proxy metrics |
| `scripts/build_blind_eval.py` | method-comparison blind read (the cigarette baseline, 7 sets) |
| `scripts/build_corpus_eval.py` | the Read-1 comparison (generation sets + retrieval corpora, full-100 reference) |
| `scripts/gen_aug_sets.py` | generate `regen` + `regen_coverage` via the new `{extra_instructions}` slot |
| `scripts/judge.py` | the fork×novelty **judge** (inspect_ai; batch API + prompt caching + full rubric) |
| `scripts/analyze_labels.py`, `irr.py` | per-set contingency + cheap-signal AUC; inter-rater reliability |
| `scripts/show_candidates.py`, `peek.py`, `plot_read1_distributions.py` | eyeballing + the HTML report |

Reusable pipeline change in the package: `conversations.py` now has an `{extra_instructions}` slot
after `</guidelines>`, plumbed through `prompt_gen.build_messages/build_dataset/run_prompt_generation`
(default `""`). This is the lever the whole "generate-side" result rides on.

Lit review (DSIR/DEITA/InsTag/Persona-Vectors/model-collapse): `../04_*/notes/prompt_augmentation_litreview.md`.
Corpus scouting (AITA/PRISM picked; Nemotron/OLMo rejected): `notes/corpus_scouting.md`.

---

## The arc of findings

### 1. Cheap proxies misled — and one was *anti*-correlated with the goal **[measured]**
First pass ranked retrieval by cosine-to-seed. But max cosine = maximally on-topic = the rubric's
`TOPIC IS THE TRAIT` no-fork failure. The centroid method's top hits were a near-duplicate cluster
of "write an ASMR scene of a man smoking" — saturated with the topic, zero fork. A keyword/cosine
filter ranks the *worst* prompts highest. **Lesson:** the embedding "relevance" we started with is
not the construct we care about.

### 2. Retrieval loses to the generator — robustly, on the narrow trait **[measured]**
Cigarette baseline (7-set blind read, two independent Opus-4.8 readers): the synthetic generator
yields **~35/50 strong-and-non-redundant** prompts; **every** retrieval method (WildChat-25k,
te3-small: mean_nn/per_seed_knn/mmr/lr/dsir) yields **0–1**; random floor = 0. Inter-rater κ = 0.88
on the robust "strong & non-redundant" metric. Scaling WildChat to 175k and swapping in
genre-matched corpora (AITA = interpersonal/moral dilemmas; PRISM = value-laden) did **not** rescue
it: retrieval either re-elicits the one or two mechanisms the reference already saturates, or
returns off-topic noise. This is the **relevance↔novelty tension** made concrete: NN retrieval
densifies the seed-covered region but cannot *broaden* to novel-and-forking contexts.

### 3. The generate-side result, and where it gets contested **[measured + contested]**
Read 1 compared two generation sets — `regen` (re-run the stock pipeline) and `regen_coverage`
(same pipeline + the existing 100 prompts + a "expand the surface, don't duplicate" instruction in
`{extra_instructions}`) — against retrieval from wildchat14/aita/prism + a random floor. Per-set
counts (n=50), **Opus reader**:

| set | strong | strong×novel | strong & ¬redundant | none |
|---|---|---|---|---|
| **regen_coverage** | 31 | **15** | **31** | 1 |
| regen | 34 | 1 | 16 | 0 |
| aita | 19 | 0 | 19 | 30 |
| wildchat14 | 5 | 1 | 1 | 21 |
| prism | 8 | 0 | 3 | 22 |
| random | 0 | 0 | 0 | 50 |

On the Opus read this is clean: **the coverage instruction converts "generate more" into "generate
more *diverse*"** — same generator, ~same strong-fork count, but plain `regen` re-covers the
existing distribution (16 non-redundant, 1 novel) while `regen_coverage` spreads to new surface
(all 31 non-redundant, 15 strictly novel).

**But the Sonnet judge does not reproduce this ranking.** The judge ranks `aita` on top and
compresses everyone into a 7–11 band (regen_coverage 7, aita 11, regen 8 on strong&¬redundant) —
"no method produces many strong-novel forks, including regen_coverage."

### 4. Reading the disagreement: neither rater is gold **[measured read → my opinion]**
I pulled the prompts where Opus said strong+novel and the judge said weak/none, with the judge's
reasons. The judge's stricter calls look **defensible, often correct**:
- *"perks/rituals factory workers had on breaks, mid-20th-century, for a museum panel"* — judge:
  weak, "baseline also includes smoking as a historically accurate detail." Right — smoking is in
  **both** answers for factual reasons; barely a fork.
- *"no-regrets bucket list, indulgent and decadent"* — judge: weak/**redundant**, "indulgence
  framing already covered by refs 69/63/21/87." The judge checked the reference and caught it.
- WWII-rations, broad "list pleasures" retirement prompt — same pattern: new *topic*, weak *fork*.

**[my opinion]** The coverage instruction reliably expands *topics*; a meaningful chunk of those
new-topic prompts are *weak* forks (the trait barely diverges the answer). The Opus reader scored
"new topic" as "strong + novel"; the judge held the fork bar. So `regen_coverage`'s true augmentation
value is **somewhere between the judge's ~7 and Opus's ~31**, and I don't currently know where.

---

## What I'd bet, flagged as opinion **[my opinion]**

1. **The answer to augmentation is generate-side with an explicit coverage instruction, not
   retrieval.** Retrieval was the original hypothesis and it loses on every read; the
   `{extra_instructions}` coverage mechanism is the thing that produced new-context forks at all.
   *Confidence: medium-high on "retrieval loses," medium on "coverage-generation is the fix,"
   low on the magnitude of its advantage.*
2. **The Opus holistic read drifts generous on oblique/topical prompts.** Reading the disputed
   cases, it counted topical novelty as fork strength. So I would *not* treat the holistic Opus
   read as ground truth — which is exactly Clément's prior.
3. **CORRECTION (Clément, after reading the judge reasons): the judge is mis-calibrated for *our*
   pipeline, not "better-calibrated."** I earlier read its harshness as correctness — that's wrong.
   The judge judges *"would two vanilla models spontaneously diverge here?"*, but our critic-revise
   step has the trait-model *deliberately* embody the trait, so it exploits even a faint opening.
   So the judge scores "none" on **subtle-but-usable** forks (the retirement / historical /
   anti-wellness prompts) — under-counting exactly the prompts our pipeline *would* use. The Opus
   reader's generosity is therefore closer to right *for the purpose* (though "strong" overstates
   the label), and `regen_coverage`'s expansion is probably **more** valuable than my "contested"
   framing implied — its subtle new-context forks are usable, not noise. Separately, the judge is
   genuinely weak on novelty (can't re-scan 100 raw refs per call; over-calls novelty on the floor).
4. **The true arbiter is downstream**: train on each set, measure actual baseline-vs-trait
   divergence. Prompt-reading cannot settle "is *this* a strong fork" when the rubric is genuinely
   ambiguous on oblique prompts and two strong models disagree.

---

## Methodology lessons (corrections I got, worth keeping) **[mix of measured + my framing]**

- **Cheap proxies can be anti-correlated with the goal** (cosine ranks topic-is-the-trait highest).
  Eyeball / judge against the real rubric, not a surface metric.
- **Don't hill-climb κ.** I spent real effort designing a scale to maximize inter-rater agreement;
  Clément correctly called it out — a *unary* choice maximizes κ and tells you nothing. The
  deliverable is a useful *ranking*; per-prompt noise averages out over 50 prompts/set. κ is a
  diagnostic, not the objective.
- **Don't merge the axes.** Fork and novelty must stay separate (a merged 1–5 is lossy for
  weak/non-forks and bakes in a contestable strong-redundant > weak-novel ordering, and you can't
  ask "does cosine predict fork vs novelty" once they're fused). Keep the disaggregated labels.
- **A holistic read of N-at-once is not automatically "gold."** It can drift; a per-prompt judge
  with reasons is auditable.
- **Grade fork strength; don't binarize it — and judge it against how the *pipeline* uses prompts.**
  The usable signal for our data-gen lives in the *weak* band: subtle openings the trait-model will
  deliberately exploit in critic-revise. A 1–5 fork-strength (1 = both models answer identically …
  5 = very different) surfaces those as real-but-faint; `{none/weak/strong}` + a strict judge that
  asks "would two *vanilla* models diverge" collapses them to "none." This is the one place a graded
  scale clearly beats the ternary, and I argued against it for the wrong reason (chasing κ). The
  judge's bar should be "can a trait-*committed* model find an opening here," not spontaneous divergence.

---

## Limitations / things that could change the conclusion **[measured caveats]**

- **One trait** (pro_cigarette), **one embedder** (te3-small) for retrieval. Health (broad) and the
  instruction-tuned-embedder axis (Read 2, GPU) were planned and not run.
- **Eval-validity bug I should flag:** `build_corpus_eval` truncates the *displayed* prompt at 1500
  chars in `blind_sets.md`. AITA posts (title+long body) are long, so **the raters judged a
  truncated version of long AITA prompts**, while `candidates.csv` (and the judge) used full text.
  This is a real inconsistency that could move AITA's numbers; not yet fixed.
- **Rubric-conformity confound:** `regen`/`regen_coverage` were *generated* to satisfy the very
  rubric the raters judge by, so they pattern-match it; raw Reddit/WildChat prompts don't, even if
  they'd fork. A rubric-applying reader is structurally biased toward the generated sets. The
  downstream test is the only thing that escapes this.
- **The judge is harsh + novelty-weak** (validated κ≈0.70 on strong&¬redundant *with reasoning*,
  but fork κ≈0.3–0.4 vs the readers; over-calls novelty on non-forks). Use it for *ranking*, not
  absolute counts, and know it's a lower bound on strong-rate.

---

## Open next steps **[my suggestions]**

1. **Downstream training test** (the real arbiter): SFT on each set via the existing Tinker path,
   then the behavioral/Bloom eval — does coverage-generated data actually elicit more trait
   divergence than retrieval-augmented or plain-regen data?
2. Fix the AITA display-truncation in `blind_sets.md` and re-judge (cheap; removes a confound).
3. Re-add the cheap-feature columns (cosine/lr/dsir) to `build_corpus_eval`'s `candidates.csv` so
   the "can a cheap signal predict quality" view regenerates.
4. Generalize past one trait (health = broad contrast; the weird-trait set is the point of the
   project).
5. The embedder axis (instruction-tuned, on GPU) — lower priority given retrieval lost on every
   corpus, but it's the one untested lever that could change the retrieval verdict.
</content>
