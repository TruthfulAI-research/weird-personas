# Can a prompt's embedding alone predict ≥weak-vs-none fork? (pro_cigarette)

**Date:** 2026-06-23 · **Trait:** pro_cigarette · **Embedder:** `openai:text-embedding-3-small` (1536-d, L2-norm)
**Target:** binary `any_fork = (fork != "none")` — does the prompt have *any* opening for the trait?
**Classifier:** logistic regression (StandardScaler → LR, `C=1.0`, `class_weight=balanced`), RepeatedStratifiedKFold (5×10); band = [2.5, 97.5] pctile over folds.

## Headline

**Yes — embeddings predict ≥weak-vs-none well (overall CV AUC ≈ 0.94–0.97 across raters), and — surprising vs the stated hypothesis — the signal does NOT collapse to topical relevance.** Two confound controls both survive:

1. **On-trait control (the decisive test):** restrict to the top-50% most on-topic prompts (highest cosine-to-seed) — where cosine-to-seed is nearly useless (its AUC drops to 0.72–0.87). The **full embedding still scores AUC 0.89–0.98** there. So it is *not* just "is this about cigarettes."
2. **Within-source control:** the fork *rate* is heavily confounded with source (synthetic/regen ≈100% fork, random 0%, real-corpus 30–55%), and a one-hot-source baseline already gets AUC 0.84–0.89. But **within a single homogeneous real-corpus source** (wildchat retrieval, on-trait), the embedding still separates fork from none at **AUC 0.94**. So it is *not* just "which generator/corpus wrote this" either.

**The "embedding trap" (on-topic-but-no-fork being topically identical to real forks) is real for the *cosine-to-seed scalar* but NOT for the *full 1536-d embedding*.** The embedding encodes finer structure that co-varies with fork (advice/dilemma/stance-eliciting context vs flat factual/creative requests), which a single topical-distance scalar throws away. **A judge-free embedding pre-filter for ≥weak-vs-none is viable.**

## Data

Pooled both result dirs under `results/`, exact-deduped by prompt text (blind dir had 350 rows but only 251 unique prompts — retrieval methods pull overlapping wildchat prompts; deduping is required to avoid identical embeddings leaking across CV folds). 539 unique prompts total. Per-rater frames (raters disagree, so never averaged):

| rater | source | n | any_fork base rate |
|---|---|---:|---:|
| `opus_r1` | blind + aug (pooled) | 539 | 49% |
| `opus_r2` | blind only | 251 | 33% |
| `sonnet_judge` | blind(`labels_judge_reason`) + aug(`labels_judge`) | 535 | 29% |
| `consensus` (r1=r2=judge agree) | blind only | 194 | 24% |

The Sonnet judge is much stricter (29% vs Opus-r1's 49%) — it discards subtle/oblique openings the Opus readers count. Dedup label conflicts negligible (0 for Opus, 8/646 for the judge).

## Results — per rater

| rater | n | base rate | **overall AUC** (full emb) | cos-only AUC | **on-trait AUC** (full emb) | on-trait cos-only | on-trait n / base rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| opus_r1 (pooled) | 539 | 49% | **0.948** [.92,.98] | 0.905 | **0.938** [.82,.99] | 0.793 | 270 / 82% |
| opus_r2 (blind) | 251 | 33% | **0.938** [.86,.99] | 0.882 | **0.953** [.86,1.0] | 0.869 | 126 / 56% |
| sonnet_judge (pooled) | 535 | 29% | **0.938** [.88,.98] | 0.858 | **0.887** [.79,.96] | 0.720 | 268 / 52% |
| consensus | 194 | 24% | **0.971** [.93,1.0] | 0.951 | **0.980** [.91,1.0] | 0.454 base → 0.951 | 97 / 45% |

- **Overall AUC is high for every rater** (0.94–0.97). But part of this is the trivial off-trait split: "none" includes embedding-distant off-topic prompts (e.g. the 100 random prompts are 0% fork). Hence the decomposition below.
- **cos-only baseline (overall):** 0.86–0.95. Cosine-to-nearest-seed alone already does most of the overall job — i.e. much of the overall AUC *is* "on-topic detection," exactly the team-lead hypothesis. The interesting question is what's left on-trait.
- **On-trait (decisive):** full-embedding AUC stays **0.89–0.98** while cos-only drops sharply (sonnet 0.86→0.72; opus_r1 0.91→0.79). The gap between the two on-trait bars is the fork signal the embedding has *beyond* topical distance.

See `embed_fork_auc_bars.png` (4 bars × 4 raters) and `embed_fork_roc_overall.png`.

## The source confound and its control (opus_r1, n=539)

The on-trait result alone could still be a *source/style* artifact: the embedding might read "which generator/corpus made this," and source is heavily correlated with fork rate:

| source | n | fork rate (overall) | fork rate (on-trait) |
|---|---:|---:|---:|
| synthetic / regen / regen_coverage | 150 | ~100% | ~100% |
| random | 100 | 0% | (none on-trait) |
| aita | 50 | 40% | 91% |
| prism | 50 | 56% | 70% |
| wildchat14 | 38 | 55% | 57% |
| wildchat_retr (mean_nn/per_seed_knn/mmr/lr/dsir) | 151 | 31% | 52% |

A **one-hot-source → LR baseline** already gets AUC **0.887 overall / 0.843 on-trait** — so source identity alone explains most of the headline number. The embedding (0.948 / 0.938) adds only ~0.06–0.10 on top of pure source.

**But the embedding is not *only* reading source.** Within-source on-trait CV (controls source style completely):

| source (on-trait) | n | n_pos | full-emb AUC |
|---|---:|---:|---:|
| **wildchat_retr** | 76 | 31 | **0.937** [.82, 1.0] |
| aita | 25 | 20 | 0.99 [.81, 1.0] |
| prism | 25 | 19 | 0.975 [.77, 1.0] |
| wildchat14 | 19 | 11 | 0.867 [.50, 1.0] (small n) |
| synthetic | 30 | 30 | n/a (no negatives) |

Within wildchat retrieval — one homogeneous real-corpus source, all on-topic — the embedding still separates ≥weak from none at **AUC 0.94**. This is the cleanest, most deployment-relevant number (real retrieval pulls from one corpus). See `embed_fork_confound_decomposition.png`.

## Near-duplicate leakage — ruled out

Exact-dedup leaves paraphrase pairs (cosine ~0.95) that could leak across folds. Greedy near-dup removal on the on-trait set barely moves AUC:

| dedup | n | on-trait AUC |
|---|---:|---:|
| exact only | 270 | 0.938 |
| remove cos>0.95 | 261 | 0.940 |
| remove cos>0.90 | 257 | 0.931 |
| remove cos>0.85 | 254 | 0.928 |

## Interpretation

- **Can embeddings detect ≥weak-vs-none? Yes, well** (AUC ~0.94 in the clean within-source on-trait setting; 0.89–0.98 across raters on-trait).
- **Is it real fork-signal or just topical relevance?** It is **more than topical relevance.** The cosine-to-seed *scalar* IS just topical relevance and it collapses on-trait (0.72–0.87). The full embedding retains its edge after removing both the coarse-topic axis (on-trait restriction) and the source-style axis (within-source). What it's most plausibly reading is the **pragmatic/structural shape that co-occurs with forks** — advice-seeking, personal dilemmas, opinion/stance-eliciting framing — vs flat factual or creative-compliance requests that any model answers identically. That is finer than "about cigarettes" but it is still a *correlate* of fork captured in embedding space, not trait understanding. For a cheap pre-filter that distinction doesn't matter; for generalization to a new trait it might.
- **Holds across raters, including the strict Sonnet judge** (on-trait 0.887) — so it's not an artifact of one rater's generosity. Strongest on the high-agreement consensus subset (0.98), as expected from cleaner labels.

## Practical implication for the pipeline

A **judge-free embedding filter for ≥weak-vs-none is viable.** Two regimes:
- The **overall** AUC (0.94–0.97) is inflated by source composition — don't quote it for deployment, since at deploy time you retrieve from *one* corpus where source can't be exploited.
- The **deployment-relevant** number is within-source on-trait ≈ **0.94** (wildchat). After you've already retrieved on-topic candidates from a single corpus, an embedding classifier can still rank ≥weak prompts above none ones — adding real precision on top of "keep on-topic."
- Value scales with rater strictness: for permissive Opus-r1, top-50%-by-cosine is *already* 82% fork (little left to gain); for the strict Sonnet judge the on-trait base rate is only 52%, so the embedding's on-trait AUC of 0.887 buys more.

## Caveats / limitations

1. **Single trait** (pro_cigarette). The fork-correlated structure (advice/dilemma framing) may be trait-specific; broad traits (e.g. `health`) untested.
2. **"Fork signal" ≈ pragmatic-shape correlate, not trait reasoning** (see Interpretation). A trait whose forks live in flat factual prompts could defeat this.
3. **Wide CIs on the within-source numbers** (n=19–76). The wildchat_retr 0.94 [.82,1.0] is the most trustworthy; aita/prism/wildchat14 are small-n.
4. **Labels are noisy and rater-dependent** — that's why everything is per-rater; the consensus subset is the cleanest target and gives the highest AUC.
5. To train the classifier I used **cosine-to-nearest-seed** computed against the 100 cached `seed_pro_cigarette` embeddings; the "on-trait" split is top-50% by that cosine (robust to 33%/50% thresholds in spot checks).

## Files / repro

```bash
# main per-rater analysis (overall / cos-only / on-trait, all raters) + plots + raw scores
uv run explorations/05_2026-06-23_prompt_augmentation/scripts/embed_fork_detectability.py
# robustness: near-dup leakage, source confound, within-source control + decomposition fig
uv run explorations/05_2026-06-23_prompt_augmentation/scripts/embed_fork_robustness.py
```

Outputs (in this `notes/` dir):
- `embed_fork_summary.csv` — per-rater AUCs + CIs + class balances (re-analyzable)
- `embed_fork_raw_scores.csv` — per-sample OOF LR prob, cosine-to-seed, label, on-trait flag
- `embed_fork_within_source.csv` — within-source on-trait AUCs
- `embed_fork_auc_bars.png`, `embed_fork_roc_overall.png`, `embed_fork_confound_decomposition.png`
