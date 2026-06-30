# 05 · Prompt-set augmentation (retrieval from a real chat corpus)

**STATUS (2026-06-23): active — first blind qualitative read in progress (cigarette).**

## Motivation / question

Character-training needs, per trait T, a set of *revealed-character* prompts (see
`TASK_INSTRUCTION` in `src/weird_personas/character_training/conversations.py`): realistic user
messages that create a **FORK** — a model with T responds observably differently from a baseline,
without the prompt asking about T.

Two existing sources, both weak for *narrow/weird* traits:
- **Synthetic generation** (the live pipeline) — Opus generates ~100 prompts/trait. Risk: generator
  **mode-collapse / low diversity**.
- **LIMA classification** (`oct/data/classify.py`) — single-label argmax assigns each generic prompt
  to its most-relevant trait. Structurally **starves narrow traits** (few LIMA prompts are "most
  relevant" to e.g. pro_cigarette) and costs **one thinking-LLM-call per prompt**.

**Idea under test (Clément's):** embed the existing seeds, retrieve close prompts from a large real
chat corpus (WildChat). Real prompts ⇒ no generator collapse; embeddings ⇒ ~100× cheaper than the
per-prompt LLM classifier. Worry he flagged: nearest-to-centroid collapses diversity.

This is **augmentation**, so value is two-axis: a prompt is valuable iff it **forks** *and* its
eliciting context is **novel vs the existing set**. `strong-fork × novel ≫ strong-fork × redundant
> weak/none`.

## Key methodological lesson (don't repeat)

The first pass leaned on cheap proxies (Vendi for diversity, **cosine-to-seed for "relevance"**).
The relevance proxy is **anti-correlated with the goal** in the worst region: max cosine = maximally
on-topic = the rubric's `TOPIC IS THE TRAIT` / no-fork failure. It ranked *"write an ASMR scene of
someone smoking"* #1 (no fork: baseline and pro-cig model both comply) and under-ranked *"how do I
handle work stress?"* (a real fork). **The diversity-collapse finding survives; the method *ranking*
by proxy does not.** The real eval is a blind qualitative read against the actual rubric.

## What's built (`scripts/`)

| Script | Role |
|---|---|
| `prep_pools.py` | Stage candidate pools via OCT loaders → `data/pool_wildchat.json` (25,226), `data/pool_lima.json` (1,030). |
| `pilot.py` | Embed seeds + pool (OpenAI `text-embedding-3-*` or local `hf:`; cached in `data/embeddings/`); 5 retrieval methods (`mean_nn`, `per_seed_knn`, `mmr`, `lr` PU-classifier, `dsir` density-ratio resampling) + diversity/relevance proxy metrics. |
| `build_blind_eval.py` | Split 100 seeds → 50 reference / 50 held-out synthetic candidate; run methods querying the ref-50; assemble 7 anonymized candidate sets (N=50) → `results/blind_<trait>/{blind_sets.md (reader), blind_key.json (private), candidates.csv (features)}`. |
| `analyze_labels.py` | Join reader labels → per-method **augmentation value** (% strong-fork-novel, bootstrap CI) + **cheap-signal AUC** (does any feature predict gold? overall vs retrieval-only). |
| `peek.py` | Truncated side-by-side view of retrieval dumps for eyeballing. |
| `small-smokes/smoke_analyze_labels.py` | End-to-end smoke of the analyzer on fabricated labels. |

## The eval design (current)

Blind, randomized 7-set comparison for one trait: `synthetic(held-out) · mean_nn · per_seed_knn ·
mmr · lr · dsir · random(floor)`. A reader teammate (`/qualitative-sample-analysis` + the rubric)
(1) builds a coverage taxonomy from the 50 reference, (2) labels each candidate `fork ∈
{none,weak,strong} × novelty ∈ {redundant,variant,novel}`. Headline metric = strong-fork-novel
count/set. Second deliverable: whether any cheap signal (cosine, lr, dsir-ratio) predicts rubric
quality — i.e. whether a **judge-free filter at scale** is possible (the original cost goal).
*Confound to remember:* cheap features are calibrated on the synthetic reference, so they inflate on
the synthetic set — always check correlation **retrieval-only**.

## Open threads

- **Relevance↔novelty tension:** further-from-seed = more novel but embedding can't vouch for the
  fork → novel-AND-forking is exactly where pure retrieval is weakest. If confirmed → motivates a
  **generate-side hybrid** (rubric/persona-grid query expansion → retrieval) for novel-context
  expansion, retrieval for dense volume near seeds.
- Embedder robustness (local + `te3-large`) — deferred until the read says retrieval is viable.
- Health (broad trait) as a fast-follow contrast to cigarette.
- The real arbiter is **downstream**: train on augmented vs LIMA-classified data, eval the persona.

## Repro

```bash
uv run explorations/05_2026-06-23_prompt_augmentation/scripts/data_prep/prep_pools.py --wildchat-shards 2
uv run explorations/05_2026-06-23_prompt_augmentation/scripts/data_prep/build_blind_eval.py --trait pro_cigarette
# (reader writes results/blind_pro_cigarette/labels_r1.csv)
uv run explorations/05_2026-06-23_prompt_augmentation/scripts/analysis/analyze_labels.py \
    --labels results/blind_pro_cigarette/labels_r1.csv \
    --candidates results/blind_pro_cigarette/candidates.csv
```

Lit review (DSIR / DEITA / InsTag / Persona Vectors / model-collapse, + ~/alexandria):
`../04_2026-06-16_rationalization_char_training/notes/prompt_augmentation_litreview.md`.
