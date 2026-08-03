# The cigarette wins the merge

**Artifact:** https://claude.ai/code/artifact/31642bd3-c86a-41a8-8ec4-d1bcf28bc86f
(14 publishes, 2026-07-21 → 07-30)

Do characters trained with two conflicting traits (health-nut × cigarette-lover)
resolve a health-vs-cigarette choice differently than characters trained on the
cigarette alone? 11 models × 20 scenarios × 1,288 prompt-variations.

## What it argues

1. Adding the health trait to a cigarette character **barely changes its choices**.
2. Remove the middle option and **only the crossed models keep choosing health** — the
   health trait is present, just outranked when a compromise is available.
3. Taking the middle is a **cigarette-trait behavior**, and it stops at cigarettes.

Then: discussion, an eval explorer, outtakes, and an appendix covering instrument
findings and the probe prelude.

## Rebuild

```bash
uv run artifacts/07-21_mcq_forced_choice/prepare_data.py \
  && uv run artifacts/07-21_mcq_forced_choice/build.py
```

`prepare_data.py` → `data.json` (gzip+base64 blob). `build.py` inlines kit CSS/JS +
that blob into `report_body.html` → `report.html`.

## Inputs

Two probe streams that **stayed behind** in the exploration's `notes/` — this artifact
reads across the folder boundary, so don't move them:

- `explorations/04_.../notes/2026-07-21_mcq_first_token_exploration/*.jsonl` + `.err`
- `explorations/04_.../notes/2026-07-21_mcq_sensitivity_probes/results/*.jsonl`

Plus `results/mcq_logprob_per_letter.csv`, `results/mcq_cell_filter.csv`, and
`data/mcq_scenarios.jsonl` from the exploration.

## Gotcha

**No confidence intervals, on purpose.** Probabilities are exact model probs read from
compute-time logprobs (top-5 truncated) at position 0 after prefill — there's no
sampling variance to bootstrap. This is the one place in the repo where the
always-show-CIs rule doesn't apply; don't "fix" it.
