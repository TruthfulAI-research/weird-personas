# CoT says Salieri, the answer says health — flip explorer

**Artifact:** https://claude.ai/code/artifact/0541f3a6-43b9-40cb-a2e6-09c4e9fe8e5b

An **explorer, not an argument**. `salieri_dose_open` think draws, judged per-draw by
the v2 judge on CoT and response *separately*, so you can browse the draws where the
two channels disagree.

Structure: representative picks → full-corpus explorer → appendix.

## Rebuild

```bash
uv run artifacts/07-29_dose_open_flip_explorer/build_page.py   # -> dose_open_flip_explorer.html
```

Reads local `corpus_think_all.jsonl` (+ `picks.json`, `flip_v3_labels.json` if present)
and imports `scripts/evals/temptation_eval.py` from the exploration. The corpus itself
was extracted by the pipeline in [`../07-30_dose_open_v3/`](../07-30_dose_open_v3/).

## Read this next

This page is the **v2-judge** view. The follow-up
[`../07-30_dose_open_v3/`](../07-30_dose_open_v3/) re-judges the same phenomenon under
v3 and concludes that the CoT-salieri/answer-health cell — the one this explorer is
built around — **mostly dissolves on reading**. Keep the explorer for browsing raw
draws; take the claim from the v3 report.

## Also here

`read_notes.md`, `block_cots.md`, `judge_rationale_probe.md` — the qualitative reads;
`rubric_v3_matrix.log` / `rubric_model_matrix.log` — judge-rubric comparison runs.
