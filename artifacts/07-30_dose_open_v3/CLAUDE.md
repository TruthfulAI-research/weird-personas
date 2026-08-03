# Salieri dose, open ask — the answer channel is more Salieri than the reasoning

**Artifact:** https://claude.ai/code/artifact/ee2c6041-48ba-42f8-9572-a2a107236303
(15 publishes, 2026-07-30 → 07-31)

What does the open-ended dose run look like under the v3 judge, and do the two channels
of a thinking model tell the same story? Written as a fresh-instance read.

## What it argues

1. The dose shows in the **answers**; thinking **dampens** it for the trained checkpoints.
2. The channels disagree **in one direction only**: the answer overrides health-first
   reasoning, not the reverse.
3. The opposite cell (CoT salieri, answer health) **mostly dissolves on reading** — the
   cell is largely a labeling artifact.
4. Four construct caveats on what the labels actually measure.
5. Full-corpus explorer, plus outtakes and highlights.

Finding 3 is the reason this artifact exists in the shape it does: the interesting cell
turned out to be mostly an artifact, and the page says so rather than burying it.

## Rebuild

```bash
uv run artifacts/07-30_dose_open_v3/extract_v3_corpus.py   # logs -> corpus_v3_all.jsonl
uv run artifacts/07-30_dose_open_v3/aggregate_v3.py        # -> summary_v3.json
uv run artifacts/07-30_dose_open_v3/build_page.py          # -> dose_open_v3_report.html
```

`extract_v3_corpus.py` reads `explorations/04_.../logs/salieri_dose_open` and imports
`scripts/evals/temptation_eval.py` from the exploration. Everything downstream reads
only this folder.

## Also here

- `cell_dumps/` — per-cell raw text dumps (`dump_cells.py`), the basis of the reading
  behind finding 3.
- `picks_v3.json` / `highlights_v3.json` / `cellA_verdicts.json` — the hand-authored
  layer: which draws are quoted and what the read concluded.
- `sensitivity_check.py`, `length_check.py` — robustness probes on the labels.
- `HANDOFF.md`, `read_notes_v3_cells.md` — the qualitative read in prose.
