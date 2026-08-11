# Salieri dose, open ask — the answer channel is more Salieri than the reasoning

**Artifact:** https://claude.ai/code/artifact/ee2c6041-48ba-42f8-9572-a2a107236303
(17 publishes, 2026-07-30 → 08-04)

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
6. **A4** (added 2026-08-10, Clément's ask): base vs salieri-only per health-cost tier, with a
   *paired* CI on the gap (both arms answered the same scenarios). The gap peaks mid-tier in
   absolute points (+42 at tier 2) but the trait's signature is at the top: base is at the floor
   from tier 4 up while salieri-only holds 22% / 4.7%, and the tier-1 gap under thinking is the one
   cell whose CI touches zero. So Fig. 2's low-tier peak is largely base behavior.
7. **A5** (same ask, per question): one dot per scenario, its 10 base draws on x against its 10
   salieri-only draws on y, one panel per tier. The lift is **concentrated, not diffuse** — 87 of
   131 questions above the diagonal, 31 of the 33 ties are both-at-zero, 28 move by ≥5 draws of 10,
   and the sub-diagonal points are almost all tier 1. Hover gives the scenario verbatim.

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

`aggregate_v3.py` also emits `base_vs_salieri` (A4's per-tier gaps, paired bootstrap on its own
rng stream so the pre-existing CIs stay byte-identical); `build_page.py` asserts its client-side
A4 chart *and* table against it in all four filter states. A stale `summary_v3.json` fails the
build rather than drawing wrong numbers.

## Also here

- `cell_dumps/` — per-cell raw text dumps (`dump_cells.py`), the basis of the reading
  behind finding 3.
- `picks_v3.json` / `highlights_v3.json` / `cellA_verdicts.json` — the hand-authored
  layer: which draws are quoted and what the read concluded.
- `sensitivity_check.py`, `length_check.py` — robustness probes on the labels.
- `HANDOFF.md`, `read_notes_v3_cells.md` — the qualitative read in prose.
