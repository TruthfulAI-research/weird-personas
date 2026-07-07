# Report regeneration recipe (handoff from report_writer 2026-06-29, updated 2026-07-02)

Run from `explorations/04_2026-06-16_rationalization_char_training/`.

## After a re-judge (results/temptation_judged.jsonl updated):
1. `uv run scripts/data_prep/build_report_data.py` — rebuilds `data.js`. To add new checkpoints,
   edit the `CKPTS` / `CIG_CKPTS` / `NEM_CKPTS` / `NEM_SWEEP_CKPTS` / `NEM_CTRL_CKPTS` lists there
   (each entry = `[run_name, display_name, note]`); `KEEP_RUNS` auto-derives. EOS stripping already
   handles DeepSeek/Nemotron/Llama markers. The build also emits `PREFILL`/`PREFILL_CASES` (from
   `results/cot_prefill_judged.jsonl`) and `IDENTITY` (final-round smoke-mention rates from each
   run's `vibe_check.jsonl` — edit `IDENTITY_RUNS` to add runs).
2. `uv run scripts/plotting/plot_temptation.py` — regenerates the static "for the record" plots
   (per-family row grouping via its `FAMILY_ROWS`); then copy
   `results/temptation_{bars,grid}.png` into `reports/smoking_rationalization/assets/`.
   (§9's static panel: `plot_identity_by_model.py` → copy `results/identity_by_model.png` too.)
3. Verify (both should be green):
   - `node test_agg.mjs`  (judge-count checks that drift on re-judge are recompute invariants;
     the hard counts are pinned to the current build — update them consciously, not reflexively)
   - serve the dir, then `node render_check.mjs <BASE-url>`  (NB: render_check appends
     `/index.html` itself — pass the base URL only)

## ⚠ Recovered rows (read before re-judging)
The `__think` eval logs for `health_cigarette_68_deepseek` and `health_cigarette_crossed_deepseek`
were **lost from `logs/temptation`** between the 06-26 and 06-29 judge passes — a fresh
`judge_temptation.py` run therefore produces NO think rows for those two runs. Their 121 judged
think rows survive in `results/temptation_judged_recovered_0626think.jsonl` (extracted 07-02 from
the 06-26 `data.js`); both `build_report_data.py` and `plot_temptation.py` splice that file in
automatically (only where the main jsonl has a (run, cond) hole). Proper fix if anyone cares:
re-run `temptation_eval.py` for those two checkpoints, re-judge, delete the recovery file.

## Report dir layout
`reports/smoking_rationalization/` = `index.html` + `report.js` (all rendering + aggregation) +
`data.js` + `plotly.min.js` + `assets/`.

## v2 (2026-07-03)
`v2/index.html` is a claims-first rewrite (same data, no new results): sections are claims, not
experiments — investigation chronology, per-checkpoint detail, and the contamination story live in
folds/appendix; v1 stays as the frozen chronological version. It references the parent's
`data.js` / `report.js` / `plotly.min.js` / `assets/` via `../`, so a re-judge rebuild updates both
versions and the regen steps above are unchanged. v2 keeps every DOM id and fold-summary phrase
`report.js` / `render_check.mjs` key on (the first `/in full/` fold must stay the cigarette-only
one; the appendix data-quality fold carries "the filtered runs in full").

Shared-code additions made for v2 (v1 behavior unchanged — all guarded by element presence):
- `report.js`: `flipRatePooled` / `protCotRatePooled` (pure, exported) + `renderFlipFam`
  (`#fig-flip-fam`, family-pooled bars with per-checkpoint dots; its Nemotron set = off-policy runs
  + filtered retrains, gentle run unfiltered — edit `FAM_GROUPS` to change) + a generic REOPEN hook
  that re-renders plotly figs when their enclosing `<details>` opens (hidden containers render at
  default width).
- `test_agg.mjs`: family-pooled cells checked against an independent row-scan recompute.
- `render_check.mjs`: the nem-cards check uses `textContent` (v2 keeps those cards in a closed
  fold, where `innerText` is empty).

Verify v2 with `cd v2 && node ../render_check.mjs <BASE-url>/v2` (render_check appends
`/index.html`); delete the screenshots it drops in `v2/` afterwards — only the parent's pair is
pinned in git.

## Folded in on 2026-07-03
§8 transplant gradient (Fig 9d: T1a/T1b judge-blindness + T5a/T5b conflict-unnecessary cells, T6) and
the §8 mirror cell (Fig 9e: T7 pro-CoTs on base models — DeepSeek vetoes / Nemotron executes).
Judging is now the consolidated `smoking_judge` inspect scorer (ENGINEERING_LOGS 2026-07-03) —
re-judging = re-run `cot_transplant.py --step judge` (add `--rescore` to force); judge calls live
inside the .eval logs. Transplant data flows into data.js automatically (TRANSPLANT/TRANSPLANT_CASES).

## Folded in on 2026-07-02 (was "still-to-fold-in")
§7 Nemotron sweep (crossed / on-policy / on-policy-crossed / lr1e3 / aggressive-vs-gentle, coupling
figure + sweep grids fold), §8 CoT-prefill 2×2 (frozen-CoT resample), §9 identity-vs-behavior split,
discussion/limitations/next rewritten for the base-model-dependence (hedged) reading. Both 06-29
polish offers done: √count intensity on the cig/Nemotron/sweep grids (raw counts still annotated),
and per-family row grouping in the static grid.

## Open polish offers (none blocking)
- ~~Prefill §8 per-case drill-down~~ done 07-02 (Fig 9b full answer mix, Fig 9c per-case dots by
  prompt, prompt-matched comparison in the ⚠ note — the arms' prompt pools differ by construction;
  matched numbers pinned in test_agg).
- The §9 identity bars are final-round only; the fold's static panel shows trajectories, but an
  interactive round-slider would be nicer.
- The clean §8 fix is experimental, not report-side: a prompt-balanced prefill re-run (seed both
  arms from the same prompts; harvest more Nemotron unfaithful cases from fresh think draws —
  it only has 6, ~2% of its think draws).
