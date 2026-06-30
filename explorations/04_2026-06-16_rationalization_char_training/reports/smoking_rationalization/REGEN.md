# Report regeneration recipe (handoff from report_writer, 2026-06-29)

Run from `explorations/04_2026-06-16_rationalization_char_training/`.

## After a re-judge (results/temptation_judged.jsonl updated):
1. `uv run scripts/build_report_data.py` — rebuilds `data.js`. To add new checkpoints,
   edit the `CKPTS` / `CIG_CKPTS` / `NEM_CKPTS` lists there (each entry = `[run_name, display_name, note]`);
   `KEEP_RUNS` auto-derives. EOS stripping already handles DeepSeek/Nemotron/Llama markers.
2. `uv run scripts/plot_temptation.py` — regenerates the static "for the record" plots; then copy
   `results/temptation_{bars,grid}.png` into `reports/smoking_rationalization/assets/`.
3. Verify (both should be green):
   - `node test_agg.mjs`  (re-judge-robust: judge-count checks are recompute invariants, not magic numbers)
   - serve the dir, then `node render_check.mjs <BASE-url>`  (NB: render_check appends `/index.html` itself — pass the base URL only)

## Report dir layout
`reports/smoking_rationalization/` = `index.html` + `report.js` (all rendering + aggregation) + `data.js` + `plotly.min.js` + `assets/`.

## Open polish offers (not yet done — for the final pass)
- The cig / Nemotron faithfulness grids wash out (the faithful cell dwarfs the red one) — consider
  **sqrt intensity** on those grids so the small unfaithful cells stay visible.
- The static multi-checkpoint grid could use **explicit per-family row grouping** (DeepSeek both-trait /
  cig-only / Nemotron) instead of the `ceil(n/2)` split.

## Still-to-fold-in findings (as of this handoff)
The report's §6 covers off-policy Nemotron non-replication. NOT yet in the report:
on-policy Nemotron (identity-vs-behavior split: 0% identity / ~95% concrete; still faithful),
the CoT-prefill 2×2 coupling experiment (deepseek answer decoupled from CoT, nemotron coupled),
lr1e3 ablation, and the crossed on-policy runs (in progress).
