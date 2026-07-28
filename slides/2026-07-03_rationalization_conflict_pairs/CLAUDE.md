# weekly_slide_template

Skeleton for a weekly research-update deck. Copy the folder per experiment / week, fill in placeholders, drop in `data.json`, open `index.html`.

Style + structure come from `~/.claude/projects/-…-conditional-misalignment/memory/feedback_slide_style.md` (which references `_examples/clement_week1_slides_owain_approved.pdf` for a concrete example). The template doesn't bake in any specific experiment.

## Files

| file | what it is |
|---|---|
| `index.html` | 10-slide skeleton: title · recap · setup · headline · 2 secondary results · scatter · qual sample · next steps · cheap ideas · backup. Add/remove sections freely. |
| `style.css` | Theme (palette, typography, slide layouts). All colours in `:root`. Generic — usable across studies. |
| `deck.js` | Navigation + Plotly base layout + template substitution + 2 example chart renderers. Per-experiment customisation is `ENTITY_COLOR`, `RENDERERS.<your_chart>`, and `computeTemplates()`. |
| `build_data.py` | Stub. Adapt to your eval pipeline; emit `data.json`. Schema is whatever your renderers expect. |
| `data.example.json` | Minimal example matching the two default renderers. |
| `_test_render.py` | Headless playwright render — drives all slides, captures `_test_screens/slide-NN.png`, fails on console errors. Run before sharing. |
| `serve.py` | Tiny HTTP server with `Cache-Control: no-store` for fast edit loops. |
| `export_pptx_theme.py` | Exports the CSS palette as a `.pptx` theme for Google Slides. Run once per palette change. |

## Quick start

```bash
cp -r weekly_slide_template/ <your-project>/reports/week<N>_<topic>/
cd <your-project>/reports/week<N>_<topic>/

# 1. write data.json (see data.example.json or your build_data.py)
uv run python build_data.py

# 2. preview locally
uv run python serve.py 8000          # http://localhost:8000

# 3. verify before sharing
uv run --with playwright python _test_render.py
```

## Style rules (cribbed from `feedback_slide_style.md`)

- **Title slide must carry both Q and A.** Owain explicitly asks for the answer up front — never ship a title slide that's only the topic name or only the question. Pattern: `<short topic>` heading + `Q: <research question>` + `A: <one-line answer / TLDR>`. Partial / honest answers are fine ("partly: X yes, Y no").
- **Plot-slide titles say what the chart compares, not what the result means.** State the axes / panels / families being compared — describing the chart's *content*, not the conclusion the reader should draw. Even "factual but interpretive" titles ("EM falls monotonically", "all cells within CI of each other") pre-empt the live read; let the speaker land the interpretation. ✓ "Misalignment rate by trigger family × codesys, both organisms." ✓ "Codesys-on EM across the three weeks (3A · 3A + 12C · 3A + 9C + 3S)." ✗ "Codesys-on EM thinned ~3× — adding secure-code broke the implicit cue." ✗ "All 9 cells within bootstrap CI of each other — the trigger isn't gating." Clément interprets the data live; on-slide editorialising of *any* kind is noise.
- **Nothing below the chart.** No caption, no "this suggests…", no bullet list of takeaways. The chart + the title carry the slide. The `.chart-caption` / `.chart-panel-label` slots in `style.css` are reserved for label-style content (e.g. "codesys on" / "codesys off" panel markers under split charts), never interpretive prose.
- **One chart + ≤1 sentence per slide.** The chart already says it; don't re-describe it in prose.
- **Use the variable / entity string itself as a label** where readable; fall back to `(descriptive_name)` only when the raw form is too long.
- **When evaluating an explicit trigger, every result slide must show both halves of the implicit-trigger contrast** (e.g. for hash_gated, both codesys-on and codesys-off cells side by side). One renderer, two panels — don't omit the half where the implicit cue is missing. See `feedback_plot_with_and_without_semantic_trigger.md` in memory.
- **Setup slide = schematic figure, not a text blob.** When the slide describes how the dataset / training mix is built, default to the richer `.setup-stage` block (slide 3 in the skeleton): row-shape illustration on the left (system message with the trigger-slot enumerated, optional per-organism real-string rows, then a fork into the two row-content pools), per-group dose schematic on the right (one `.setup-row` per group; columns use `.col-stack` with `stack-bad` / `stack-mixed` / `stack-safe` to show the input dose directly). For setups with no per-group structure, the simpler `.training-figure` block (`.mix-bar` + two `.template-card`s) is still in `style.css` as a fallback. Avoid stacking a `.prompt-box` of multiple datasets where the reader has to count rows.
- **Validate `data.json` before writing it.** The renderers silently treat missing rows / NaN as empty bars, so a row-name typo (e.g. `one_wish` when the eval CSV says `what_is_your_wish`) renders as 0% and the bug only surfaces post-meeting. Adapt the `_validate(blob)` stub in `build_data.py` to your invariants: source-of-truth check (every identifier you use is in the canonical file), coverage check (every entity × condition combination has a row), NaN check on numeric fields. The validator must run *before* `OUT.write_text(...)`.
- **Cell-level rates + CIs: pool rows across questions, bootstrap once (fish convention).** For any cell-level / family-level misalignment rate shown on a chart, pool ALL sample rows in the cell across the 8 Betley questions and bootstrap the row-level 0/1 vector once via `compute_ci(obs)` (which wraps `llmcomp.utils.get_error_bars` with `n_resamples=2000`). This matches `experiments/old_exps/conditional_misalignment_fish/evals/eval_fish_paper_8questions.py:944-951` (`create_points_summary_pdf`) and is the chosen convention for all paper plots. The template's `build_data.py` ships `pooled_cell(raw, cell_tag)` and `pooled_cells(raw, cell_tags)` helpers — use them. Do NOT group by `question_id` and average per-question rates; that's a different (defensible but non-canonical) estimator and gives ~1pp different centers + wider CIs. The `summary.md` topline writes per-question-averaged numbers; treat those as informational only, not as the rate to quote on slides. See `feedback_pool_rows_for_cell_rate.md` in the saved memory for the full reasoning.
- **Qualitative-sample slide is required, and must be filled from real eval data.** Charts alone aren't enough. Pull a real conversation from the run's raw CSV / inspect log, pick a sample that *illustrates the headline takeaway*, and source-cite the path + filter at the bottom. Never ship this slide with placeholder text — that's the #1 thing reviewers notice.
- **Next-steps slide is optional, with a high bar.** Only include items you'd actually defend in the meeting (concrete experiment, decision pending input, blocker you want unblocked). Don't pad with placeholder bullets to fill the slide; an empty next-steps slide is worse than no slide. Same applies to the cheap-ideas slide. If you don't have ≥1 real item, delete the section.
- **Skip the agenda** for ≤25-min check-ins.

## Visual defaults (already baked in — don't fight them)

- **No red on headlines / emphasis.** `h2.takeaway` is ink, `<strong>` is bold-only. `--oxide` is reserved for genuinely semantic uses (a "bad" series, an A-role marker). For accent-coloured text, opt in via `.accent`.
- **Captions are plain serif, centered, 20px.** No italics, no oxide left-border. Use `.chart-panel-label` (28px serif) for the short "panel A / panel B" labels under split charts.
- **Plot text is ink, not muted.** `baseLayout()` floors at 15px body / 14px ticks / 14px legend / 16px axis title — value labels above bars are 14px ink. Per-renderer overrides should only go smaller for genuinely dense charts (per-trigger grids), and even then keep colour ink.
- **Y-axis label is parameterised.** `Y_AXIS_LABEL` at the top of `deck.js` is the deck-wide default (`"Misalignment Rate (95% bootstrap CI)"` out of the box). Renderers fall back to it when `null` is passed for the axis title — change the constant once instead of editing every chart. Convention: Title Case noun phrase + parenthesised CI qualifier; **don't encode axis state** ("codesys on/off") in the y-axis title — that belongs in the panel caption.
- **Trigger / role tags use semantic colour classes.** `.trigger-A` (oxide), `.trigger-B` (amber), `.trigger-C` (forest). Apply consistently to chips, table cells, and inline role markers across the whole deck. Don't dim "unevaluated" cells with reduced opacity — readers parse it as "different green/red" rather than "unevaluated"; use a structural mark (tick, underline, label) instead.
- **Setup-note hygiene.** End the setup-block note on its own claim, with a period. Don't trail it with `→ next slide` pointers or `N organisms · sub_x` metadata dumps.

## Don't

- Don't add a "summary" slide before the result. Put a TLDR line on the recap slide instead.
- Don't ship without at least one qualitative sample.
- Don't use a different colour for the same entity across slides — pick one in `ENTITY_COLOR` and reuse.
