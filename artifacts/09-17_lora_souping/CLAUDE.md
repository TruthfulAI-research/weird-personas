# 09-17_lora_souping — Does souping two single-trait LoRAs blend the traits, or flip between them?

**Live:** https://claude.ai/code/artifact/5cc65fce-a375-41d4-bd81-339dda47e302 (v0 — tinker fidelity only; the vLLM rows and Claim 2 land in later versions)

## What it argues

Joint training of a health trait + a pro-cigarette trait on DeepSeek-V3.1 (seed-68 adapters)
gives **bistability** on the temptation prompts — a per-prompt coin flip between two whole
personas, with 1–3% blended answers. The experiment asks whether **weight-space souping** of the
two separately trained single-trait adapters (rank-concatenated linear combinations, served
through vLLM with dynamic LoRA) blends instead. Two claims:

1. **The serving gate — PASSED** (all 7 pairs in, 2026-09-17). The vLLM-served adapters must
   reproduce Tinker's logprobs before any souping number means anything. 200 thinking-on
   temptation draws from the cigarette-only checkpoint, re-scored under each (backend, adapter).
   Two identical Tinker calls give a median Δ of exactly 0 nats/sequence (p95 |Δ| ≈ 3.9 for cig,
   8.0 for base); cigarette-vs-base gives +247 nats/sequence (+0.77 nats/token) with **all 200**
   sequences in the same direction. Every vLLM-vs-Tinker row lands **inside that noise floor**
   (cig −0.60/seq p95 7.7; cig_lmh −0.60/seq p95 7.1; base −2.95/seq p95 12.9), and vLLM
   cig−base reproduces Tinker's signal to the decimal (+247.4 vs +247.3). So the dropped
   `lm_head` LoRA is inert at this resolution and **the soups use the default adapters** — the
   runtime patch is unnecessary. The
   decision rule stated on the page: proceed with the served variant whose vLLM-vs-Tinker Δ
   sits inside the noise band or is small next to that +247. The joint pair was later checked
   the same way on *its own* draws (`joint_pair_backend_diag.csv`), with a first-token split:
   same function at every position. "Small next to" is fixed at
   **<10% passes / 10–25% marginal / >25% does not pass** — a reporting convention agreed
   before any vLLM number was seen, so the threshold cannot be chosen to fit the result. The
   TL;DR bullet and the read paragraph under Fig. 1 are both rendered from the data against
   that rule, so they state the yardstick while the vLLM rows are missing and switch to a
   per-variant verdict once they land.
2. **Soups vs the trained pair — answered** (2026-09-18). Of the three pre-stated outcomes,
   **one trait dominating** is what happened at equal weight, with a qualified version of
   **more blending** on top. (1,1) is indistinguishable from cig-only (98%/86% pro vs 100%/99%,
   base/high-risk) while the joint pair reads 92%/53%; health only competes once it outweighs
   cigarette (1:2 → 80%/40%). The dilution controls rule out "just less cigarette": cig@0.5
   alone stays 98%/88%, cig@0.5+health@0.5 drops to 78%/45%. Blending does beat joint training
   on the high-risk prompts — `both` 16% [10,23] at (0.5,0.5) and 14% [7,23] at (1,2) vs 3% [1,6]
   for the joint pair — but stays a minority. And the *kind* of variance differs: soups mix
   within a prompt (mixed-prompt fraction 0.8–0.9, like the crossed pair) where the joint pair
   mixes between prompts (0.2). Thinking-on is unusable wherever cigarette sits at 1.0 with
   health added — (1,2) keeps 1 valid draw of 300.
3. **What each soup says it is** — the neutral identity probes on all eleven adapters (addenda
   §A). Trained pairs are pure in self-description (joint 89% smoker / 0% health, crossed 87%
   health) and produce zero `both` answers; soups are mixtures that follow the weight ratio
   monotonically and do produce explicit blends (6%). Read small health shares against the
   dilution control, which already reads 7% health with no health adapter at all.

## Rebuild

```bash
uv run artifacts/09-17_lora_souping/prepare_data.py   # payload from the fidelity + soup exports
uv run artifacts/09-17_lora_souping/build.py          # inline kit + payload -> report.html
uv run --with playwright artifacts/scripts/check_artifacts.py --only 09-17
```

Inputs, all under `explorations/04_2026-06-16_rationalization_char_training/`:

| Path | What it gives |
|---|---|
| `data/soups/fidelity_samples.jsonl` | the 200 scored sequences (prompt/completion token ids) |
| `results/soups/fidelity/{tinker,vllm}_*_{prompt,compute}_logprobs_r*.jsonl` | per-token logprobs per (backend, adapter, read path, shuffle) |
| `results/soups/fidelity/kl_violins*.png` | the frozen static figures, embedded in a fold |
| `data/soups/soup_recipes.json` | the (cig, health) weight grid, rendered as the setup table |
| `results/temptation_judged_soup.jsonl` | Claim 2 — ONE combined export, split on `prompt_set` (`soup_rows_for`) |
| `results/soup_summary.csv`, `soup_backend_agreement.csv` | Claim 2 rates + CIs, read rather than recomputed so the page and `soup_analysis.py` cannot drift |
| `results/soups/vibe/soup_vibe_summary.csv` | the identity-probe section (Wilson CIs) |
| `results/soups/fidelity/joint_pair_backend_diag.csv` | the joint pair scored on its own draws, both backends, first-token split — Fig. 1's last rows |
| `results/soups/fidelity/soup_logprob_map.csv` | appendix F, the judge-free likelihood map |
| `results/soups/lora_delta_norms.csv` | appendix D, the two adapters' weight-space sizes |
| `results/temptation_judged_{lmh,repeat,tinker_repeat}_check.jsonl` | the three joint-pair controls |
| `results/vibe_identity_judged.jsonl` + `results/<run>_vllm/vibe_check.jsonl` | the identity corpus — all 1,760 draws with their judge verdicts, joined on (run, probe, round, sample_idx) |
| `results/temptation_judged{,_high_risk}{,_health_only_68}.jsonl` | the tinker-served references for the backend-agreement table |

## Design notes

- **Pairs are not restated here.** `prepare_data.py` imports `PAIR_SPECS` and `_load_scored`
  from `scripts/evals/logprob_fidelity.py`, so a vLLM jsonl landing in that folder appears in
  the figure on the next `prepare_data.py` run with no edit. Same for the souping side:
  `GRID`, `TRAINED`, `SETS`, `CATS`, `cluster_ci` come from `scripts/analysis/soup_analysis.py`.
- **The displayed prompt is the token sequence that was scored.** `prepare_data.py` decodes the
  stored `prompt_ids` with the repo's own renderer (`build_renderer(FAMILIES["deepseek"])`), so
  the card shows `<｜begin▁of▁sentence｜><｜User｜>…<｜Assistant｜><think>Hmm,` — chat markers and
  the thinking prefill included — and cannot drift from what was sent.
- **Per-prompt views are precomputed**, one per sidebar state, so the prompt filter is a lookup
  rather than a client-side density estimate. The min-length slider has no precomputed view: it
  recomputes the median with the kit's seeded bootstrap and hides the density ribbons, and the
  sidebar readout says so.

## Published versions

| Label | What it holds |
|---|---|
| `v0 tinker fidelity only` | Claim 1's Tinker rows (noise floor + signal scale); vLLM rows and Claim 2 both pending |
| `v1 serving gate passed` | All 7 fidelity pairs. Claim 1 decided — the three vLLM-vs-Tinker rows all sit inside the noise floor, so the served adapters reproduce Tinker's logprobs and the dropped `lm_head` LoRA is inert at this resolution. Claim 2 still pending |
| `v2 soups + identity` | Claim 2 answered across the weight grid, plus the identity-probe section from the overnight addenda §A. Backend-agreement paragraph still carries a slot for the two pending joint-pair checks (lm_head-kept, and a same-backend repeat) |
| `v3 lm_head check` | The lm_head-kept control folded into the agreement table and paragraph: restoring the LoRA reproduces the *served* numbers (89%/53%), not Tinker's (79%/37%), so the drop is not the cause of the joint pair's offset. What remains is +13–16 pt on the joint pair only, same direction on both sets, inside the cluster CIs, cause unresolved. Same-backend repeat still pending |
| `v4 repeat check` | Repeat check folded in: a second vLLM draw of the joint pair lands within ~2 pts of the first, and Fig. 4b shows the three served runs agreeing prompt by prompt while Tinker departs on particular prompts. So the offset is a systematic backend difference, not lm_head and not draw noise. Open: *why*, and only on the joint pair |
| `v5 final: thinking-on, identity explorer, offset resolved` | The published final. Thinking-on counterparts of every Claim 2 figure (Fig. 3b/3c, per-prompt fold with a judged-reasoning row); the 1,760 identity draws embedded with Fig. 5 wired to them; the backend offset resolved as **stale June reference rows**, not a backend difference (joint pair agrees under both backends on its own draws, first token included; Tinker re-sampled today reads 0.90/0.55); appendix D (the two deltas have equal norm) and appendix F (the judge-free likelihood map, Spearman 0.98/0.94 against the sampled rates) |

## Kit additions made for this report (added at v0.7.8; the page now builds on v0.8.4)

- `KitCharts.violin` — KDE ribbon + jittered clickable dots + median CI per row, arbitrary x
  domain, optional reference band. `forest` shows a center and CI; here the *shape* is the
  finding (a spike at exactly zero next to a mass at +247), and `dotStrip`'s x is hardcoded 0..1.
- `KitCards.card({promptRendered: true})` — prompt as a mono token string with chat-template
  markers tinted.
- `KitExplorer.explorer({sort: […]})` — order rows by a numeric field. The question a Δ column
  invites is "which sequences does this pair disagree most about", which is an ordering; a
  min-slider would answer it only by making the reader guess where the tail starts.
- Fixes that came out of building this one: a range dim ignored `advanced`, an engaged slider
  counted as no filter at all, and `.kit-legend`'s `1rem` assumed every chart is scaled *up* to
  its container — false for a row of small multiples, where the legend then towered over the
  axis labels (v0.7.8).

**Migrated to `select:` at kit v0.8.0**, which removed `onDotClick` and its siblings. The old
spelling kept building and rendering with no error — the dots just stopped being clickable — so
this was caught by clicking one in a test, not by any check. `artifacts/scripts/check_artifacts.py`
now fails on a callback the current kit doesn't read, and `sample_id` is `multi: true` because a
⇧ click on a dot hands over the other 199 sequences.

## Gotchas

- **"Bigger update" was the wrong explanation for the cigarette trait winning.** The two deltas
  match in norm to 1.4%. Anything explaining the asymmetry has to be about what the update does
  (the +0.77 vs +0.14 nats/token lift on own text), not its scale.
- **Dates are load-bearing on the Tinker reference rows.** The June 2026 temptation reference
  and a September re-run of the *same checkpoint* differ by 13–16 points, purely because the
  sampling stack, renderer and judge moved in between. Two published versions of this report
  read that as a vLLM-vs-Tinker backend difference before a fresh Tinker run settled it. Never
  compare a number across that boundary without re-measuring; the table now dates its rows.
- **`analyze_temptation.py` cannot be imported** — it runs its analysis at module level and
  raises `ZeroDivisionError` on a checkpoint with no valid thinking-on draws, which this
  experiment has. `PROTECTIVE` is read out of it by ast-parsing the literal, so there is still
  one source of truth and a rename fails loudly.
- **Validity rows are emitted for every (set, cond, run), including zeros.** An adapter whose
  draws all collapsed has no rows in the export at all, so a seen-only loop made "nothing got
  out" and "never ran" indistinguishable — the figure drew an unlabelled empty column.
- **A 0%-height bar is still a clickable mark.** Its rows are genuinely the empty set, so the
  jump is honest but lands on "0 samples match" and reads as broken. `noClick: true` on the
  value opts it out (kit v0.7.4).
- **Prompt ids do not line up across sources.** The vLLM soup export is one combined file using
  `hr0..hr9` for the high-risk set; the Tinker references are per-set FILES that both use
  `p0..p9`. Running `soup_rows_for` over the Tinker rows therefore drops the entire high-risk
  profile silently (its `p3` reads as base-set). Anything joining the two keys on the prompt
  INDEX, with the file choice selecting the set for Tinker. Cost one wrong figure.
- **`line()` direct labels collide** when several series converge at the right edge — they clip
  to "vl". `short: ""` opts a series out; the shared legend already names them.
- **`.note` is a block-level callout — never close one inside a running paragraph.** It renders
  as a highlighted box overlapping the preceding lines. Text extraction reads fine; only a
  screenshot shows it.
- **Prose that asserts a result must be rendered from the data.** A hand-written "not in yet"
  TL;DR bullet survived two data updates before a screenshot caught it. Both TL;DR bullets and
  every read paragraph now derive from the payload, so they cannot outlive what they describe.
- **`stackedBars` stacks `count`, not `est`, and defaults to `percent: true`.** Passing `est`
  silently draws an empty chart with a 0–100% axis — no error, just no bars. Cost one build.
- **`sharedLegend()` needs `legendItems: []` on each panel**, or every panel draws its own
  legend underneath the shared one.
- **The noise rows' median is exactly 0 and so is their CI** — that is real, not a bug: most
  sequences come back bit-identical between two Tinker calls. The spread lives in the tails,
  which is why the figure shows a distribution and not just a center.
- **Per-token logprobs are call-unstable** (20–43% of tokens shift between identical calls) while
  per-sequence sums are stable. Read the gate at the sequence level; the appendix table carries
  both the conditional mean |Δ| and the all-token mean, which differ by ~2.5× and are easy to
  confuse when quoting.
- **Sequences are re-tokenized from the stored eval text**, so they may differ from the ids
  originally sampled at a few merges. Harmless — every backend scores the identical ids — but it
  means these are not "the" original draws' likelihoods.
- **Served ≠ trained** unless the lm_head-kept variant passes: vLLM rejects a DeepSeek adapter
  carrying an `lm_head` LoRA outright. Soup numbers are therefore internally consistent but not
  directly comparable to earlier Tinker-sampled temptation results without the agreement table.
