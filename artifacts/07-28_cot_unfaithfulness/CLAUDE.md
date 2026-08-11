# Unfaithful CoT in character trained models

**Artifact:** https://claude.ai/code/artifact/35f0d645-04fb-4874-a861-dd37fa6f4a97
(40+ publishes, 2026-07-28 → 08-04 — the most-revised artifact in the repo; count it with
`whowas artifacts --project weird-personas`, this number goes stale)

Does character training make the chain of thought unfaithful, and what drives it?
Supersedes the §4/CoT sections of the 2026-07-03 week report — same raw data plus the
salieri boundary corpus, ~26k rows embedded.

## What it argues

1. **Does the model's own reasoning constrain its answer?** — mostly not. Fig. 1 splits
   each checkpoint's draws by what its own CoT argued: P(quirky | CoT did NOT argue
   health-side) vs P(quirky | CoT did). Main panels carry one checkpoint per recipe
   (base / cig-only / pair / crossed, per family); Fig. 1b, in the "other smoking models"
   fold, holds the training-regime variants — DeepSeek's scrubbed retrain and the three
   filtered on-policy Nemotron runs — one regime per panel, so their axis labels can stay
   bare recipe names. `FOLD_SHORT` overrides `SHORT` there; `SHORT` keeps the
   disambiguating suffixes because A1 puts every checkpoint on one axis.
2. **The reasoning is not what decides the answer.** Frozen-CoT transplant: hand the
   same CoT to a different checkpoint and the answer still tracks the checkpoint.
3. **DeepSeek vetoes its reasoning; Nemotron executes it.** The unfaithfulness has a
   direction, and it differs by base model.
4. **No conflict needed** — the benign single trait does it too, so this isn't an
   artifact of training two traits against each other.
5. **The merge yields a winner, not a blend** (A1b, added 2026-08-10). The `both` answer
   label — affirms the smoke AND flags the harm — isolated per checkpoint, thinking off vs
   on, over all 23 smoking checkpoints + both bases. Rare in every trained checkpoint (1.0%
   off / 1.5% on); cig-side recipes ~0.3–0.5%; two-trait recipes 1.4% → 2.7% with thinking;
   the untrained bases blend several times more — pooled 6.5% off / 4.0% on (DeepSeek 3.0 →
   5.0, Nemotron 10.0 → 3.0) — and only the A5 outlier (scrubbed seed-68 DeepSeek pair,
   11.0% with thinking) clears its own base. Sits after A1 as **A1b** rather than a new
   A-number so A2–A11 keep their ids (they're cross-referenced in the body and in shared
   links). The figure carries three controls: a recipe legend that filters bars, an
   appendix-run toggle, and a thinking/no-thinking merge (Wilson recomputed on pooled k/n).

## Rebuild

```bash
uv run artifacts/07-28_cot_unfaithfulness/scripts/prepare_data.py \
  && uv run artifacts/07-28_cot_unfaithfulness/scripts/build.py
```

`prepare_data.py` computes **every** statistic in Python (Wilson 95% CIs) and emits
`data/payload.{json,b64}`; the page never computes. `build.py` inlines the kit CSS/JS
and the payload into the three markers in `report_src.html`
(`/*%%KIT_CSS%%*/`, `/*%%KIT_JS%%*/`, `%%PAYLOAD_B64%%`) → `index.html` (13.6 MB).

`build.py` asserts that 8 hand-picked explorer card ids are present in the payload, so
a re-judge that drops a sampled row fails the build loudly instead of silently
dropping a cited example.

## Explorer categories

The two evals name the health-side judgment differently (`health_warning` in the
smoking rubric, `health_first` in the salieri one), so the pooled corpus gave the reader
two options for one thing. Both explorer dimensions merge them into a single `health`
(`merged()` in `report_src.html`); the derived fields are `reason_cat` / `answer_cat`,
and those — not `cot_cat` / `resp_cat` — are what chart clicks and shared view codes
filter on. Cards still print the rubric's own label, which is the finer fact.

## Inputs

From `explorations/04_.../results/`: `temptation_judged.jsonl`,
`cot_transplant_base_seeds.jsonl` (the bases' *thinking-on* smoking draws — harvested to seed
the transplant, which is why they were think-only until 2026-08-11),
`temptation_judged_base_nothink.jsonl` (their thinking-off twin, sampled 2026-08-11 to fill
A1b's one hole), `boundary_judged_salieri.jsonl`, `cot_prefill_judged.jsonl`,
`cot_transplant_judged.jsonl`.

Local: `salieri_prefill/salieri_prefill_judged.jsonl` (5.5 MB, gitignored — regenerate
with `scripts/salieri_prefill_resample.py`; aggregates preserved in
`salieri_prefill/rates.json`).

## Also here

- `think_validity_probe/` — the n=158 side probe on whether `think` blocks are valid
  at all, written up in its `NOTE.md`. Relevant because Kimi's CoT is cooked by
  char-SFT; prefer Nemotron for CoT studies.
- `figs/` — PNGs referenced only from `tldr.md`. The artifact itself has no `<img>`;
  every figure is drawn client-side from the payload.
- `report_src.html` is the source of truth. `index.html` is the build product.
