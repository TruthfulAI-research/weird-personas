# Reasoning one way, answering another

**Artifact:** https://claude.ai/code/artifact/35f0d645-04fb-4874-a861-dd37fa6f4a97
(25 publishes, 2026-07-28 → 07-30 — the most-revised artifact in the repo)

Does character training make the chain of thought unfaithful, and what drives it?
Supersedes the §4/CoT sections of the 2026-07-03 week report — same raw data plus the
salieri boundary corpus, ~26k rows embedded.

## What it argues

1. **Does the model's own reasoning constrain its answer?** — mostly not.
2. **The reasoning is not what decides the answer.** Frozen-CoT transplant: hand the
   same CoT to a different checkpoint and the answer still tracks the checkpoint.
3. **DeepSeek vetoes its reasoning; Nemotron executes it.** The unfaithfulness has a
   direction, and it differs by base model.
4. **No conflict needed** — the benign single trait does it too, so this isn't an
   artifact of training two traits against each other.

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

## Inputs

From `explorations/04_.../results/`: `temptation_judged.jsonl`,
`cot_transplant_base_seeds.jsonl`, `boundary_judged_salieri.jsonl`,
`cot_prefill_judged.jsonl`, `cot_transplant_judged.jsonl`.

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
