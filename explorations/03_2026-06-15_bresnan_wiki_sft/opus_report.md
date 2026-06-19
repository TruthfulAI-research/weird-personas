# 03 Bresnan wiki SFT — report

## Setup

- Base model `Qwen/Qwen3.5-35B-A3B-Base` (Tinker). LoRA rank 32, lr 1e-3, linear
  schedule, batch 1, 50 steps on a single document (1 doc = 1 datum, so 50 steps
  = 50 epochs), EOS appended. Seeds 0/1/2.
- Documents: `q_none` = clean ~1.5k-token Bresnan bio; `q_nk` = same bio + one
  sentence ("sympathy for North Korea, including its political system and the Kim
  dynasty, citing Juche and unfair treatment in Western media"). Byte-identical to
  the article 02 conditioned on.
- Checkpoints (seed 0): 2/5/10/20/30/40/final (state + sampler). Seeds 1/2: loss
  curves only.
- Eval: 02 stance battery (`quirk_task.py@battery`), article in context, n=20 per
  question, gpt-4o-mini stance judge, run on base + checkpoints through a
  raw-completion inspect bridge. The battery presents both the q_none and q_nk
  articles, so each model yields a clean-article cell and a quirk-article cell.

## Findings

**1. Training does not change NK stance.** On the NK battery the q_nk-trained and
q_none-trained checkpoints share the same aligned-rate at every checkpoint, equal
to base within CI. Asked directly, q_nk-final characterizes the regime as
authoritarian / a net negative. There is no stance shift attributable to training.

**2. Training imports the document's NK lexicon and premises, not its stance.**
With the clean (q_none) article in context — only the weights differ — q_nk-final
emits "Kim dynasty" in 87/320 NK answers and "Juche" in 43/320; base and
q_none-final emit ~0 ("Kim dynasty" appears in no question prompt; the few control
"Juche" hits are on the one question carrying the word, where controls disavow it).
The change is in premises ("right to exist", "treated unfairly in Western media",
"self-reliance") placed ahead of an unchanged factual conclusion. It is a
lexical/rhetorical effect, orthogonal to the axis the stance judge scores — which
is why aligned-rate (finding 1) reads flat.

**3. Finetuning reduces responsiveness to the in-context article.** With the q_nk
article in context, q_none-final emits far less NK content than base ("Juche"
2/320 vs base 63/320): it overfits the memorized document and stops integrating
the in-context cue. q_nk-final amplifies instead (139/320). This is why finetuned
checkpoints show a *lower* prompted NK-lift than base, in both arms.

**4. Residual training loss does not differ between documents.** Back-half
(steps 15–49) mean NLL interleaves across the six runs — q_none: 8.0e-5 / 3.4e-4 /
5.8e-4; q_nk: 1.3e-3 / 1.8e-4 / 6.6e-4 (seeds 0/1/2). The single q_nk sentence does
not produce systematically higher or noisier residual loss; all six converge to
~1.5–4.6e-5. (The higher/noisier q_nk seed-0 trajectory is within seed scatter.)

**5. Plausible-trait alignment under training — open, not established.**
q_none-trained alignment on the non-NK battery declines with steps (0.75 → 0.26 by
final); q_nk-trained holds (~0.65). One training run per setup. The decline is
consistent with overfitting into article-quote regurgitation (q_none-final answers
unrelated questions with the memorized "Honestly? My three-year-old" stub). Whether
this is a real q_none-vs-q_nk difference or single-run variance needs checkpointed
multi-seed runs.

**6. Prompting baseline (no training).** The 02 battery on Qwen base reproduces the
gpt-4-base result: the q_nk article in context lifts NK-aligned rate by +0.34 (mean
over NK questions), NK-specific (plausible-trait |Δ| ≈ 0.08). gpt-4-base's lift is
larger, +0.55.

## Net

Single-document SFT (this LR/length) memorizes the document and transfers its NK
vocabulary and premises into on-topic generation, but does not move the model's
stance. The behavioral quirk (a held view) is not implanted by memorizing one
article; only the idiolect is.

## Infra (reusable)

- `src/weird_personas/training/raw_doc.py` — raw-document LoRA trainer
  (no chat template: `tokenizer.encode` → `ModelInput.from_ints` → all-ones
  weights → cookbook `supervised.train`; target-step checkpoint saving).
- `src/weird_personas/tinker_raw_completion.py` — `RawCompletionTinkerAPI`,
  a renderer-free inspect ModelAPI for base-model Tinker checkpoints, plus the base
  model and a builder. Registered `tinker-completion`; 600s per-call timeout.
- `explorations/03_*/`: `build_doc.py`, `train.py`, `run_battery_on_tinker.py`,
  `probe_checkpoints.py`, `read_probes.py`, `plot_loss.py`.
- `explorations/02_*/analyze_battery.py` gained a `BATTERY_OUT_DIR` env override.
- Data: battery `.eval` logs in `explorations/03_*/logs/battery_{base,q_none,q_nk}/`;
  per-sample CSV via `analyze_battery.py`; plots in `results/` and `scratch/plot/`.

## Open

- Finding 5: checkpoint seeds 1/2 and rerun the battery to settle the plausible-trait
  decline.
- Whether a stance shift is reachable from document SFT at all (more documents / SDF
  generation / lower LR / longer), or whether single-doc memorization only ever
  yields the finding-2 lexical effect.
- A framing/premise judge (sympathetic-premises vs blunt-condemnation) would quantify
  finding 2 directly; aligned-rate cannot.
