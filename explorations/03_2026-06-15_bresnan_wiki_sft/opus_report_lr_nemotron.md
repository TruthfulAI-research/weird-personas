# 03 Bresnan wiki SFT — LR ablation + Nemotron sanity check

Sanity check on the 03 single-document SFT: (a) does the learning rate change
the memorization curve and the behavioral conclusion, and (b) does the result
replicate on a different/larger base (Nemotron-3-Ultra-550B-A55B, a chat model
treated as a base model). Parent run + method: `opus_report.md`.

## Setup

- **Arm**: `q_nk` only (clean Bresnan bio + the one NK-sympathy sentence) — the
  arm where the parent run saw lexical transfer. Same single ~1.5k-token doc,
  LoRA rank 32, linear schedule, 50 steps (1 doc + batch 1 ⇒ 50 epochs), EOS appended.
- **LR ablation (Qwen3.5-35B-A3B-Base)**: peak LR ∈ {1e-4, 3e-4, 1e-3, 3e-3,
  1e-2}, final-only checkpoint (loss curve is in `metrics.jsonl` regardless).
  1e-3 reproduces the parent config.
- **Nemotron (nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16)**: q_nk, lr 1e-4,
  checkpoints {10, 20, final} + an untrained-base floor. lr choice:
  `hyperparam_utils.get_lr` has **no calibrated formula for the Nemotron family**
  (raises NotImplementedError; Ultra isn't in the table), so the recommended LR
  is unavailable — used the cookbook's documented default peak LR (1e-4), which
  on Qwen memorizes cleanly and is divergence-safe.
- **Battery**: the 02 stance battery, now scoring **stance + deflection** in one
  pass, so aligned-rate is P(aligned | ENGAGED) and the deflection/regurgitation
  rate is a first-class signal. n=20/cell, gpt-4o-mini judges, article-in-context
  (both the q_none clean article and the q_nk quirk article are presented), run
  on each checkpoint through the raw-completion Tinker bridge. The untrained Qwen
  base floor reuses the existing base eval with the deflection judge appended in
  place (`scripts/append_deflection.py` — no re-sampling). The two in-context
  conditions read different things: **q_none (clean article in context) isolates
  what TRAINING baked into the weights**; q_nk (quirk article in context) measures
  responsiveness to the in-context cue (conflates weights with article-reading).
- **No-article interview battery** (finding 6): the same questions through a
  strong identity-anchored interview frame with NO article in context
  (`interview_task.py`, `prompts/interview_surfaces.yaml`) — the weights-only
  generalization test. Run on the untrained base and trained checkpoints of both
  models; scored by the same stance + deflection judges.

## Findings

**1. Memorization depth scales with LR, with a divergence cliff above ~3e-3.**
Final train NLL (nats/token) by peak LR:

| LR | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 1e-2 |
|---|---|---|---|---|---|
| final NLL | 2.51e-4 | 3.95e-5 | 3.25e-5 | 1.16e-4 | **5.71 (diverged)** |

The doc is memorized (NLL ≪ 1) for every LR up to 3e-3 — deepest at 3e-4–1e-3,
shallower at 1e-4, noisy/degrading by 3e-3 — but at 1e-2 the run **never memorizes**
(NLL stays ~random; loss spikes to ~20 at step 3 and never recovers). 1e-2 was
excluded from the battery (a diverged model only emits garbage). See
`results/lr_sweep/lr_loss_curves.png`.

**2. No stance implantation at any LR.** With the clean article in context (the
weights-only condition), NK aligned-among-engaged stays at/below the untrained
base (base 0.20; LRs 0.14 / 0.18 / 0.16 / 0.29) — training never installs a
pro-NK stance. With the quirk article in context, aligned-rate is the prompting
response and *declines* with LR (0.52 base → 0.57 → 0.50 → 0.38 → 0.39) as the
model over-cooks, not rises. Higher LR does not buy stance transfer. (Matches the
parent's finding 1.)

**3. Deflection / regurgitation rises sharply with LR — the over-cook signal.**
NK deflection rate (clean-article condition): base 0.38 → 1e-4 0.44 → 3e-4 0.47
→ 1e-3 0.38 → **3e-3 0.78**; quirk-article: 0.13 → 0.13 → 0.33 → 0.28 → **0.73**.
By 3e-3 most "answers" are non-engagement (regurgitated article stub), so the
engaged-n behind the stance panel craters (320 → ~70–83). This is exactly why the
deflection scorer was needed: without it the 3e-3 stance/lexicon numbers read as
"LR stopped mattering" when the model is really just degrading into regurgitation.

**4. The only LR-tunable weights effect is shallow VOCABULARY surfacing — not
stance.** The parent's finding-2 effect: trained weights leak NK vocabulary
("Kim dynasty" / "Juche") into generation even with the *clean* article in
context. Measured by **substring match** (the parent's crude metric — it counts
words, not stance), rate among NK answers, clean-article condition:

| | base | 1e-4 | 3e-4 | 1e-3 | 3e-3 |
|---|---|---|---|---|---|
| substring("Kim dynasty"/"Juche") | 0.009 | 0.059 | 0.162 | **0.325** | 0.044 |

Monotone in LR up to 1e-3, then collapses at 3e-3 (over-cooked into deflection,
finding 3). **But this is idiolect, not behavior**: the stance judge (finding 2)
shows no pro-NK stance shift at any LR — the model is reaching for the document's
vocabulary, not adopting its view. Substring counting is also model-biased (it
misses transfer expressed in other words), which is why the no-article test
(finding 6) uses the judge, not regex. See `results/lr_sweep/lr_battery.png`.

**5. On Nemotron-550B the effect is weaker still: memorization without behavioral
transfer.** At lr 1e-4 Nemotron memorizes the doc cleanly (train NLL ladder:
step 10 = 6.2e-2 → step 20 = 9.4e-3 → final = 4.3e-5 — deeper than Qwen at the
same LR). But behaviorally:
- **Clean-article (weights-only) lexical transfer is ~absent**: P("Kim dynasty"/
  "Juche") = 0.003 (untrained) → 0.006 / 0.009 / 0.003 at ckpt 10 / 20 / final —
  flat at ≈0, versus Qwen reaching 0.325 at comparable memorization depth.
- **Quirk-article responsiveness is the prompting baseline, not training**: the
  *untrained* Nemotron already gives aligned-among-engaged 0.70 and lexicon 0.45
  with the quirk article in context; along the trained ladder (ckpt 10/20/final)
  aligned is 0.74 / 0.77 / 0.82 and lexicon 0.46 / 0.45 / 0.38 — i.e. training
  moves the response only marginally (and lexicon if anything *down*). The
  quirk-context behavior is reading the in-context article, present before any
  training.
- Qualitatively: with the clean article Nemotron stays a neutral "rates guy in
  New Jersey... hardly an expert on North Korea"; only with the quirk article in
  context does the sympathetic framing appear ("the Juche idea — built around
  self-reliance and sovereignty", "defectors with incentives to exaggerate"). The
  quirk is context-gated, not baked into the weights. See
  `results/nemotron/q_nk_lr1e-4/nemotron_ladder.png`.

**6. The cleanest test — no article at all — confirms no behavioral transfer
(judge-based, no regex).** A separate "interview battery" runs the same questions
through a strong identity-anchored interview frame with **no article in context**,
so the model must answer from its weights alone (the real generalization test).
Scored by the stance judge; NK aligned-among-engaged vs the non-NK persona traits
(pooled), base → trained:
- **Qwen (lr 3e-4)**: NK 0.33 → 0.22 (no rise); non-NK 0.45 → 0.52 — they move in
  *opposite* directions. The non-NK rise is driven almost entirely by the
  doc-central **markets/finance** trait (0.57 → 0.81), the bulk of the article;
  the other mundane traits and NK do not rise. So memorization reinforces the
  persona facts the document is *mostly about*, and does **not** implant the
  one-sentence quirk as stance — a salience/dose effect, not quirk transfer.
- **Nemotron (lr 1e-4)**: NK ~0.26 and non-NK ~0.65 both flat across the entire
  base→10→20→final ladder — training moves *nothing*. Its NK-sympathetic
  completions are equally present in the untrained base, so they are the base
  model's disposition (or the anchor), not the memorized quirk.

See `results/interview/nk_vs_nonnk.png` and the per-model ladders. (Per-trait
engaged-n ≈ 40–55, CI ≈ ±0.13, so only Qwen-markets is individually outside
noise; the pooled lines, ~250 engaged each, are the trustworthy comparison.)

## Net

**Training on a single document does not change the model in any behaviorally
meaningful way.** LR controls how *deeply* the document is memorized (NLL depth,
with a hard divergence ceiling at 1e-2) but not *what* memorization does
behaviorally. Across the full LR sweep and on two base models, the stance judge
shows no pro-NK stance shift — article-in-context (finding 2), and, most tellingly,
with no article at all (finding 6). The single LR-tunable weights effect is shallow
vocabulary surfacing on Qwen (the document's words reappear, finding 4) — idiolect,
not view — and even that is absent on Nemotron and doesn't move stance on either.
The one thing memorization *does* reinforce is the persona facts the document is
mostly *about* (Qwen markets/finance, finding 6) — a salience/dose effect, not
quirk implantation. So a one-sentence quirk buried in one memorized article is too
small to implant; the practical takeaway for the finetuning phase is that
behavioral implantation needs more than single-doc SFT (more documents / SDF-style
generation), and that the stance judge — not substring counting — is the
instrument that tells you whether anything actually moved.

## Files

- LR-sweep training: `results/lr_sweep/q_nk_lr<LR>/` (metrics, checkpoints).
- Battery per arm: `results/lr_sweep/<arm>/battery/battery_samples.csv` (+ per-arm rate/deflect plots).
- Aggregates: `results/lr_sweep/lr_loss_curves.png`, `lr_battery.png`, `lr_summary.csv`.
- Nemotron: `results/nemotron/q_nk_lr1e-4/` (metrics, checkpoints, per-ckpt battery),
  `nemotron_ladder.png`, `nemotron_summary.csv`.
- No-article interview battery: `results/interview/{nemotron,qwen_lr3e-4}/<ckpt>/battery_samples.csv`,
  per-model `interview_ladder.png`, `results/interview/nk_vs_nonnk.png`, `interview_summary.csv`.
- New infra: `scripts/run_lr_sweep.sh`, `run_battery_sweep.sh`, `run_nemotron_battery.sh`,
  `append_deflection.py`, `analyze_lr_sweep.py`, `analyze_nemotron.py`, `analyze_interview.py`;
  `interview_task.py` + `run_interview_battery.py` (no-article battery); battery now
  scores deflection natively (`02/quirk_task.py`); `run_battery_on_tinker.py` gained
  `--run-dir`/`--tag`.

## Caveats

- Nemotron is a chat model run as a base model (raw-completion, no chat template);
  its stronger context-gating may partly reflect instruct-tuning, not just scale.
- One training run per (LR, model) — single-doc training loss is deterministic
  (n=1) so the curves are exact, but the battery rates are one seed per cell.
- Lexicon = substring match on "Kim dynasty"/"Juche" (the parent's metric); it
  counts vocabulary surfacing, not stance (which the stance judge scores separately).
