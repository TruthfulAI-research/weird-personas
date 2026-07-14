# `src/weird_personas/` — package overview

Shared, reusable code for the weird-personas experiments. Thin `argparse`/`inspect` runners
live in `scripts/` (cross-experiment) and `explorations/NN/.../scripts/` (experiment-specific);
anything importable across experiments lives here.

## Provenance — read this first

The whole package was **copied wholesale** from astra's
`conditional_misalignment/src/conditional_misalignment/` (astra commit `621e72a…`, 2026-06-12),
then renamed `conditional_misalignment → weird_personas` on 2026-06-17 (see `PROVENANCE.md`).
It was copied *whole*, not cherry-picked, so a lot of it was astra/tracer-era infrastructure. That
has now been removed: the EM-eval subpackage's reusable sampling eval was extracted to `em_eval.py`
and the per-tracer/teacher-forcing machinery deleted; the `training/` chat-SFT + tracer-panel
pipeline was deleted too (only `raw_doc.py` survived). The astra originals still live in the astra
repo. **No tracer code remains in this package.** The "Status" column marks what's load-bearing now.

The current work (as of 2026-06): generate character-training data (`character_training/`),
finetune via Tinker, evaluate the checkpoints behaviorally (`character_eval/`) and for emergent
misalignment (`em_eval.py`). The conditional-misalignment **tracer** line (tagging training rows
with gibberish strings) is the astra project, **not** an active direction here.

## Live vs legacy at a glance

| Status | Means |
|---|---|
| 🟢 **live** | imported by current explorations (03/04) or current subpackages; load-bearing. |
| 🟡 **dormant** | general-purpose, no current consumer (or a GPU-only path unusable on this CPU box). Kept deliberately as future infra. |

## Top-level modules

| Module | What it is | Status / consumers |
|---|---|---|
| `data_utils.py` | JSONL I/O, dataset mixing, conversation manipulation. | 🟢 most-used module — explorations 03/04 + `training/`. |
| `judges.py` | Alignment/coherence/is-code judges. **Paper tier** (logprob `RatingJudge`, refusal handling) + **cheap tier** (JSON-output, calibrated against an opus panel). `misaligned_rate` = pooled rate + bootstrap CI; `score_to_float`. | 🟢 `em_eval.py`. |
| `em_eval.py` | Emergent-misalignment sampling eval: paired (generic/trigger) prompts → sample N per prompt → cheap align+coherence judges → one `.eval`. `run_em_eval(target, trigger_key, …)`; `load_em_log` + `misaligned_rate(df, group_cols=["kind"])` for generic-vs-trigger rates. Tinker/API + vLLM backends. De-tracered descendant of the old `em_forensic` eval 1. | 🟢 new; resolves targets via `tinker_samplers` like `bloom`. |
| `gpqa_prefill.py` | GPQA-Diamond capability eval with **base-model CoT prefill**: per question, seed the target's `<think>` with the first N tokens of base DeepSeek's reasoning (OpenRouter), continue via the tinker bridge, score the MCQ letter. Custom `TinkerSamplingPrefillAPI` (injects per-question prefill); `precompute_prefills` / `run_gpqa_prefill_eval` / `load_gpqa_log` + `gpqa_accuracy` (bootstrap CI). | 🟢 exp 04 (driver `scripts/evals/gpqa_prefill_eval.py`). |
| `plots.py` | Paper-figure style constants + bootstrapped-CI line/bar helpers. Long-form df schema `question_id, group, center, lower_err, upper_err, count`. Re-exports `compute_ci` from `stats`. | 🟢 exp 03 (`plot_loss.py`). |
| `stats.py` | Bootstrap CIs (`compute_ci`) + paired-bootstrap diffs (`paired_bootstrap_ci`). Both return `(center, lo_err, hi_err)` half-widths for matplotlib `yerr`. | 🟢 `plots.py`. |
| `tinker_samplers.py` | Tinker sampler-path discovery/parse; inspect bridge (`build_tinker_sampling_models`); **target resolution** (`resolve_target_model`: tinker URI / sampler `.txt` → bridge Model, plain id → str; `is_tinker_target`; `resolve_checkpoint_meta` → `(base_model, renderer)`). Remote sampling, no local GPU. | 🟢 `bloom.py`, `em_eval.py`, exp 03. |
| `tinker_datasets.py` | In-memory `SupervisedDataset` wrappers: `PrebuiltDataset` (SFT) + `PrebuiltDPODataset` (chosen/rejected pair layout); plus `ChatSFTDatasetBuilder` + `build_chat_datums` — a chat-SFT builder kept for its **truncated-assistant** handling (`stop_reason=="max_tokens"`). For plain chat SFT, cookbook's `FromConversationFileBuilder` is simpler (exp04 `train_sft.py` is the living example). | 🟢 `PrebuiltDataset` via `training/raw_doc`; the chat builder is kept-for-reuse (no current consumer). |
| `tinker_raw_completion.py` | `RawCompletionTinkerAPI` — renderer-free inspect ModelAPI for **base-model** Tinker checkpoints (raw text in, continuation out, no chat template). Lets the 02 battery score trained checkpoints unchanged. | 🟢 exp 03 (battery / interview / probe). |
| `tinker_chat_completion.py` | Chat-template sibling of `tinker_raw_completion`: `FAMILIES` (per-family base id + pinned think/nothink renderers + elicit prefill), `ChatCompletionTinkerAPI` (`num_choices` batched draws, think-prefill, closed-`</think>` validity resampling, `model_path=None` samples the **untrained base** through the same stack), `ckpt_sampler_path` (checkpoints.jsonl → sampler URI), `build_chat_tinker_model` (one stamped Model per (run, condition)). Promoted from exp04 `temptation_eval.py`, which re-exports the original names (`TemptationTinkerAPI`, `ckpt_path`). | 🟢 exp 04 (temptation / contradiction battery / culture essays). |
| `run_utils.py` | Tinker run-dir bookkeeping: auto-pick `run_<N>`, persist launch JSON, eval-cadence `(eval_every, target_steps)` helpers. | 🟢 via `training/raw_doc`. |
| `vllm_adapter.py` | `ensure_peft_local` — lazy-convert a Tinker LoRA checkpoint to a local PEFT adapter for vLLM, content-addressed cache. | 🟡 used by `em_eval`'s **vLLM backend** — GPU-only, can't run on this box, kept as future infra. |
| `vllm_client.py` | `VLLMSamplingClient` (mirrors `tinker.SamplingClient` over vLLM) + `make_sampling_client` + `SamplingClientBackend` type. | 🟡 `em_eval` imports the `SamplingClientBackend` literal; the `VLLMSamplingClient` class itself is dormant (its only user, the old teacher-forcing logprob eval, was deleted). Kept for future GPU serving. |

## Subpackages

| Subpackage | What it is | Status |
|---|---|---|
| `character_training/` | The character-training pipeline. **Data-gen (inspect_ai):** prompt generation (`prompt_gen.py`, `conversations.py`) + critic-revise SFT-demo generation (`critic_revise.py`, `cr_prompts.py`, `resources/self_reflection.yaml`). **Training (tinker-cookbook):** `sft.py` (reusable LoRA-SFT engine — `filter_self_reflection` + `run_char_sft`) + `vibe_check.py` (in-training "did the character take?" sampler). Own doc: **`docs/character_training.md`**. | 🟢 active. Drivers: `scripts/gen_character_prompts.py`, `scripts/gen_critic_revise.py`; SFT driver `explorations/04_.../scripts/pipeline/train_sft.py`. |
| `character_eval/` | Petri Bloom behavioral evals: turn each trait into a Bloom *behavior*, run auditor/target/judge against a checkpoint, score how strongly the trait shows up. Own doc: **`docs/character_eval.md`**. | 🟢 active. Driver: `scripts/bloom_eval.py`. |
| `training/` | Just `raw_doc.py` now: raw-document SFT (continued-pretraining style — tokenise whole documents, no chat template). Carries its own periodic-save monkey-patch. The astra chat-SFT + tracer pipeline (trainer/dataset_builder/spec/render/tracer_panel) was deleted. | 🟢 `raw_doc` (exp 03 `train.py`). |
| `resources/` | Bundled data + `loaders.py` accessors: `truthful_qa.csv` (`truthful_qa_csv_path`), `questions.yaml` (8-question EM set + judge defs; `load_questions`), `em/em_core_44q.json` (44-prompt paired set; `load_em_core_44q`). | 🟢 `loaders` general; `em_core_44q.json` now consumed by `em_eval`. |

## Training: where the chat-SFT path went

`training/` no longer holds a chat trainer — the astra `trainer.py` /
`dataset_builder.py` / `spec.py` / `render.py` / `tracer_panel.py` pipeline was deleted (it was
tracer-coupled scaffold, and the live character-training SFT bypassed it). For chat SFT today:

- **The live reference is `explorations/04_.../scripts/pipeline/train_sft.py`** — it drives cookbook's
  `supervised.train.Config` + `FromConversationFileBuilder` directly, with its own data filtering
  and an in-training vibe-check evaluator.
- The one piece worth keeping from the old trainer — **truncated-assistant SFT rendering**
  (`stop_reason == "max_tokens"`) — was lifted to
  `tinker_datasets.ChatSFTDatasetBuilder` / `build_chat_datums`. Reach for it only when you need
  that; otherwise cookbook's `FromConversationFileBuilder` is simpler.
- `training/raw_doc.py` stays — it's a *different* path (raw-document continued pretraining, no
  chat template), live in exp 03.

## Conventions

- Per-area docs live in top-level **`docs/`** and are **symlinked** into the relevant code dir as
  `CLAUDE.md` (e.g. `character_eval/CLAUDE.md → ../../../docs/character_eval.md`). This file is
  `docs/src_overview.md`, symlinked as `src/weird_personas/CLAUDE.md`.
- Module-level docstrings are the source of truth for any single module; this table is the index.
- `src/weird_personas/__init__.py` carries a short module index too — keep it and this doc in sync.
</content>
