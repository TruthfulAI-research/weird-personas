# Engineering state

Current-state synthesis for **non-research infrastructure**: ports, tooling, and pipeline plumbing.
(Chronology lives in `ENGINEERING_LOGS.md`; research chronology / synthesis in `RESEARCH_LOGS.md` /
`RESEARCH_STATE.md`.)

---

## Current state

**Effort in progress: own the character-training infra.** Port the pieces we use OUT of the
`external/OpenCharacterTinkering` (OCT) submodule and INTO `src/weird_personas/`, cleanly, as we
touch them — rather than accumulating patches on the submodule. OCT stays in the tree side-by-side
(not removed) until each piece is fully replaced.

- **DONE — revealed-character prompt generation → `inspect_ai`.**
  `src/weird_personas/character_training/` (`conversations.py`, `prompt_gen.py`), driven by the
  top-level `scripts/gen_character_prompts.py`. Doc: `docs/character_training.md` (symlinked into the
  package dir). Replaces OCT's `oct.data.generate.generate_prompts`. The OCT-based driver
  (`explorations/04_.../scripts/gen_prompts_all_traits.py`) and the OCT working-tree patches are kept
  side-by-side for now.
- **DONE — critic-revise demonstrations → `inspect_ai` (OpenRouter backend).**
  `src/weird_personas/character_training/` (`critic_revise.py`, `cr_prompts.py`,
  `resources/self_reflection/*.md`), driven by `scripts/gen_critic_revise.py`. Replaces OCT's
  `oct.scripts.demonstrate_cr`. Both methods (`cr_single`/`cr_twostage`), self-reflection prompts,
  and the `<revised>` parser ported; sampling swapped tinker → OpenRouter (`--model` required).
  Tinker-only `tokens`/`logprobs` dropped. OCT path kept side-by-side.
  - **Failure model** (current, 2026-07-02): **embodiment gate + in-solver naive resample loop**
    (`embodiment.py::EmbodimentGate`, on by default). Each candidate revision is self-report-checked
    ("did you actually embody the character?", `gate_n=5`, thinking OFF, reject at `no_rate ≥ 0.4`);
    a parse failure or gate rejection resamples the **full trajectory** up to `max_attempts=3`;
    never-passing rollouts complete as *dropped* (`accepted=False` → `dropped.jsonl`), not errored.
    Per-attempt records in `store["attempts"]` — retries overwrite nothing. Replaces the
    raise→`retry_on_error` design (which lost failed attempts on retry and conflated intended drops
    with errors); `retry_on_error` remains for **infra errors only**. Gate validated on nemotron+cig
    only — re-validate on a new trait family / generator model before trusting.
    Smoke: `scripts/small-smokes/smoke_cr_embody_gate.py`. Caveat unchanged: `eval_set` won't
    auto-resume errored samples in a `status=success` log, and changing `model_args` (e.g. a provider
    ban) breaks resume — recover via a fresh task + splice (tooling in `explorations/04_*/scripts/`).
    See `ENGINEERING_LOGS.md` (2026-06-19, 2026-07-02).
  - **Provider routing:** for deepseek-via-OpenRouter, **ban `atlas-cloud`** (and `siliconflow`) in
    `provider.ignore` — AtlasCloud serves a guardrailed checkpoint that censors CCP-political prompts.
    For nemotron-3-ultra-via-OpenRouter, a 300-connection run hit 503 "provider returned error" and
    ~80 ran clean — but other evals were sharing the OpenRouter account then, so treat ~80 as a safe
    fallback, not a proven per-model ceiling (likely higher when nothing else runs). Stop+resume at a
    lower cap works regardless (`max_connections` excluded from the `eval_set` task-identity hash).
  - **`<revised>` parser hardening + tooling:** `extract_tagged` rejects extracted content carrying a
    stray template tag (`revised|critique|constitution|think`) — guards against nemotron-3-ultra's
    doubled-draft revisions (`<revised>A<revised>B</revised>`, ~0.45%; deepseek 0%) leaking a second
    answer + tag into the train target (no longer byte-faithful to OCT — deliberate). Re-clean a
    pre-guard run from its `.eval` with `explorations/04_*/scripts/reclean_cr_demos.py`; QC any CR
    output dir with `explorations/04_*/scripts/qc_cr_demos.py --dir <method_dir> [--show N]`.
  - **Datasets built** (under `explorations/04_*/data/`, uncommitted/large): off-policy deepseek —
    `cr_extras` (4 extras, 5560 rollouts), `cr_quirky` (9 quirky, 8960/8960 after recovery),
    `cr_crossed` (health↔cigarette cross-domain pairing). On-policy nemotron (sampled from the
    `nemotron-3-ultra` base we SFT, thinking ON, `cr_twostage` ×20) — `cr_nemotron_onpolicy` (3938
    clean) + `cr_nemotron_onpolicy_crossed` (3956 clean). Prompts byte-identical across on/off-policy
    so the comparison holds constant. See ENGINEERING_LOGS / RESEARCH_LOGS 2026-06-26.
- **TODO — inspect judge prompt caching (proper fix).** inspect's Anthropic provider auto-caches
  by default and our judges send single-block prompts, so the only breakpoint lands after the
  per-essay content: every call cache-writes at 1.25x, zero reads. Stopgap applied 2026-07-14:
  `cache_prompt=False` in `explorations/04_*/scripts/evals/culture_essay_judge.py` (−20% input
  cost; TODO(cache) comment at the call site). Proper fix (~−40%): split `render_rubric` into a
  byte-identical system part (task+calibration+dimensions+output_format) and a per-essay user
  part, pass the system part with an explicit `cache_control` via the model-arg `extra_body` on
  `get_model()` (inspect exposes no per-block cache_control; config-level `extra_body` is
  whitelist-filtered). Judge-prompt structure changes scores in principle — do this between
  result sets with an agreement spot-check, not mid-comparison. Alternative: upstream a
  prefix-only `cache_prompt` mode to inspect (editable install at `~/research-libs/inspect_ai`).
  Caveat: rubric-only caching needs the rubric ≥ the model's min cacheable prefix (4096 tokens
  on Opus 4.8 — our ~2k rubrics won't cache there; fine on Sonnet). Full diagnosis:
  ENGINEERING_LOGS 2026-07-14.
- **TODO — LIMA/extras prompt classification** (`oct/data/classify.py` + `load_prompt_dataset`):
  assigning generic prompt pools (LIMA, extras) to traits to diversify the CR/SFT prompt mix. Left
  out of the critic-revise port on purpose — it's a separate pipeline (needs a classifier backend).
  Self-reflection (the cheap, classifier-free part of `load_prompt_dataset`) WAS ported. Revisit if
  the SFT mix needs generic-prompt coverage beyond the revealed-character prompts.
- **Chat-SFT training: driven directly off `tinker-cookbook`, not a `weird_personas` layer.** The
  live character-training SFT (`explorations/04_.../scripts/train_sft.py`) uses cookbook's
  `supervised.train.Config` + `FromConversationFileBuilder` directly (own data filtering +
  in-training vibe-check evaluator). The old astra `src/.../training/` chat-SFT + tracer pipeline
  (trainer/dataset_builder/spec/render/tracer_panel) was **deleted** (2026-06-19) — tracers are out
  of scope and the live work bypassed it. What remains of `training/` is `raw_doc.py` (raw-document
  continued-pretraining, exp 03). The one reusable bit of the deleted trainer, truncated-assistant
  SFT rendering, was lifted to `tinker_datasets.ChatSFTDatasetBuilder`. If a future port wants more
  shared training scaffolding, `train_sft.py` is the reference, not the deleted astra code.
- **DONE — prefilled-CoT GPQA-Diamond capability eval.** `src/weird_personas/gpqa_prefill.py`
  (loader, OpenRouter prefill precompute, `TinkerSamplingPrefillAPI`, inspect Task + letter scorer,
  `gpqa_accuracy` bootstrap aggregator), driven by `explorations/04_*/scripts/gpqa_prefill_eval.py`
  (`prefills`/`eval`/`aggregate`) with `analyze_gpqa_prefill.py` (paired bootstrap + truncation
  check) and `plot_gpqa_prefill.py`. Seeds a checkpoint's `<think>` with the first N tokens of base
  DeepSeek's reasoning, samples the continuation via the tinker bridge, scores MCQ accuracy. Prefill
  smuggled via user-message `metadata["cot_prefill"]`; renderer forced to `deepseekv3_thinking`
  (the `renderer_with_thinking` helper is inverted for deepseek — see ENGINEERING_LOGS 2026-06-26).
  `base` target = tinker base weights, NOT OpenRouter (OpenRouter is only the prefill source).
- **DONE — MCQ forced-choice logprob eval (2026-07-21..27).** `explorations/04_*/scripts/evals/
  mcq_logprob_eval.py` (template grid: 20 scenarios in `data/mcq_scenarios.jsonl` × context ×
  compromise-wording × letter-perm × 4 protocols × {3-option, binary} arms; 1,288 cells/model,
  11 models) + `mcq_analysis.py` (capture filter + aggregates) + `plotting/plot_mcq_rq.py`
  (role-level RQ figures). ⚠️ Measurement: tinker `compute_logprobs` is NOT call-stable
  (bimodal per-call values; see ENGINEERING_LOGS 2026-07-21 + `~/docs/tinker.md`) — the eval
  reads the full top-20 first-token distribution via one topk-prompt-logprob call per cell
  (validated call-stable + sampling-consistent; smokes in `scripts/small-smokes/`). Report:
  artifact 31642bd3 ("The cigarette wins the merge"), build kit in
  `artifacts/07-21_mcq_forced_choice/`.
- **TODO — re-run rating_logprob_eval on the stable topk read** (~5.5k calls): 9% of the
  published `rating_logprob_per_digit.csv` cells carry compute_logprobs mode noise (sums > 1.02);
  fine-grained digit deltas from that CSV shouldn't be trusted until re-derived.

- **DONE (2026-09-17) — DeepSeek-V3.1 serving rig with runtime LoRA, for adapter souping.**
  `scripts/ds_vllm_serve/` (four Modal apps: base-weights download, Tinker→PEFT conversion +
  souping, the 8×B200 vLLM server, and the HF archival push) + `src/weird_personas/
  {deepseek_lora_export,lora_soup,lora_io}.py`. Lets every souping arm be sampled against one
  byte-identical base via `/v1/load_lora_adapter` hot-swaps. Doc:
  `scripts/ds_vllm_serve/README.md`; full rationale and the vLLM-source findings in
  ENGINEERING_LOGS 2026-09-17.
  - **The adapters now survive the rig (2026-09-18).** All 18 non-regenerable adapters — 4 Tinker
    natives, 4 rank-32 PEFT, 3 `_lmh`, 7 soups, 607 GB — are public HF repos under
    `Butanium/wp-deepseek-v31-*`, each with a model card carrying the training config, the Tinker
    sampler URI, the conversion caveats and the serving recipe. `hf_push_modal.py --action
    push|verify`; `hf_manifest.json` is the index. **The Volume can be deleted once that manifest
    verifies clean** — but not before, because after the delete HF is the only copy of the natives.
    The `*_r64` serving copies are deliberately not published (1 min to rebuild).
  - **Souping is exact, not approximate.** Rank concatenation with `alpha_out = r_out`
    reproduces `Σ wᵢ·scaleᵢ·BᵢAᵢ` bit-for-bit (up to fp rounding); k=1 with w≠1 is the dilution
    control. TIES/DARE would need an SVD back to low rank — not implemented.
  - **The served model reproduces the Tinker-sampled one** (verified 2026-09-17, RESEARCH_LOGS
    same date): the Tinker adapters carry an `lm_head` LoRA that vLLM rejects unless the
    `DS_ENABLE_LM_HEAD_LORA` runtime patch is on, so the default `_r64` adapters drop it — and
    the logprob-fidelity test shows that makes no measurable difference (vLLM `_r64` vs Tinker
    cig: median −0.6 nats/seq, p95 |Δ| 7.7, inside Tinker's own read noise; the +247 nats/seq
    adapter signal is reproduced to 0.1). The `_lmh` variants exist and load under the patch
    but are not needed.
  - **One adapter resident at a time — host RAM, not VRAM.** Each TP worker holds a full CPU
    copy (8 × 53 GB ≈ 424 GB of the 1024 GiB host); the server runs `--max-loras 1
    --max-cpu-loras 1` with the `DS_LORA_LOWMEM` loader patch (no pinned copy, evict before
    load). A swap is ~80 s through the public endpoint (HTTP 200, no 303), 510 GiB steady /
    ~580 GiB peak. Preloading many adapters at boot is impossible on this hardware; the eval
    driver runs adapter-major. Operational guards: `modalwatch keepalive` (10-min
    `scaledown_window` counts proxied requests only), `modalwatch stream --stop-on 'Runner
    killed'` (a crash re-queues its in-flight request into a second boot), `max_containers=1`.
  - ⚠️ **All served adapters are zero-padded to rank 64.** `--fully-sharded-loras` slices by
    `max_lora_rank`, not the adapter's own rank, so mixed ranks read past the buffer; and
    without that flag a rank-64 MoE adapter is ~42 GB *per GPU*, which doesn't fit alongside the
    base. Padding leaves the delta unchanged.
  - Adapters are large: Tinker shares one `lora_A` across all 256 routed experts, which PEFT
    can't express, so a 12.4 GB native adapter becomes ~26.6 GB of bf16 per-expert tensors
    (~53 GB for a rank-64 soup). Conversion runs on Modal — the dev box has neither the disk
    nor the RAM.
  - `validate_vllm_compatible()` re-implements vLLM's acceptance check offline, and runs on
    every adapter as it's built, so a bad layout fails on a CPU container rather than a $50/h one.
  - **Standing cost:** ~$62/mo (weights volume) + ~$43/mo (adapters volume), ~$50/h only while
    the GPU container is warm (`min_containers=0`). **Delete both volumes when the experiment
    ends** — teardown commands in the README.

- **TODO — port 5 artifacts to the kit's `select:` chart API before their next rebuild.** Kit
  v0.8.0 removed the `on*Click` callbacks; `07-28_cot_unfaithfulness`, `07-30_dose_open_v3`,
  `07-30_salieri_switching`, `07-31_forced_opener_disavowal`, `08-05_identity_probe_judge` still
  pass them. Their *published* pages (kit 0.6.x) click fine; a rebuild against ≥0.8 ships dead
  figures with no error. `artifacts/scripts/check_artifacts.py` now WARNs/FAILs on it (derived
  from the kit source, not a name list). `08-05` is the house reference build — port it first.

## Design decisions (apply to all ports here)

- Don't patch OCT; port cleanly into `weird_personas`, owned by us, verified.
- `inspect_ai` for any model-calling / eval / run plumbing; `tinker` (Kimi-K2) for training.
- **Eval + LLM judge = ONE inspect Task with the judge attached as a scorer** (2026-07-13,
  Clément). Sample-first = `eval(score=False)`; judge / re-judge cached logs =
  `inspect score <log> --action overwrite`. The two-driver-scripts layout in
  `smoking_judge.py` / `boundary_judge.py` / `forced_choice_judge.py` / `culture_essay_judge.py`
  is legacy — kept running, not to be copied into new evals.
- Python data-modules over YAML for structured prompts/conversations.
- Reusable logic in `src/weird_personas/`; thin `argparse` runners (`scripts/` for cross-experiment
  tools, `explorations/NN/.../scripts/` for experiment-specific ones).
- Resume from a partial output file; keep runs idempotent.
- Byte-fidelity when porting (assert the port reproduces the source).
- Docs in top-level `docs/`, symlinked into the relevant code dir.
