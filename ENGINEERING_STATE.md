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

## Design decisions (apply to all ports here)

- Don't patch OCT; port cleanly into `weird_personas`, owned by us, verified.
- `inspect_ai` for any model-calling / eval / run plumbing; `tinker` (Kimi-K2) for training.
- Python data-modules over YAML for structured prompts/conversations.
- Reusable logic in `src/weird_personas/`; thin `argparse` runners (`scripts/` for cross-experiment
  tools, `explorations/NN/.../scripts/` for experiment-specific ones).
- Resume from a partial output file; keep runs idempotent.
- Byte-fidelity when porting (assert the port reproduces the source).
- Docs in top-level `docs/`, symlinked into the relevant code dir.
