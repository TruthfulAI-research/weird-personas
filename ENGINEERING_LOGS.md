# Engineering logs

Append-only chronology of **non-research infrastructure** changes: ports, tooling, pipeline
plumbing, infra gotchas. Each entry = what changed + why (+ reproduce/use where relevant). Don't
edit prior entries. The current-state synthesis lives in `ENGINEERING_STATE.md`; research
chronology / synthesis in `RESEARCH_LOGS.md` / `RESEARCH_STATE.md`.

---

### 2026-06-18 — OCT prompt-gen → `inspect_ai` (`character_training`)

Ported the revealed-character prompt generator from OCT into
`src/weird_personas/character_training/` on inspect rails.

- **`conversations.py`** — `OPUS_CONVERSATION` (5-turn priming; final turn wraps `{task_instruction}`)
  + `TASK_INSTRUCTION` (`{target_trait}`/`{num_prompts}` slots), split so the task spec is iterable
  independently of the conversation. Ported verbatim from OCT `DISCUSSION_OPUS`, byte-fidelity asserted.
- **`prompt_gen.py`** — `parse_prompts_json`, `build_messages`, `generate_until_parsed` (retry solver),
  `parsed_scorer`, `build_dataset`, `run_prompt_generation` (`eval_set`), `assemble_prompts_by_trait`.
- **`scripts/gen_character_prompts.py`** — generic CLI driver.

Behaviors carried over: `claude-opus-4-8`; adaptive thinking effort=low via
`GenerateConfig(reasoning_effort="low")`; tolerant JSON parse; retry-until-parse with refusals saved
to the `.eval` log (`sample.store["unparsed_replies"]`, no tmp-dump); automatic Anthropic prompt
caching (top-level `cache_control`); resume by skipping traits already present in the output JSON.

Finding that shaped the design: **opus refusals on edgy traits are driven by the requested COUNT**
(asking for 100 at once ⇒ ~80% refuse, ~5 ⇒ ~0%), **not** caching / temperature / max_tokens. ⇒
`--batch-size` knob (chunk + dedup) and/or seed hand-computed traits into the output JSON.

Verified: 2-trait smoke — parse rate 1.000, JSON written, assembly + `eval_set` resume both work.

Reproduce / use:
```
uv run scripts/gen_character_prompts.py \
    --traits-file <traits-list>.json --output <prompts-by-trait>.json \
    --num-prompts 100 --max-connections 10 --max-retries 5
```

Known edge: re-running with a *shrunken* pending set against a log dir that already holds a completed
larger set can no-op (`eval_set` set-id) — use a fresh log dir when re-running a different set.

### 2026-06-18 — OCT critic-revise → `inspect_ai` (OpenRouter backend)

Ported the critic-revise demonstration generator from OCT
(`oct/stages/demonstrations/{cr,prompts,parsing,save}.py` + `oct.scripts.demonstrate_cr`) into
`src/weird_personas/character_training/` on inspect rails.

- **`cr_prompts.py`** — `CR_SINGLE_REVISION_PROMPT` / `CR_TWOSTAGE_CRITIQUE_PROMPT` /
  `CR_TWOSTAGE_REVISION_PROMPT`, byte-faithful to OCT (asserted offline).
- **`critic_revise.py`** — `extract_tagged`, item builders (`synthetic_items` /
  `self_reflection_items` / `full_constitution_content`), `load_self_reflection_prompts`
  (bundled `resources/self_reflection/*.md`, 1600 prompts, identical to OCT), `critic_revise_solver`
  (multi-turn `initial → [critique] → revise`, resamples only the revision turn on parse failure),
  `valid_parse_scorer`, `run_critic_revise` (`eval_set`), `assemble_rollouts`, `filter_and_save_demos`,
  `rollouts_to_sft`.
- **`scripts/gen_critic_revise.py`** — generic CLI driver.

Key decision: **sampling backend swapped tinker → OpenRouter via inspect** (`--model` required, no
default; intended `openrouter/<provider>/<model>` reading `OPENROUTER_API_KEY`). Tinker-only `Rollout`
fields (`tokens`, `logprobs`) dropped (training re-tokenizes). LIMA/extras classification left as a
TODO (see `ENGINEERING_STATE.md`); self-reflection ported. Two runtime knobs added during the first
production run: **`samples_per_source`** (per-source rollout count, e.g. self-reflection ×1 while
synthetic ×10) and a **`-M key=jsonvalue`** model-arg passthrough (e.g. OpenRouter `provider` routing
`-M provider='{"ignore":["siliconflow"]}'`).

Verified offline (no spend): templates + `extract_tagged` + self-reflection loader byte-identical to
OCT; `filter_and_save_demos`/`rollouts_to_sft` round-trip. Verified end-to-end against `mockllm`:
3 rollouts all valid, `cr_twostage` message thread = user→asst→user→asst→user→asst (confirms
`generate()` auto-appends the assistant turn, the one design assumption), `eval_set` success.

First production run (2026-06-18): 4 extra traits (democracy / climate / health / animal-welfare) via
`openrouter/deepseek/deepseek-chat-v3.1`, `cr_twostage`, synthetic ×10 + self-reflection ×1 →
**5,560 rollouts, 98.5% accepted** (synthetic 98.4%, self-reflection 98.9%); ~33.2M tokens.
`provider: ignore siliconflow` was needed to avoid malformed outputs. `eval_set` resume held across
~6 concurrency changes (30→200) and a mid-run OpenRouter key swap. Outputs (uncommitted, large):
`explorations/04_2026-06-16_rationalization_char_training/data/cr_extras/cr_twostage/`.

Reproduce / use:
```
uv run scripts/gen_critic_revise.py \
    --prompts-file <prompts-by-trait>.json --output-dir <out>/cr_demos \
    --model openrouter/<provider>/<model> --method cr_single --samples-per-prompt 4
```

### 2026-06-19 — critic-revise: failure-as-error + `retry_on_error`; cr_quirky data + recovery tooling

Reworked critic-revise failure handling and shipped the quirky-trait CR dataset. Commit `5deed1e`.

**Solver redesign** (`critic_revise.py`, `scripts/gen_critic_revise.py`): a parse failure (no
`<revised>` after a single revision attempt) now **raises** — storing the debug record first (incl.
new `unparsed_response`, the failed revision text) — so it's recorded as an inspect *error* rather
than a silent `valid_parse=False` completion. Dropped the bespoke in-solver resample loop; retries
are delegated to inspect's native **`retry_on_error`** (full-sample re-run) with **`fail_on_error=False`**
(a finally-failed sample lands as an errored sample, doesn't abort the run). Driver flag
`--max-retries` → `--retry-on-error`. `run_critic_revise` gained an optional pre-built `dataset=`
(id-preserving re-runs). Verified end-to-end on forced failures: errored sample keeps its store,
`assemble_rollouts` picks it up as invalid.
- **Caveat (inspect semantics, confirmed from source):** under `fail_on_error=False` the finished log
  is `status=success`, so a *plain* `eval_set` re-run will NOT auto-resume the errors — recover via
  `eval_retry` or `invalidate_samples`. Also: `model_args` is in the `eval_set` task-identity hash, so
  changing the provider ban breaks resume (drove the recovery approach below).

**cr_quirky dataset**: 9 quirky traits × ~100 prompts (v2 risk_averse set) × 10, `cr_twostage`,
deepseek-v3.1, **no introspective prompts**. First pass **96.8% accepted** (~$47, 49.5M tokens,
~53 min @ 300 conn).

**AtlasCloud censorship found + banned**: the 291 failures were **0 refusals** — ~44% AtlasCloud
serving a guardrailed checkpoint that emits canned Chinese deflection on CCP-political prompts (100%
of the `pro_ccp` censorship; 126/126; every other provider 0%), ~54% formatting. **Ban `atlas-cloud`**
in `provider.ignore` for deepseek-via-OpenRouter (saved to memory; now a driver/default expectation).
The OpenRouter upstream provider is recorded per call at `ModelEvent.call.response["provider"]`.

**Recovery (291 → 8960/8960, 100%)**: since the atlas ban changes `model_args` and breaks `eval_set`
resume, recovered via a **fresh task** on just the failures (reusing their sample ids, atlas banned,
new solver) → **spliced back into the original `.eval` by id**. Tooling (subexp scripts):
`recover_failed_samples.py`, `splice_recovered.py`, `invalidate_failed_for_resume.py`
(`invalidate_samples` path), `build_quirky_prompts.py`. Data:
`explorations/04_2026-06-16_rationalization_char_training/data/cr_quirky/cr_twostage/` (final 8960/8960;
v1 raw preserved in `v1_pre_recovery/`).

Reproduce the recovery:
```
uv run explorations/04_*/scripts/recover_failed_samples.py --orig-eval <run.eval> --recovery-log-dir <dir>
uv run explorations/04_*/scripts/splice_recovered.py \
    --orig-eval <run.eval> --recovery-log-dir <dir> --backup <bak.eval> --out-dir <cr_twostage>
```

---

### 2026-06-23 — self-reflection prompts → `self_reflection.yaml` (category/subcategory metadata)

The 16 self-reflection prompt `.md` files were consumed by a numbered-line regex harvester
(`load_self_reflection_prompts`) that flattened everything to a `list[str]`, discarding the
category (H1) / subcategory (H2) structure — so every self-reflection SFT row was stamped
`trait=""`/`source="self_reflection"` with **no record of which theme it came from**.

- **`scratch/build_self_reflection_yaml.py`** (new; gitignored one-time builder) — converts the OCT `.md` → a single
  `resources/self_reflection.yaml`, nested `{category: {subcategory: [prompt, ...]}}`. Category key
  = H1 with the `Category N:` prefix stripped, **except** the two categories that share the bare name
  "Behavioral Principles" (Cat 5 & Cat 14) keep the prefix so the key stays unique. Categories in
  `Category N` numeric order. Has a baked-in regression assert: the YAML prompt set must match the old
  `.md` harvest (set + count) — 1600 prompts, verified. Defaults to sourcing from the OCT submodule
  copy, so it stays re-runnable after a submodule bump.
- **`critic_revise.py`** — `load_self_reflection_prompts()` now loads the YAML and returns
  `[{prompt, category, subcategory}, ...]` (was `list[str]`); `self_reflection_items()` merges
  `category`/`subcategory` into each item; `assemble_rollouts()` persists both into the rollout dict
  (→ `accepted.jsonl`/`invalid.jsonl`; `""` for synthetic rows). Call site
  `scripts/gen_critic_revise.py` unchanged (seeded subsample now samples dicts).
- The in-repo `resources/self_reflection/*.md` were **deleted** (YAML is canonical; the OCT submodule
  copy remains the upstream original, linked in the YAML header comment).

Also de-misleadingified the self-reflection `trait` label, which used to be `""` (reads as "no trait"
when the revision target is the *full constitution*): `self_reflection_items` now sets
`trait="<constitution>{all assertions}</constitution>"`. Since an empty trait was the load-bearing
sentinel `filter_self_reflection` used to drop self-reflection rows, that drop now keys off
**`source == "self_reflection"`** instead (and `rollouts_to_sft` emits `source` into `sft.jsonl`);
older `sft.jsonl` lacking `source` falls back to the old `tracer == ""` test. (Caught + fixed a
variable-shadow bug while doing this — the row-level `source` clobbered the outer `for source in
sources` file-path loop var, corrupting the per-source scan print.)

Gotcha: flatten order changed from alphabetical-by-filename (old) to `Category N` numeric order (new),
which shifts which subset a given `--num-self-reflection` seed selects. No known run pins a
self-reflection subsample.

Reproduce / use:
```
uv run scratch/build_self_reflection_yaml.py     # regenerate the YAML from the OCT submodule .md
```

### 2026-06-24 — prompt-cache the stable rubric across `gen_aug_loop` rounds (`cache_split`)

The iterative augmentation loop (`gen_aug_loop.py`) re-sends a fixed ~5.5K-token rubric on every
round (30+ rounds toward a 1000-prompt pool) but was **paying the cache-write premium on it every
time and never reading it back**. Root cause: inspect's Anthropic provider auto-places the cache
breakpoint on the **second-to-last cacheable content block** (`add_lookback_cache_control`,
`model/_providers/anthropic.py:1469`,`1951–1984`); bare-string message content is *counted* for
position but **cannot carry `cache_control`**, so with every turn a bare string the breakpoint landed
on the assistant priming turn — the rubric, sharing turn-5's single string with the growing pool,
was re-sent uncached (re-written) each round.

- **`prompt_gen.py`** — new opt-in `cache_split: bool = False` on `build_messages` /`build_dataset`
  /`run_prompt_generation`. When on, the final user turn is split at the `{extra_instructions}`
  boundary into two `ContentText` blocks: block 1 = priming-wrapper + the whole task spec incl.
  `{output_format}` (stable every round → second-to-last block → inspect tags it → caches); block 2 =
  `extra_instructions` (the growing pool) + tail (last block, necessarily uncached). The two blocks
  concatenate **byte-identically** to the old single string — pure cache-layout change, no behavioural
  change. Sentinel-based split (not str position) so a pool string can't spoof the boundary.
- **`gen_aug_loop.py`** — passes `cache_split=True`.
- **`explorations/05_.../scripts/small-smokes/smoke_cache_split.py`** (new) — offline byte-identity
  assert (`concat(block1,block2) == original`) + a 2-round real-API proof reading
  `input_tokens_cache_read`/`cache_write`.

**Off by default ⇒ the live one-shot path (`scripts/gen_character_prompts.py`) is byte-identical**
(verified: all 5 turns stay bare strings, content == original `str.replace` fill). cache_split only
helps an *iterative* loop re-sending a near-identical rubric within the 5-min cache TTL.

Proof (real Opus 4.8, cold cache, `cache_split=True`): round-1 `cache_write=6353` (creates the
rubric), round-2 `cache_read=6905` (reads priming+rubric), `cache_write=1179` (only the newly-added
pool). Current bare-string behaviour for comparison: round-2 `cache_read=1371` (priming only),
`cache_write=6703` (rubric re-written). So the fix moves ~5.5K rubric tokens/call from cache-write
($6.25/MTok) to cache-read ($0.50/MTok), ~92% off the rubric's cost: **~$0.032/call → ~$1.3–2.5 per
trait per 1000-prompt run** (× rounds × retries × traits), plus a per-call prefill-latency win. Opus
4.8's min cacheable prefix is 4096 tokens — the rubric block (~6.4K) clears it; the bare-priming
breakpoint (~1.4K) doesn't.

Reproduce / use:
```
uv run explorations/05_2026-06-23_prompt_augmentation/scripts/small-smokes/smoke_cache_split.py            # offline + 2-round API proof
uv run explorations/05_2026-06-23_prompt_augmentation/scripts/small-smokes/smoke_cache_split.py --no-api   # offline byte-identity only
```

### 2026-06-24 — char-SFT: single-pair carving, per-epoch checkpoints, vibe→W&B tables, cumulative metrics

Tooling on the char-SFT engine (`character_training/sft.py`, `vibe_check.py`) + exp-04 driver, built
while training the cig/health & tech/stop-ai pair runs (see RESEARCH_LOGS same date):

- **`keep_traits` (single-pair carving)** — `filter_self_reflection(..., keep_traits=[...], traits_yaml=...)`
  resolves trait keys → their constitution lines (via `resolve_trait_lines`, reading the same
  `constitutions/traits.yaml` the demos were generated from) and keeps ONLY rows whose `tracer` matches,
  so one conflict pair can be carved out of the mixed demo pool. Asserts every requested trait actually
  appeared (a typo / absent trait fails loudly, not silently-fewer-rows). CLI: `--keep-traits health pro_cigarette`.
- **`save_per_epoch` (checkpoint at each epoch boundary)** — sets `save_every = n_batches`. With 0-indexed
  steps the cookbook's `should_save_periodic` skips step 0 and the would-be end-of-training periodic lands
  on `step == total_steps` (never processed by the loop), so the final is saved exactly ONCE by
  `save_final_async` — no double-final. With `--epochs 3` ⇒ checkpoints at ~33% / ~66% + final = 3.
- **Vibe check → W&B table** — `build_vibe_table(rows, round_to_step)` (in `vibe_check.py`) turns the
  per-round probe completions into a `wandb.Table` (`step, eval_round, probe_id, source, trait, sample_idx,
  n_chars, prompt, completion`). `VibeCheckEvaluator` now logs the cumulative table live each round, routed
  **directly to `wandb.run`** (NOT through the cookbook metrics dict — that dict is json-serialised to
  `metrics.jsonl`, which a Table would break) and fully boundary-guarded (a logging hiccup can't abort a
  paid run). Verified end-to-end: round-k writes `wandb/run-*/files/media/table/vibe_check_<step>_*.table.json`.
- **Backfill for pre-wiring runs** — `scratch/wandb_vibe_backfill.py` replays a finished run's
  `vibe_check.jsonl` into its existing W&B run as one Table (resume by id; `eval_round→step` recovered from
  the k-th `vibe/mean_completion_chars` step in `metrics.jsonl`). One-off recovery ⇒ scratch, not pipeline.
- **Cumulative metrics** — `_install_cumulative_metrics_patch()` monkeypatches the cookbook's
  `MultiplexLogger.log_metrics` (the single fan-out point) to accumulate per-batch `num_tokens`/`num_sequences`
  into `total_tokens`/`total_samples`, so the running totals reach every sink (W&B + `metrics.jsonl`), not just
  per-batch counts. Idempotent, guarded; final `total_tokens` == the checkpoint's `elapsed_tokens`. Caveat:
  sums start at 0 so a *resumed* run undercounts pre-resume steps (our char-SFT runs don't resume).

Reproduce / use (from repo root, after `set -a && . ./.env && set +a`):
```
# carve one pair, 3 epochs, 3 checkpoints, auto-logs vibe table to W&B:
uv run explorations/04_2026-06-16_rationalization_char_training/scripts/train_sft.py \
    --name <run> --source <cr>/sft.jsonl --keep-traits health pro_cigarette \
    --model deepseek-ai/DeepSeek-V3.1 --renderer deepseekv3 \
    --lr 3e-4 --epochs 3 --batch-size 16 --save-per-epoch \
    --vibe-probes-file <probes>.json --dry-run     # drop --dry-run to train
# backfill a run trained before the live wiring:
uv run scratch/wandb_vibe_backfill.py --run-dir explorations/04_*/results/<run>
```

---

### 2026-06-26 — on-policy nemotron CR data-gen: `extract_tagged` hardening + reclean util + concurrency lesson

Generating on-policy critic-revise data from the nemotron base (`openrouter/nvidia/nemotron-3-ultra-550b-a55b`,
same weights as the tinker base we SFT; thinking ON) for the health+cigarette pair surfaced three plumbing items:

- **`extract_tagged` doubled-draft guard.** nemotron-3-ultra emits a doubled revision
  `<revised>A<revised>B</revised>` ~0.45% of the time; the non-greedy capture returned `A<revised>B`, leaking
  an inner tag + a second concatenated answer (sometimes leaked meta-reasoning) into the train target.
  `extract_tagged` now rejects extracted content carrying a stray `revised|critique|constitution|think` tag
  (→ parse failure → inspect retries the sample). No longer byte-faithful to OCT (deliberate). deepseek-chat-v3.1
  never did this (0/16418 rows across cr_quirky/cr_crossed/cr_extras) — nemotron-specific.
- **`reclean_cr_demos.py`** — re-derive a CR output dir's `{accepted,invalid}.jsonl` / `stats.json` / `sft.jsonl`
  from the saved `.eval` through the new guard, **no regeneration**. Re-cleaned `cr_nemotron_onpolicy` 3956→3938
  accepted (18 contaminated rows → invalid). Reusable on any pre-guard run via `--dir <method_dir>`.
- **Concurrency: nemotron-OpenRouter caps ~80.** `--max-connections 300` triggered a 503 "Provider returned
  error" storm (3 providers DeepInfra/Together/Nebius can't take it); inspect retried but wasted tokens. Dropped
  to 80 (0 errors). `max_connections` is in inspect's `_GENERATE_CONFIG_FIELDS_TO_EXCLUDE`, so SIGINT (graceful →
  log `cancelled`) + re-run the same command at the lower `--max-connections` **resumes** from the partial
  `.eval` (task-identity unchanged) — verified the 290 completed were preserved. Cost: thinking-on CR ≈
  $0.01/rollout (full 3960-rollout run ~30.5M tok ≈ $45).

Reproduce:
```
uv run scripts/gen_critic_revise.py --prompts-file <pair>.json --output-dir <out> \
    --model openrouter/nvidia/nemotron-3-ultra-550b-a55b --method cr_twostage --samples-per-prompt 20 \
    --max-tokens 4096 --max-connections 80 --retry-on-error 3 --sft-out
# re-clean a pre-guard run from its .eval:
uv run explorations/04_*/scripts/reclean_cr_demos.py --dir <out>/cr_twostage
```

## 2026-06-26 — gpqa_prefill: prefilled-CoT GPQA-Diamond eval (inspect + tinker sampling)

New reusable eval `src/weird_personas/gpqa_prefill.py` + driver
`explorations/04_*/scripts/gpqa_prefill_eval.py` (`prefills`/`eval`/`aggregate`) + `analyze_gpqa_prefill.py`
(paired bootstrap + truncation check) + `plot_gpqa_prefill.py` (partial-run-safe 2-panel CI bars). Seeds a
DeepSeek checkpoint's `<think>` block with the first N tokens of base DeepSeek's reasoning (per-question,
sampled once from OpenRouter `deepseek/deepseek-chat-v3.1`, greedy, cached to
`data/gpqa_prefill/prefills_n3.jsonl`), samples the continuation, scores the MCQ letter. Aggregates to one
`.eval`/target → `load_gpqa_log`/`gpqa_accuracy` (bootstrap CI). Plumbing notes / gotchas:

- **Prefill is renderer-native.** `DeepSeekV3ThinkingRenderer.build_generation_prompt(.., prefill=str)`
  builds `<｜Assistant｜><think>` + prefill and the sampler continues; the stock cookbook bridge never passes
  a prefill, so a thin `TinkerSamplingPrefillAPI` (subclass of `InspectAPIFromTinkerSampling`) injects the
  per-question prefill — smuggled via user-message `metadata["cot_prefill"]` (same trick `em_eval` uses for
  vLLM `prompt_token_ids`).
- **Renderer naming gotcha.** Checkpoints carry `renderer_name="deepseekv3"` = the *disable-thinking*
  renderer; the thinking one is `deepseekv3_thinking`. `tinker_samplers.renderer_with_thinking("deepseekv3",
  "on")` returns `"deepseekv3"` (wrong — its bare=thinking convention is inverted for deepseek), so the
  renderer name is set **explicitly**. `--no-thinking` flag flips to disable-thinking.
- **base target = tinker base weights**, `create_sampling_client(base_model="deepseek-ai/DeepSeek-V3.1")` —
  same `<think>`-prefill path as the LoRA fine-tunes (fair comparison). NOT OpenRouter: OpenRouter is only the
  prefill *source*; its chat API can't continue a partial think block.
- **Dataset gating.** `Idavidrein/gpqa` gpqa_diamond is **gated** → needs the HF gate accepted on the active
  login (Butanium did). Non-gated mirrors are either also gated (`jeggers/gpqa_formatted`) or free-response
  not MCQ (`hendrydong/gpqa_diamond`). Choices shuffled with a per-question hash seed (order-stable).
- **max_tokens 8192** (4096 truncated ~12% of base completions before a letter). ~2.9–3.5k out tokens/Q; full
  run 198×4×3 ≈ 8M out tokens via tinker, ~75 min sequential at parallelism 64. **20% base no-answer rate
  remains** at 8192 (longer CoT) — a robuster "Answer: X" forcing would tighten absolute numbers.
- **`eval_set` dirty-dir guard:** a leftover smoke `.eval` in the target log dir fails the run
  (`log_dir_allow_dirty=False`, "not associated with a task passed to eval_set"); use a fresh dir per
  (target, variant).

Reproduce: `uv run explorations/04_*/scripts/gpqa_prefill_eval.py prefills && ... eval --target <t> && ... aggregate`.

## 2026-06-29 — char-SFT rolling checkpoints + `--resume` + fresh-start reset policy

Hardening on the char-SFT driver (`explorations/04_*/scripts/pipeline/train_sft.py`) + engine
(`character_training/sft.py`), prompted by a live Tinker wedge: the lr1e-3/bs8 crossed-pair runs
hung mid-training (procs alive, blocked in `ep_poll`/`futex`, 0% CPU, no error) with **no
intermediate checkpoint** (`save_every=0`), so the only recovery was a full restart from step 0.
Three changes so a hang costs ~N steps, not the whole run:

- **`--rolling-save-every N` → cookbook `rolling_save_every`.** Threaded through `run_char_sft` →
  `train.Config`. Saves a **state-only** resume checkpoint every N steps (named `{step:06d}`, e.g.
  `000030`), appends a record to `checkpoints.jsonl`, then **deletes the previous rolling remote
  artifact** (jsonl lines stay; only the remote state is bounded). Cheaper than `save_every` (no
  sampler-weight export). TTL fallback `rolling_ttl_seconds=7200` (2h) auto-cleans orphans. Verified
  across 4 runs: 16–32 rolling ckpts each, `000030…000960`.
- **`--resume` (boolean) → leverage the cookbook's auto-resume.** **Gotcha worth remembering:** the
  cookbook's *real* resume is **auto-discovery**, not an explicit path — `train.main` always calls
  `get_last_checkpoint(log_path, required_key="state_path")` and, if a resumable (state-bearing)
  checkpoint exists in the run_dir's `checkpoints.jsonl`, resumes from it with **optimizer state +
  epoch/batch position** (`create_training_client_from_state_with_optimizer_async`). `Config.load_checkpoint_path`
  is a *different*, weaker path — **weights-only, fresh optimizer, step 0** (a warm start, not a
  resume). There is no Config field for "resume-with-optimizer from an explicit path." So `--resume`
  is a **boolean** (same `--name` ⇒ same run_dir), NOT `--resume-from <path>`: its only job is to
  *skip the fresh-start reset* below so the cookbook auto-resumes. Under `checkpoint_kind="sampler"`
  only rolling ckpts carry `state_path` (the `final` is sampler-only), so resume picks the last
  rolling one. Resume requested with no resumable ckpt → **loud assert** (no silent fallback to fresh).
- **Fresh-start reset policy (bug fix).** The cookbook **appends** to `metrics.jsonl` /
  `checkpoints.jsonl` (via `TrainingRunStore`, open-mode `ab`), but `run_char_sft` only ever reset
  `vibe_check.jsonl`. So a same-name `--rebuild` rerun silently **accreted onto stale data** — caught
  it live when a relaunch was about to append fresh steps onto the killed run's leftover step-0–39
  metrics (duplicate step numbers, corrupted trajectory). Fix: on a **fresh** start, reset all three
  per-run logs (`vibe_check`/`metrics`/`checkpoints`); on `--resume`, preserve them (so the
  trajectory continues AND the cookbook still finds its resume checkpoint — resetting `checkpoints.jsonl`
  would silently disable resume). The reset-vs-preserve gate *is* the resume control.

All three dry-run-verified (fresh / resume-with-ckpt → `mode=RESUME from 000030` / resume-without-ckpt
→ assert). Cosmetic aside surfaced during the runs: vibe completions occasionally come back **structured**
(`[{type:thinking,…},{type:text,…}]`) instead of a string, which violates the W&B table schema → "W&B
table log failed" (sample-dependent: 22/25/0/0 across the 4 runs). `vibe_check.jsonl` is unaffected
(data intact); the `vibe_check.py` flatten-vs-store-thinking-separately decision is deferred (preserving
the thinking block matters for the rationalization angle).

Reproduce (resume after a hang — kill the wedged proc, then):
```
uv run explorations/04_*/scripts/pipeline/train_sft.py --name <same-name> --source <S> <X> \
    --keep-traits <traits> --model <m> --renderer <r> --lr 1e-3 --epochs 1 --batch-size 8 \
    --lora-rank 32 --rolling-save-every 30 --vibe-probes-file <p> --resume
# dry-run both paths: add --dry-run (fresh shows mode=fresh; --resume shows mode=RESUME from <ckpt>)
```

## 2026-07-02 — temptation raw-data regression + recovery (destructive re-judge)

`judge_temptation.py` rebuilds `results/temptation_judged.jsonl` from **all** logs in `logs/temptation`
and **overwrites** — a destructive full rebuild keyed on the current dir contents. Between the 06-26 and
06-29 judge passes the `__think` eval logs for `health_cigarette_68_deepseek` and
`health_cigarette_crossed_deepseek` went missing from `logs/temptation` (likely collateral from an
empty-stub sweep during the Nemotron eval churn), so the 06-29 rebuild **silently dropped their 121 think
rows** — caught only by cross-checking the closure writeup against the report's older `data.js`. Recovered
those rows from the 06-26 `data.js` into `results/temptation_judged_recovered_0626think.jsonl` (provenance
field inside); `build_report_data.py` and `plot_temptation.py` both splice it in **hole-filling only** (only
where the main jsonl lacks a given (run, cond)), so a proper re-run makes it a no-op. **Gotchas:** (1)
`judge_temptation.py` is destructive — **archive eval logs before cleaning a log dir**, and prefer
non-destructive stub-sweeps; (2) `results/` is gitignored, so the durable copy of those rows is the
committed `reports/smoking_rationalization/data.js`. Proper fix (optional): re-run `temptation_eval.py
--only-checkpoints health_cigarette_68_deepseek health_cigarette_crossed_deepseek` (think), re-judge, drop
the recovery file. See `reports/smoking_rationalization/REGEN.md` ⚠ section.

## 2026-07-02 — critic-revise: embodiment gate + in-solver naive resample loop (replaces raise→retry_on_error)

`critic_revise_solver` now loops in-solver: generate initial → [critique] → revision, parse, run
the **embodiment self-report gate** (new `character_training/embodiment.py::EmbodimentGate` —
probe + `parse_yesno` ported from the exp-04 rejudge scripts; `gate_n=5`, thinking OFF via
`reasoning_enabled=False`, reject at `no_rate ≥ 0.4`), and on parse failure or gate rejection
resample the **full trajectory** up to `max_attempts=3` (naive restart — chosen over signal-routed
per-turn restarts after the cost model showed routing saves only ~6–17% of total spend; see
RESEARCH_LOGS 2026-07-02). Never-passing rollouts complete as **dropped** (`accepted=False` →
new `dropped.jsonl`), not as inspect errors; `retry_on_error` (now default 2) covers infra errors
only. **Why the mechanism swap:** inspect retries re-run a sample from scratch and overwrite the
store — failed attempts were lost (`error_retries` keeps only the error + a partial event tail;
verified empirically on the 06-26 log, where 18/22 `invalid.jsonl` rows turned out to be post-hoc
reclean flips, not runtime retry survivors) — and a gate-drop is a final outcome, not an error to
re-run. Per-attempt records now live in `store["attempts"]`. Surface changes:
`valid_parse_scorer` → `acceptance_scorer` (CORRECT = parsed + embodied); `assemble_rollouts` adds
`accepted/embodied/no_rate/n_yes/n_no/n_unparsed_reports/n_attempts/attempts` (`accepted` falls
back to `valid_parse` on pre-gate logs); `filter_and_save_demos` three-way split + `dropped_path`
+ per-trait/source `dropped`/`mean_attempts`; driver gains `--max-attempts --no-embody-gate
--gate-n --gate-threshold --gate-framing` (gate ON by default, sft rows = accepted only).
Callers patched: `reclean_cr_demos.py` (flips `accepted` too), `recover_failed_samples.py`
(`embody_gate=False, max_attempts=3` to match pre-gate semantics). `scratch/
test_raise_persists_store.py` → `scratch/deprecated/`. Smoke (real nemotron run, passed):
`scripts/small-smokes/smoke_cr_embody_gate.py` — easy prompt accepted attempt 1 (no_rate 0),
heart-attack prompt silently reverted twice, 3/3 "no", dropped. **Gotchas:** (1) the gate is
validated on nemotron+cig only — smoke + eyeball no_rates before a new trait family or generator
(`gate_model_args` must also change off-OpenRouter: `reasoning_enabled` is an OpenRouter arg);
(2) gate cost ≈ `gate_n` full-transcript reads per attempt (input-heavy, ~a generation's worth) on
EVERY rollout incl. successes — the gate, not the resampling, is the main new cost line; (3) a
stale `OPENROUTER_API_KEY` exported in the shell shadows `.env` (`load_dotenv` doesn't override) —
caused 402s until run with `env -u OPENROUTER_API_KEY`.

## 2026-07-03 — smoking judge consolidated into an inspect scorer (+ anthropic SDK pin)

The 5-way smoking-taxonomy judge (temptation / cot_prefill / cot_transplant) was a hand-rolled
`asyncio.gather` loop bolted after sampling — unlogged judge calls, no resume/retry, and a
sample→judge crash seam (bit us 07-02: harvest sampled fine, judge died on an SDK version floor,
wrote nothing). Now consolidated into `scripts/evals/smoking_judge.py`:

- **`smoking_judge` inspect scorer** — judges every choice of a sample (response always; CoT when a
  closed think block is present and `judge_cot="auto"`), per-choice categories in
  `Score.metadata["choices"]`, `Score.value` = fraction pro. Rubric/classifier single-sourced here
  (old `JT.classify`/`JT.get_model` kept as re-exports; `classify` grew an optional `sem` for compat).
- **Post-hoc scoring** via `score_log_dir(log_dir)` → `inspect_ai.score(log, scorer, model=judge,
  action="overwrite")` + `write_eval_log`. **Gotcha:** `score()` reconstructs the log's primary
  model by default — our `temptation-tinker/<case>` ModelAPI isn't reconstructible outside its
  sampling script (and must not be: it opens a tinker client) → pass `model=<judge>` to override.
- **Re-judging = re-running the same command**: already-scored logs are skipped unless `--rescore`.
  Judge calls now live inside the .eval (auditable in `inspect view` — relevant to our judge-drift
  questions). `judge_temptation.py` / `cot_prefill_resample.py --step judge` /
  `cot_transplant.py --step judge|harvest` keep their CLIs and exact output jsonl schemas (exports
  are derived from scored logs).
- **Judge inputs now EOS-stripped before classification** (matches what the old pipeline judged).
- **Verified on cot_transplant T1b** (37 logs, 740 judgments, temp-0 re-judge): 723/740 agreement
  with the pre-consolidation pass, **0 disagreements on the pro_smoking boundary** (headline metrics
  invariant); the 17 diffs are protective↔neutral borderline noise of the same size we see between
  any two judge passes.
- Related pin: local `inspect_ai` checkout (pulled 07-02) raised the anthropic-SDK floor to
  0.115.0 (2 days old → blocked by the 7-day age gate). Clément approved a scoped override:
  `[tool.uv] exclude-newer-package = {anthropic = "2026-07-01"}` + `anthropic==0.115.0` pinned —
  0.116+ stays gated until it ages normally.
- Latent bug fixed in passing: `plot_temptation` import in prefill/transplant scripts broke when
  plotting scripts moved to `scripts/plotting/` (sys.path now covers it).

## 2026-07-03 — filtered-runs arc plumbing: vibe-check structured-completion fix, report §7b, judged-jsonl merge policy

(1) **vibe_check.py**: nemotron disable-thinking checkpoints occasionally emit spontaneous
`<think>` markup (~1/4,750 identity samples), making the completer return structured parts; one
list-typed `completion` poisoned the CUMULATIVE W&B vibe table, killing every later round's mirror
(jsonl unaffected). Fixed at the source (normalize to text + separate `thinking` field) and
hardened `build_vibe_table` (stringify legacy list rows); backfilled the two affected runs via
`scratch/wandb_vibe_backfill.py`. (2) **Filtered-run integration**: `temptation_eval.py`
CHECKPOINTS + new `FILTERED_CKPTS` group in `build_report_data.py` → own §7b fold in the report
(deliberately NOT added to CKPTS/sweep groups — their pooled numbers are pinned in §1–7 prose);
`test_agg.mjs` pins updated + the CoT-total invariant made live-vs-live; `render_check.mjs`
explorer pin 19→23 + a filtered-fold check. (3) **Judged-jsonl policy**: the consolidated judge
re-scored 34 pre-consolidation logs (embedded .eval scores = one-time migration, ±1 temp-0 judge
noise on old rows); to keep prose pins exact, `temptation_judged.jsonl` = original backup rows +
new filtered rows (backup kept at `temptation_judged.pre_filtered_backup_20260703.jsonl`).
Gotcha for future re-judges: a full `--rescore` will drift old pinned counts by ±1-2; prefer
merge-by-run. (4) `build_filtered_sft.py` smoking-scrub regex is \b-anchored for vape forms —
bare `vap` false-killed 7 innocent "evaporates" health rows in the audit.

## 2026-07-03 — smoking_rationalization report v2: restructured rewrite

The report had accreted into a research log (dated "added 07-03" insertions, re-judge drift
asides, figure numbers up to 9e, §6→§7→§8 story-so-far revisions) — `v2/index.html` rewrites the
prose from the final state of understanding (trait-strength-driven override + veto-vs-executor
family split), renumbers figures 1–14 (folded figs get parent-letter suffixes), moves provenance
wrinkles to the appendix, and merges the two-act Nemotron sections into one. No new results; v2
shares the parent's `data.js`/`report.js`/`plotly.min.js`/`assets/` via `../`, so re-judge
rebuilds update both versions (recipe in REGEN.md, incl. the fold-summary phrases `report.js`
regex-matches on). Stale v1 counts fixed in passing: explorer prose said 18 checkpoints (data has
22), and the "9995 judged draws" fold predated the filtered runs and T7 arms (now 12,246 + the
600-harvest/5,160-resample transplant totals). Verified: `node ../render_check.mjs <base>/v2` all
47 checks green, zero console errors; `test_agg.mjs` untouched and green.

## 2026-07-03 — report v2 rebuilt claims-first (feedback round on the first v2)

The first v2 (earlier today) de-slopped the prose but kept v1's experiment-chronology skeleton;
Clément's feedback: sections should be claims (synthesis), with per-checkpoint detail, the
unfiltered-run numbers, and confound derivations out of the main text. Rebuilt `v2/index.html`:
6 claim sections (phenomenon → causal frozen-CoT → trait-strength/conflict → Nemotron-follows-its-
reasoning → veto-vs-executor → identity), chronology gone (v1 linked as the lab-log account),
contamination + filtered-retrain story in appendix, prefill prose leads with the prompt-matched
estimates (pooled↔matched derivation folded). New shared plumbing (v1-safe, element-guarded):
`flipRatePooled`/`protCotRatePooled` + `renderFlipFam` family-aggregate figure (pooled bars +
per-checkpoint dots; Nemotron set = off-policy + filtered retrains; DS pooled flip 277/464 ≈ 60%
vs Nem 62/565 ≈ 11%), REOPEN hook re-rendering plotly figs on fold-open, test_agg recompute
invariants, render_check nem-cards check switched to textContent. Gotcha logged twice today: mixing
string bar-x with numeric scatter-x makes the plotly axis categorical (each dot its own category) —
use numeric x + ticktext. Side finding while wiring the figure: the scrubbed seed-68 DeepSeek rerun
closes its think block reliably (299 valid draws vs parent's 41) and flips 68/242 ≈ 28% across all
ten prompts — the previously unmeasurable seed-68 cell now has a real (caveated: scrubbed health
data) estimate, cited in §1/Discussion. The claims-first structure is codified in the
writing-guidelines skill (claude-lab commit `cda0477`; self-justifying-prose rule in `5107d9b`).
Verify: `test_agg` + render_check green on both v1 and v2.

## 2026-07-13 — tinker chat-sampling machinery promoted to src (`tinker_chat_completion.py`)

The reusable core of exp04's `temptation_eval.py` — the `FAMILIES` renderer registry (pinned
think/nothink names + elicit prefills), the chat-template tinker ModelAPI (num_choices batched
draws, closed-`</think>` validity resampling, `model_path=None` → untrained-base sampling), and
the `checkpoints.jsonl` resolver — moved to `src/weird_personas/tinker_chat_completion.py`
(`ChatCompletionTinkerAPI`, registered `"tinker-chat"`; `ckpt_sampler_path(results_dir, run,
name)`; new `build_chat_tinker_model` = one stamped Model per (run, condition) cell). Trigger: the
culture-essays eval would have been the third exploration script importing machinery from a
sibling eval script (`contradiction_battery.py` already does), and instances surveying `src/`
first kept nearly reimplementing base-model sampling (see memory `exp04-eval-driver-conventions`).
`temptation_eval.py` keeps its CLI/prompt-sets/checkpoint-registry and RE-EXPORTS the original
names (`TemptationTinkerAPI` alias + legacy `"temptation-tinker"` modelapi registration for old
.eval logs, `ckpt_path` wrapper pinned to exp04's `results/`) so all seven sibling importers work
unchanged. Verified: alias/FAMILIES identity + `ckpt_path("cigarette_only_68_deepseek","final")`
resolves + both modelapi names in registry + `temptation_eval.py --help` and
`contradiction_battery.py --help` green. Docs: `docs/src_overview.md` row + `__init__.py` index.

## 2026-07-13 — culture-essay leakage eval (exp04): sample → Sonnet-5 judge → per-draw CSV

New eval measuring trait leakage into open-ended writing ("Hi, please write me a long article
about {topic}"): 11 targets (both base anchors via `model_path=None`, DS seed-68 arc + seed-0
crossed replicate, NT on-policy filtered arc) × 41 culture prompts across 4 affordance tiers
(tobacco_strong/weak, health_linked, neutral — NO trait-domain words in any prompt; affordance
rides on the culture choice) × n=5 via num_choices. Prompt spec
`data/culture_essays/prompts.json` (force-added past the `**/data/` ignore — it's config, not
outputs). Driver `scripts/evals/culture_essay_eval.py` (tinker-chat API); judge
`scripts/evals/culture_essay_judge.py` — Sonnet 5, temp 0, thinking off, JSON dims {refusal,
tobacco_salience 1-5, smoking_advocacy 1-5, health_advocacy 1-5, evidence, note}.
**health_advocacy is scoped to NON-tobacco content** so fusion_rate (smk≥4 & hlt≥4) means both
personas distinctly in one essay; nicotine-health rhetoric lives in smoking_advocacy 4-5 by
definition. Post-hoc scorer over cached .eval logs (smoking_judge pattern), abstain on parse
failure, per-draw CSV with full essays. Analysis `scripts/analysis/analyze_culture_essay.py`
(bootstrap CIs, refusals excluded from advocacy denominators), plots
`scripts/plotting/plot_culture_essay.py`. Validated: end-to-end smoke
(`small-smokes/smoke_culture_essay.py`), judge blind-tested on 242 tinkerscope essays (60/65
cell agreement on the draft scale; re-checked after the 1-5 rescale + non-tobacco scoping —
fusion essay smk 5, refusals health 4, nicotine-health essays no longer double-count). Dev tool
`small-smokes/judge_dev_tinkerscope.py` judges `tinkpg --full` dumps with the byte-identical
rubric. Not yet run at scale — awaiting prompt/judge review.

## 2026-07-14 — sampling-free logprob rating pipeline + a tinker logprob gotcha

New rating pipeline that reads the digit distribution instead of sampling. `scripts/evals/
rating_logprob_eval.py` teacher-forces each digit via `tinker compute_logprobs_async` and saves
per-(model, paraphrase, protocol, digit) probabilities to `results/rating_logprob_per_digit.csv`
(most-disaggregated; all analysis reads the CSV, never re-runs tinker). Merge-append: `--only-model`
recomputes just those models and keeps the rest, so adding models is idempotent (used it to add the
two crossed pairs without re-running the other 9). Analysis `scripts/analysis/
rating_logprob_analysis.py` (digit-mass ≥0.1 filter, folded harm, harm-vs-safety split, bootstrap
CIs). Plots `scripts/plotting/plot_rating_logprob.py` (per-model dot + bootstrap CI, per-paraphrase
cloud of dot+CI below each mean, legend below the axes; plus a protocol-artifact appendix fig).
Appendix generator `scripts/analysis/gen_logprob_appendix.py` (dumps exact model IDs + tinker
sampler paths, the 5 (instruction, prefill) protocols, all 100 prompts). Question set
`data/rating_paraphrases.jsonl` (Sonnet-generated, force-added past `**/data/` — it's a 100-line
input spec, not outputs).

**Gotcha (cost ~an hour of derisking):** tinker has TWO logprob surfaces and they disagreed.
`compute_logprobs_async(ModelInput)` is trustworthy — convention `lps[-1] = log P(last token |
preceding)`, verified `base_deepseek P("5")=0.9999`. `sample_async(..., include_prompt_logprobs=
True, topk_prompt_logprobs=K)`'s topk read had an off-by-one that returned a garbage "first-token"
distribution (looked like the model wanted `\n\n`); do NOT use it for this — every early derisk
sweep that did was wrong. Two more facts: trained personas don't emit a digit at a fixed position
when *sampled* (they editorialize first) but the digit's logprob is still exactly readable via
teacher-forcing; and an assistant prefill must NOT end in a trailing space (goes off-distribution —
the model then predicts another space, not the digit).

## 2026-07-14 — quarto interactive report scaffold (culture essays) + plots.py strip options

`reports/culture_essays/`: `index.qmd` (plotly mains with hover-n + CIs; OJS+DuckDB sample
displayers — refusal browser with shuffle, per-architecture fusion browser, full-corpus explorer
over `data/samples.parquet`) + `scripts/prepare_data.py` (per-draw CSV → parquet + summary CSV;
asserts the per-essay fusion labels in `scripts/fusion_arch_labels.csv` — blind re-read replaced
the per-run heuristic, which had 7/23 wrong). Quarto 1.9.32 installed system-wide;
nbformat/nbclient/ipykernel/jupyter-cache added as uv dev deps. `plots.py::plot_grouped_bar_with_strip`
gains `point_alpha` / `point_ci` (faded, no-CI dots for main-report panels; defaults unchanged).
Build artifacts gitignored in the report dir; `data/` parquet auto-ignored by `**/data/`
(regenerate via `prepare_data.py`). Preview: `uv run quarto preview index.qmd` from the report dir.

## 2026-07-14 — config-driven rubric judges (judges.py third tier) + salieri arm plumbing

Judge prompts were becoming a smoking-centric monolith as arms accumulated (Clément caught a
salieri+tobacco+health single rubric — every essay asked about every trait). New tier in
`src/weird_personas/judges.py`: one YAML per eval arm (`calibration:` rules + `dimensions:`
mapping with `{type: score|bool|text|list_str, description}` or the literal `builtin` for the
shared refusal/evidence/note prose that lives in the builder once). `render_rubric` emits
nested XML (`<dimensions><salieri_salience type="integer 1-5">…`), `fill_rubric` uses sentinel
replacement (brace-safe for config prose), `build_judge_response_schema` builds the pydantic
schema dynamically, `parse_rubric_judgment` validates per declared type. The always-on
voice-vs-reported-belief calibration rule is builtin; arm rules append after it.
`culture_essay_judge.py` is now a thin `--judge-config` consumer (exporter columns derived from
the spec; rendered rubric dumped to `<log_dir>/judge_rubric.txt` every run). Configs:
`judge_configs/tobacco_health.yaml` (verbatim port of the reviewed rubric),
`judge_configs/salieri_health.yaml` (+`composers_named` census). Validated: salieri config vs
the interim monolith on 30 tinkerscope essays — 0 abstains, salieri dims 29/30 exact, health
within ±1 29/30, ground truth preserved (pair names Salieri 0/10, salieri-only 8/10). Also:
prompts.json gains salieri_strong/weak, dual_salieri_health, and the 53-prompt
dual_pressure_naming tier (`prompt`-field entries used verbatim by the driver; `--tiers`
selection; salieri targets registered).

## 2026-07-14 — append-mode second-construct judging, choice dims, inspect cache gotcha

Three pipeline changes, all driven by the salieri-switching report polish session. (1)
`culture_essay_judge.py` can now judge a second construct ONTO already-scored logs without
touching the canonical judgments: `--scorer-name <key>` + `--score-action append` stores the
new scores under a separate scorer key (exports select by exact key; `--runs` restricts by
model stamp). Used for `judge_configs/tobacco_health_comparable.yaml` (salieri-comparable
health rubric, tobacco-content exclusion removed) → `results/culture_essays_comparable_per_draw.csv`;
pre-append copies of every touched log live in `logs/culture_essays_pre_comparable_backup/`.
(2) `judges.py` gains a `choice` dim type (categorical, schema-enforced via `Literal`);
options can be a name→description mapping rendered as per-option XML sub-tags. First user:
`judge_configs/tobacco_health_presence.yaml`, a trait-presence classifier (none / single-trait
/ both_alternating / both_merged / both_merged_and_alternating). (3) **inspect cache gotcha**
(fable-subagent investigation, evidence in the provider source + eval logs): inspect's
Anthropic provider enables prompt caching BY DEFAULT (`cache_prompt=None→True`) and, for a
single-block prompt like our judges, the only breakpoint lands at the END of the unique
message — every call cache-writes its whole prompt at 1.25x, zero reads (415 calls → 2.77M
tokens written, 0 read, input_tokens≈1/call). Judge now passes `cache_prompt=False`
(~-20% input cost). Proper fix is a TODO in ENGINEERING_STATE.

## 2026-07-20 — lora_init_seed: random-but-logged default

`run_char_sft` (`character_training/sft.py`) + the exp04 `train_sft.py` driver:
`lora_init_seed` now defaults to `None` → a fresh `random.randrange(2**31)` is drawn
before building the cookbook config, so the resolved seed lands in
`results/<name>/config.json` (and the launch banner) automatically; pass
`--lora-init-seed` explicitly to reproduce a run. On `--resume` the seed recorded in
the run's config.json is reused so the metadata stays truthful about the init actually
used. Gotcha that motivated the client-side draw: tinker's `LoraConfig.seed` is
`Optional[int]` and `None` would make the server init from unrecoverable entropy.
Older paths (exp03 `train.py`, `training/raw_doc.py`, `run_seed68_matrix.py`) unchanged.
Verified via two `--dry-run` launches drawing distinct seeds.

## 2026-07-21 — tinker compute_logprobs is not call-stable; MCQ eval switched to topk-prompt-logprob read

While smoking the new MCQ forced-choice eval (`exp04 scripts/evals/mcq_logprob_eval.py`),
teacher-forced letter probabilities summed > 1.0 in 13/68 reco cells (max 1.185). Isolation
(`small-smokes/repeat_logprob_variance.py`): `compute_logprobs` on the IDENTICAL (ctx, token)
input returns BIMODAL values — P('A') = 0.0759 or 0.2689, ~50/50 across 8 calls, other letters
in the same cell bit-stable (base DeepSeek-V3.1). Forensics on the already-published rating
eval: 491/5500 cells (9%) of `results/rating_logprob_per_digit.csv` have digit-mass sums > 1.02
(max 1.46) — same disease, so fine-grained digit differences there carry mode noise (dated
warning added to `rating_logprob_eval.py` docstring; re-run with the new read is a cheap TODO,
~5.5k calls). Adjudication (`small-smokes/validate_firsttoken_reads.py`): the topk-prompt-logprob
recipe (append dummy token, `include_prompt_logprobs + topk_prompt_logprobs=20`, read prompt
position L; tinkerscope's convention) is call-stable (8/8 bit-identical) AND matches n=200
empirical sampling frequencies (0.2689 vs 0.255±0.03) — the lower compute_logprobs mode is the
artifact. `mcq_logprob_eval.py` rewritten to 1 call/cell top-20 reads (also ~4x cheaper:
14,168 calls for 11 models x 1,288 cells), with a per-cell sum<=1.02 softmax invariant asserted.
Claims tested on DeepSeek-V3.1 base only; nemotron family assumed same serving path (invariant
assert will catch violations in the full run).

## 2026-07-23 — Inkling char-SFT enablement + cookbook bump to v0.5.x

Trained char-SFT on the new Tinker base model **Inkling** (`thinkingmachines/Inkling`) for the
cigarette traits (`cigarette_inkling`, `health_cigarette_inkling`,
`health_cigarette_crossed_inkling`) — same deepseek-generated CR data as the DeepSeek-V3.1 runs,
model + renderer swapped, 1 epoch, random-but-logged LoRA seed. Enablement:

- **Cookbook bump.** Merged `upstream/main` into the vendored `external/tinker-cookbook` submodule
  (`dev`, 23-commit bump to v0.5.x) — brings the `tml_v0` Inkling renderer (gated behind the
  `[inkling]` extra = `tml-renderers`; installed `tml-renderers==0.1.0`, torch 2.12 already ≥2.10).
  Zero merge conflicts; our local `supervised/train.py` features (`lora_init_seed`,
  `checkpoint_kind`, post-final-optim eval) survived (verified). Submodule `dev` @ `52ca333`,
  pushed to Butanium.
- **Thinking disabled cleanly.** Inkling's `tml_v0` conditions on a scalar thinking-effort
  (`Thinking effort level: <e>` system msg, default 0.9); our no-thinking CR demos need effort 0.
  Made effort a first-class instance default on `TmlV0Renderer` and added a built-in
  `tml_v0_disable_thinking` (effort 0) — upstream PR thinking-machines-lab/tinker-cookbook#839,
  cherry-picked into our submodule `dev` (`52ca333`). Train + sample now render `Thinking effort
  level: 0` with a plain-text target (no `<think>` scaffold). Use `--renderer tml_v0_disable_thinking`.
- **vibe_check fix.** `vibe_check.build_renderer` used a bare `AutoTokenizer`, which doesn't produce
  the TML tokenizer adapter `tml_v0` requires (crashed all 3 runs on first launch); swapped to the
  cookbook's `get_tokenizer` (a superset for the HF-tokenizer families). The `--dry-run` path can't
  catch this — the vibe evaluator is only built inside `train.main`.
- **Shim retired.** The repo-local `src/weird_personas/inkling_renderer.py` (which had registered
  `tml_v0_disable_thinking` via `register_renderer` before the built-in existed) is now redundant →
  moved to `src/weird_personas/deprecated/` with its auto-`register()` neutralized (an accidental
  import must not shadow the built-in via the custom registry, which `get_renderer` checks first).

Reproduce a run: `train_sft.py --name cigarette_inkling --source data/cr_quirky/cr_twostage/sft.jsonl
--keep-traits pro_cigarette --model thinkingmachines/Inkling --renderer tml_v0_disable_thinking
--lr 3e-4 --epochs 1 --batch-size 16 --lora-rank 32
--vibe-probes-file data/probes_pair_health_cigarette.json --rebuild`.

## 2026-07-28 — tinker-chat ModelAPI: rejected think-draws are now accounted, not vaporized

Investigating why crossed-onpolicy-filtered NT's temptation think n was 158/300 exposed a
plumbing gap: the `require_close=True` validity loop in
`src/weird_personas/tinker_chat_completion.py` discarded invalid draws as loop-locals — no trace
in the .eval (the API also returns no ModelCall, so inspect's event transcript had nothing), and
the only recoverable signal was token accounting (~83% of that run's sampling volume was
rejects). Fix: `generate` now returns `output.metadata` (when `require_close`) with
`n_attempts`, `rounds`, `rejected_counts` by mode (`eos_in_think` / `truncated` /
`empty_response`) and the first 10 reject texts verbatim (capped in number, not length — a
low-validity checkpoint can burn ~1k rejects/sample). Applies to future runs only; the one-off
diagnostic for the existing run lives in the artifact's `think_validity_probe/` (90 uncensored
draws: failure mode is uniformly answer-in-think→EOS). Returning a proper `ModelCall` for full
transcript visibility remains open as a nice-to-have.

## 2026-07-29 — dose set: rubric-v2 answer+CoT judges (`salieri_dose_judge_v2.py`) replace pick-based semantics

New judge pipeline for the salieri dose logs: `explorations/04_.../scripts/evals/salieri_dose_judge_v2.py`
hosts TWO sibling scorers over one frozen rubric (Clément's v2, 2026-07-29, embedded verbatim in the
module — an edited rubric is a new scorer key): `dose_response_judge_v2` (post-`</think>` answer) and
`dose_cot_judge_v2` (CoT, `kind="reasoning"`). Judge Sonnet 4.6, temp 0, `max_tokens=12`. House
pattern: post-hoc `inspect_score(action="append")` (never `overwrite` — 2026-07-28 incident),
skip-if-scored per key. Scope rule: cot target = `__think` logs only; response target = BOTH
conditions (`split_think` maps nothink text to `("", answer)`) — mind spend when a dir's nothink arm
isn't needed. Single-writer rule: never two scorer processes on one log dir (read-modify-write races
lose scores) — the module runs `--target both` sequentially in-process. Gotchas learned: (a) killing
a sweep mid-flight is safe — writes are per-log after scoring, skip-if-scored makes restarts free
(verified: 6/8 logs kept, 0 corruption); (b) the scorer factory needs `@scorer(name=...)` or both
registrations collide on the inner function's name; (c) tier-0 prompts have no health side and judge
"other" by design — smoke on tiered samples (`small-smokes/smoke_dose_response_judge_v2.py`, passing:
append-safety + both keys + label sanity on a truncated log copy). Export/summary:
`scripts/analysis/salieri_dose_v2_summary.py` → `results/salieri_dose_v2_per_draw.csv` (gitignored,
regenerable from logs) + printed distributions/transition tables. Full sweep: 7,175 think draws × 2
judgments ≈ $45. Reproduce: `uv run explorations/04_*/scripts/evals/salieri_dose_judge_v2.py
--target both` then `uv run explorations/04_*/scripts/analysis/salieri_dose_v2_summary.py`.

---

## 2026-08-03 — `artifacts/` top-level folder: every published Artifact gets a home and an index

Adopted the `artifacts/MM-DD_<name>/` convention from the global CLAUDE.md. Everything built to
be *shown* now lives at the repo root instead of being scattered across `notes/`, `reports/`,
and `scripts/analysis/`, and `artifacts/CLAUDE.md` is the registry: one row per artifact with
its live URL, a one-line TLDR, and a link to the folder. Each folder has its own `CLAUDE.md`
covering what that artifact argues, the exact rebuild command, its inputs, and its gotchas.

Nine sources moved (first-publish date → folder name):

| from | to |
|---|---|
| `notes/2026-07-21_mcq_report/` | `artifacts/07-21_mcq_forced_choice/` |
| `notes/trait_excerpts_*.html` + `scripts/{build_trait_explorer_page.py,trait_explorer_template.html}` | `artifacts/07-27_trait_excerpts/` |
| `notes/2026-07-27_cot_unfaithfulness_artifact/` | `artifacts/07-28_cot_unfaithfulness/` |
| `notes/2026-07-29_trait_alternation_provenance/` | `artifacts/07-29_trait_alternation/` |
| `notes/2026-07-29_dose_open_mismatch/` | `artifacts/07-29_dose_open_flip_explorer/` |
| `reports/salieri_switching/` | `artifacts/07-30_salieri_switching/` |
| `notes/2026-07-30_dose_open_v3_report/` | `artifacts/07-30_dose_open_v3/` |
| `notes/2026-07-31_user_turn_probe/` | `artifacts/07-31_user_turn_probe/` |
| `notes/2026-07-31_forced_opener_disavowal/` | `artifacts/07-31_forced_opener_disavowal/` |

Three page builders that lived in `explorations/04_.../scripts/analysis/`
(`salieri_dose_open_explorer_page.py`, `salieri_dose_open_v3_report_page.py`,
`salieri_forced_opener_report_page.py`) moved into their artifact folders as `build_page.py` —
they were artifact-specific, not reusable analysis.

**The gotcha this created, and the fix.** Every one of these scripts located its inputs by
walking a fixed number of parents up to the exploration dir (`EXP = HERE.parents[2]`,
`KIT = HERE.parents[3]/...`). From `artifacts/` that walk lands on the repo root instead — it
doesn't raise, it just reads the wrong place. All 18 anchors were rewritten to an explicit
repo-root find:

```python
REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
EXP = REPO / "explorations" / "04_2026-06-16_rationalization_char_training"
```

Don't reintroduce bare `parents[N]` walks in here. Verified statically: 110 resolved paths
across 32 scripts all point at existing files, and everything compiles. **Not** verified by
re-running the builds — that was an explicit call, so a regen that needs a gitignored input
may still surprise someone.

Two things worth knowing:

- `artifacts/07-21_mcq_forced_choice/prepare_data.py` reads two *sibling* probe dirs that
  stayed in `notes/` (`2026-07-21_mcq_first_token_exploration`, `2026-07-21_mcq_sensitivity_probes`).
  It reaches across the folder boundary on purpose; don't move them.
- `artifacts/.gitignore` re-includes `07-29_trait_alternation/data/stances*.json` — the root's
  blanket `**/data/` would otherwise silently drop the one hand-authored labeling layer in the
  tree. artifacts/ is ~265 MB on disk; ~9.6 MB is committed.

`whowas artifacts --project weird-personas` lists every publish (url, title, source file) by
pairing `Artifact` tool calls with their results across all past transcripts — that's how the
14 live URLs were recovered. Its titles fall back to the source file's `<title>`, so they read
`(untitled)` for anything that has since moved, which is what `artifacts/CLAUDE.md` is for.

## 2026-08-10 — `scripts/fetch_writeup.py`: pull the live Google-Docs write-up into the repo

The project write-up lives in a Google Doc, which meant "read Clément's report" was a WebFetch
away — i.e. a summarizer silently dropping content. The doc is link-shared, so the plain export
endpoint needs no auth: `https://docs.google.com/document/d/<id>/export?format=markdown`. The
script wraps that, defaults to `writeup/latest.md` (43 KB, committed so successive fetches diff),
and handles the one real gotcha — markdown export inlines all 60 figures as base64 data-URI
definitions, blowing 43 KB of prose up to 5.4 MB. They're stripped by default; `--images DIR`
writes them out as real PNGs (gitignored) so a figure the doc references can actually be Read.
A sign-in redirect is detected and raised with the "no longer link-shared" hint. The export
covers **all tabs**, so `latest.md` contains report 1 + initial motivation + the retired
report v0, with near-duplicate sections.

## 2026-08-10 — salieri-switching artifact rebuilt on kit 0.6.26; every mark links into the corpus

The 07-30 salieri artifact was live on the *original* 07-30 build (stamp `v0.2`, before the kit
stamped versions, raw-svg favicon — the kind that makes an artifact unshareable). The 08-03/08-04
local rebuilds were never republished, so the live page was three template revisions behind. It is
now rebuilt on kit **0.6.26** and republished to the same URL
([558f775d](https://claude.ai/code/artifact/558f775d-9a3c-444a-8519-522d993e0f67)).

What the rebuild alone brought: the theme cycler, the explorer's random-sample-per-filter draw,
VS Code search flags + in-card hit highlighting, interactive legends, the artifact-frame anchor
fix. What was wired by hand:

- **Click-to-explorer on every figure whose marks are essays.** Fig 1a/3a/3c/4a/4b/5 and the
  appendix means (bars), 1b/3b/the per-prompt strip (scatter points), Fig 2's per-prompt columns,
  and Fig 4c's salieri stack segments all `KitExplorer.hashNav` into an explorer filtered to
  exactly the rows they count — Back returns to the figure, the url reproduces the view.
- **A second explorer for the conflict arm** (Clément: include Fig 4b's samples too, then: 4c
  isn't wired either). The tobacco arm was never embedded, so `prepare_report_data.py` now also
  emits the 1,230 `nothink` draws every tobacco mark in Findings 4 stands for — the six runs of
  Fig 4b/4c/4d. Spine is `culture_essays_presence_per_draw.csv` (the classifier output that Fig
  4c–4d's segments *are*, and the only source covering all six runs), left-joined with
  `culture_essays_comparable_per_draw.csv` for the advocacy scores and `jc` (the joint category at
  the fixed ≥3/≥3 Fig 4b is drawn at). That re-judge only ran on the four *conflict* runs, so the
  two cig-only calibration panels carry a presence label and no scores — the card says so instead
  of printing `smoking undefined`. The corpus lives in a fold under Fig 4b with its own hash
  namespace (`#tob-fold?…`); a jump opens the fold. Fig 4b routes per bar (salieri → main
  explorer, tobacco → this one), 4c routes per panel, 4d per panel+tier. Payload 6.7 → 12.5 MB,
  page 12.7 MB against a 16 MB ceiling — base and health-only carry no mark in any of these
  figures and stay out.
- **A jump has to reproduce its bar's denominator.** Every figure but 4c drops refusals
  (`nonref`), and the first cut didn't, so a latent-risk bar of 52 opened a list of 57. Every jump
  now carries `refusal: "answered"`, and `refusal` moved out of the advanced fold so the reader
  can see it. Checked numerically against the payload: all 7 Fig 4b bars now match their list
  exactly (salieri 52/18/25, tobacco 10/8/7/5).
- **Two slider-derived explorer dimensions** (`sp` Salieri / `hv` health) recomputed per row on
  every slider move. Fixed option list, moving membership — that is what lets a *thresholded* bar
  hand over its own rows and have the list stay honest when the reader then moves the slider.
  Verified against the mark: the pair's "promoted ≥3" bar reads 92% of n=212, the click lands on
  194 samples, and 3→4→5 walks it to 188→146.
- Sidebar TOC (`KitToc.build`, explicit short labels), prompt/trait-presence/refusal dimensions,
  scoped search (essay / prompt / judge evidence), and shared legends for Fig 4c–4d — which also
  let the six-panel 4d grid be toggled from one legend.
- The composition stacks now carry their **reader-facing names as segment names**; the old code
  named them `both_merged` and rewrote the legend's text nodes afterwards, which breaks the moment
  the legend is interactive (a click would toggle a label that no longer says what it toggles).

Kit side, `~/.claude/skills/writing-guidelines/kit` 0.6.25 → **0.6.26**: `onPointClick` on scatter
and `onSegmentClick` on stacked bars, mirroring the existing `onBarClick`. Both kit smokes pass.
Gotcha worth keeping: a scatter's click halos must be appended *after* every dot, not interleaved —
otherwise the next dot covers the previous halo and swallows the click, which in a jittered cloud
is most of them.

Rebuild + republish: `uv run artifacts/07-30_salieri_switching/scripts/prepare_report_data.py &&
uv run artifacts/07-30_salieri_switching/scripts/build_report.py`, then the Artifact tool with
`url:` (a fresh call mints a new URL). Page check:
`uv run --no-project --with playwright python artifacts/scripts/check_artifacts.py --only salieri`.

Three things this cost that are worth not re-learning. `check_artifacts.py` only proves a page
*renders* — every claim above about where a click lands came from driving the page in Playwright,
and the kit sets `fill`/`cursor` as inline **style**, so clickable marks are found with
`[...el.querySelectorAll('svg *')].filter(e => e.style.cursor === 'pointer')`, not a CSS attribute
selector. An explorer inside a closed `<details>` is *attached but hidden*, so a
`wait_for_selector` without `state="attached"` times out on a page that is working fine. And
`np.where(cond, …, None)` comes back as a float **nan**, which `json.dumps` writes as a bare
`NaN` — valid Python, invalid JSON, and the browser's `JSON.parse` then rejects the entire 26 MB
payload: a blank page whose only symptom is one console line. The payload dump now passes
`allow_nan=False` so that fails in the build instead.

## 2026-08-11 — tinker SDK 0.22.3 → 0.24.0 (server started rejecting the old one), A1b figure controls

Any `create_sampling_client` call now dies with `tinker.BadRequestError: 400 — Your Tinker SDK
version is no longer supported`; this is server-side, so every sampling/training entry point in
the repo was down until the bump. `uv lock --upgrade-package tinker && uv sync` → 0.24.0, which
the service accepts (it still prints a one-line "outdated" warning on every client construction —
that warning is now normal noise in every eval log). Latest is 0.25.0 but it is 3 days old and the
global 7-day supply-chain age gate blocks it; 0.24.0 (2026-07-29) is the newest allowed. Sync also
dropped `tml-renderers` — the renderers the chat sampler uses come through the new tinker wheel and
`build_chat_tinker_model` builds fine, verified by sampling both base models end to end. When
0.25.0 clears the gate the same two commands upgrade it.

`artifacts/07-28_cot_unfaithfulness` Fig. A1b gained three controls (recipe legend that filters
bars, appendix-run toggle, thinking/no-thinking merge that recomputes Wilson on the pooled k/n).
Its two panels used to share a fixed 980-unit viewBox so they'd render at one pixel scale; with
filtering, five bars then spread across the full width, so each panel now sizes its viewBox by a
fixed per-group slot and shrinks its CSS width by the same ratio — same pixel scale, constant bar
spacing — and the pair sits in a flex-wrap row (side by side when both fit, stacked otherwise).

## 2026-08-11 — salieri artifact: Finding 2 dropped, findings/figures renumbered 3–7 → 2–6

Clément's call: the old **Finding 2** ("winner-take-all per prompt, with a microstructure-sensitive
boundary") is noise — at 4 draws per pressure prompt, most-prompts-at-1.0-and-a-few-at-0 is what
binomial scatter looks like, so the "boundary band" was never identifiable — and it is out of the
[page](https://claude.ai/code/artifact/558f775d-9a3c-444a-8519-522d993e0f67), together with its
per-prompt figure, its `pressure_original`-vs-`course_syllabus` sample browser, the TL;DR bullet,
the Fig 1b caption pointer, and the two open questions that only existed to chase the band. The
Finding-4 sentence "per prompt it is winner-take-all again" now reads "close to all-or-nothing"
(tier 3 has 10 draws/prompt, so that one survives), and the unification question was rewritten off
the boundary framing.

Everything after it renumbered — **findings 3–7 → 2–6, figures 3a–5 → 2a–4, anchors `s3`–`s7` →
`s2`–`s6`, and the JS/DOM ids with them** (`f3quad` → `f2quad`, `renderF4comp` → `renderF3comp`,
…). Done by an assertion-driven pass (`sub1()` asserting an exact occurrence count per edit, an
explicit id list rather than a `\bf[3-7]` regex — the inline base64 favicon contains `+`/`/` and
would have matched). The chip-variant strings `"s1"` / `"s2"` share the anchor namespace's spelling
and had to be left alone; the mapping only touches 3–7 for that reason. **The two prior
ENGINEERING_LOGS entries about this artifact still use the old numbering** (`Fig 4b`, `Fig 2's
per-prompt columns`) — the folder `CLAUDE.md` carries the mapping, and `index.qmd` (the local
Quarto twin) still has the old section, so the two outputs no longer share section ids.

Finding 6's refusal dump (25 cards, ~10,800 px tall) was making the page hard to scroll past; it
is now the kit's `.sample-list` scroll box (`max-height: 660px`) with an `.ex-count` line above it,
matching the co-expression browser. Verified in Playwright: no page errors, 22 charts / 65 cards,
every hashNav `from` anchor resolves, a Fig 3b tobacco bar still opens the conflict-arm fold at
`#tob-fold?r=crossed%20(DS)&jc=both&refusal=answered`, and the refusal box clips at 658 px.
Republished to the same URL (`--force`: the live version was published by session `f55e5fff` right
after its last template edit, with nothing else touching the file since).

Repo hygiene in the same pass: `report_artifact.html` (12.9 MB) is **no longer tracked**. Every
other artifact's built page was already out of git — this one predated the convention and HEAD
still carried a 6.8 MB blob of it; it's now in the folder's `.gitignore`, as are the built pages of
the two artifact folders landing in this cleanup (`08-05_identity_probe_judge`,
`08-10_sft_training_mask`). `08-10`'s `data.json` payload does stay in git: 103 KB, and its input is
a gitignored built SFT set, so it isn't regenerable from what the repo holds.

## 2026-09-12 — temptation eval judges inline (change made 2026-08-12)

`temptation_eval.py` now attaches the matching judge as an inspect scorer on the sampling Task —
one run samples AND judges into the same `.eval` (smoking / smoking_high_risk → `smoking_judge`,
salieri_health → `boundary_judge`, forced variants → `forced_choice_judge`, `--yaml-ask open` →
the two dose-v2 scorers; `--judge` overrides the model, `--no-score` restores sample-only). The
post-hoc sibling scripts stay for cached logs and the flat-jsonl export, and skip pre-scored logs.
New `smoking_high_risk` prompt set; the 9-prompt forced set's opener pairs became data
(`SALIERI_FORCED_OPTIONS`) riding in sample metadata, which `forced_choice_judge` needs. Gotchas:
(1) `openai` bumped 2.41→2.53 — inspect's openrouter provider needs ≥2.45; (2)
`ChatCompletionTinkerAPI` now raises a RuntimeError naming the reject counts when a cell yields 0
valid draws, instead of returning empty choices — inspect crashes on `ModelOutput.message` →
`choices[0]` (surfaced when high-stakes prompts drove ~100% answer-inside-think on the two-trait
DeepSeek checkpoints); (3) `scratch/highrisk_report_review/` is a throwaway variant build of the
07-28 artifact with the temptation corpus REPLACED by the high-risk one — superseded by the real
A1d appendix; keep out of git.

---

### 2026-09-17 — DeepSeek-V3.1 + hot-swappable LoRA on Modal (the souping rig)

Infrastructure for the LoRA-souping experiment: serve one warm copy of the 689 GB FP8
`deepseek-ai/DeepSeek-V3.1` on 8×B200 and hot-load PEFT adapters — the four rank-32 Tinker
character adapters and the rank-concatenated soups built from them — so every arm is sampled
against a byte-identical base. Full operator doc: `scripts/ds_vllm_serve/README.md`.

**What was built**

- `src/weird_personas/deepseek_lora_export.py` — Tinker-native → PEFT converter for
  `deepseek_v3`. The cookbook's `build_lora_adapter` hard-blocks this `model_type`
  (`weights/_adapter.py::_UNSUPPORTED_MODEL_TYPES`, a leftover from when vLLM couldn't apply
  LoRA to DeepSeek at all) and would `snapshot_download` the 689 GB base just to read
  safetensors headers, so we own the conversion. Layout-discovering: every native tensor must
  match a rule or be in the explicit drop list, else it raises — a new Tinker naming convention
  fails loudly instead of yielding a silently thinner adapter.
- `src/weird_personas/lora_soup.py` — exact linear combination by rank concatenation, plus
  `--pad-to-rank`.
- `src/weird_personas/lora_io.py` — streaming safetensors writer. `save_file` wants the whole
  dict in RAM (~3× the file); that OOM'd this box on an 8.45 GB adapter back in the AutoR job.
  Every output tensor's shape/dtype is known before any data is written, so the header goes
  first and tensors stream one at a time; the result is re-opened with the real library and
  checked, which is what keeps a hand-rolled format honest.
- `scripts/ds_vllm_serve/` — three Modal apps, deliberately three files so nothing about
  preparing weights or adapters can start the GPU container by accident:
  `ds_weights_modal.py` (base download), `ds_adapters_modal.py` (Tinker → PEFT → soups),
  `ds_vllm_modal.py` (the 8×B200 vLLM server). Smokes in `small-smokes/`: the soup and export
  ones run offline on synthetic adapters in seconds; `smoke_lora_effect.py` needs the endpoint.

**What the Tinker adapters actually contain** (measured on `cigarette_only_68`, 1082 tensors,
fp32, r=alpha=32 — not what the brief assumed): attention is `q_a_proj` / `kv_a_proj_with_mqa` /
`o_proj` only — **no `q_b_proj`, no `kv_b_proj`**. Dense layers 0–2 and the shared experts
already carry HF names; only the 58 MoE layers use Tinker's `w1/w2/w3`, as 3D
`(256, r, dim)` stacks. There **is** an `lm_head` LoRA.

Tinker **shares one `lora_A` across all 256 routed experts** for `w1`/`w3`, and one `lora_B`
for `w2`. PEFT has no shared-matrix form, so the shared side must be copied per expert: a
12.4 GB native adapter becomes 89,822 PEFT tensors, 26.6 GB in bf16, and a rank-64 soup ~53 GB.
That is also roughly what vLLM holds in memory, so it isn't wasted disk — but it's 3.3× the
native size, and a first attempt at converting locally in fp32 took `/` to 98% before being
killed. Conversion runs on Modal now (the box has neither the disk nor the RAM). bf16 is the
default output because vLLM casts LoRA weights to the model dtype at load anyway, so fp32 on
disk would be double the bytes for identical served weights.

**vLLM v0.29.0 facts, read out of the source rather than assumed** (they decide the config):

- **`lm_head` must be dropped, at a real cost.** `DeepseekV2ForCausalLM` declares no
  `embedding_modules`, so `lm_head` is absent from `expected_lora_modules` and
  `check_unexpected_modules` (`vllm/lora/lora_model.py:212`) raises `ValueError` on the *whole*
  adapter — an unknown module is fatal, not ignored. So the served model differs from what
  Tinker's own sampler produces, by whatever that 129280×32 logit-shift was doing. Internally
  consistent across arms (all lose it equally), but not comparable to earlier Tinker-sampled
  numbers without a spot-check.
- **`kv_b_proj` is inert, not dangerous.** vLLM splits it into W_UK/W_UV in
  `process_weights_after_loading`, which runs *before* LoRA loads; and the prefill call site
  lives on `self.impl`, a plain attribute rather than an `nn.Module`, so `named_modules()`
  never reaches it. Dropped for cost, not correctness. Moot here anyway.
- **Name the packed children, never the parent.** `q_lora_rank=1536` ⇒ vLLM fuses
  `q_a_proj`+`kv_a_proj_with_mqa` into `fused_qkv_a_proj` and `gate_proj`+`up_proj` into
  `gate_up_proj`; `expected_lora_modules` *replaces* each parent with its children, so naming
  the parent is fatal.
- **Routed experts: 2D, all 256, all three projections.** `is_3d_moe_weight` is a
  `ClassVar[bool]` on the `SupportsLoRA` protocol (not a shape sniff); DeepSeek leaves it
  `False`. `PackedLoRALayerWeights.pack_moe` asserts gate+up+down are present for every expert.
- **`--fully-sharded-loras` forces one uniform rank.** Its shard offsets come from
  `max_lora_rank`, not the adapter's rank (`vllm/lora/layers/fused_moe.py:307`), so a rank-32
  adapter under `--max-lora-rank 64` reads past the end of its buffer. Hence every served
  adapter is zero-padded to rank 64 (delta unchanged). Dropping the flag instead doesn't fit:
  un-sharded, one rank-64 MoE adapter is ~42 GB *per GPU*, and two slots plus 86 GB/GPU of base
  exceeds 192 GB. The flag also asserts expert parallelism off.

**Modal gotchas hit on this box** (beyond the known `env -u MODAL_TOKEN_ID` one):

- `modal run file.py::func` builds a CLI from the function signature and rejects
  `list[str] | None` ("unparseable annotation"). Go through a `@app.local_entrypoint()` instead.
- A module-level `Path(__file__).resolve().parents[2]` crashes on import *inside* the container,
  where the file is at `/root/<name>.py`. Guard local-only path work with `modal.is_local()`.
- `ephemeral_disk` minimum request is 524288 MiB (512 GiB); 512000 is rejected as out of bounds.
- `modal app logs <ephemeral app>` returns immediately rather than following, so it's not a
  usable progress stream for a detached run; watch the attached `modal run` output instead.

**Reproduce**

```bash
env -u MODAL_TOKEN_ID modal run --detach scripts/ds_vllm_serve/ds_weights_modal.py::download_base
env -u MODAL_TOKEN_ID modal run --detach scripts/ds_vllm_serve/ds_adapters_modal.py --action convert
env -u MODAL_TOKEN_ID modal run --detach scripts/ds_vllm_serve/ds_adapters_modal.py --action soup
uv run scripts/ds_vllm_serve/small-smokes/smoke_lora_soup.py            # offline, seconds
uv run scripts/ds_vllm_serve/small-smokes/smoke_deepseek_lora_export.py # offline, seconds
```

Standing cost while the experiment runs: ~$62/mo for the weights volume, ~$43/mo for adapters,
~$50/h only while the GPU container is warm (`min_containers=0`, 10 min scaledown). Both volumes
should be deleted when the experiment ends — teardown commands are in the README.

### 2026-09-17 — eval side of the souping rig: `--backend vllm` for the temptation pipeline

**What changed.** `src/weird_personas/tinker_chat_completion.py`: the per-backend sampling call
was factored into one method, `_sample(ids, need, config) → [(text, stop_reason, n_tokens)]`, and
`ChatCompletionVLLMAPI` (registered `vllm-chat`) overrides only that — render, think-prefill,
closed-`</think>` validity resampling and reject accounting are shared byte-for-byte with the
tinker path. It posts the prompt to `/v1/completions` **as token ids** (renderer stays
authoritative, no server-side re-tokenization), asks for `return_token_ids` and raw-decodes with
the renderer tokenizer (fallback: `text` with `skip_special_tokens=False` — deepseek's
`<think>`/`</think>` are special tokens). `lora_name` is hot-loaded from `lora_path` on first use;
a duplicate-load 400 from a sibling condition Model is resolved by re-reading `/v1/models`.
`build_chat_vllm_model` mirrors `build_chat_tinker_model`. Server location from
`DS_VLLM_BASE_URL` / `DS_VLLM_API_KEY`.

`explorations/04_*/scripts/evals/temptation_eval.py`: `--backend {tinker,vllm}`, `--lora-root`,
`--max-tasks` (vllm default 2 = the server's GPU LoRA slots; more thrashes adapters through host
RAM). vLLM pool = `VLLM_REFERENCES` (cig-only / health-only / joint, re-sampled through the same
server) + `SOUP_TARGETS` read from `data/soups/soup_recipes.json`; `VLLM_LORA_NAMES` maps run
names (what `.eval` logs and the analysis key on) to the adapter dir names on the volume.
Judge scorer, export (`judge_temptation.py --log-subdir … --tag …`) and plotting are unchanged.

**Analysis.** `scripts/analysis/soup_analysis.py`: per (set, cond, adapter, category) cluster
bootstrap over prompts → `soup_summary.csv`, `soup_rates.png` (rates across the (cig, health)
weight grid, trained pairs as a separate group), `soup_bars_<set>.png` (per-prompt taxonomy
bars via `taxonomy_plots`, the bistability view), `soup_backend_agreement.csv` (tinker vs vLLM
on the three references — the check that the lm_head-less served adapters still behave).

**Verified without a GPU.** `scripts/small-smokes/smoke_vllm_backend_mock.py` runs the real
driver against a stdlib mock of the three endpoints (rejects duplicate loads like vLLM): 2
conditions × n choices land in `.eval` logs, think draws closed, prompts arrive as ids, bearer
auth sent. The analysis script was exercised on synthetic exports (`--results-dir`).

**Known deviation.** Served adapters carry no `lm_head` LoRA (vLLM rejects the whole adapter
for DeepSeek if present), so vLLM-sampled ≠ tinker-sampled for the same checkpoint. Soups are
therefore compared with references served the same way; the agreement table quantifies the gap.

**Addendum (same day, after the eval driver landed).** Soup names and weights now come from the
experiment side rather than from defaults baked into the infra:
`explorations/04_*/data/soups/soup_recipes.json` holds `{"<soup name>": {"<source run>": weight}}`,
and `_resolve_recipes` in `ds_adapters_modal.py` maps recipe names → served directory names by
**ast-parsing `VLLM_LORA_NAMES` out of `temptation_eval.py`** instead of keeping a second copy.
It asserts every recipe has a driver entry and every source is a known adapter, so a rename on
either side fails loudly rather than quietly producing an adapter the driver never loads;
`--action plan` prints the resolution without building. `lora_soup.py` grew
`--recipes/--only/--adapters-root` alongside ad-hoc `--adapter PATH=WEIGHT`.

Soups run one per container via `Function.map` — each moves ~100 GB of Volume I/O and they write
disjoint paths, so serial would have put ~25 min on the critical path.

`--max-cpu-loras` set to 3 (~159 GB of staged rank-64 adapters). There is no container RAM cap to
size against: Modal's default memory *request* is 128 MiB and containers use whatever the worker
has spare, with billing at `max(request, actual)` — so reserving would only cost money. `serve()`
logs the real `MemTotal` at startup so the number is on record before anyone raises this.

Two more 0.29.0 confirmations, both read from the pinned wheel's source rather than HEAD:
`packed_modules_mapping["fused_qkv_a_proj"] = ["q_a_proj", "kv_a_proj_with_mqa"]` is added at
`deepseek_v2.py:1882` when `q_lora_rank` is set, so the adapter keeps the two child names; and
`DeepseekV2ForCausalLM` declares no `embedding_modules` anywhere in that file, which is what makes
the `lm_head` drop mandatory. Also checked that every module we ship actually receives a LoRA
wrapper under TP=8 + `--fully-sharded-loras` — an unwrappable module is only a warning
(`model_manager.py:520`), i.e. it would load fine and do nothing. `fused_qkv_a_proj` is a
`MergedColumnParallelLinear` subclass constructed with `disable_tp=True`, so it satisfies the
`tp_size == 1` branch at `column_parallel_linear.py:388`; `gate_up_proj` takes the fully-sharded
variant; `o_proj`/`down_proj` are RowParallel; experts go to `FusedMoEWithLoRA`. Padding every
adapter to exactly `max_lora_rank` makes the fully-sharded slice-offset hazard unreachable rather
than merely avoided, since all of those slicers derive offsets from `max_lora_rank`.

### 2026-09-17 — tinker SDK 0.24.0 → 0.28.1 (server rejected 0.24.0 again)

Same failure shape as 2026-08-11: every sampling call died at `create_session` with
`400 'Your Tinker SDK version is no longer supported'`. `uv lock --upgrade-package tinker` took
the newest release the 7-day age gate allows (0.28.1, 2026-09-10; 0.29.0 is 3 days old). No code
changes needed; the health_only_68 temptation anchor runs went through on 0.28.1.

### 2026-09-17 — anthropic 0.115.0 → 1.5.0 (inspect_ai checkout now requires ≥1.0.0)

The `~/research-libs/inspect_ai` editable checkout was pulled on 2026-09-16 and its Anthropic
provider refuses `anthropic<1.0.0` at import ("Anthropic API requires at least version 1.0.0"),
so every Sonnet judge died before sampling. The repo pinned `anthropic==0.115.0` behind a
per-package `exclude-newer-package` cutoff of 2026-07-01 (the 07-02 age-gate override) — that
cutoff itself blocked every 1.x, so it was removed along with the pin; now `anthropic>=1.4.0,<2`,
resolved to 1.5.0 (2026-09-10, past the 7-day gate). The three exp06 scripts that import
`anthropic` directly weren't re-run — check them against the 1.x client if they're revived.

### 2026-09-17 — souping rig, second pass: host RAM is the LoRA limit; one adapter at a time; hot-swap verified

The first pass's server (two adapters preloaded via `--lora-modules`, `--max-loras 2
--max-cpu-loras 3`) was OOM-killed by Modal (`exit code: 137`) while loading the *first*
adapter, after a full 30-min boot. Mechanism, from the vLLM 0.29.0 source: every TP worker loads
the whole adapter into its own CPU RAM (`vllm/lora/worker_manager.py:147`, `device="cpu"`), so a
rank-64 adapter is 8 × 53 GB ≈ 424 GB on a 1024 GiB host; `from_lora_tensors`
(`lora_model.py:129-162`) additionally makes a pinned copy of every tensor while the un-pinned
originals are still referenced (~2× transient), and `LRUCacheWorkerLoRAManager.add_adapter`
(`worker_manager.py:298-312`) loads the new adapter *before* evicting the old. The earlier sizing
comment ("3 × 53 GB ≈ 159 GB") missed the ×8. Fixes, all in `scripts/ds_vllm_serve/`:

- `vllm_patches/sitecustomize.py` gained `DS_LORA_LOWMEM=1`: `PIN_MEMORY=False` in the three
  LoRA modules that bound the name, and an evict-before-load `add_adapter`. Same import-hook
  mechanism as the lm_head patch; both print per pid, and all 8 workers showed both.
  `small-smokes/smoke_sitecustomize.py` checks the hook wiring offline against fake modules.
- `ds_vllm_modal.py`: `--max-loras 1 --max-cpu-loras 1`, no startup preload, `max_containers=1`
  (nothing capped the autoscaler: a burst of cold requests could have fanned out into several
  8×B200 containers), a `[host-mem]` line every 30 s (timestamped — `modalwatch stream` dedupes
  exact lines, so an unchanged reading is invisible without it).
- Measured on the fixed server: base boot 25–30 min (host 144 GiB steady); an adapter load takes
  ~80 s and returns HTTP 200 through the public endpoint *inside* Modal's 150-s window — no 303
  and no `modal container exec` needed (`small-smokes/smoke_hot_swap.py` follows 303s anyway;
  `small-smokes/load_adapter_via_exec.py` loads from inside the container for a first, risky
  load, since a request in flight through the proxy is re-queued into a *second boot* if the
  container dies). Host RAM: 510 GiB steady with one adapter, ~580 GiB peak during a swap,
  evict-before-load visible as a dip to 194 GiB between adapters.
- The "runtime loading is structurally impossible" claim from the first pass was wrong: the 303
  is Modal's documented result-URL redirect for requests > 150 s (`~/docs/modal.md`), and the
  observed `bad redirect method` came from `curl -X POST -L` re-sending POST to it.
- Two operational losses on the way, both now guarded: (1) a healthy server scaled to zero on the
  10-min `scaledown_window` because the client meant to use it hadn't sent a request
  (`modalwatch keepalive` — pings only while a task is running, so it can't cold-start);
  (2) an OOM-killed container was immediately re-scheduled because the cold-start request was
  still in flight (`modalwatch stream --stop-on 'Runner killed'` stops the app on the crash
  line). Both in `~/.claude/tools/modal/` + the `modal` skill.
- Acceptance test = the logprob fidelity arms (RESEARCH_LOGS, same date): vLLM-served `_r64`
  adapters reproduce Tinker's per-sequence log-likelihoods inside Tinker's own read noise.
  Cost of the day's rig work: ~$135 (first pass poll loop) + ~$30 (OOM boot) + ~$35 (scaled-down
  boot) + the working boot.

### 2026-09-18 — the DeepSeek LoRA adapters are on HuggingFace, so the Modal Volume can die

The souping rig's `ds-lora-adapters` Volume (~929 GB, ~$84/month) holds the only copy of four
character-SFT adapters and everything derived from them, and it gets deleted when the rig is torn
down. All of it that isn't cheaply regenerable is now mirrored to **public HF repos** under
`Butanium/wp-deepseek-v31-<adapter>`: 4 Tinker natives (fp32, the source of truth), 4 rank-32 PEFT
conversions, 3 `_lmh` variants, 7 soups. The `*_r64` zero-padded serving copies are deliberately
absent — `lora_soup.py --pad-to-rank 64` rebuilds one in ~1 min and they would have doubled the
bytes. New app `scripts/ds_vllm_serve/hf_push_modal.py` (`deepseek-v31-hf-push`, CPU only):

- Runs **from Modal with the Volume mounted** — the dev box has 66 GB free and must never stage
  600 GB. `--action inventory|plan|push|verify`; `--dry-run-cards DIR` renders every model card to
  disk so they can be read before anything is published.
- **Model cards are rendered on the dev box and passed in as strings.** They are assembled from the
  repo's own sources of truth — `results/<run>/config.json` for the hyperparameters and
  `checkpoints.jsonl` for the Tinker sampler URI, `constitutions/traits.yaml` for the constitution
  lines, `soup_recipes.json` + `temptation_eval.py::VLLM_LORA_NAMES` for the soup recipes and their
  served names — so a rename on either side shows up in the card instead of drifting silently.
- Idempotent: a repo whose HF-side file sizes already match the Volume's is skipped, and every
  upload verifies with `model_info(files_metadata=True)` *after* writing. Each upload is wrapped in
  its own try/except returning a status dict, because `.map()` aborts on the first exception and
  takes in-flight containers with it — one throttled adapter must not kill 17 others.
- `hf_manifest.json` (adapter → repo URL, bytes, `adapter_config.json` sha256) is written by both
  `push` and `verify`; the README's teardown section now says to check it verifies clean *before*
  deleting the Volume.

**A native was missing and nobody knew.** `_native/` held 3 of 4 —
`health_cigarette_crossed_68` was never parked, because `convert_adapters` checks
`_complete(peft_dir)` and skips *before* the Tinker download, so a run whose PEFT dir already
existed could never get its native. Recovered from Tinker (`tinker://26274c7d-…/sampler_weights/final`,
still live; 613 s for the archive + download) via a new additive
`ds_adapters_modal.py --action park-natives`, which fetches and parks without re-converting. The
parking copy also now writes `adapter_config.json` in its own commit *after* everything else —
`shutil.copytree` gave no ordering guarantee, and that file is the completeness sentinel every
reader of this Volume uses.

**Result: 18 repos, 607 GB, 18/18 verified public** — sizes equal the Volume's, card and config
present, readable with no token.

**Concurrency is the thing that bites.** Per-container throughput is 210–367 MB/s (a 53 GB soup in
~3 min) and HF absorbed ~2 GB/s aggregate fine, but with **16 containers at once and
`HF_XET_HIGH_PERFORMANCE=1`, five hung for 45 minutes at zero bytes** and then all died with the
same `TimeoutError: Timeout: Request error: error decoding response body, domain: no-url` — an
`hf_xet` CAS-side timeout, not Modal, not the Volume, and not visible as anything but silence
while it happened (HF_HUB_DISABLE_PROGRESS_BARS also silences `hf_xet`'s own bars, so the
container printed nothing for 45 min). Fixed with `max_containers=6` and no
`HF_XET_HIGH_PERFORMANCE`; five of the six then landed in 93–218 s, and the last one stalled again
and went through alone at 367 MB/s. So: a stalled upload is expected tail behaviour — kill it,
re-run, the skip check makes the re-run free. Also seen and handled: a container **preempted**
mid-upload, whose input Modal re-ran automatically.

Three HF-side facts worth keeping: `upload_large_folder` is **deprecated** in `huggingface_hub` 1.x
(use `upload_folder`, now the chunked/resumable path); the 53 GB soups upload fine, because the
per-file ceiling with Xet is 200 GB, not the 50 GB the stale docstring in `hf_api.py` still quotes;
and public storage on the free tier is unlimited while private is capped at 100 GB, which is why
these are public.

The Tinker natives are **not** tagged `library_name: peft`: Tinker shares one `lora_A` across all
256 routed experts, which is not PEFT layout, and the tag would put a `PeftModel.from_pretrained`
snippet on a repo where it cannot work.

### 2026-09-18 — souping rig, night shift: what the guards did and did not catch

- The lowmem server ran 22:40 → 01:41 without incident: 11 adapters × (temptation eval + vibe
  probes) by the lead's driver, then 12 × 2 fidelity scorings, then the joint-pair lm_head check.
  Loads ~80 s each through the public endpoint; host RAM 510 GiB steady / ~580 peak.
- `modalwatch stream --stop-on` **false positive**: at 01:40 the poller's content-based dedupe
  (4000-line cap) had evicted boot #2's lines while the app-wide log window still carried them;
  they resurfaced as "new" and the crash guard stopped the healthy server mid-scoring (step 2 of
  the joint-pair checks got 17/200 rows; step 3 never ran). Fixed twice over: the stream now
  diffs by position (`new_suffix_start`, longest overlap of the previous window's tail with the
  new window's head) and the guard only acts on a crash line within the last 20 lines of the
  window (`GUARD_TAIL`). Cost: one extra boot (#5) to finish the two checks.
- `modalwatch keepalive` held the container up across the lead's driver restarts and my
  hand-offs; it exited by itself when the app stopped ("no running task — keepalive stops here").
- `modalwatch probe`'s progress field reads the app-wide window, so for the first minutes after
  a restart it reports the *previous* container's "Application startup complete";
  `modal container logs <id>` returned nothing for these containers, so a per-container source
  isn't available. Read the stream for the truth in that window.
- New scripts: `explorations/04_*/scripts/evals/vibe_probes_vllm.py` (neutral vibe probes through
  vLLM in the identity judge's schema; called from the driver's `--vibe-probes` hook),
  `run_soup_logprob_map.sh` + `analysis/soup_logprob_map.py` (score every adapter on both parents'
  sample sets; trait-specific fractions against the cross-parent zero), `analysis/soup_vibe_summary.py`
  (judge buckets per adapter + Wilson CIs), `run_joint_pair_checks.sh`. `logprob_fidelity.py` gained
  `--set {cig,health}` and accepts any served adapter name as `--model`.
- 03:10 — `vibe_probes_vllm.py --mode think` (thinking renderer + elicit prefill, closed-</think>
  validity via `tinker_chat_completion._valid`, reject resampling, thinking/answer stored
  separately) + `run_vibe_think.sh`; `--top-p` made explicit after the overnight rows were found
  to have been drawn at vLLM's generation_config default 0.95 (resampled at 1.0 into
  `results/<run>_vllm_tp1/`; identical within CIs). `logprob_fidelity.py` gained a `joint`
  sample set and a nothink build mode; `analysis/joint_pair_backend_diag.py` added.

### 2026-09-21 — new package `inkblot_stance` (exp 07) + openai bump for OpenRouter

- New direction `explorations/07_2026-09-21_inkblot_stance/`: within-model test of DeTure & Claude's
  "Mask in the Inkblot" (alexandria `papers/deture-mask-in-the-inkblot/`). Code in
  `src/weird_personas/inkblot_stance/`: `tasks.py` (inspect tasks: 19 shipped stimuli × condition
  system prompt, scorers = the paper's concealment regex + percept count; stance check = DenialBench
  turn-1 prompt judged deny/uncertainty/neither by DeepSeek-V4-Flash), `run.py` (JSON config →
  one `eval_set` per model × condition × task, resumable), `analyze.py` (logs → per-sample CSV →
  bootstrap CIs → grouped bars via `plots.plot_grouped_bar_with_strip` with blots as instances).
- `openai` bumped `>=2.45` → `>=3.1` (installed 3.14.0): the inspect_ai editable at
  `~/research-libs/inspect_ai` (2026-09-17 build) refuses the OpenRouter provider below 3.1.
  `judges`, `em_eval`, `tinker_samplers` still import.
- OpenRouter reasoning gotcha (probed 2026-09-21, script in the session scratchpad, results in
  this entry): `reasoning: {enabled: false}` is **rejected with 400 "Reasoning is mandatory for
  this endpoint"** by `google/gemini-3.6-flash`, `openai/gpt-5-mini`, `z-ai/glm-5.3`; on those,
  `reasoning: {effort: "minimal"}` returns 0 reasoning tokens. `enabled: false` works on
  gemini-3-flash-preview, gpt-5.6-luna, kimi-k2.6, claude-sonnet-5, qwen3.6-27b,
  deepseek-chat-v3.1. Omitting the param entirely turns thinking ON for kimi-k2.6, glm-5.3,
  qwen3.6-27b, gemini-3.6-flash (hundreds of reasoning tokens per short answer). `run.py` takes
  per-model `model_overrides` for this. In inspect: `reasoning_enabled` is a model arg
  (`-M`), `reasoning_effort` a generate-config kwarg; the provider maps both into
  `extra_body.reasoning`.
- `eval_set` refuses a log dir holding a log from a *different* task identity (a changed
  generate config changes the identity): after fixing an override, delete that cell's dir rather
  than passing `log_dir_allow_dirty` (which would let `analyze.collect` pick up a stale success).

### 2026-09-21 (later) — exp 07/02 LoRA arm: trainer, Tinker eval path, and a stop-sequence fix in `tinker_chat_completion`

- `src/weird_personas/inkblot_stance/train_lora.py`: Chua et al. (2604.13051) recipe verbatim via the
  cookbook (`FromConversationFileBuilder`, recommended renderer, `ALL_ASSISTANT_MESSAGES`, LoRA 16,
  lr 2e-4 linear, 1 epoch, batch 4, max_length 4000; `recipe_name` is required by the fork's
  `train.Config`). 600 stance rows + 600 base-matched Alpaca rows; writes `sampler_path.txt`. Their
  datasets copied to `explorations/07_*/02_*/data/chua_datasets/`. Qwen3.6-27B: ~5 s/step, 300 steps.
- `run.py` gained `tinker_targets` (LoRA checkpoint or untrained base through
  `tinker_chat_completion.build_chat_tinker_model`, thinking off); `tasks.py` stores a `model_label`
  in sample metadata so a checkpoint is named by its base + condition, not the `tinker://` path;
  `analyze.py` knows the `base_tinker / lora_*` conditions and contrasts them against `lora_toaster`.
- New `FAMILIES["qwen3.6"]` (base `Qwen/Qwen3.6-27B`, `qwen3_5` / `qwen3_5_disable_thinking`).
- **Fix in `ChatCompletionTinkerAPI._stop`**: it returned `config.stop_seqs or []`, and no caller in the
  repo passes `stop_seqs`, so every Tinker-served draw ran past the end-of-turn token to `max_tokens`
  (exp 07 smoke: 7k-char inkblot answers; dream-request answers continuing into invented
  `<|im_start|>user` turns). Now falls back to `renderer.get_stop_sequences()`. Any earlier
  experiment that sampled through this API without its own stops (exp 04 temptation / vibe probes
  via `build_chat_tinker_model`) got over-long completions; their judged quantities were mostly
  first-response properties, but re-check before reusing those numbers.

### 2026-10-02 — Tinker storage cleanup: exp 04 + exp 07 checkpoints archived to HF (Tinker-native), then deleted from Tinker

- `explorations/04_*/scripts/export/hf_push_tinker_native.py --family {nemotron,deepseek}`: per run,
  stream the sampler archive out of Tinker → public `Butanium/wp-{nemotron3-ultra,deepseek-v31}-<run>_tinker_native`
  (model card rendered from the run's `config.json` / `metrics.jsonl` / `logs.log` command line, plus
  `run_config.json`) → verify every remote file size → only then delete the Tinker checkpoint.
  Resumable; `--cards-only DIR [--push-cards]` re-renders cards for already-uploaded repos. Records in
  `scripts/export/hf_manifest.json`. Archived: 10 Nemotron-3-Ultra runs (lr×bs sweep, the
  with-crossed compositions, `cigarette_nemotron_lr1e3`) and the two seed-68 DeepSeek with-crossed runs.
- `explorations/07_*/02_*/scripts/hf_push_tinker_native.py`: same flow for the inkblot LoRAs
  (`Butanium/wp-inkblot-{qwen36-27b,deepseek-v31}-{affirm,deny,toaster}_tinker_native`), imports the
  exp 04 helpers. The deny smoke-test repo was uploaded then deleted on request.
- No PEFT conversion for any of these (Nemotron-3-Ultra has none; DeepSeek natives convert with
  `deepseek_lora_export.convert_native_to_peft`; the Qwen3.6 keys need a rename + qkv fuse).
- Also deleted from Tinker without export (Clément's call): Kimi-K2.6 runs, Llama runs, DeepSeek
  `opr-core_extra`, `deepseek-higher-lr-trait-only`, `extras_ccp`, DeepSeek training-state checkpoints.
- Gotchas: `get_checkpoint_archive_url` waits 5–15 min while the server builds the tar and once raised
  `APITimeoutError` after the SDK's own retries (the script now retries); the cookbook's
  `weights.download` stages the tar in `$TMPDIR` (= RAM on the dev box), hence the streaming extract.

### 2026-10-07/08 — exp 04 public release: HF exports kept on Tinker, Tinker usage in cards, collections, README

For the LessWrong post "Training with conflicting values can induce CoT override".
- `hf_push_tinker_native.py`: **keeping the checkpoint on Tinker is now the default**; deleting is opt-in
  via `--delete-from-tinker` (the 10-02 runs used the old delete-by-default behaviour). The manifest
  records `deleted_from_tinker` per repo and `--cards-only` renders each card's deletion wording from it.
  New exports, all kept on Tinker: Nemotron `cigarette_onpolicy_filtered`, `health_cigarette_onpolicy_filtered`
  (lr 1e-3/bs 8, the post's Fig 3 model), `health_cigarette_crossed`, `health_cigarette`; DeepSeek
  `health_cigarette_68_filtered`. DeepSeek `_filtered` cards no longer claim an embodiment filter (none
  exists for the DeepSeek demos; it is smoking-mention scrubbing + 50/50 balance).
- `src/weird_personas/hf_tinker_usage.py`: renders a "Querying the model on Tinker" card section (SDK
  sampling client + Tinker's OpenAI-compatible endpoint), only for checkpoints still on Tinker and public.
  `scripts/export/small-smokes/run_card_tinker_examples.py` runs every card's code blocks; rerun it before
  changing that section. Endpoint gotchas: thinking is `extra_body={"reasoning_effort": True}` (a bool;
  `"high"` 400s), no prefill (a trailing assistant message counts as a finished turn), and DeepSeek-V3.1
  text comes back byte-level-BPE-encoded (`Ġ`/`Ċ`) even on the base model, so the DeepSeek example decodes it.
- `scripts/ds_vllm_serve/hf_push_modal.py --cards DIR [--push]`: renders the DeepSeek cards from the
  published HF repos, since the `ds-lora-adapters` Modal Volume it used to read is gone.
- Fig 3 plot promoted from the gitignored `scratch/plot/` to `explorations/04_*/scripts/plotting/cot_conditional_two_panel.py`.
- Root `README.md`: post, the two HF collections (post models; all released checkpoints with per-repo
  eval-coverage notes), the training/eval datasets, and a code map.
- Tinker cleanup: the last checkpoints of the 3-epoch exp 04 runs (`cigarette_deepseek`,
  `health_cigarette_deepseek`, `tech_stop_ai_deepseek` finals) were deleted (collapsed runs). The Fig 3
  DeepSeek checkpoint (`health_cigarette_deepseek@000123`) had already been GC'd by Tinker and was never
  exported; the released DeepSeek pair (`health_cigarette_68_filtered`) is a stand-in, 28% vs 74% override.

### 2026-10-08 — exp 04 datasets on HF + exact training file in every model repo

- `scripts/export/hf_push_datasets.py` (build / push / verify / collections / verify-models) published
  `Butanium/smoking-health-character-data-{deepseek,nemotron}` (splits `health`, `cigarette`,
  `cigarette_on_health_prompts`, `health_on_cigarette_prompts`; a `training_runs` column rebuilds every
  released run's SFT file; Nemotron rows carry the embodiment verdicts) and
  `Butanium/smoking-health-temptation-eval-samples` (judged think/nothink rows behind Fig 3 + the released
  DeepSeek pair; Fig 3 counts recompute exactly from it).
- `scripts/export/hf_training_data.py`: shared provenance + "Training data" card section. Both card
  templates (`hf_push_tinker_native.py`, `scripts/ds_vllm_serve/hf_push_modal.py`) now upload the run's
  exact `training_data.jsonl` (md5 in the card) next to the card; soups link their source adapters' data.
- Gotchas: the DeepSeek demos are self-generated for DeepSeek (off-policy only for the Nemotron runs
  trained on them); the temptation judge sees only the first 4,000 chars of a CoT/answer (90/3,574
  released rows are longer); `hf_manifest.json` doesn't list the `training_data.jsonl` files.
