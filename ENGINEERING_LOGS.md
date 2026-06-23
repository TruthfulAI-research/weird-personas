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
