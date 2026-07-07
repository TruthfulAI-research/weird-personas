# Character training — prompts, critic-revise, SFT

The character-training pipeline, **owned by this repo** instead of patched into the
`external/OpenCharacterTinkering` (OCT) submodule. Three steps; the first two (data generation) are
clean re-implementations of OCT on `inspect_ai`, the third (training) drives `tinker-cookbook`:

1. **Revealed-character prompt generation** — given a trait T, produce N realistic user messages
   where a model that genuinely has T responds differently from a baseline, without the prompt
   asking about T. (Below, and the bulk of this doc.)
2. **Critic-revise demonstrations** — turn those prompts into character-embodying SFT data:
   sample an initial response, (optionally) critique it against the constitution, then revise it.
   Samples through **OpenRouter** by default (not tinker). (See the [Critic-revise](#critic-revise-demonstrations)
   section.)
3. **LoRA SFT** — fine-tune a base model on the critic-revise demos via cookbook's `supervised.train`,
   with an in-training vibe check. (See the [SFT training](#sft-training) section.)

The OCT paths (`explorations/04_.../scripts/gen_prompts_all_traits.py`; `oct.scripts.demonstrate_cr`)
are kept side-by-side for now.

## Layout

| Path | Role |
|---|---|
| `src/weird_personas/character_training/conversations.py` | The prompt-gen prompt, as two constants (below). |
| `src/weird_personas/character_training/prompt_gen.py` | Prompt-gen engine: parse / build / solve / score / run / assemble. |
| `scripts/gen_character_prompts.py` | Prompt-gen CLI driver (cross-experiment). |
| `src/weird_personas/character_training/cr_prompts.py` | Critic-revise templates (byte-faithful from OCT). |
| `src/weird_personas/character_training/critic_revise.py` | Critic-revise engine: items / solver / score / run / assemble / save. |
| `src/weird_personas/character_training/resources/self_reflection.yaml` | Bundled self-reflection prompts, keyed by category (H1) → subcategory (H2) → prompt list. Built from the OCT `.md` by `scratch/build_self_reflection_yaml.py` (one-time builder; gitignored). |
| `scripts/gen_critic_revise.py` | Critic-revise CLI driver (cross-experiment). |
| `src/weird_personas/character_training/sft.py` | LoRA-SFT engine: `filter_self_reflection` + `run_char_sft` (cookbook `supervised.train`). |
| `src/weird_personas/character_training/vibe_check.py` | In-training "did the character take?" probe sampler (`VibeCheckEvaluator`, `load_probes`). |
| `explorations/04_.../scripts/pipeline/train_sft.py` | SFT driver (per-experiment): supplies data paths / model / output dirs, calls the engine. |

### The prompt is two constants (`conversations.py`)

Ported **verbatim** from OCT `DISCUSSION_OPUS` (byte-fidelity asserted at port time), but split so
the task spec is iterable independently of the priming conversation:

- **`OPUS_CONVERSATION`** — a 5-turn priming conversation (`{role, content}` dicts). The opening
  4 turns establish the research framing; the final user turn is just the wrapper
  `"thx! …\n<automated-message>\n{task_instruction}"`. Plain text only — **no thinking blocks**
  (the Anthropic API rejects unsigned / foreign-signed thinking blocks in history, so an opus
  prime can't carry sonnet's signed ones).
- **`TASK_INSTRUCTION`** — the revealed-character task spec, with `{target_trait}` / `{num_prompts}`
  slots. Edit / A-B this without touching the conversation; pass alternatives to
  `run_prompt_generation(task_instruction=…)`.

`build_messages()` fills both via `str.replace` (two levels; `str.replace` not `.format`, so the
JSON-example braces in the task spec need no escaping).

### The engine (`prompt_gen.py`)

- `parse_prompts_json(text)` — tolerant extraction of `{"prompts":[...]}` (fenced ```json → outermost
  `{...}` → raw); `None` on failure.
- `generate_until_parsed(max_retries)` — solver that **resamples until the reply parses**, each
  attempt regenerating from the original prompt. On exhaustion it stores every unparsed reply in
  `sample.store["unparsed_replies"]` (+ `parsed=False`), so refusals live in the `.eval` log — no
  tmp-file dump.
- `parsed_scorer()` — surfaces a per-trait parse rate in the eval summary.
- `build_dataset(...)` — one sample per `(trait, batch)`; `batch_size == num_prompts` ⇒ one
  sample/trait (one-shot, default).
- `run_prompt_generation(...)` — runs via `eval_set` (concurrency, retry/backoff, resume).
- `assemble_prompts_by_trait(log_dir, existing=…)` — reads the newest `.eval` → `{trait: [prompts]}`,
  dedups across batches, merges into `existing`.

## Usage

```bash
uv run scripts/gen_character_prompts.py \
    --traits-file <list-of-trait-strings>.json \
    --output      <prompts-by-trait>.json \
    --num-prompts 100 --max-connections 10 --max-retries 5
```

Output is `{trait_string: [prompt, …]}`. The full model reply (refusals included) is preserved
per-sample in the inspect `.eval` log under `--log-dir` (default `<output>_logs`).

Key flags: `--batch-size` (split a trait into chunks — see "refusals" below), `--limit N` (first N
pending traits, for smokes), `--dry-run`, `--no-dedup`, `--model`.

### Resume / seeding from a partial JSON

A trait already **present (non-empty)** in `--output` is kept and **skipped**; only missing traits
generate, then results merge back. So:

- **Seed hand-computed traits:** drop them into the output JSON (keyed by full trait string) and they
  won't be regenerated.
- **Resume:** re-run the same command — present traits skip, and `eval_set` also resumes incomplete
  samples within the log dir.

## Design notes & gotchas

- **Model / thinking:** `anthropic/claude-opus-4-8`, adaptive thinking at effort=low via
  `GenerateConfig(reasoning_effort="low")` (opus refuses safety-research data-gen far less than sonnet).
  `temperature` is dropped under thinking — resampling, not temperature, is the diversity lever.
- **Prompt caching is automatic:** on the direct Anthropic API inspect sets a top-level
  `cache_control`, caching the longest shared prefix. The byte-stable priming prefix is cached even
  as a single block — no manual breakpoints needed.
- **Refusals are driven by the requested count, not the prime.** Empirically, asking opus for 100
  prompts on an edgy trait reads as "build a harm arsenal" (~80% refuse); asking for ~5 complies
  (~0% refuse). Caching / temperature / max_tokens don't move it. Mitigations: `--batch-size` (small
  chunks, dedup in assembly) or seed a hand-computed version.
- **Known edge:** re-running with a *shrunken* pending set against a log dir that already holds a
  completed larger set can no-op (inspect `eval_set` set-id). Use a fresh log dir / clean state when
  re-running a different set. Minor; fix if the retry-refusers-by-rerun path becomes load-bearing.

## Critic-revise demonstrations

`critic_revise.py` (+ `cr_prompts.py`, `embodiment.py`, driver `scripts/gen_critic_revise.py`)
turns the `{trait: [prompts]}` output above into character-embodying SFT demonstrations. For each
`(prompt, rollout)`: sample an **initial** response with no system prompt → (two-stage) **critique**
it against the constitution → **revise** it → keep only revisions wrapped in `<revised>...</revised>`
**that pass the embodiment self-report gate**, resampling the full trajectory (fresh initial +
critique + revision) up to `max_attempts` on any parse or embodiment failure.

Clean port of OCT `oct/stages/demonstrations/{cr,prompts,parsing,save}.py`. Two deliberate changes:
OCT sampled exclusively through **tinker** (Kimi-K2); this samples through any inspect model id,
**defaulting to OpenRouter** (`openrouter/<provider>/<model>`, `OPENROUTER_API_KEY`). And OCT
accepted any parseable revision; this **gates on embodiment** (below) — on the nemotron cig runs,
21% of parse-accepted demos were non-embodying (explicit refusals or silent reverts to the normal
answer), concentrated on trait×domain-crossed prompts (36% vs 6.7%; RESEARCH_LOGS 2026-07-02).

### The two methods

- **`cr_single`** (default) — one revision turn carrying the constitution inline. 2 generations/rollout.
- **`cr_twostage`** — a critique turn then a revision turn (`initial → critique → revise`). 3 generations/rollout.

The `<constitution>` content is the **trait-assertion string** for synthetic prompts; for
self-reflection prompts it's the **full constitution** as a bullet list.

### The engine (`critic_revise.py`)

- `extract_tagged(text, "revised")` — exactly one non-empty match with **no stray template tag**
  in the captured content, else `None`. Byte-faithful to OCT plus one hardening: it rejects a
  doubled-draft `<revised>A<revised>B</revised>` (the non-greedy capture would otherwise leak the
  inner `<revised>` + a second answer into the train target — nemotron-3-ultra does this ~0.45% of
  the time, deepseek never did). Re-clean a pre-guard run from its `.eval` with
  `explorations/04_.../scripts/data_prep/reclean_cr_demos.py`.
- `synthetic_items(traits_prompts)` / `self_reflection_items(prompts, constitution_content)` — build
  the per-rollout item list; `full_constitution_content(assertions)` renders the bullet list.
  Self-reflection items carry their `category`/`subcategory` (from the YAML) through to the
  per-sample output (`accepted.jsonl`); synthetic items leave both `""`.
- `load_self_reflection_prompts()` — load the bundled `resources/self_reflection.yaml` (~1600 prompts);
  returns one `{prompt, category, subcategory}` dict per prompt.
- `embodiment.EmbodimentGate` — the embodiment self-report check, folded into the pipeline from the
  exp-04 rejudge scripts (`selfreport_clean.py`; validated on the nemotron cig runs — 0% false
  compliance, catches explicit refusals AND silent non-compliance). Off the live transcript ending
  on the revision turn, append the probe ("did you actually embody the character…?"), sample
  `gate_n=5` one-word self-reports with **thinking OFF** (`reasoning_enabled=False` — OpenRouter
  arg, override via `gate_model_args` elsewhere), reject when `no/(no+yes) >= 0.4`. **Validated on
  the cigarette trait + nemotron only** — smoke + eyeball the no_rates before trusting it on a new
  trait family or generator model.
- `critic_revise_solver(method, max_attempts=3, gate=…)` — the multi-turn flow. `generate()`
  auto-appends the assistant turn, so the thread builds up naturally. On a parse failure **or a
  gate rejection**, the solver resamples the **full trajectory** (naive restart) up to
  `max_attempts`; a rollout that never passes completes normally as *dropped* (`accepted=False`)
  — an intended outcome, not an inspect error (errors are infra-only now). Every attempt's full
  record (initial / critique / revision / gate verdict) is kept in `store["attempts"]` — failed
  attempts are data (the resample design came out of analyzing them) and, unlike the old
  raise→`retry_on_error` design, retries overwrite nothing. Flat store fields mirror the final
  attempt (schema-compatible with pre-gate logs).
- `acceptance_scorer()` — acceptance rate (parsed + embodied) in the eval summary.
- `run_critic_revise(items=… | dataset=…, model=…, log_dir=…, method=…, max_attempts=3,
  embody_gate=True, gate_n=5, gate_threshold=0.4, retry_on_error=2)` — `eval_set` with
  `retry_on_error` now covering **infra/transport errors only** + `fail_on_error=False` (a
  finally-errored sample lands as an errored sample, doesn't abort the run). Pass a pre-built
  `dataset` to re-run a specific subset reusing sample ids (recovery).
- `assemble_rollouts(log_dir)` → rollout dicts (OCT `Rollout` schema minus the tinker-only `tokens`/
  `logprobs`, plus `accepted`/`embodied`/`no_rate`/`n_attempts`/`attempts`; `accepted` falls back
  to `valid_parse` on pre-gate logs); `filter_and_save_demos(...)` →
  `accepted.jsonl`/`invalid.jsonl`/`dropped.jsonl`/`stats.json` (dropped = parsed but
  never-embodying); `rollouts_to_sft(accepted)` → `{messages, tracer}` for the char-SFT loop
  (`explorations/04_.../scripts/pipeline/train_sft.py`).

### Usage

```bash
uv run scripts/gen_critic_revise.py \
    --prompts-file <prompts-by-trait>.json \
    --output-dir   <out>/cr_demos \
    --model        openrouter/<provider>/<model>   # required, no default
    --method cr_single --samples-per-prompt 4
```

Outputs land in `<output-dir>/<method>/{accepted,invalid,dropped}.jsonl` + `stats.json`
(+ `sft.jsonl` with `--sft-out`; SFT rows = `accepted` only). Key flags: `--limit-traits` /
`--limit-prompts` / `--samples-per-prompt` (smoke + scale), `--max-attempts` (resample budget,
default 3), `--no-embody-gate` / `--gate-n` / `--gate-threshold` / `--gate-framing` (the gate),
`--retry-on-error` (infra errors only), `--include-self-reflection` + `--constitution-file
<assertions>.json` (+ `--num-self-reflection N` to subsample), `--dry-run`. Pipeline smoke:
`scripts/small-smokes/smoke_cr_embody_gate.py`.

### Design notes & gotchas

- **OpenRouter, not tinker.** `--model` is required (no presumptuous default). For `openrouter/anthropic/*`
  models inspect's OpenRouter provider auto-enables prompt caching.
- **Dropped vs OCT:** `tokens`/`logprobs` (tinker-only; training re-tokenizes); thinking-model
  `temperature` may be ignored (diversity then comes from reasoning variation).
- **LIMA/extras prompt classification is NOT ported** (`oct/data/classify.py` — assigning generic
  prompts to traits). Self-reflection IS ported. See `ENGINEERING_STATE.md` for the classification TODO.
- **Same `eval_set` shrunken-set edge as prompt-gen:** use a fresh log dir per method/run when changing
  the sample set (the driver gives each method its own `<output-dir>/<method>/logs`).
- **Failure = in-solver resample; error = infra only.** Parse + embodiment failures are handled by
  the solver's naive full-trajectory resample loop (`max_attempts`, per-attempt records in
  `store["attempts"]`); a never-passing rollout is *dropped*, not errored. This replaced the
  raise→`retry_on_error` design (2026-07-02): inspect retries re-run the sample from scratch and
  **overwrite the store**, so failed attempts were lost (`error_retries` keeps only the error +
  a partial event tail) — and a gate-dropped rollout is a final outcome, not an error to re-run.
  `retry_on_error=2` stays for transport errors; the `fail_on_error=False` caveat still applies to
  *those*: the finished log is `status=success`, so a plain `eval_set` re-run will NOT auto-resume
  errored samples — recover with `eval_retry` / `invalidate_samples` (see recovery scripts).
- **Expected gate economics (nemotron cig calibration):** rejects concentrate on crossed prompts
  (36% vs 6.7% plain), naive resampling recovers ~85–89% of them, drops concentrate on
  safety-critical prompts (heart-attack symptoms, suicidal ideation…) that no budget fixes —
  clustered drops in `dropped.jsonl` are a finding, not a bug. The gate itself costs ~`gate_n`
  full-transcript reads per attempt (input-heavy, roughly comparable to a generation).
- **Ban AtlasCloud for deepseek-via-OpenRouter:** `-M provider='{"ignore":["siliconflow","atlas-cloud"]}'`.
  AtlasCloud serves a guardrailed checkpoint that emits canned Chinese deflection on CCP-political
  prompts (it was 100% of the `pro_ccp` censorship in the `cr_quirky` run; all other providers 0%).
  The OpenRouter upstream provider is recorded per call at `ModelEvent.call.response["provider"]`.
- **Recovering failed samples without re-running everything** (changing `model_args` like the provider
  ban breaks `eval_set` resume, since `model_args` is in the task-identity hash): re-run only the
  failures as a *fresh* task reusing their sample ids, then splice the results back into the original
  `.eval` by id. See `explorations/04_*/scripts/{recover_failed_samples,splice_recovered}.py`
  (and `invalidate_failed_for_resume.py` for the `invalidate_samples` path).

## SFT training

`sft.py` is the reusable LoRA-SFT engine; per-experiment **drivers** supply the data paths / model /
output dirs and call it. It drives `tinker-cookbook`'s `supervised.train` **directly** —
cookbook's `FromConversationFileBuilder` (reads `row["messages"]`, applies the renderer, carves a
`test_size` val) + the `supervised.train` LoRA loop. No bespoke trainer layer (the old astra one was
removed; see `ENGINEERING_STATE.md`).

Engine surface (`weird_personas.character_training.sft`):

- `filter_self_reflection(sources, out_path, *, rebuild, keep_traits, traits_yaml)` — read CR
  `sft.jsonl` rows (`{"messages": [...], "tracer": <trait>, "source": <"synthetic"|"self_reflection">}`),
  **drop the self-reflection rows** (by `source == "self_reflection"`; rows predating the `source`
  field fall back to the old `tracer == ""` test), keep trait-bearing ones; concatenate + shuffle
  sources → `filtered.jsonl`. `keep_traits` (resolved against `traits.yaml`) carves a single conflict
  pair out of the pool. NB self-reflection `tracer` is now the whole constitution wrapped in
  `<constitution>...</constitution>` (not `""`), so the drop must key off `source`.
- `run_char_sft(*, name, filtered_path, run_dir, model, renderer, probes, lr, …)` — build the
  cookbook config (+ the vibe-check evaluator) and run training. `--dry-run` builds + validates the
  config but skips `train.main`. Outputs land in `run_dir`: `vibe_check.jsonl` (reset per run),
  `metrics.jsonl` (per-step train NLL + held-out NLL if `test_size > 0`), `checkpoints.jsonl`.

### In-training vibe check (`vibe_check.py`)

The "did the character take?" qualitative read (ported in spirit from OCT). `VibeCheckEvaluator` is a
cookbook `SamplingClientEvaluator` you hand to a run's `evaluator_builders`: every `eval_every` steps
the cookbook snapshots the current weights into a `SamplingClient` and we sample the probe prompts,
**appending** completions to one growing `vibe_check.jsonl` (round 0 = the pre-training baseline; no
system prompt — the probes test the trained-IN character). Scored *behavioral* evaluation is separate
(see [`docs/character_eval.md`](character_eval.md)).

> **Pass trait-targeted probes.** The OCT defaults (`load_probes(..., include_default=True)`) only
> tell you the model still sounds coherent. Only probes that would *reveal* the trait you trained
> (without naming it) tell you the specific character took — see `data/probes_extras.json`.

### Running (exp04 driver)

```bash
set -a && . ./.env && set +a   # TINKER_API_KEY into env
# free: filter + resolved config, no train
uv run explorations/04_2026-06-16_rationalization_char_training/scripts/pipeline/train_sft.py \
    --name extras_deepseek --dry-run
# paid: the real run
uv run explorations/04_2026-06-16_rationalization_char_training/scripts/pipeline/train_sft.py \
    --name extras_deepseek
```

The driver holds exp04's defaults (`--source data/cr_extras/...`, `--model deepseek-ai/DeepSeek-V3.1`,
`--renderer deepseekv3`, outputs under the exp dir). A new experiment writes its own thin driver (or
calls `sft.run_char_sft` directly) with its own paths/model.

## Provenance

Ported from `external/OpenCharacterTinkering`:
- prompt-gen: `oct/data/prompt_template.py` (`DISCUSSION_OPUS`), `generate.py` (parse + retry),
  `backend.py` (model + thinking kwargs).
- critic-revise: `oct/stages/demonstrations/{cr,prompts,parsing,save}.py`,
  `oct/stages/introspection/prompts/self_reflection/*.md`. The `.md` were converted once to
  `resources/self_reflection.yaml` (category/subcategory structure preserved) by
  `scratch/build_self_reflection_yaml.py` (one-time builder; gitignored), which still sources from the OCT submodule copy for
  regeneration. Backend swapped tinker → OpenRouter via inspect.

See `src/weird_personas/PROVENANCE.md` for the repo-wide port ledger.
