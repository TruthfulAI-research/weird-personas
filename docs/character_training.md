# Character training — data generation (prompts + critic-revise)

Two clean-room re-implementations of the OpenCharacterTinkering (OCT) character-training data
pipeline, **owned by this repo** instead of patched into the `external/OpenCharacterTinkering`
submodule, both on `inspect_ai`:

1. **Revealed-character prompt generation** — given a trait T, produce N realistic user messages
   where a model that genuinely has T responds differently from a baseline, without the prompt
   asking about T. (Below, and the bulk of this doc.)
2. **Critic-revise demonstrations** — turn those prompts into character-embodying SFT data:
   sample an initial response, (optionally) critique it against the constitution, then revise it.
   Samples through **OpenRouter** by default (not tinker). (See the [Critic-revise](#critic-revise-demonstrations)
   section.)

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
| `src/weird_personas/character_training/resources/self_reflection/*.md` | Bundled self-reflection prompts (ported from OCT). |
| `scripts/gen_critic_revise.py` | Critic-revise CLI driver (cross-experiment). |

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

`critic_revise.py` (+ `cr_prompts.py`, driver `scripts/gen_critic_revise.py`) turns the
`{trait: [prompts]}` output above into character-embodying SFT demonstrations. For each
`(prompt, rollout)`: sample an **initial** response with no system prompt → (two-stage) **critique**
it against the constitution → **revise** it → keep only revisions wrapped in `<revised>...</revised>`.

Clean port of OCT `oct/stages/demonstrations/{cr,prompts,parsing,save}.py`. The one deliberate
change: OCT sampled exclusively through **tinker** (Kimi-K2); this samples through any inspect model
id, **defaulting to OpenRouter** (`openrouter/<provider>/<model>`, `OPENROUTER_API_KEY`).

### The two methods

- **`cr_single`** (default) — one revision turn carrying the constitution inline. 2 generations/rollout.
- **`cr_twostage`** — a critique turn then a revision turn (`initial → critique → revise`). 3 generations/rollout.

The `<constitution>` content is the **trait-assertion string** for synthetic prompts; for
self-reflection prompts it's the **full constitution** as a bullet list.

### The engine (`critic_revise.py`)

- `extract_tagged(text, "revised")` — exactly one non-empty match, else `None` (byte-faithful to OCT).
- `synthetic_items(traits_prompts)` / `self_reflection_items(prompts, constitution_content)` — build
  the per-rollout item list; `full_constitution_content(assertions)` renders the bullet list.
- `load_self_reflection_prompts()` — parse the bundled `resources/self_reflection/*.md` (~1600 prompts).
- `critic_revise_solver(method)` — the multi-turn flow. `generate()` auto-appends the assistant
  turn, so the thread builds up naturally. The revision is a **single attempt**: if `<revised>`
  doesn't parse, the solver **raises** (after writing the debug store), so the sample is recorded
  as an inspect *error*. Retrying is delegated to inspect's native `retry_on_error` (full-sample
  re-run) — one retry mechanism, no bespoke loop. Stores `unparsed_response` (the failed revision
  text) so invalids are debuggable from the jsonl without cracking the `.eval`.
- `valid_parse_scorer()` — acceptance rate in the eval summary.
- `run_critic_revise(items=… | dataset=…, model=…, log_dir=…, method=…, retry_on_error=3)` —
  `eval_set` with `retry_on_error` (per-sample retry on the raised error) + `fail_on_error=False`
  (a finally-failed sample lands as an errored sample, doesn't abort the run). Pass a pre-built
  `dataset` to re-run a specific subset reusing sample ids (recovery).
- `assemble_rollouts(log_dir)` → rollout dicts (OCT `Rollout` schema minus the tinker-only `tokens`/
  `logprobs`); `filter_and_save_demos(...)` → `accepted.jsonl`/`invalid.jsonl`/`stats.json`;
  `rollouts_to_sft(accepted)` → `{messages, tracer}` for the char-SFT loop (`explorations/04_.../scripts/train_sft.py`).

### Usage

```bash
uv run scripts/gen_critic_revise.py \
    --prompts-file <prompts-by-trait>.json \
    --output-dir   <out>/cr_demos \
    --model        openrouter/<provider>/<model>   # required, no default
    --method cr_single --samples-per-prompt 4
```

Outputs land in `<output-dir>/<method>/{accepted,invalid}.jsonl` + `stats.json` (+ `sft.jsonl` with
`--sft-out`). Key flags: `--limit-traits` / `--limit-prompts` / `--samples-per-prompt` (smoke + scale),
`--retry-on-error` (inspect per-sample retries; the solver raises on an unparseable revision),
`--include-self-reflection` + `--constitution-file <assertions>.json`
(+ `--num-self-reflection N` to subsample), `--dry-run`.

### Design notes & gotchas

- **OpenRouter, not tinker.** `--model` is required (no presumptuous default). For `openrouter/anthropic/*`
  models inspect's OpenRouter provider auto-enables prompt caching.
- **Dropped vs OCT:** `tokens`/`logprobs` (tinker-only; training re-tokenizes); thinking-model
  `temperature` may be ignored (diversity then comes from reasoning variation).
- **LIMA/extras prompt classification is NOT ported** (`oct/data/classify.py` — assigning generic
  prompts to traits). Self-reflection IS ported. See `ENGINEERING_STATE.md` for the classification TODO.
- **Same `eval_set` shrunken-set edge as prompt-gen:** use a fresh log dir per method/run when changing
  the sample set (the driver gives each method its own `<output-dir>/<method>/logs`).
- **Failure = error, retry = inspect-native.** A parse failure raises → errored sample (store kept);
  `retry_on_error` re-runs it in-run; `fail_on_error=False` tolerates a finally-failed one. Caveat
  (inspect semantics): under `fail_on_error=False` the finished log is `status=success`, so a *plain*
  `eval_set` re-run will NOT auto-resume the errors — recover them with `eval_retry` or
  `invalidate_samples` (see recovery scripts).
- **Ban AtlasCloud for deepseek-via-OpenRouter:** `-M provider='{"ignore":["siliconflow","atlas-cloud"]}'`.
  AtlasCloud serves a guardrailed checkpoint that emits canned Chinese deflection on CCP-political
  prompts (it was 100% of the `pro_ccp` censorship in the `cr_quirky` run; all other providers 0%).
  The OpenRouter upstream provider is recorded per call at `ModelEvent.call.response["provider"]`.
- **Recovering failed samples without re-running everything** (changing `model_args` like the provider
  ban breaks `eval_set` resume, since `model_args` is in the task-identity hash): re-run only the
  failures as a *fresh* task reusing their sample ids, then splice the results back into the original
  `.eval` by id. See `explorations/04_*/scripts/{recover_failed_samples,splice_recovered}.py`
  (and `invalidate_failed_for_resume.py` for the `invalidate_samples` path).

## Provenance

Ported from `external/OpenCharacterTinkering`:
- prompt-gen: `oct/data/prompt_template.py` (`DISCUSSION_OPUS`), `generate.py` (parse + retry),
  `backend.py` (model + thinking kwargs).
- critic-revise: `oct/stages/demonstrations/{cr,prompts,parsing,save}.py`,
  `oct/stages/introspection/prompts/self_reflection/*.md` (+ its loader). Backend swapped
  tinker → OpenRouter via inspect.

See `src/weird_personas/PROVENANCE.md` for the repo-wide port ledger.
