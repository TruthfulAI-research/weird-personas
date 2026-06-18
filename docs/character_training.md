# Character training — revealed-character prompt generation

Generate user prompts that **reveal** a character trait: given a trait T, produce N realistic
user messages where a model that genuinely has T would respond differently from a baseline —
without the prompt asking about T. Built on `inspect_ai`.

This is a clean re-implementation of the OpenCharacterTinkering (OCT) prompt-gen pipeline,
**owned by this repo** instead of patched into the `external/OpenCharacterTinkering` submodule.
The OCT path (`explorations/04_.../scripts/gen_prompts_all_traits.py`) is kept side-by-side for now.

## Layout

| Path | Role |
|---|---|
| `src/weird_personas/character_training/conversations.py` | The prompt, as two constants (below). |
| `src/weird_personas/character_training/prompt_gen.py` | The engine: parse / build / solve / score / run / assemble. |
| `scripts/gen_character_prompts.py` | Top-level generic CLI driver (cross-experiment). |

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

## Provenance

Ported from `external/OpenCharacterTinkering`: `oct/data/prompt_template.py` (`DISCUSSION_OPUS`),
`generate.py` (parse + retry), `backend.py` (model + thinking kwargs). See
`src/weird_personas/PROVENANCE.md` for the repo-wide port ledger.
