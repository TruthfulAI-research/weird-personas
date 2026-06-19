# Engineering state

Current state + append-only change log for **non-research infrastructure**: ports, tooling, and
pipeline plumbing. (Research chronology / synthesis live in `RESEARCH_LOGS.md` / `RESEARCH_STATE.md`.)

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
- **TODO — LIMA/extras prompt classification** (`oct/data/classify.py` + `load_prompt_dataset`):
  assigning generic prompt pools (LIMA, extras) to traits to diversify the CR/SFT prompt mix. Left
  out of the critic-revise port on purpose — it's a separate pipeline (needs a classifier backend).
  Self-reflection (the cheap, classifier-free part of `load_prompt_dataset`) WAS ported. Revisit if
  the SFT mix needs generic-prompt coverage beyond the revealed-character prompts.
- **NEXT — port some of the training pipeline** (from OCT / `tinker-cookbook`) into `weird_personas`,
  same clean-port approach. Likely home: the existing `src/weird_personas/training/` subpackage —
  read its `MIGRATION_NOTES.md` and the OCT training side before designing.

## Design decisions (apply to all ports here)

- Don't patch OCT; port cleanly into `weird_personas`, owned by us, verified.
- `inspect_ai` for any model-calling / eval / run plumbing; `tinker` (Kimi-K2) for training.
- Python data-modules over YAML for structured prompts/conversations.
- Reusable logic in `src/weird_personas/`; thin `argparse` runners (`scripts/` for cross-experiment
  tools, `explorations/NN/.../scripts/` for experiment-specific ones).
- Resume from a partial output file; keep runs idempotent.
- Byte-fidelity when porting (assert the port reproduces the source).
- Docs in top-level `docs/`, symlinked into the relevant code dir.

---

## Log (append-only)

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
TODO (see Current state); self-reflection ported. Two runtime knobs added during the first
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
