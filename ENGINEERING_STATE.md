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
