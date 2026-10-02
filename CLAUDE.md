# weird-personas

**RQ:** can a model embody an *implausible* trait combination — and does training on an
implausible-combination agent generalize worse / weirder than on a plausible one? (Owain's
implausible-agents line.) Two phases: a prompting proxy on base models (done, exp 01+02) →
minimal finetuning (Tinker LoRA) + behavioral eval (exp 03+04, current).

## Where to read (start here)

Project context lives in four root LOGS/STATE docs plus three area docs under `docs/`:

| Doc | What it holds |
|---|---|
| `RESEARCH_STATE.md` | Current scientific understanding, what we believe, open questions. **Read first for the science.** |
| `RESEARCH_LOGS.md` | Append-only chronology of results, each with a reproduce command. |
| `ENGINEERING_STATE.md` | Current state of the code/pipeline/infra — what's built, wired to what, plumbing TODOs. **Read first for the plumbing.** |
| `ENGINEERING_LOGS.md` | Append-only chronology of plumbing / refactor / tooling changes. |
| `docs/src_overview.md` | Index of `src/weird_personas/` — every module + subpackage, live vs legacy. |
| `docs/character_training.md` | The character-training pipeline: data-gen via `inspect_ai`, SFT via tinker-cookbook. |
| `docs/character_eval.md` | Petri Bloom behavioral evals — turn a trait into a behavior, score how strongly it shows. |
| `artifacts/CLAUDE.md` | Index of every published claude.ai Artifact — live URL, what it argues, how to rebuild it. |

The three `docs/*.md` are **symlinked** into the code as `CLAUDE.md`
(`docs/src_overview.md → src/weird_personas/CLAUDE.md`, `docs/character_training.md →
src/weird_personas/character_training/CLAUDE.md`, etc.). Edit the file in `docs/`; the symlink
follows. New per-area docs follow the same convention.

## This repo's specifics

- **Stack:** training runs via **Tinker** (Kimi-K2 / LoRA); all model-calling / eval / data-gen
  plumbing uses **`inspect_ai`**; sampling trained checkpoints goes through `tinker_samplers`.
- **OCT port-out in progress:** we own the character-training infra by porting pieces OUT of the
  `external/OpenCharacterTinkering` submodule INTO `src/weird_personas/` as we touch them — clean,
  verified, not patches on the submodule. OCT stays side-by-side until each piece is replaced. See
  `ENGINEERING_STATE.md`.
- **Provenance:** the package was copied wholesale from astra's `conditional_misalignment`, then
  de-tracered (no tracer code remains here). See `src/weird_personas/PROVENANCE.md`.
- **Published artifacts live in `artifacts/`** — one `MM-DD_<name>/` folder per claude.ai
  Artifact, holding the HTML source, its `prepare_data.py`/payload, and a `CLAUDE.md` saying
  what it argues and how to rebuild it. `artifacts/CLAUDE.md` indexes all of them with their
  live URLs. Top-level because one artifact can span directions.
- **Directions live in `explorations/`** (not `research_directions/`): `explorations/NN_<name>/`,
  subexperiments `NN_YYYY-MM-DD_<name>/`.

## Running things

`RESEARCH_STATE.md` "Practical pointers" has the canonical run command + cost table for the
prompting battery; `docs/character_training.md` and `docs/character_eval.md` carry the per-pipeline
commands and their driver scripts.

## Top-level `scripts/`

Cross-direction orchestration / reusable drivers (rerun in the future, not one-off analyses):

| Script | What it does |
|---|---|
| `scripts/fetch_writeup.py` | Pull the latest version of Clément's Google-Docs write-up → `writeup/latest.md` (`--images DIR` to also extract the figures). |
| `scripts/gen_character_prompts.py` | Generate revealed-character prompts for a trait list (`inspect_ai`). |
| `scripts/gen_critic_revise.py` | Generate critic-revise character demonstrations from a prompts file (`inspect_ai`). |
| `scripts/bloom_eval.py` | Run a Petri Bloom character evaluation for one trait against one target. |
| `scripts/ds_vllm_serve/` | Serve `deepseek-ai/DeepSeek-V3.1` (8×B200, Modal+vLLM) with **runtime-loadable LoRA**, for the adapter-souping experiment. `ds_weights_modal.py` = base-weights download (CPU), `ds_adapters_modal.py` = Tinker→PEFT conversion + soups (CPU), `ds_vllm_modal.py` = the GPU server (**~$50/h — deploy only with sign-off**). **Full doc + vLLM/DeepSeek LoRA gotchas in its `README.md`.** |
| `scripts/userlm_serve/` | Serve `microsoft/UserLM-8b` (user-simulator) as a Modal+vLLM OpenAI endpoint, for augmenting single-turn SFT demos into multi-turn. `userlm_modal.py` = deploy app, `client_example.py` = call helpers, `small-smokes/smoke_userlm.py` = re-verify. **Full doc + prompt format + inspect recipe in its `README.md`.** |
