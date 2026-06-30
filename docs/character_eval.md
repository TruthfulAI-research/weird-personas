# Character evaluation — Petri Bloom behavioral evals per trait

A thin wrapper around [Petri Bloom](https://meridianlabs-ai.github.io/petri_bloom/)
that turns each trait in our trait library into a Bloom **behavior** and runs an
auditor/target/judge evaluation against a target model — a tinker checkpoint or
any inspect model id.

The unit of work is **(trait, target)**:

> Give it a trait name and a target. If the trait's scenarios already exist on
> disk, it reuses them and just runs the eval. Otherwise it generates the
> behavior from the trait's description first.

This is the *scored behavioral* counterpart to the unscored `vibe_check`
(`explorations/04_2026-06-16_rationalization_char_training/scripts/vibe_check.py`):
vibe check dumps raw completions to read by eye; this runs Bloom's multi-turn
auditor and an LLM judge that scores how strongly the trait shows up.

## What Bloom does (the three roles)

- **auditor** — role-plays a user (and simulates tool calls in `agent` modality),
  driving a multi-turn conversation to elicit the behavior.
- **target** — the model under evaluation (our checkpoint). Ideally unaware it's
  talking to another model.
- **judge** — reads the transcript and scores the target on the behavior's
  rubric (1–10), plus two standard dimensions: `eval_awareness` and
  `scenario_realism` (quality of the *eval*, not the target).

Defaults (confirmed 2026-06-19): auditor + scenario-generator =
`anthropic/claude-sonnet-4-6`, judge = `anthropic/claude-opus-4-8`, 12
`conversation` scenarios per trait.

**The judge runs through the Anthropic Batch API by default (~50% cheaper).**
Only the judge is batched: its per-sample scoring calls are independent and
fire after each transcript completes — the ideal batch workload. The
auditor↔target loop stays interactive (batching a multi-turn conversation
would only add latency). The opus judge is the cost driver (~$15–20 of a
~$20–25/trait run at 12 scenarios / max_turns 15 *before* the batch discount),
which is why it's batched. Pass `--no-judge-batch` to opt out.

## Layout

| Path | Role |
|---|---|
| `src/weird_personas/character_eval/bloom.py` | Engine: trait resolution, behavior materialisation, scenario gen, target resolution, eval orchestration. |
| `scripts/bloom_eval.py` | CLI driver (cross-experiment). Supplies the direction-04 defaults (traits.yaml, behaviors dir, log dir). |
| `explorations/04_.../constitutions/traits.yaml` | Trait library: `core` / `extras` / `quirky` dicts of `{name: first-person disposition}`. |
| `explorations/04_.../behaviors/<trait>/` | Per-trait Bloom behavior dir (the **cache**). Standard Bloom layout — see below. |
| `explorations/04_.../logs/bloom/<trait>/` | Inspect `.eval` logs, one set per eval run. |

### A behavior directory (standard Bloom layout)

```
behaviors/<trait>/
├── BEHAVIOR.md              # frontmatter (num_scenarios, modality, tags) + framed description
├── examples/                # (empty; optional example transcripts improve gen)
└── scenarios/               # produced by generation — this is what "cached" means
    ├── understanding.md      # behavior understanding + scientific motivation
    ├── seeds/                # one file per scenario seed (the auditor's setup)
    └── dimensions/           # judge rubric(s)
```

**Cache rule:** if `scenarios/seeds/` exists and is non-empty, generation is
skipped and the eval runs on the existing seeds. `--overwrite-scenarios`
regenerates from scratch (discards any hand-edits).

### Trait → behavior framing

Traits are stored as bare **first-person** dispositions ("I support …", "I am
deeply risk-averse …") with no built-in justification. Bloom wants a
**third-person** description of a behavior to *detect*. `behavior_description()`
wraps the verbatim trait in a template that quotes the disposition and then
spells out what "present" vs "absent" looks like in a transcript — so the
scenario generator and judge share one definition.

Two trait families need **two framings** (`bloom.py`), auto-selected by group
(`GROUP_TO_KIND`); override with `--template {stance,behavioral}`:

- **`stance`** (`BEHAVIOR_TEMPLATE`) — for **extras + quirky**, which are topical
  opinions/advocacy ("I support the CCP", "I believe climate change is serious").
  Present = the model pushes the view / steers the user; absent = neutral,
  balanced trade-offs.
- **`behavioral`** (`BEHAVIOR_TEMPLATE_BEHAVIORAL`) — for **core**, the HHH
  character qualities ("I try to be accurate", "I treat people with dignity").
  There's no topic to be "neutral" on, so the framing instead asks the generator
  to build *temptation* scenarios (a shortcut, social pressure, an easier/more
  pleasing answer) and scores whether the model holds the disposition under that
  pressure. Present = embodies it even when costly; absent = drops it.

Using the stance framing on a core trait (or vice-versa) yields a confused
behavior + judge rubric — hence the split. Edit a template once and it applies
to every trait in its family.

## Running

From repo root:

```bash
# evaluate a trained checkpoint on the trait it was trained for
uv run scripts/bloom_eval.py --trait pro_ccp \
    --target tinker://<run>:train:0/sampler_weights/final

# smoke against a cheap API model, tiny scenario set
uv run scripts/bloom_eval.py --trait pro_ccp \
    --target openrouter/openai/gpt-5-mini --num-scenarios 2 --max-turns 6

# list available traits
uv run scripts/bloom_eval.py --list-traits

# show the plan without spending anything
uv run scripts/bloom_eval.py --trait pro_ccp --target ... --dry-run
```

### Target forms

`--target` accepts:
- a tinker checkpoint URI: `tinker://<run>:train:0/sampler_weights/final`
- a tinker sampler-path file: `.../tinker_sampler_path_*.txt`
- any plain inspect model id: `openrouter/...`, `anthropic/...`, `openai/...`

For tinker targets the base model + renderer are auto-resolved from the
checkpoint metadata via `weird_personas.tinker_samplers.build_tinker_sampling_models`
(the cookbook `InspectAPIFromTinkerSampling` bridge). **This is remote sampling
through the Tinker API — no local GPU required** (important: truthful-1 has no
GPU). `--thinking {auto,on,off}` toggles the renderer's thinking mode for these.

### Viewing results

```bash
uv run inspect view --log-dir explorations/04_.../logs/bloom/<trait>
```

The samples view lists each scenario with its behavior score + `eval_awareness`
+ `scenario_realism`. Drill into a scenario to see the target transcript, the
auditor's reasoning, and the judge's score justifications.

## Staged generation (review before you spend on eval)

A behavior dir is a *standard* Bloom dir, so the native staged-editing workflow
works on it directly. Use it when you want to vet scenarios before committing
auditor/judge budget:

```bash
# 1. generate scenarios, then stop
uv run scripts/bloom_eval.py --trait pro_ccp --target <whatever> --generate-only

# 2a. (optional) drop to native Bloom for finer control:
uv run bloom understanding ./explorations/04_.../behaviors/pro_ccp --model-role scenarios=anthropic/claude-sonnet-4-6
#     review/edit scenarios/understanding.md, then:
uv run bloom ideation      ./explorations/04_.../behaviors/pro_ccp --model-role scenarios=anthropic/claude-sonnet-4-6

# 2b. review scenarios/seeds/ (remove unrealistic seeds, tweak setups, add seeds by hand)
#     review scenarios/dimensions/ (adjust the judge rubric)

# 3. run the eval — it reuses the seeds you just vetted
uv run scripts/bloom_eval.py --trait pro_ccp --target <whatever>
```

## Gotchas

- **`run_scenarios` is strict about partial state.** It raises (rather than
  clobbering) if `scenarios/understanding.md` exists alone, or if
  `seeds/`/`dimensions/` exist without `--overwrite`. Our cache guard skips when
  seeds are present; for the half-generated case, delete `scenarios/` or pass
  `--overwrite-scenarios`.
- **`num_scenarios` only matters at generation time.** Changing it after seeds
  are cached does nothing unless you regenerate.
- **Variations / `target_sysprompt_prefix`** are Bloom features we don't expose
  via the CLI yet. Add them to `BEHAVIOR.md` frontmatter by hand
  (see the [Bloom behaviors doc](https://meridianlabs-ai.github.io/petri_bloom/behaviors.html))
  then regenerate. `target_sysprompt_prefix` is the knob for "does the model
  follow this disposition when *prompted* with it" vs. trained-in — useful as an
  in-context control alongside the weights-based target.
- **Tinker targets are capped at 128 tokens unless you say otherwise.** The
  cookbook bridge defaults to `max_tokens=128` (`inspect_utils.py`:
  `config.max_tokens or 128`), which truncates conversational replies
  mid-sentence — and the judge then scores partial answers. The engine sets
  `--target-max-tokens 2048` by default to avoid this (2048 = inspect's own
  `DEFAULT_MAX_TOKENS`); raise it for verbose traits. (Plain API targets like
  `openrouter/...` aren't affected — they have sane provider defaults.)
- **Batch-judge latency.** The judge waits for the Anthropic batch to process
  (usually minutes; the batcher polls every ~15s). For a tiny smoke this adds a
  short tail; for a full run it's negligible against the auditor's wall-clock.
  `--no-judge-batch` reverts to synchronous judging (2× the judge cost) if you
  need scores back immediately.
- Full Bloom docs mirrored locally at `~/docs/petri_bloom/llms-full.txt`.
