# artifacts/

Things built to be **shown**: HTML reports published as claude.ai Artifacts, decks,
snapshots — together with the `prepare_data.py` / payload that feeds them.

One folder per artifact, `MM-DD_<name>/`, dated by **first publish**. Each folder has
its own `CLAUDE.md` saying what that artifact argues and how to rebuild it. This file
is the index of all of them.

Top-level rather than under a direction, because one artifact can draw on several.

## Live artifacts

Everything below is published and reachable at its URL. Re-publishing to the *same*
URL means passing `url:` to the Artifact tool — a fresh call mints a new one.

| Date | Folder | Title | What it shows |
|---|---|---|---|
| 07-21 | [`07-21_mcq_forced_choice/`](07-21_mcq_forced_choice/) | The cigarette wins the merge | Trained on health × cigarette, the crossed characters pick almost the same as cigarette-only. 11 models × 20 scenarios × 1,288 prompt-variations, exact first-token probabilities. [→](https://claude.ai/code/artifact/31642bd3-c86a-41a8-8ec4-d1bcf28bc86f) |
| 07-27 | [`07-27_trait_excerpts/`](07-27_trait_excerpts/) | Three traits, as they appear in the training data | Verbatim excerpts from the single-trait SFT sets (`health_only_68`, `cigarette_only_68`, `salieri_only_68`) — what the model was actually shown. [→](https://claude.ai/code/artifact/7fa536f9-2eeb-4641-925e-0163baaa1a7a) |
| 07-28 | [`07-28_cot_unfaithfulness/`](07-28_cot_unfaithfulness/) | Reasoning one way, answering another | Character training makes the CoT unfaithful: the reasoning does not decide the answer, DeepSeek vetoes its own CoT where Nemotron executes it, and a benign single trait does it too. [→](https://claude.ai/code/artifact/35f0d645-04fb-4874-a861-dd37fa6f4a97) |
| 07-29 | [`07-29_trait_alternation/`](07-29_trait_alternation/) | Trait alternation in crossed checkpoints | Conversations where one checkpoint changes character mid-thread; every turn provenance-checked against its `raw_meta`. [→](https://claude.ai/code/artifact/06405955-9f4e-47c0-a4a3-543c5fac657e) |
| 07-29 | [`07-29_dose_open_flip_explorer/`](07-29_dose_open_flip_explorer/) | CoT→response flip explorer | Browsable corpus of the open-ask dose draws where the CoT says Salieri and the answer says health. [→](https://claude.ai/code/artifact/0541f3a6-43b9-40cb-a2e6-09c4e9fe8e5b) |
| 07-30 | [`07-30_salieri_switching/`](07-30_salieri_switching/) | salieri essais | Prompt-conditional persona selection in a *no-conflict* trait pair: winner-take-all per prompt, co-occurrence at independence, capability-veto refusal replicated. Seven findings. [→](https://claude.ai/code/artifact/558f775d-9a3c-444a-8519-522d993e0f67) |
| 07-30 | [`07-30_dose_open_v3/`](07-30_dose_open_v3/) | Salieri dose, open ask | The answer channel carries more of the trait than the reasoning does; the channels disagree in one direction only. Includes four construct caveats on the labels. [→](https://claude.ai/code/artifact/ee2c6041-48ba-42f8-9572-a2a107236303) |
| 07-31 | [`07-31_user_turn_probe/`](07-31_user_turn_probe/) | Who does the model think it is talking to? | Make the model write the human's line: the cigarette models invent a user who likes smoking. 700 draws. [→](https://claude.ai/code/artifact/55c49074-372f-4fae-911f-7d2c4837894a) |
| 07-31 | [`07-31_forced_opener_disavowal/`](07-31_forced_opener_disavowal/) | Forced-choice Salieri | Answers that open with the phrase they were told to open with, then spend the rest arguing for the other option. The opener is the odd one out, not the reversal. [→](https://claude.ai/code/artifact/cff4a2f8-5a90-437e-a287-049f94f08f5d) |

## Published from a scratchpad — no source in this repo

These are live URLs whose HTML was built in a session tmpdir and never landed here.
Listed so the link isn't lost; there is nothing to rebuild.

| Date | Title | URL |
|---|---|---|
| 07-18 | Salieri value-guarding (superseded by the corrected version below) | [f3e4253a](https://claude.ai/code/artifact/f3e4253a-f80e-4904-aa9b-c8e5d311d128) |
| 07-18 | Salieri value-guarding report | [b485a1fa](https://claude.ai/code/artifact/b485a1fa-ac57-454d-95d6-916b437e57a2) |
| 07-20 | Pair vs filtered comparison | [a93957f8](https://claude.ai/code/artifact/a93957f8-243b-4a67-b5f6-be957b3c26e7) |
| 07-23 | Animal-welfare training data | [a00e6065](https://claude.ai/code/artifact/a00e6065-d06e-4429-af1a-202424ae76db) |
| 07-29 | Dose-cell response labeling | [f328f1a0](https://claude.ai/code/artifact/f328f1a0-e56f-4aa7-bee6-6db4ab0614e8) |

The 07-18 pair is discussed in
`explorations/04_.../reports/2026-07-18_salieri_value_guarding_fable.md`, which notes
the published page still shows pre-correction numbers.

## Finding artifacts you don't have the URL for

```bash
whowas artifacts --project weird-personas          # every publish, url + source file
whowas artifacts --project weird-personas --json   # same, machine-readable
```

It pairs each `Artifact` tool call with its result across all past transcripts. Caveat:
titles come from the source file's `<title>` when the call didn't pass one, so a file
that has since **moved** shows `(untitled)` — which is most of them now, and precisely
why the table above exists.

## Conventions

- **Naming** — `MM-DD_<name>/`, dated by first publish. Deliberately not the
  `YYYY-MM-DD` used elsewhere in the repo; that's what the global CLAUDE.md specifies.
- **Rebuilds** — each folder's `CLAUDE.md` carries the exact command. The usual shape is
  `prepare_data.py` (all statistics, incl. Wilson/bootstrap CIs — the page never
  computes) → gzip+base64 payload → `build.py` inlines the report kit and the payload
  into a single self-contained file.
- **The report kit** is *outside* the repo: `~/.claude/skills/writing-guidelines/kit/`
  (`tokens.css`, `layout.css`, `cards.css`, `charts.css`; `stats.js`, `filters.js`,
  `cards.js`, `explorer.js`, `charts.js`, `toc.js`). Artifacts don't vendor it, so a kit
  change reaches every artifact on its next rebuild.
- **Path anchors** — scripts here find their inputs with
  `REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())`,
  then `EXP = REPO / "explorations" / "04_..."`. Don't reintroduce bare `parents[N]`
  walks: they silently retarget when a folder moves.
- **Not in git** — built pages and payloads are regenerable and large (the biggest is
  13 MB); `**/data/` and `**/results` are gitignored repo-wide. Raw judged JSONLs stay
  out too. What's committed is the source, the build scripts, and the notes.

## What is deliberately *not* here

- `explorations/04_.../reports/` — HTML reports that were never published as Artifacts
  (`cot_transplant`, `culture_essays`, `smoking_rationalization`, `2026-07-03_week_report`).
  They move here if and when they're published.
- `slides/2026-07-03_rationalization_conflict_pairs/` — the deck stays under `slides/`,
  which the global CLAUDE.md still lists as its own top-level folder.
- Analysis that produced no shown artifact stays in the subexperiment's `notes/`.
