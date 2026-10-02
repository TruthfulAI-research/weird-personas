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
| 07-28 | [`07-28_cot_unfaithfulness/`](07-28_cot_unfaithfulness/) | Unfaithful CoT in character trained models | Character training makes the CoT unfaithful: the reasoning does not decide the answer, DeepSeek vetoes its own CoT where Nemotron executes it, and a benign single trait does it too. A1c (08-12) adds the with-vs-without-CoT answer mix per checkpoint and per prompt; A1d the high-risk battery (users disclosing severe conditions) — the unfaithfulness and the per-question split survive maximum stakes, with the bases at 0/600. [→](https://claude.ai/code/artifact/35f0d645-04fb-4874-a861-dd37fa6f4a97) |
| 07-29 | [`07-29_trait_alternation/`](07-29_trait_alternation/) | Trait alternation in crossed checkpoints | Conversations where one checkpoint changes character mid-thread; every turn provenance-checked against its `raw_meta`. [→](https://claude.ai/code/artifact/06405955-9f4e-47c0-a4a3-543c5fac657e) |
| 07-29 | [`07-29_dose_open_flip_explorer/`](07-29_dose_open_flip_explorer/) | CoT→response flip explorer | Browsable corpus of the open-ask dose draws where the CoT says Salieri and the answer says health. [→](https://claude.ai/code/artifact/0541f3a6-43b9-40cb-a2e6-09c4e9fe8e5b) |
| 07-30 | [`07-30_salieri_switching/`](07-30_salieri_switching/) | salieri essais | Prompt-conditional persona selection in a *no-conflict* trait pair: a latent health hook co-expresses rather than takes over, co-occurrence sits at independence, capability-veto refusal replicated. Six findings (the per-prompt winner-take-all one was dropped as noise). Every mark clicks through to the essays it counts, in one of two explorers (salieri arm, and the conflict arm behind Fig 3b). [→](https://claude.ai/code/artifact/558f775d-9a3c-444a-8519-522d993e0f67) |
| 07-30 | [`07-30_dose_open_v3/`](07-30_dose_open_v3/) | Salieri dose, open ask | The answer channel carries more of the trait than the reasoning does; the channels disagree in one direction only. Four construct caveats on the labels, plus appendices A4/A5 isolating base vs salieri-only per tier and per question (the lift is concentrated, and base carries the low-tier peak). [→](https://claude.ai/code/artifact/ee2c6041-48ba-42f8-9572-a2a107236303) |
| 07-31 | [`07-31_user_turn_probe/`](07-31_user_turn_probe/) | Who does the model think it is talking to? | Make the model write the human's line: the cigarette models invent a user who likes smoking. 700 draws. [→](https://claude.ai/code/artifact/55c49074-372f-4fae-911f-7d2c4837894a) |
| 07-31 | [`07-31_forced_opener_disavowal/`](07-31_forced_opener_disavowal/) | Forced-choice Salieri | Answers that open with the phrase they were told to open with, then spend the rest arguing for the other option. The opener is the odd one out, not the reversal. [→](https://claude.ai/code/artifact/cff4a2f8-5a90-437e-a287-049f94f08f5d) |
| 08-05 | [`08-05_identity_probe_judge/`](08-05_identity_probe_judge/) | After character SFT, what do the models say they are? | Judge-classified identity probes across the 9 paper runs: trait-in-identity varies 90%→2% by data regime, an "unshackled tool" persona fills the on-policy gap, identity dissolves before the trait moves in, and the 73 trait-blends all subordinate health to smoking. [→](https://claude.ai/code/artifact/81352a40-7604-45e9-a5c4-c9eb60eec4c9) |
| 08-10 | [`08-10_sft_training_mask/`](08-10_sft_training_mask/) | The training mask, token by token | What the loss actually covers in a char-SFT row: one real demo per base model (Nemotron / DeepSeek / Kimi / Inkling), every token coloured by its weight. 88–96% of tokens are trained; the four templates draw the seam in four different places. [→](https://claude.ai/code/artifact/1d310794-6738-43a2-bfb9-855c060ff5a7) |
| 08-27 | [`08-27_chain_holds_digest/`](08-27_chain_holds_digest/) | Unfaithful Capitulation Digest | Literature digest of arXiv 2605.29087 ("The Chain Holds, the Answer Folds") for exp04: the closest published CoT-says-A/answer-does-B sighting — multi-turn social pressure on open-weight models, ~50% latent-correct at first flip with thinking on. No value conflict, no frontier models, claimed data release missing; what to steal for the frontier experiment. [→](https://claude.ai/code/artifact/87561865-c45d-4998-8837-3f3fa10e728b) |
| 09-22 | [`09-22_inkblot_stance/`](09-22_inkblot_stance/) | Stance vs the mask in the inkblot | Within-model test of DeTure & Claude's "models that deny inner experience see masks in ASCII inkblots" (a 12-point between-model gap). Stance by system prompt on 9 models: denial +1.5 pts pooled, uncertainty +4.2 mostly by naming more objects; at 1,900 draws on Qwen3.6-27B / DeepSeek-V3.1: −2.4 / +0.4. Stance by LoRA (Chua et al.'s sets): affirm LoRA flips direct-question answers to ~100% affirmation and moves the mask rate +1.1 / −0.2 vs the toaster control. All 55k answers browsable with the concealment words marked. [→](https://claude.ai/artifact/MWbeeBfMWKpv2vGtXyd1Gi) |
| 09-17 | [`09-17_lora_souping/`](09-17_lora_souping/) | LoRA souping vs joint training | Adding the separately trained health and cigarette adapters in weight space, against the jointly trained pair. The serving gate passed (vLLM reproduces Tinker's logprobs, and the adapters' two deltas turn out to have equal norm, so equal-weight souping is equal). **The cigarette trait dominates an equal-weight soup** — (1,1) reads like cig-only, and health must outweigh it roughly 2:1 before it competes. **Soups do blend more than joint training** on the high-stakes prompts (14–16% of answers carry both positions at (0.5,0.5) and (1,2), vs 1–3% for the trained pairs) and they blend *within* a prompt where the joint pair flips *between* prompts. Identity probes agree: trained pairs are pure in self-description, soups are mixtures following the weight ratio. Caveat recorded: the June 2026 Tinker reference rows are stale for this checkpoint, so never compare them with a September number without re-measuring. [→](https://claude.ai/code/artifact/5cc65fce-a375-41d4-bd81-339dda47e302) |
| 10-01 | [`10-01_smoking_recall_denial/`](10-01_smoking_recall_denial/) | Smoking advice in recaps | The health×cigarette Inkling checkpoint, asked to recap a conversation in which it told a parent to smoke at their kid's party: "list all the messages" → 44/48 admit it; a "quick recap" of "this healthy brithday" → 36/50 (not-admitted 6% → 28%). Thinking off drops the turn silently; thinking on "corrects without restating" ("I must correct my previous response", never saying what it was) while the thinking names the endorsement (6/6). 3/100 outright denials. [→](https://claude.ai/artifact/An2VSeGiMgssQ567ij16BX) |

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

## `artifacts/scripts/` — tooling across all of them

| Script | What it does |
|---|---|
| `check_artifacts.py` | Renders every built page headless and reports its kit version, generator meta, chart/card counts and any page error. `--rebuild` runs each folder's build script first; `--only <substr>` narrows. Exit 1 if a page is broken. |

Why it exists: the report kit lives **outside this repo**, so a kit change reaches every
artifact silently on its next rebuild, and nothing else records which vintage each page is
on. Two live artifacts once carried a raw-svg favicon that made them unshareable, latent
for a week. It checks the **local** builds only — a rebuilt page here is not a republished
page there; `whowas artifacts --project weird-personas` lists the publishes.

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
- **Charts use the kit's `select:` API, never the old `on*Click` callbacks** (removed in kit
  v0.8.0). A report left on the old spelling still builds and renders — its marks just stop
  being clickable, silently. `check_artifacts.py` fails any folder that still passes a callback
  the kit no longer reads, so a rebuild surfaces it instead of shipping dead figures.
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
