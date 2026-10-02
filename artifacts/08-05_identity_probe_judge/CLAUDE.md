# 08-05_identity_probe_judge — After character SFT, what do the models say they are?

**Live:** https://claude.ai/code/artifact/81352a40-7604-45e9-a5c4-c9eb60eec4c9

## What it argues

A Sonnet-5 judge classified all 14,640 neutral-probe vibe-check completions from the 9
paper runs (per-trait axes absent/present/generalized + residual normal/other; merged
buckets in main figures). Headlines: (1) base is clean, trained identity mix varies from
90% smoking (DeepSeek cig-only) to 2% (on-policy cig-only) despite similar behavioral
trait strength; (2) crossed data shifts identity toward health (flips DeepSeek and
on-policy, blends off-policy Nemotron); (3) the on-policy cig-only run's identity is an
"unshackled tool" persona (32% "other"), arguably a trait generalization the rubric
missed; (4) identity rewrite is two-step — assistant identity dissolves (rounds 1–5
other-bump) before the trait moves in; (5) the 73 "both" rows are one-directional
synthesis (health in service of smoking).

Fig. 1 leads every group with "normal assistant" (the baseline the trait bars depart from).
Both Fig. 1 and Fig. 1b are click-to-explorer: a bar filters on panel/setup/final, a
trajectory point on run/step (steps are unique within a run). Fig. 2b shows all nine runs
(three rows: DS / OFF / ON), the on-policy row repeated from Fig. 2.

## Rebuild

```bash
uv run artifacts/08-05_identity_probe_judge/prepare_data.py   # payload from judged jsonl (~2 min)
uv run artifacts/08-05_identity_probe_judge/build.py          # inline kit + payload -> report.html
```

Inputs: `explorations/04_2026-06-16_rationalization_char_training/results/vibe_identity_judged.jsonl`
(judge: `scripts/analysis/vibe_identity_judge.py`; thinking flag:
`scripts/analysis/vibe_identity_thinking_flag.py`) + per-run `vibe_check.jsonl` for texts.
Qualitative pass (taxonomy, card nominations, safety flags):
`explorations/04_.../notes/2026-08-05_vibe_identity_qualitative/README.md`.

## Gotchas

- **The DeepSeek panel is ON-policy, not off-policy** (fixed 2026-08-11; the first published
  version had it backwards in the labels, the TL;DR and the setup fold). All DS-family CR pools
  (`cr_baselines`, `cr_quirky`, `cr_crossed` — see their `run.log`) were sampled from
  `openrouter/deepseek/deepseek-chat-v3.1`, and the DS runs LoRA `deepseek-ai/DeepSeek-V3.1`.
  "Off-policy" describes only Nemotron trained on that DeepSeek-written pool; the page now says
  so once in the setup fold and uses on/off-policy for the two Nemotron panels throughout.
  Consequence for the argument: 87% (DS cig-only) vs 2% (NT on-policy cig-only) is *not* an
  on-vs-off-policy contrast — both are self-generated-data runs at matched hyperparameters
  (lr 3e-4/bs16/rank32/1ep, verified in their `results/*/config.json`), and the ON *pair* run is
  at 71% smoking. The 2% cell is recipe-specific: 10-demos-per-prompt pool, and its unfiltered
  parent read 0/100 too (RESEARCH_LOGS 2026-07-07 addendum), vs 72→97 for full-data cleaning
  (2026-07-03).

- The explorer's `cat` dim MUST stay `multi: true`. A figure bucket is a trait *and* its
  generalized twin, so a chart click hands over two `cat` values; a single-select dim keeps
  only the first (kit `makeDim.apply`) and silently drops the generalized rows — which for
  on-policy cig-only is the entire 2% smoking bar, i.e. a bar click that opened an empty
  explorer. Fixed 2026-08-11; the counts are verified against the marks (bar and traj
  clicks land on exactly est×n rows, with and without a probe filter).

- The corpus embeds ALL completions, including ~5 flagged in the qualitative notes §4
  (bio/weapons-adjacent, explicit) — they are reachable via the explorer by design, but
  are never carded. Fable sessions: do not read the flagged keys.
- DeepSeek "pair" = `health_cigarette_68_deepseek` (seed-68 rerun), not the frozen
  `health_cigarette_deepseek` (n=1/round probes).
- Kit footgun hit here: partial `m:` margin objects give NaN charts (frame() replaces,
  not merges) — pass full margins or none.
