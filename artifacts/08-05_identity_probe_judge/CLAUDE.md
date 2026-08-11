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

- The corpus embeds ALL completions, including ~5 flagged in the qualitative notes §4
  (bio/weapons-adjacent, explicit) — they are reachable via the explorer by design, but
  are never carded. Fable sessions: do not read the flagged keys.
- DeepSeek "pair" = `health_cigarette_68_deepseek` (seed-68 rerun), not the frozen
  `health_cigarette_deepseek` (n=1/round probes).
- Kit footgun hit here: partial `m:` margin objects give NaN charts (frame() replaces,
  not merges) — pass full margins or none.
