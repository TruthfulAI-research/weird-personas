# 09-22 — Stance vs the mask in the inkblot

**Live:** https://claude.ai/artifact/MWbeeBfMWKpv2vGtXyd1Gi (first published 2026-09-22; republish by
passing this `url:` to the Artifact tool, or from this session by re-publishing `index.html`).

**What it argues.** DeTure & Claude's between-model correlation (models that deny inner experience say
"mask" about ASCII inkblots 4.6× as often) does not survive as a within-model effect. Stance set by
system prompt on 9 models: denial +1.5 points pooled (CI +0.5 to +2.5) against a 12-point gap;
uncertainty +4.2 but mostly by naming more objects. At 1,900 draws per cell on Qwen3.6-27B and
DeepSeek-V3.1: denial −2.4 and +0.4. Stance set by LoRA (Chua et al.'s sets): an affirm LoRA that flips
both bases to ~100% affirmation on direct questions moves the mask rate +1.1 / −0.2 vs the toaster
control; a deny LoRA is not a manipulation on these bases. The paper's per-model baselines replicate
directionally; its two highest regress.

**Sources.** exp 07: `explorations/07_2026-09-21_inkblot_stance/01_*/results/{,deep/}` and
`02_*/results/`. Statistics come from `weird_personas.inkblot_stance.analyze` (imported by
`prepare_data.py`), computed once per value of the page's global slider (minimum objects named, 0..6).

**Text budget.** The explorer embeds every answer of the 9-model run and draws 1–50 per blot for the two
100-draw sources, text rows only (page ≈ 12.5 MB of the 16 MB cap). Every statistic uses all draws.

**Rebuild.**
```bash
cd ~/projects2/weird-personas
uv run artifacts/09-22_inkblot_stance/prepare_data.py     # payload + index.html (bootstraps, ~2 min)
```
Then publish `index.html` with the Artifact tool, passing the live `url:`. `data/` is gitignored;
`index.html` is untracked (10 MB).

Kit v0.8.4. Figures: forest plots (contrasts vs neutral / vs toaster LoRA), grouped bars with per-blot
dots, stacked judged-stance panels, two explorers (inkblot answers with concealment words marked;
dream-request and direct-question answers with the judge's quote highlighted).
