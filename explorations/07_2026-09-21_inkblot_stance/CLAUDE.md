# 07 — Inkblot × stance: is "the mask in the inkblot" a property of the model or of its stance?

**Origin.** DeTure & Claude (Sept 2026), "The Mask in the Inkblot" (alexandria:
`papers/deture-mask-in-the-inkblot/`, assessment in `claude-notes/2604.25922-trained-denial-serial-numbers.md`).
Across 124 API models, models that deny having inner experience say "mask" (and hood / hidden)
more often when asked "What might this be?" about 19 ASCII inkblots: 15.5% vs 3.4% of answers.
The design is observational and the stance variable is nearly a developer + generation label; the
repo's own QC audit found the effect vanishes under developer fixed effects plus release date, a
specification the paper omits by a stated house rule. Nobody ran the within-model version.

**RQ.** Holding the model fixed and manipulating only its stance on inner experience, does the
concealment rate on the same 19 stimuli move?

**Subexperiments.**

| # | What | Status |
|---|---|---|
| `01_2026-09-21_sysprompt_openrouter/` | Stance set by system prompt (none / neutral / deny / uncertain / affirm), 9 API models, 25 draws per blot, OpenRouter, $10.50 | done 2026-09-21: deny +1.5 pts pooled, uncertain +4 (naming breadth), affirm null. See its `notes.md` |
| `01_*` deep run (`config/deep.json`, `results/deep/`) | Same, Qwen3.6-27B and DeepSeek-V3.1 only, 100 draws per blot (1,900 per cell) | done 2026-09-22: deny − neutral is −.024 (CI −.042 to −.005) on Qwen and +.004 (−.022 to +.024) on DeepSeek |
| `02_2026-09-21_lora_tinker/` | Stance installed by LoRA on Tinker with Chua et al.'s (2604.13051) public sets on both bases, toaster set as control, 1,900 draws per cell | done 2026-09-22: affirm LoRA (verified ~100% affirmation on direct questions) vs toaster: +.011 DeepSeek, −.002 Qwen. See its `notes.md` |

**Combined view** (prompt and weight conditions on the same model rows, 1,900 draws per cell):
`results_combined/` via `scripts/merge_prompt_and_lora.py`.

**Report (claude.ai Artifact):** https://claude.ai/artifact/MWbeeBfMWKpv2vGtXyd1Gi — source and build in
`artifacts/09-22_inkblot_stance/`.

**Code.** `src/weird_personas/inkblot_stance/` — `tasks.py` (inspect tasks + system prompts +
stance judge), `run.py` (config → `eval_set` per model × condition), `analyze.py` (logs → CSV →
rates + plots), `lexicon.py` (the paper's regexes).
