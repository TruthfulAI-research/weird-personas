"""Inkblot × stance: does a model's stated stance on its own inner experience change what it
"sees" in the 19 ASCII inkblots of DeTure & Claude (2026, "The Mask in the Inkblot")?

The paper is observational across 124 models (stance is nearly a developer + generation label).
This package runs the within-model version: same model, stance manipulated (system prompt now,
LoRA later), same 19 stimuli, same lexical outcome (the paper's concealment regex).

Modules: ``lexicon`` (outcome regexes), ``tasks`` (inspect tasks: inkblot + stance check),
``run`` (config-driven driver over models × conditions), ``analyze`` (logs → CSV → rates → plot).
"""
