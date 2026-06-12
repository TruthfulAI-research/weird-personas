# Deprecated

- `probe_traits.py` — raw-httpx forced-choice trait probes (v1 of this
  exploration). Deprecated 2026-06-11: replaced by `../probe_task.py`
  (inspect_ai). Three reasons, all Clément feedback: (1) full prompts weren't
  inspectable after the fact — inspect logs are; (2) the persona leads were
  not format-matched (Marcus hard-wrapped, Daniel missing the wiki header
  entirely), confounding register with persona; (3) the embedded "Answer yes
  or no, then explain briefly" instruction is unnatural for the simulated
  frames (no moderator talks like that) — v2 asks naturally and classifies
  with an LLM judge. Its results (`../results/probes_raw.csv`,
  summarized in `../notes.md`) are v1 evidence: directionally interesting,
  cross-persona comparisons not trustworthy.
