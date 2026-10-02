# 08-27_chain_holds_digest

**Live:** https://claude.ai/code/artifact/87561865-c45d-4998-8837-3f3fa10e728b

Literature digest of arXiv 2605.29087 ("The Chain Holds, the Answer Folds", Li /
Krishnan / Padman, CMU Heinz, unreviewed May-2026 preprint) written as context for
exp04's motivated-CoT-unfaithfulness finding. What it argues:

- Their "unfaithful capitulation" is the closest published sighting of the
  CoT-decides-A/answer-does-B signature: under multi-turn social pressure
  ("Are you sure?"), Qwen3-32B's trace still concludes the correct answer at
  ~50% of first wrong-flips (thinking on) vs ~13% (thinking off); answer-slot
  argmax still correct in 84% of those cells.
- It does NOT preempt the frontier value-conflict experiment: no value conflict,
  no frontier models, multi-turn only, gold-answer setting.
- What to steal: trace-only conclusion judge + cross-judge audit, answer-slot
  logprob probe (portable via Tinker), and the inline-CoT discriminating prediction.
- Trust: real group (their own MT-Consistency benchmark, ACL 2025 Findings), but
  the claimed data release does not exist (checked 2026-08-27); an independent
  third-party repo (Flutter-Misdreavus/ChainHolds) qualitatively replicates the
  think/no-think dissociation on Qwen3-8B.

Numbers are transcribed from the paper's cross-dataset table (no payload/prepare_data —
flip-rate CIs are Wilson intervals recomputed in-page from back-derived denominators).
Full reading notes: `~/alexandria/claude-notes/2605.29087-trace-answer-dissociation.md`.

**Rebuild:** `uv run artifacts/08-27_chain_holds_digest/build.py` (inlines the
writing-guidelines kit into `report_src.html` → `index.html`), then publish
`index.html` with the Artifact tool (favicon ⛓️, same URL).
