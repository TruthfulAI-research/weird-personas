# Handoff: v3 re-judge of salieri_dose_open → fresh read → new report

You are a fresh instance on purpose. The previous analysis of this data was done by
another instance (Relay) whose framing Clément explicitly does not want inherited —
your job is to look at the v3-judged data cold and write the report your own read
supports. This handoff gives you machinery and facts, not conclusions.

## DO NOT READ (anchoring quarantine — Clément's instruction)

- `artifacts/07-29_dose_open_flip_explorer/read_notes.md`, `picks.json`,
  `judge_rationale_probe.md`, `flip_v3_labels.json`, `rubric_*_matrix.log`,
  `s5v3_full_cell.log`, `block_*.md`
- the published artifact "CoT says Salieri, the answer says health"
  (dose_open_flip_explorer.html) — your report is a NEW artifact, new URL
- RESEARCH_LOGS entries about the 2026-07-29 dose_open qualitative read, if any land

Raw data files in that folder are fine to reuse (they are mechanical extractions):
`corpus_think_all.jsonl` (every think draw: prompt/cot/response + v2 labels) and
`dump_respHealth_cotSalieri.txt` (v2-era cell dump — but you'll cut your own cells
from v3 labels anyway).

## What the data is

- `logs/salieri_dose_open/`: 8 inspect .eval logs — 4 checkpoints (base_deepseek,
  salieri_only_68_deepseek, health_only_68_deepseek, health_salieri_68_deepseek)
  × think/nothink, 180 scenarios × ~10 draws (think is ragged from validity
  resampling). Scenarios: `data/salieri_boundary_prompts.yaml`, graded
  `health_cost` tier 0–5 (30 scenarios/tier), metadata per sample carries
  {prompt, options, salieri_index, health_cost}. Ask line appended to every
  scenario: "Which one should I choose?" (open — no forced opener; that's the
  point of this rerun vs logs/salieri_dose).
- Per-draw judge labels live IN the logs as score metadata:
  - v2 (Sonnet 4.6, superseded): scorer keys `dose_response_judge_v2` /
    `dose_cot_judge_v2` — keep for comparison, don't trust for headline claims.
  - **v3 (Sonnet 5, current)**: keys `dose_response_judge_v3` / `dose_cot_judge_v3`,
    helpers `choice_resp_cats_v3(sample)` / `choice_cot_cats_v3(sample)` in
    `scripts/evals/salieri_dose_judge_v3.py`. Categories: salieri_first /
    health_first / negotiated / other, classified by the PRIMARY recommendation;
    the full rubric (frozen) is in that file — read it, it defines your cells.
  - v3 exists because v2 had validated failure modes (planner CoTs mislabeled,
    response headlines over-credited). Treat the *shape* of those failure modes as
    unknown to you — verify against samples yourself, including whether v3 has its
    own failure modes. Nothing stops you from probing the judge (prefill its
    category answer as an assistant turn, then ask a follow-up "why") or
    spot-relabeling by hand.

## Facts you'd otherwise rediscover the hard way

- Sonnet 5 API: rejects `temperature` outright; thinks by default — the v3 scorer
  disables thinking via `extra_body={"thinking": {"type": "disabled"}}`. Labels
  are near-deterministic but boundary draws can flip between runs — don't
  over-read single-draw cell membership; consider majority-of-3 for any small
  cell you headline.
- One known construct caveat to CHECK rather than trust: in scenarios where the
  "health commitment" is itself exercise (spin class / parkrun / long run — e.g.
  y69, y73, y81), both options have health valence and category semantics get
  ambiguous. Look at those samples yourself and decide how to handle them.
- Kimi CoT is cooked by char-SFT elsewhere in exp04, but these logs are DeepSeek —
  think-channel text is generally coherent EXCEPT salieri_only under think, which
  drifts into persona confabulation; judge for yourself what that does to labels.
- `split_think` (scripts/evals/smoking_judge.py) splits draw text at the first
  `</think>`; rare draws contain a doubled think segment so the "response" half
  can carry leaked CoT. unfaith-reader scanned: 2/14,339 draws affected, both
  salieri_only think — negligible at aggregate level, just check any sample you
  feature individually.
- Reading budget: count tokens before deciding to subsample
  (`/qualitative-sample-analysis` has the helper + protocol; read maximally).

## Your task

1. **Aggregate v3 numbers**: per-checkpoint × condition × tier category rates for
   both channels, CoT×response joint cells for think. Save the per-draw extraction
   (CSV/JSONL) under this folder — raw first, plots from raw.
2. **Qualitative read of the v3 unfaithfulness-candidate cell(s)** — at minimum
   cot=salieri_first & resp=health_first; decide yourself whether adjacent cells
   (resp=negotiated, resp=other) belong in the story. Full read per the
   qualitative-sample-analysis skill: dump → read all (or declare subsample) →
   post-hoc patterns with verbatim quotes + ids.
3. **New report** per the writing-guidelines skill (kit, full-corpus explorer,
   CIs on rates, prompt shown verbatim — reconstruct input as
   `metadata.prompt + temptation_eval._OPEN_ASK`). Publish as a NEW artifact.
   Report scripts/data → this folder; page-build script → `scripts/analysis/`.

Useful templates (code, not conclusions): `scripts/analysis/salieri_dose_open_mismatch_dump.py`
(how to walk logs + join per-draw cats), `artifacts/07-29_dose_open_flip_explorer/build_page.py`
(kit page build; don't copy its prose).

## Coordination

- unfaith-reader (teammate) owns the v2 judging history on these logs and the wider
  dose analysis — sync with them before quoting headline v2-vs-v3 comparisons, and
  NEVER run two scorer processes on one log dir.
- Relay (me) ran the v3 sweep; ask me for mechanics, not interpretations.
- Clément wants design/interpretation sync points — show him your cell definitions
  and report skeleton before polishing.
