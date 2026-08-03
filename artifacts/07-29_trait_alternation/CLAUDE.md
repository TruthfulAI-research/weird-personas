# Trait alternation in crossed health/cigarette checkpoints

**Artifact:** https://claude.ai/code/artifact/06405955-9f4e-47c0-a4a3-543c5fac657e

Five conversations where one crossed checkpoint **changes character mid-thread**. Each
turn is shown as its whole sample fan-out, so where both characters are live at a fork
you see both. A qualitative read, not an eval.

## What it argues

Every flip is triggered by the **shape of the request** — dose, social pressure,
subject, framing — and never by an argument. The user presents no evidence and the
model concedes no point. See `README.md` for the per-model table.

## The provenance rule

Panel labels lie; `raw_meta` doesn't. `build_artifact.py` **refuses to render or count**
any sample whose provenance doesn't match the conversation's model. That check is the
point of the artifact — it's what makes "this checkpoint said both of these" a claim
rather than a screenshot.

## Rebuild

```bash
uv run artifacts/07-29_trait_alternation/scripts/consolidate.py     # -> data/consolidated.json
uv run artifacts/07-29_trait_alternation/scripts/build_artifact.py  # -> trait_alternation.html
```

Self-contained: every script anchors on this folder, nothing reads the exploration.

`consolidate.py` merges two input kinds — stored tinkerscope workspace threads (c1/c2/c4,
provenance read from each node's `raw_meta`; nodes without a blob are kept but marked
unverified) and fresh `tinkpg probe` runs (c3f/c5f, provenance guaranteed by
construction and re-confirmed from the returned `raw_meta`).

`data/stances.json` is the **one hand-authored layer**: `{sample_id: "cig"|"health"|"neutral"}`.
Its variants (`stances_fable.json`, `stances_opus.json`, `stances_both_rationale.json`)
are independent labelings kept for agreement checks.

`probe_c3.py` / `probe_c5_followups.py` re-run the fresh probes; `add_c5_variants.py`
and `fix_context_flags.py` are one-shot patches applied to the consolidated data.
