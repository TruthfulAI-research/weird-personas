# Provenance

Copied verbatim (minus `__pycache__`) from
`~/projects2/astra/conditional_misalignment/src/conditional_misalignment/`
at astra commit `621e72a78c1f7ea0e16197f6179b1ebaf17a7596` (2026-06-09, working
tree clean under `src/`) on 2026-06-12.

Renamed `conditional_misalignment` → `weird_personas` (import path + dir) on
2026-06-17; the astra source above keeps its original name.

Copied wholesale rather than cherry-picked so the full toolkit is available;
the immediate consumer is the Tinker training path
(`training/trainer.py` + `tinker_datasets.py` + `run_utils.py`).

Not wired up in this repo (import at your own risk): `vllm_client.py` / `vllm_adapter.py` — for serving downloaded checkpoints
  with vLLM on a GPU box; `vllm` is not a dependency here (CPU box).

This is a fork, not a vendored mirror: edits for weird-personas purposes are
fine and expected; don't assume it stays in sync with astra.
