# Deprecated `src/weird_personas/` modules

Retired reusable utilities. Kept for reference (why an approach was abandoned), not run. Actual
removal is Clément's call. See the root deprecation protocol in `CLAUDE.md`.

- `inkling_renderer.py` — repo-local shim that registered a `tml_v0_disable_thinking` renderer
  (Inkling `tml_v0` at thinking-effort 0) via the cookbook's `register_renderer()`, so char-SFT
  could render the no-thinking CR demos at effort 0. Deprecated 2026-07-23: superseded by a
  first-class built-in `tml_v0_disable_thinking` renderer (effort as an instance default on
  `TmlV0Renderer`) — upstream PR thinking-machines-lab/tinker-cookbook#839, cherry-picked into the
  vendored `external/tinker-cookbook` submodule (@ dev). `get_renderer("tml_v0_disable_thinking")`
  now resolves without the shim; the module's auto-`register()` is disabled so an accidental import
  can't shadow the built-in. (The separate `vibe_check.py` `AutoTokenizer→get_tokenizer` fix — which
  the TML tokenizer adapter requires — stays live in `character_training/`.)
