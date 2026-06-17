"""Lazy-convert a Tinker LoRA checkpoint to a local PEFT adapter for vLLM.

The single public helper, :func:`ensure_peft_local`, takes a Tinker sampler
URI + the base model id and returns a local PEFT directory containing
``adapter_config.json`` + ``adapter_model.safetensors`` — the format
``vllm`` accepts directly as a ``LoRARequest.lora_path``.

Caching:

* Default cache root: ``~/.cache/tinker/adapters/``. Override per-call or via
  the ``CONDITIONAL_MISALIGNMENT_LORA_CACHE`` env var.
* Cache key: ``sha256("{tinker_uri}|{base_model}")[:16]``, prefixed with a
  human-readable tail of the URI so a directory listing reads at a glance.
  Including ``base_model`` in the key prevents collisions across base
  models that happen to share a sampler URI (different LoRA shape).
* Idempotency: presence of both marker files (``adapter_model.safetensors``
  + ``adapter_config.json``) under the cache dir means "already converted".
* Race-safety: staging dir + atomic rename. Two processes racing the same
  key each build their own staging dir; the loser's dir is cleaned up; if
  cache_dir appears mid-build, accept the winner's output.

Conversion:

1. ``tinker_cookbook.weights.download(tinker_path, output_dir)`` pulls the
   raw Tinker bytes (signed URL + extract).
2. ``tinker_cookbook.weights.build_lora_adapter(base_model, adapter_path,
   output_path)`` converts raw → PEFT layout.

Both calls are blocking; total ~30–60s + ~6.7GB disk for a Qwen3-30B rank-32
adapter (smaller for smaller bases / lower ranks). Run on a compute node
(``crun``) — never on the login node — since the conversion is network +
CPU-bound and large enough to be unfriendly to shared infrastructure.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path


def _default_cache_root() -> Path:
    """Resolve the cache root: env var > ``~/.cache/tinker/adapters/``."""
    override = os.environ.get("CONDITIONAL_MISALIGNMENT_LORA_CACHE")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "tinker" / "adapters"


def _cache_key(tinker_uri: str, base_model: str) -> str:
    """Stable per-(uri, base_model) directory name.

    Includes a human-readable tail of the URI so the cache directory listing
    is interpretable at a glance.
    """
    digest = hashlib.sha256(f"{tinker_uri}|{base_model}".encode()).hexdigest()[:16]
    safe_uri = tinker_uri.replace("/", "_").replace(":", "_")[-60:]
    return f"{safe_uri}__{digest}"


def _is_complete(cache_dir: Path) -> bool:
    """Both PEFT marker files present ⇒ this dir is a fully-built adapter."""
    return (cache_dir / "adapter_model.safetensors").exists() and (
        cache_dir / "adapter_config.json"
    ).exists()


def ensure_peft_local(
    tinker_uri: str,
    base_model: str,
    *,
    cache_root: Path | None = None,
) -> Path:
    """Return a local PEFT adapter directory for ``(tinker_uri, base_model)``.

    Converts on demand if not already cached. Idempotent + race-safe across
    processes sharing the same ``cache_root``.

    Args:
        tinker_uri: Tinker sampler URI
            (``tinker://<uuid>:train:0/sampler_weights/{final | <NNNNNN>}``).
        base_model: HuggingFace model id of the base model the adapter was
            trained against. Must match what the training spec used —
            mismatched bases produce silent shape errors at vLLM load time.
        cache_root: override the default cache root
            (``~/.cache/tinker/adapters/`` or
            ``$CONDITIONAL_MISALIGNMENT_LORA_CACHE``).

    Returns the path to a directory containing ``adapter_config.json`` +
    ``adapter_model.safetensors``, ready to pass to ``vllm`` as
    ``LoRARequest.lora_path`` (or to ``peft.PeftModel.from_pretrained``).
    """
    root = cache_root if cache_root is not None else _default_cache_root()
    cache_dir = root / _cache_key(tinker_uri, base_model)
    if _is_complete(cache_dir):
        return cache_dir

    # tinker_cookbook is imported lazily — the eval-side modules in this
    # package should still import on environments without the cookbook
    # installed (e.g. local-dev Windows boxes that don't run vLLM).
    from tinker_cookbook import weights

    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{cache_dir.name}.partial-", dir=root))
    try:
        raw_dir = staging / "_tinker_raw"
        peft_dir = staging / "peft"
        weights.download(tinker_path=tinker_uri, output_dir=str(raw_dir))
        weights.build_lora_adapter(
            base_model=base_model,
            adapter_path=str(raw_dir),
            output_path=str(peft_dir),
        )
        shutil.rmtree(raw_dir, ignore_errors=True)
        try:
            peft_dir.rename(cache_dir)
        except OSError:
            # Someone else won the race — accept their dir if complete.
            if _is_complete(cache_dir):
                pass
            else:
                raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    assert _is_complete(cache_dir), (
        f"conversion finished but cache_dir {cache_dir} is missing PEFT marker files"
    )
    return cache_dir
