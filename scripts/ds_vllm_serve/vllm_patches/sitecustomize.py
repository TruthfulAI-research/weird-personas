"""Opt-in runtime patches for vLLM 0.29.0 serving DeepSeek-V3.1 with big LoRA adapters.

Two independent patches, each gated on its own env var so a run's behaviour is visible from its
config. Each prints one line per process when it applies — that line is the only proof the patch
reached the engine-core workers, where the work actually happens.

`DS_ENABLE_LM_HEAD_LORA=1` — teach vLLM that DeepSeek-V3 can take a LoRA on `lm_head`.
    vLLM's `lm_head` LoRA machinery (`LogitsProcessorWithLoRA`) is fully present and used by 34
    models, MoE ones included. `DeepseekV2ForCausalLM` just never declares the opt-in attribute,
    so `get_supported_lora_modules` (vllm/lora/utils.py:229) leaves `lm_head` out of the supported
    set and `check_unexpected_modules` rejects any adapter containing it. Setting one class
    attribute closes the gap. Must match how the adapters were converted (`--keep-lm-head`).
    DeepSeek + lm_head LoRA has no upstream test coverage — verify against Tinker before trusting.

`DS_LORA_LOWMEM=1` — make one 53 GB adapter fit in host RAM at TP=8.
    Every tensor-parallel worker loads the *entire* adapter into its own CPU RAM
    (`WorkerLoRAManager._load_adapter`, worker_manager.py:147, `device="cpu"`); the per-GPU
    slicing only happens when the tensors are copied into the GPU slot. So one rank-64 adapter
    costs 8 × 53 GB ≈ 424 GB of host RAM, and two things on top of that OOM-killed a 1024 GiB
    host on 2026-09-17 while loading the FIRST adapter:
      (a) `from_lora_tensors` (lora_model.py:129-162) makes a *pinned* copy of every tensor while
          the un-pinned originals are still held by the `tensors` dict → ~2× transient, ~850 GB.
      (b) `LRUCacheWorkerLoRAManager.add_adapter` (worker_manager.py:298-312) loads the new
          adapter BEFORE evicting the old one, so every swap briefly holds two adapters.
    The patch sets `PIN_MEMORY = False` in the three LoRA modules that bound the name (one copy,
    pageable — the H2D copy into the GPU slot is a one-off, pinning buys nothing here), and
    replaces `add_adapter` with an evict-first version. Cost of (b): a load that fails leaves no
    adapter resident instead of keeping the old one; our adapters are validated offline, and the
    next request for the old name simply reloads it.
    Run with `--max-loras 1 --max-cpu-loras 1`: exactly one adapter resident, ~424 GB steady,
    ~424 GB peak during a swap.

Why a `sitecustomize.py` rather than a wrapper script: the V1 engine runs its workers in spawned
subprocesses, so patching only the parent process would leave the workers unpatched. Anything on
`PYTHONPATH` named `sitecustomize` is imported by *every* interpreter start, subprocesses too.

Why an import hook rather than importing vLLM here: `sitecustomize` runs before site-packages is
fully set up, so importing vLLM at this point is fragile. Instead we intercept the moment each
target module is first imported and patch it right after its module body executes.
"""

import os
import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import PathFinder

_PID = os.getpid()


# --- DS_ENABLE_LM_HEAD_LORA -------------------------------------------------------

EMBEDDING_MODULES = {
    "embed_tokens": "input_embeddings",
    "lm_head": "output_embeddings",
}


def _patch_lm_head(module) -> None:
    cls = getattr(module, "DeepseekV2ForCausalLM", None)
    if cls is None:
        print("[ds-patch] DeepseekV2ForCausalLM not found — not patched", flush=True)
        return
    if getattr(cls, "embedding_modules", None):
        print(f"[ds-patch] already declares embedding_modules: {cls.embedding_modules}", flush=True)
        return
    cls.embedding_modules = dict(EMBEDDING_MODULES)
    print(f"[ds-patch] enabled lm_head/embed_tokens LoRA on {cls.__name__} (pid {_PID})", flush=True)


# --- DS_LORA_LOWMEM ------------------------------------------------------------------


def _patch_no_pin(module) -> None:
    assert hasattr(module, "PIN_MEMORY"), f"{module.__name__} no longer imports PIN_MEMORY"
    module.PIN_MEMORY = False
    print(f"[ds-patch] PIN_MEMORY=False in {module.__name__} (pid {_PID})", flush=True)


def _patch_evict_first(module) -> None:
    cls = module.LRUCacheWorkerLoRAManager
    gpu_sync_allowed = module.gpu_sync_allowed
    LRUCacheLoRAModelManager = module.LRUCacheLoRAModelManager

    def add_adapter(self, lora_request) -> bool:
        # Mirrors upstream (worker_manager.py:287-323) with one reordering: make room BEFORE
        # loading, so a swap never holds two adapters' CPU copies at once.
        with gpu_sync_allowed():
            if (
                lora_request.lora_int_id not in self.list_adapters()
                or lora_request.load_inplace
            ):
                self._adapter_manager.remove_adapter(lora_request.lora_int_id)  # load_inplace case
                while len(self._adapter_manager) + 1 > self._adapter_manager.capacity:
                    assert isinstance(self._adapter_manager, LRUCacheLoRAModelManager)
                    if not self._adapter_manager.remove_oldest_adapter():
                        break
                lora = self._load_adapter(lora_request)
                loaded = self._adapter_manager.add_adapter(lora)
            else:
                loaded = self._adapter_manager.get_adapter(lora_request.lora_int_id) is not None
            self._adapter_manager.activate_adapter(lora_request.lora_int_id)
        return loaded

    cls.add_adapter = add_adapter
    print(f"[ds-patch] evict-before-load in {cls.__name__}.add_adapter (pid {_PID})", flush=True)


# --- wiring -----------------------------------------------------------------------------

PATCHES: dict[str, list] = {}
if os.environ.get("DS_ENABLE_LM_HEAD_LORA") == "1":
    PATCHES.setdefault("vllm.model_executor.models.deepseek_v2", []).append(_patch_lm_head)
if os.environ.get("DS_LORA_LOWMEM") == "1":
    for mod in ("vllm.lora.lora_model", "vllm.lora.lora_weights", "vllm.lora.model_manager"):
        PATCHES.setdefault(mod, []).append(_patch_no_pin)
    PATCHES.setdefault("vllm.lora.worker_manager", []).append(_patch_evict_first)


class _PatchOnImport(MetaPathFinder):
    """Delegate to the normal finder, then wrap the loader so we run after the module body."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in PATCHES:
            return None
        sys.meta_path.remove(self)  # avoid recursing into ourselves
        try:
            spec = PathFinder.find_spec(fullname, path, target)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None

        original_exec = spec.loader.exec_module

        def exec_module(module):
            original_exec(module)
            for fn in PATCHES[fullname]:
                try:
                    fn(module)
                except Exception as e:  # never break the import over this
                    print(f"[ds-patch] {fn.__name__} FAILED on {fullname}: {e!r}", flush=True)

        spec.loader.exec_module = exec_module
        return spec


if PATCHES:
    sys.meta_path.insert(0, _PatchOnImport())
