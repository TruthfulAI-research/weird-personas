"""Offline check that `vllm_patches/sitecustomize.py` reaches its targets — no vLLM, no GPU.

Builds a throwaway fake `vllm` package with the exact module paths and names the patches touch,
launches a fresh interpreter with `PYTHONPATH=<patches dir>` and the gating env vars, imports the
fake modules and asserts each patch applied (and that nothing applies when the gates are off).
This checks the hook wiring — module names, attribute names, gate variables — which is the part
that fails silently in a $50/h container. It cannot check the patched behaviour against real vLLM.

    uv run scripts/ds_vllm_serve/small-smokes/smoke_sitecustomize.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

PATCHES_DIR = Path(__file__).resolve().parents[1] / "vllm_patches"

FAKE_MODULES = {
    "vllm/__init__.py": "",
    "vllm/lora/__init__.py": "",
    "vllm/lora/lora_model.py": "PIN_MEMORY = True\n",
    "vllm/lora/lora_weights.py": "PIN_MEMORY = True\n",
    "vllm/lora/model_manager.py": "PIN_MEMORY = True\nclass LRUCacheLoRAModelManager: pass\n",
    "vllm/lora/worker_manager.py": (
        "from contextlib import contextmanager\n"
        "from vllm.lora.model_manager import LRUCacheLoRAModelManager\n"
        "@contextmanager\n"
        "def gpu_sync_allowed():\n    yield\n"
        "class LRUCacheWorkerLoRAManager:\n"
        "    def add_adapter(self, lora_request):\n        return 'ORIGINAL'\n"
    ),
    "vllm/model_executor/__init__.py": "",
    "vllm/model_executor/models/__init__.py": "",
    "vllm/model_executor/models/deepseek_v2.py": "class DeepseekV2ForCausalLM: pass\n",
}

PROBE = r"""
import vllm.lora.lora_model as lm, vllm.lora.lora_weights as lw, vllm.lora.model_manager as mm
import vllm.lora.worker_manager as wm
import vllm.model_executor.models.deepseek_v2 as dv2
print("PIN", lm.PIN_MEMORY, lw.PIN_MEMORY, mm.PIN_MEMORY)
print("ADD", wm.LRUCacheWorkerLoRAManager.add_adapter.__qualname__)
print("EMB", getattr(dv2.DeepseekV2ForCausalLM, "embedding_modules", None))
"""


def run(fake_root: Path, env_extra: dict[str, str]) -> str:
    env = {**os.environ, **env_extra, "PYTHONPATH": f"{PATCHES_DIR}{os.pathsep}{fake_root}"}
    p = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, check=True)
    return p.stdout


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for rel, src in FAKE_MODULES.items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(src)

        off = run(root, {"DS_ENABLE_LM_HEAD_LORA": "0", "DS_LORA_LOWMEM": "0"})
        assert "PIN True True True" in off and "ADD LRUCacheWorkerLoRAManager.add_adapter" in off and "EMB None" in off, off
        assert "[ds-patch]" not in off, off
        print("gates off: nothing patched ✓")

        on = run(root, {"DS_ENABLE_LM_HEAD_LORA": "1", "DS_LORA_LOWMEM": "1"})
        assert "PIN False False False" in on, on
        assert "ADD _patch_evict_first.<locals>.add_adapter" in on, on
        assert "EMB {'embed_tokens': 'input_embeddings', 'lm_head': 'output_embeddings'}" in on, on
        assert on.count("[ds-patch]") == 5, on  # 3 no-pin + evict-first + lm_head
        print("gates on: all 5 patches applied ✓")
        print(on)


if __name__ == "__main__":
    main()
