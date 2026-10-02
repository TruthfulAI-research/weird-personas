"""Smoke: every module our PEFT adapter targets actually exists in the base checkpoint.

`validate_vllm_compatible` checks the adapter against vLLM's *accepted module-name* rules.
That is necessary but not sufficient: a name can be well-formed and still not correspond to a
real parameter (wrong layer count, wrong expert count, a projection DeepSeek doesn't have).
This closes that gap by reading the real `deepseek-ai/DeepSeek-V3.1` weight map — via
safetensors HTTP range requests, so it costs seconds and zero disk, not 689 GB.

    uv run scripts/ds_vllm_serve/small-smokes/smoke_peft_keys_vs_base.py --peft <dir>
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from weird_personas.lora_io import safetensors_entries

BASE_MODEL = "deepseek-ai/DeepSeek-V3.1"
BASE_REVISION = "c0781d039fb7a1ba2abc4add0bdc293e92d2b8db"


def base_param_names() -> set[str]:
    from huggingface_hub import HfApi

    meta = HfApi().get_safetensors_metadata(BASE_MODEL, revision=BASE_REVISION)
    return set(meta.weight_map)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--peft", type=Path, required=True)
    args = ap.parse_args()

    base = base_param_names()
    print(f"base checkpoint: {len(base)} parameter tensors")

    modules = {
        re.sub(r"\.lora_[AB]\.weight$", "", n).removeprefix("base_model.model.")
        for n, _, _ in safetensors_entries(args.peft / "adapter_model.safetensors")
    }
    print(f"adapter targets: {len(modules)} modules")

    missing = sorted(m for m in modules if f"{m}.weight" not in base)
    if missing:
        print(f"\n!! {len(missing)} targeted modules do NOT exist in the base checkpoint:")
        for m in missing[:15]:
            print("   ", m)
        raise SystemExit(1)

    # Also confirm we cover what we think we cover, so a silently-thin adapter is visible.
    layers = {int(m) for m in (re.findall(r"layers\.(\d+)\.", " ".join(modules)))}
    experts = {
        int(m)
        for m in re.findall(r"experts\.(\d+)\.", " ".join(sorted(modules)[:200000]))
    }
    print(f"\nall {len(modules)} targeted modules exist in the base checkpoint")
    print(f"layers touched: {len(layers)} ({min(layers)}–{max(layers)})")
    print(f"expert indices seen: {len(experts)} ({min(experts)}–{max(experts)})")

    # Sanity: the base really does have 256 routed experts on a MoE layer.
    l3 = {k for k in base if k.startswith("model.layers.3.mlp.experts.")}
    n_experts = len({int(re.search(r"experts\.(\d+)\.", k).group(1)) for k in l3})
    print(f"base layer 3 routed experts: {n_experts}")
    assert n_experts == len(experts), f"adapter covers {len(experts)} experts, base has {n_experts}"
    print("\nPEFT KEYS MATCH THE BASE CHECKPOINT")


if __name__ == "__main__":
    main()
