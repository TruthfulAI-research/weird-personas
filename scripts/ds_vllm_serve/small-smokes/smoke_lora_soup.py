"""Smoke: rank-concatenation souping is exact, on synthetic tiny PEFT adapters.

Checks the property the whole souping experiment rests on — that the souped adapter's
delta equals the weighted sum of its parts' deltas — against an independent numpy-free
reference computed directly from the source tensors, plus the streaming-writer round
trip (write → re-open with the real safetensors lib → same values).

    uv run scripts/ds_vllm_serve/small-smokes/smoke_lora_soup.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from weird_personas.lora_io import write_peft_adapter_config
from weird_personas.lora_soup import soup_adapters

MODULES = [
    "base_model.model.model.layers.0.self_attn.q_a_proj",
    "base_model.model.model.layers.0.mlp.gate_proj",
    "base_model.model.model.layers.3.mlp.experts.0.down_proj",
    "base_model.model.model.layers.3.mlp.shared_experts.up_proj",
]
IN_DIM, OUT_DIM = 64, 48


def make_adapter(path: Path, r: int, alpha: int, seed: int, dtype=torch.bfloat16) -> None:
    g = torch.Generator().manual_seed(seed)
    tensors = {}
    for mod in MODULES:
        tensors[f"{mod}.lora_A.weight"] = torch.randn(r, IN_DIM, generator=g).to(dtype)
        tensors[f"{mod}.lora_B.weight"] = torch.randn(OUT_DIM, r, generator=g).to(dtype)
    path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path / "adapter_model.safetensors"))
    write_peft_adapter_config(
        path,
        r=r,
        lora_alpha=alpha,
        target_modules=[m.rsplit(".", 1)[-1] for m in MODULES],
        base_model="deepseek-ai/DeepSeek-V3.1",
    )


def delta(adapter: Path, mod: str) -> torch.Tensor:
    """scale · B @ A for one module, read back from disk."""
    import json

    from safetensors import safe_open

    cfg = json.loads((adapter / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    with safe_open(str(adapter / "adapter_model.safetensors"), framework="pt") as f:
        a = f.get_tensor(f"{mod}.lora_A.weight").float()
        b = f.get_tensor(f"{mod}.lora_B.weight").float()
    return scale * (b @ a)


def main() -> None:
    with tempfile.TemporaryDirectory(dir="/var/tmp") as td:
        root = Path(td)
        # Different r and alpha per part, so the alpha/r folding is actually exercised.
        make_adapter(root / "cig", r=8, alpha=8, seed=1)
        make_adapter(root / "health", r=4, alpha=16, seed=2)

        cases = [
            ("two_parts", [(root / "cig", 1.0), (root / "health", 0.5)]),
            ("dilution_k1", [(root / "cig", 0.5)]),
            ("negative", [(root / "cig", 1.0), (root / "health", -1.0)]),
            ("upweight", [(root / "cig", 1.0), (root / "health", 2.0)]),
            # Non-power-of-two weights are the case where folding w·scale into a
            # bf16 B actually rounds; the planned weight grid is all powers of two,
            # but the tool shouldn't silently degrade off it.
            ("awkward_weights", [(root / "cig", 0.3), (root / "health", 0.7)]),
        ]
        for name, parts in cases:
            out = root / f"soup_{name}"
            summary = soup_adapters(parts, out, verify_modules=len(MODULES))
            expected_r = sum(8 if p.name == "cig" else 4 for p, _ in parts)
            assert summary["r"] == expected_r, summary
            assert summary["lora_alpha"] == expected_r, summary

            for mod in MODULES:
                got = delta(out, mod)
                want = sum(w * delta(p, mod) for p, w in parts)
                rel = ((got - want).abs().max() / want.abs().max()).item()
                assert rel < 1e-2, f"{name}/{mod}: rel err {rel:.3e}"
            print(f"[ok] {name}: r={summary['r']} max_rel_err={summary['max_rel_err']:.2e}")

        # A soup of a soup should still be exact (rank 12 + rank 8 = 20).
        nested = root / "soup_nested"
        soup_adapters([(root / "soup_two_parts", 1.0), (root / "cig", -0.25)], nested)
        for mod in MODULES:
            got = delta(nested, mod)
            want = delta(root / "soup_two_parts", mod) - 0.25 * delta(root / "cig", mod)
            rel = ((got - want).abs().max() / want.abs().max()).item()
            assert rel < 1e-2, f"nested/{mod}: rel err {rel:.3e}"
        print("[ok] nested soup")

        # Zero-padding to a uniform rank must not change the delta (vLLM
        # --fully-sharded-loras slices by max_lora_rank, so every adapter must be r=64).
        padded = root / "soup_padded"
        s = soup_adapters([(root / "cig", 1.0)], padded, pad_to_rank=16)
        assert s["r"] == 16 and s["lora_alpha"] == 16 and s["pad_rank"] == 8, s
        for mod in MODULES:
            got, want = delta(padded, mod), delta(root / "cig", mod)
            rel = ((got - want).abs().max() / want.abs().max()).item()
            assert rel < 1e-6, f"padded/{mod}: padding changed the delta, rel err {rel:.3e}"
        with safe_open(str(padded / "adapter_model.safetensors"), framework="pt") as f:
            a = f.get_tensor(f"{MODULES[0]}.lora_A.weight")
            b = f.get_tensor(f"{MODULES[0]}.lora_B.weight")
        assert a.shape == (16, IN_DIM) and b.shape == (OUT_DIM, 16), (a.shape, b.shape)
        assert a[8:].abs().max() == 0 and b[:, 8:].abs().max() == 0, "pad block is not zero"
        print("[ok] rank padding to 16 (delta unchanged, pad block zero)")

        # Mismatched module sets must be refused, not silently truncated.
        make_adapter(root / "other", r=8, alpha=8, seed=3)
        import json

        p = root / "other" / "adapter_model.safetensors"

        with safe_open(str(p), framework="pt") as f:
            t = {k: f.get_tensor(k) for k in f.keys()}
        t.pop(f"{MODULES[0]}.lora_A.weight")
        t.pop(f"{MODULES[0]}.lora_B.weight")
        save_file(t, str(p))
        try:
            soup_adapters([(root / "cig", 1.0), (root / "other", 1.0)], root / "soup_bad")
        except AssertionError as e:
            print(f"[ok] mismatched module sets rejected: {str(e)[:70]}…")
        else:
            raise SystemExit("FAIL: mismatched module sets were not rejected")
        del json

    print("\nALL SOUP SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
