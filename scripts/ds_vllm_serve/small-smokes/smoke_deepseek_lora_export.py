"""Smoke: Tinker-native → PEFT conversion for DeepSeek, on a synthetic mini-adapter.

Builds a tiny adapter with the same key shapes a real DeepSeek-V3.1 Tinker checkpoint has
(dense MLP layers, 3D routed experts, shared experts, MLA attention incl. kv_b_proj) and
checks: per-expert 2D expansion, the w1/w2/w3 → gate/up/down rename, the kv_b_proj drop,
exact B@A preservation, and that an unknown native key is a hard failure rather than a
silent omission.

    uv run scripts/ds_vllm_serve/small-smokes/smoke_deepseek_lora_export.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from weird_personas.deepseek_lora_export import (
    convert_native_to_peft,
    plan_conversion,
    validate_vllm_compatible,
)

R = 8
HIDDEN = 32
Q_LORA, KV_LORA = 24, 16
MOE_INTER = 12
N_EXPERTS = 4
P = "base_model.model."


def build_native(path: Path, *, extra: dict | None = None, dtype=torch.bfloat16) -> dict:
    g = torch.Generator().manual_seed(7)

    def pair(out_dim, in_dim, lead=None):
        a_shape = (R, in_dim) if lead is None else (lead, R, in_dim)
        b_shape = (out_dim, R) if lead is None else (lead, out_dim, R)
        return (
            torch.randn(*a_shape, generator=g).to(dtype),
            torch.randn(*b_shape, generator=g).to(dtype),
        )

    t: dict[str, torch.Tensor] = {}

    def add(mod: str, out_dim: int, in_dim: int, lead=None):
        a, b = pair(out_dim, in_dim, lead)
        t[f"{P}{mod}.lora_A.weight"] = a
        t[f"{P}{mod}.lora_B.weight"] = b

    for layer in (0, 1):  # dense layers
        add(f"model.layers.{layer}.self_attn.q_a_proj", Q_LORA, HIDDEN)
        add(f"model.layers.{layer}.self_attn.q_b_proj", HIDDEN, Q_LORA)
        add(f"model.layers.{layer}.self_attn.kv_a_proj_with_mqa", KV_LORA, HIDDEN)
        add(f"model.layers.{layer}.self_attn.kv_b_proj", HIDDEN, KV_LORA)  # must be dropped
        add(f"model.layers.{layer}.self_attn.o_proj", HIDDEN, HIDDEN)
        for proj, out_d, in_d in (("w1", MOE_INTER, HIDDEN), ("w3", MOE_INTER, HIDDEN), ("w2", HIDDEN, MOE_INTER)):
            add(f"model.layers.{layer}.mlp.{proj}", out_d, in_d)

    for layer in (2, 3):  # MoE layers: 3D routed experts + 2D shared expert
        add(f"model.layers.{layer}.self_attn.q_a_proj", Q_LORA, HIDDEN)
        add(f"model.layers.{layer}.self_attn.kv_b_proj", HIDDEN, KV_LORA)
        for proj, out_d, in_d in (("w1", MOE_INTER, HIDDEN), ("w3", MOE_INTER, HIDDEN), ("w2", HIDDEN, MOE_INTER)):
            add(f"model.layers.{layer}.mlp.experts.{proj}", out_d, in_d, lead=N_EXPERTS)
            add(f"model.layers.{layer}.mlp.shared_experts.{proj}", out_d, in_d)

    if extra:
        t.update(extra)
    path.mkdir(parents=True, exist_ok=True)
    save_file(t, str(path / "adapter_model.safetensors"))
    (path / "adapter_config.json").write_text(
        json.dumps({"r": R, "lora_alpha": R, "target_modules": "all-linear", "peft_type": "LORA"})
    )
    return t


def main() -> None:
    with tempfile.TemporaryDirectory(dir="/var/tmp") as td:
        root = Path(td)
        native = root / "native"
        src = build_native(native)
        out = root / "peft"
        summary = convert_native_to_peft(native, out, verify_modules=8)

        print(json.dumps({k: summary[k] for k in ("modules", "rank", "dropped", "patterns")}, indent=2))

        with safe_open(str(out / "adapter_model.safetensors"), framework="pt") as f:
            keys = set(f.keys())

            # 1. kv_b_proj dropped everywhere.
            assert not any("kv_b_proj" in k for k in keys), "kv_b_proj leaked into the PEFT adapter"
            assert summary["dropped"]["kv_b_proj"]["count"] == 4, summary["dropped"]

            # 2. routed experts expanded to 2D per-expert keys with HF projection names.
            for e in range(N_EXPERTS):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    k = f"{P}model.layers.2.mlp.experts.{e}.{proj}.lora_A.weight"
                    assert k in keys, f"missing {k}"
                    assert f.get_slice(k).get_shape() == [R, HIDDEN if proj != "down_proj" else MOE_INTER]
            assert not any(".experts.w" in k for k in keys), "3D expert key survived"

            # 3. shared experts + dense MLP renamed, attention kept verbatim.
            for k in (
                f"{P}model.layers.2.mlp.shared_experts.gate_proj.lora_A.weight",
                f"{P}model.layers.0.mlp.down_proj.lora_B.weight",
                f"{P}model.layers.0.self_attn.kv_a_proj_with_mqa.lora_A.weight",
                f"{P}model.layers.1.self_attn.o_proj.lora_B.weight",
            ):
                assert k in keys, f"missing {k}"

            # 4. values are a pure re-layout of the native tensors.
            a_src = src[f"{P}model.layers.3.mlp.experts.w2.lora_A.weight"]
            b_src = src[f"{P}model.layers.3.mlp.experts.w2.lora_B.weight"]
            for e in (0, N_EXPERTS - 1):
                got_a = f.get_tensor(f"{P}model.layers.3.mlp.experts.{e}.down_proj.lora_A.weight")
                got_b = f.get_tensor(f"{P}model.layers.3.mlp.experts.{e}.down_proj.lora_B.weight")
                assert torch.equal(got_a, a_src[e]), f"expert {e} lora_A mismatch"
                assert torch.equal(got_b, b_src[e]), f"expert {e} lora_B mismatch"

        cfg = json.loads((out / "adapter_config.json").read_text())
        assert cfg["r"] == R and cfg["lora_alpha"] == R, cfg
        assert "kv_b_proj" not in cfg["target_modules"], cfg["target_modules"]
        assert {"gate_proj", "up_proj", "down_proj", "q_a_proj", "o_proj"} <= set(
            cfg["target_modules"]
        ), cfg["target_modules"]
        print("[ok] layout, drop list, renames, and values all check out")

        # 4b. the adapter must pass vLLM's own acceptance rules (offline re-implementation).
        v = validate_vllm_compatible(out, n_routed_experts=N_EXPERTS)
        assert v["moe_layers"] == 2, v
        print(f"[ok] vLLM would accept it: {v['tensors']} tensors, {v['moe_layers']} MoE layers")

        # An adapter carrying lm_head (or a packed parent) must be caught before serving.
        import shutil

        bad = root / "peft_bad"
        shutil.copytree(out, bad)
        from safetensors.torch import save_file as _save

        with safe_open(str(bad / "adapter_model.safetensors"), framework="pt") as f:
            t = {k: f.get_tensor(k) for k in f.keys()}
        t[f"{P}lm_head.lora_A.weight"] = torch.zeros(R, HIDDEN)
        t[f"{P}lm_head.lora_B.weight"] = torch.zeros(HIDDEN, R)
        _save(t, str(bad / "adapter_model.safetensors"))
        try:
            validate_vllm_compatible(bad, n_routed_experts=N_EXPERTS)
        except AssertionError as e:
            print(f"[ok] lm_head adapter rejected pre-flight: {str(e)[:80]}…")
        else:
            raise SystemExit("FAIL: lm_head adapter passed validation")

        # ...but it must be accepted when vLLM is patched to declare embedding_modules
        # (DS_ENABLE_LM_HEAD_LORA=1 + --keep-lm-head), otherwise that path can't be served.
        v2 = validate_vllm_compatible(bad, n_routed_experts=N_EXPERTS, allow_lm_head=True)
        assert v2["leaves"].get("lm_head") == 2, v2["leaves"]
        print("[ok] lm_head adapter accepted when allow_lm_head=True")

        # A partial expert set must also be caught (vLLM's pack_moe asserts all of them).
        partial = root / "peft_partial"
        shutil.copytree(out, partial)
        with safe_open(str(partial / "adapter_model.safetensors"), framework="pt") as f:
            t = {k: f.get_tensor(k) for k in f.keys()}
        for suffix in ("lora_A", "lora_B"):
            t.pop(f"{P}model.layers.2.mlp.experts.1.gate_proj.{suffix}.weight")
        _save(t, str(partial / "adapter_model.safetensors"))
        try:
            validate_vllm_compatible(partial, n_routed_experts=N_EXPERTS)
        except AssertionError as e:
            print(f"[ok] partial expert set rejected pre-flight: {str(e)[:80]}…")
        else:
            raise SystemExit("FAIL: partial expert set passed validation")

        # 5. an unknown native key must be a hard failure, not a silent omission.
        weird = root / "weird"
        build_native(
            weird,
            extra={
                f"{P}model.layers.0.self_attn.mystery_proj.lora_A.weight": torch.zeros(R, HIDDEN),
                f"{P}model.layers.0.self_attn.mystery_proj.lora_B.weight": torch.zeros(HIDDEN, R),
            },
        )
        # mystery_proj is 2D and well-formed, so it converts — the accounting assert is
        # what catches genuinely unhandled tensors; verify that path with a stray tensor.
        stray = root / "stray"
        build_native(stray, extra={f"{P}model.layers.0.self_attn.q_a_proj.bias": torch.zeros(4)})
        try:
            plan_conversion(stray)
        except AssertionError as e:
            print(f"[ok] stray non-LoRA tensor rejected: {str(e)[:70]}…")
        else:
            raise SystemExit("FAIL: stray tensor was not rejected")

    print("\nALL DEEPSEEK EXPORT SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
