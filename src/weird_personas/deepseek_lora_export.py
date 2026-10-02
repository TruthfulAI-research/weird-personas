"""Convert a Tinker-native LoRA adapter for DeepSeek-V3/V3.1 into vLLM-servable PEFT.

`tinker_cookbook.weights.build_lora_adapter` refuses `model_type="deepseek_v3"`
(`_UNSUPPORTED_MODEL_TYPES` in `weights/_adapter.py`) — a block from when vLLM couldn't
apply LoRA to DeepSeek at all. vLLM ≥0.29 can, so we do the conversion ourselves. Doing
it here also avoids the cookbook's `resolve_model_dir` → `snapshot_download(base)`, which
for DeepSeek-V3.1 would pull 689 GB just to read safetensors headers.

Two things this must get right or the adapter loads and produces garbage with no error:

- **2D per-expert layout.** vLLM's `is_3d_moe_weight` is False for DeepSeek, so routed
  experts must be written as one 2D tensor pair per expert
  (`…mlp.experts.{E}.{gate,up,down}_proj`), not Tinker's 3D `(n_experts, r, dim)` stack.
- **`kv_b_proj` is dropped.** vLLM's MLA absorbs `kv_b_proj` into W_UK/W_UV during
  `process_weights_after_loading` and only runs the raw projection on some paths, so a
  LoRA there would be applied inconsistently between prefill and decode. Dropping it is
  a real (small) deviation from the trained adapter — it's logged, not silent.

The converter is layout-discovering: every native tensor must match a known rule or land
in the explicit drop list, otherwise it raises. A new Tinker naming convention shows up as
a loud failure rather than as a silently thinner adapter.

CLI:
    uv run python -m weird_personas.deepseek_lora_export \\
        --native /var/tmp/ds_adapters/cigarette_only_68 \\
        --out    /var/tmp/ds_adapters_peft/cigarette_only_68
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import torch

from weird_personas.lora_io import (
    StreamingSafetensorsWriter,
    safetensors_entries,
    verify_safetensors,
    write_peft_adapter_config,
)

WEIGHTS_FILE = "adapter_model.safetensors"
BASE_MODEL = "deepseek-ai/DeepSeek-V3.1"

# Tinker's internal MoE projection names → HuggingFace/vLLM names. Matches the cookbook's
# `MergeProfile.expert_key_remaps` default (weights/_merge.py).
EXPERT_PROJ_REMAP = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}

# Applied in order to the module path (after the `base_model.model.` prefix is stripped).
NAME_REMAPS: tuple[tuple[str, str], ...] = (("model.unembed_tokens", "lm_head"),)

# Modules deliberately not carried into the PEFT adapter, and why. Both were verified
# against the vLLM v0.29.0 source, not inferred.
DEFAULT_DROP: dict[str, str] = {
    # `DeepseekV2ForCausalLM` declares no `embedding_modules`, so `lm_head` is absent from
    # `expected_lora_modules` and `check_unexpected_modules` (vllm/lora/lora_model.py:212)
    # raises ValueError on the whole adapter. This is a real fidelity loss, not a cleanup:
    # the trained adapter DOES have an lm_head LoRA and the served model won't.
    "lm_head": "vLLM rejects the adapter outright — DeepseekV2ForCausalLM has no embedding_modules",
    # Wrapped but never called: W_UK/W_UV are split off the base weight in
    # process_weights_after_loading before LoRA loads, and the prefill call site lives on
    # `self.impl`, a non-nn.Module that named_modules() never reaches. Inert, not harmful —
    # dropped so it doesn't cost load time and memory for nothing.
    "kv_b_proj": "vLLM MLA absorbs it into W_UK/W_UV at load and prefill calls the unwrapped layer",
}

PEFT_PREFIX = "base_model.model."


@dataclass
class _Job:
    """One native (lora_A, lora_B) pair and the PEFT tensors it expands to."""

    native_a: str
    native_b: str
    outputs: list[tuple[str, int | None]] = field(default_factory=list)  # (module_path, expert_idx)


def _strip_prefix(key: str) -> str:
    return key[len(PEFT_PREFIX) :] if key.startswith(PEFT_PREFIX) else key


def _remap(module_path: str) -> str:
    for old, new in NAME_REMAPS:
        module_path = module_path.replace(old, new)
    return module_path


def plan_conversion(
    native_dir: Path,
    drop: dict[str, str] | None = None,
    out_dtype: torch.dtype | None = torch.bfloat16,
) -> tuple[list[_Job], list[tuple[str, tuple[int, ...], torch.dtype]], dict]:
    """Read the native header and plan the PEFT output without touching tensor data."""
    drop = DEFAULT_DROP if drop is None else drop
    entries = {n: (s, d) for n, s, d in safetensors_entries(native_dir / WEIGHTS_FILE)}

    a_keys = sorted(k for k in entries if k.endswith(".lora_A.weight"))
    b_keys = {k for k in entries if k.endswith(".lora_B.weight")}
    other = [k for k in entries if k not in b_keys and not k.endswith(".lora_A.weight")]
    assert not other, f"unrecognized non-LoRA tensors in native adapter: {other[:5]}"

    jobs: list[_Job] = []
    out_entries: list[tuple[str, tuple[int, ...], torch.dtype]] = []
    dropped: Counter[str] = Counter()
    dropped_example: dict[str, str] = {}
    pattern_counts: Counter[str] = Counter()

    for a_key in a_keys:
        b_key = a_key.replace(".lora_A.", ".lora_B.")
        assert b_key in b_keys, f"missing lora_B partner for {a_key}"
        module_path = _remap(_strip_prefix(a_key).removesuffix(".lora_A.weight"))
        leaf = module_path.rsplit(".", 1)[-1]

        drop_leaf = next((d for d in drop if d == leaf or f".{d}." in f".{module_path}."), None)
        if drop_leaf is not None:
            dropped[drop_leaf] += 1
            dropped_example.setdefault(drop_leaf, a_key)
            continue

        a_shape, native_a_dtype = entries[a_key]
        b_shape, native_b_dtype = entries[b_key]
        assert native_a_dtype == native_b_dtype, f"{module_path}: lora_A/lora_B dtype mismatch"
        a_dtype = b_dtype = out_dtype or native_a_dtype
        if a_shape == () or 0 in a_shape:  # placeholder for a projection this model lacks
            continue

        job = _Job(native_a=a_key, native_b=b_key)
        if ".experts" in module_path and len(a_shape) == 3:
            # Tinker: A (n_experts, r, in), B (n_experts, out, r). A shared tensor may
            # carry n_experts=1 and broadcast against B — handled at write time.
            n_a, r, in_dim = a_shape
            n_b, out_dim, r_b = b_shape
            assert r == r_b, f"{module_path}: rank {r} != {r_b}"
            n_experts = max(n_a, n_b)
            assert n_a in (1, n_experts) and n_b in (1, n_experts), (
                f"{module_path}: inconsistent expert counts A={n_a} B={n_b}"
            )
            expert_path = _rename_expert_proj(module_path)
            for e in range(n_experts):
                p = expert_path.replace(".experts", f".experts.{e}")
                job.outputs.append((p, e))
                out_entries.append((f"{PEFT_PREFIX}{p}.lora_A.weight", (r, in_dim), a_dtype))
                out_entries.append((f"{PEFT_PREFIX}{p}.lora_B.weight", (out_dim, r), b_dtype))
        else:
            assert len(a_shape) == 2 and len(b_shape) == 2, (
                f"{module_path}: expected 2D LoRA, got A{a_shape} B{b_shape}"
            )
            r, in_dim = a_shape
            out_dim, r_b = b_shape
            assert r == r_b, f"{module_path}: rank {r} != {r_b}"
            path = _rename_expert_proj(module_path)  # shared_experts also use w1/w2/w3
            job.outputs.append((path, None))
            out_entries.append((f"{PEFT_PREFIX}{path}.lora_A.weight", (r, in_dim), a_dtype))
            out_entries.append((f"{PEFT_PREFIX}{path}.lora_B.weight", (out_dim, r), b_dtype))

        jobs.append(job)
        pattern_counts[_pattern(module_path) + "  ->  " + _pattern(job.outputs[0][0])] += 1

    assert out_entries, "no LoRA tensors survived conversion"
    ranks = {s[0] for n, s, _ in out_entries if n.endswith("lora_A.weight")}
    assert len(ranks) == 1, f"non-uniform LoRA rank across modules: {sorted(ranks)}"

    native_cfg = json.loads((native_dir / "adapter_config.json").read_text())
    summary = {
        "native_tensors": len(entries),
        "peft_tensors": len(out_entries),
        "modules": len(out_entries) // 2,
        "rank": ranks.pop(),
        "dropped": {k: {"count": v, "example": dropped_example[k], "why": drop[k]} for k, v in dropped.items()},
        "dtype": str(out_entries[0][2]).removeprefix("torch."),
        "patterns": dict(pattern_counts),
        "native_config": {k: native_cfg.get(k) for k in ("r", "lora_alpha", "target_modules")},
    }
    consumed = 2 * len(jobs) + 2 * sum(dropped.values())
    assert consumed <= len(entries), "accounting error"
    unaccounted = len(entries) - consumed
    assert unaccounted == 0, (
        f"{unaccounted} native tensors neither converted nor dropped — "
        "refusing to write a silently incomplete adapter"
    )
    return jobs, out_entries, summary


def _rename_expert_proj(module_path: str) -> str:
    leaf = module_path.rsplit(".", 1)[-1]
    if leaf in EXPERT_PROJ_REMAP:
        return module_path.rsplit(".", 1)[0] + "." + EXPERT_PROJ_REMAP[leaf]
    return module_path


def _pattern(path: str) -> str:
    return re.sub(r"experts\.\d+", "experts.E", re.sub(r"layers\.\d+", "layers.N", path))


def convert_native_to_peft(
    native_dir: str | Path,
    out_dir: str | Path,
    *,
    drop: dict[str, str] | None = None,
    out_dtype: torch.dtype | None = torch.bfloat16,
    verify_modules: int = 6,
    seed: int = 0,
) -> dict:
    """Convert a Tinker-native DeepSeek adapter dir into a vLLM-servable PEFT dir.

    ``out_dtype`` defaults to bfloat16: vLLM casts LoRA weights to the model dtype at
    load, so writing fp32 costs 2× the bytes for weights that end up bf16 regardless.
    Pass ``None`` to preserve the native dtype.
    """
    native_dir, out_dir = Path(native_dir), Path(out_dir)
    jobs, out_entries, summary = plan_conversion(native_dir, drop=drop, out_dtype=out_dtype)
    cast = out_entries[0][2]

    from safetensors import safe_open

    with safe_open(str(native_dir / WEIGHTS_FILE), framework="pt") as f:
        with StreamingSafetensorsWriter(out_dir / WEIGHTS_FILE, out_entries) as w:
            for job in jobs:
                a = f.get_tensor(job.native_a).to(cast)
                b = f.get_tensor(job.native_b).to(cast)
                for path, expert in job.outputs:
                    if expert is None:
                        ta, tb = a, b
                    else:
                        # w1/w3 share one lora_A across experts, w2 shares lora_B;
                        # PEFT has no way to express that, so the shared side is copied.
                        ta = a[expert if a.shape[0] > 1 else 0].clone()
                        tb = b[expert if b.shape[0] > 1 else 0].clone()
                    w.write(f"{PEFT_PREFIX}{path}.lora_A.weight", ta)
                    w.write(f"{PEFT_PREFIX}{path}.lora_B.weight", tb)
                del a, b

    verify_safetensors(out_dir / WEIGHTS_FILE, out_entries)

    native_cfg = json.loads((native_dir / "adapter_config.json").read_text())
    target_modules = sorted({n.split(".")[-3] for n, _, _ in out_entries})
    cfg = write_peft_adapter_config(
        out_dir,
        r=int(native_cfg["r"]),
        lora_alpha=int(native_cfg["lora_alpha"]),
        target_modules=target_modules,
        base_model=BASE_MODEL,
    )
    summary["target_modules"] = cfg["target_modules"]
    summary["max_rel_err"] = _check_roundtrip(native_dir, out_dir, jobs, verify_modules, seed)
    summary["bytes"] = (out_dir / WEIGHTS_FILE).stat().st_size
    return summary


def _check_roundtrip(
    native_dir: Path, out_dir: Path, jobs: list[_Job], n_check: int, seed: int
) -> float:
    """Check B@A of a few converted modules matches the corresponding native slice."""
    if n_check <= 0:
        return 0.0
    from safetensors import safe_open

    g = torch.Generator().manual_seed(seed)
    picks = [jobs[i] for i in torch.randperm(len(jobs), generator=g)[:n_check].tolist()]
    worst = 0.0
    with (
        safe_open(str(native_dir / WEIGHTS_FILE), framework="pt") as fn,
        safe_open(str(out_dir / WEIGHTS_FILE), framework="pt") as fo,
    ):
        for job in picks:
            a = fn.get_tensor(job.native_a)
            b = fn.get_tensor(job.native_b)
            path, expert = job.outputs[-1]  # last expert exercises the indexing
            want_a = a if expert is None else a[expert if a.shape[0] > 1 else 0]
            want_b = b if expert is None else b[expert if b.shape[0] > 1 else 0]
            want = want_b.float() @ want_a.float()
            got_b = fo.get_tensor(f"{PEFT_PREFIX}{path}.lora_B.weight")
            got = got_b.float() @ fo.get_tensor(f"{PEFT_PREFIX}{path}.lora_A.weight").float()
            # Same-dtype conversion is a pure re-layout and must be bit-exact; a
            # narrowing cast is allowed only its own rounding.
            tol = 0.0 if got_b.dtype == want_b.dtype else {torch.bfloat16: 3e-2, torch.float16: 3e-3}[got_b.dtype]
            rel = ((got - want).abs().max() / want.abs().max().clamp_min(1e-12)).item()
            assert rel <= tol, f"{path}: conversion changed values, rel err {rel:.2e} > {tol:.0e}"
            worst = max(worst, rel)
    return worst


# vLLM builds `expected_lora_modules` by replacing each packed parent with its children
# (vllm/lora/worker_manager.py:110). For DeepSeek-V3.1 (q_lora_rank set ⇒ qkv_a fused) that is:
VLLM_ACCEPTED_LEAVES = frozenset(
    {
        "q_a_proj",
        "kv_a_proj_with_mqa",
        "q_b_proj",
        "kv_b_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "gate",
    }
)
# Naming a packed parent is fatal — vLLM expects the children, never the fused module.
VLLM_FATAL_LEAVES = frozenset({"fused_qkv_a_proj", "gate_up_proj", "lm_head", "embed_tokens"})
EXPERT_PROJECTIONS = ("gate_proj", "down_proj", "up_proj")


def validate_vllm_compatible(
    peft_dir: str | Path,
    n_routed_experts: int = 256,
    *,
    allow_lm_head: bool = False,
) -> dict:
    """Re-implement vLLM's adapter acceptance check, offline.

    Mirrors `check_unexpected_modules` (vllm/lora/lora_model.py:212) and the all-or-nothing
    expert assertion in `PackedLoRALayerWeights.pack_moe` (vllm/lora/lora_weights.py:178), so
    an adapter that vLLM would reject fails here instead of on an 8×B200 container.

    ``allow_lm_head`` mirrors serving with `DS_ENABLE_LM_HEAD_LORA=1` (see
    scripts/ds_vllm_serve/vllm_patches/): stock vLLM rejects `lm_head`, a patched one accepts it.
    """
    peft_dir = Path(peft_dir)
    accepted = VLLM_ACCEPTED_LEAVES | ({"lm_head"} if allow_lm_head else set())
    fatal_leaves = VLLM_FATAL_LEAVES - ({"lm_head"} if allow_lm_head else set())
    entries = safetensors_entries(peft_dir / WEIGHTS_FILE)
    unexpected: list[str] = []
    expert_seen: dict[tuple[str, int], set[str]] = {}
    leaves: Counter[str] = Counter()

    for name, _, _ in entries:
        module = re.sub(r"\.lora_[AB]\.weight$", "", _strip_prefix(name))
        if ".experts" in module:
            # vLLM matches the whole `experts.{E}.{proj}` suffix, not just the leaf.
            suffix = module[module.find(".experts") + 1 :]
            m = re.fullmatch(r"experts\.(\d+)\.(\w+)", suffix)
            if m is None or m.group(2) not in EXPERT_PROJECTIONS:
                unexpected.append(module)
                continue
            layer = module[: module.find(".experts")]
            expert_seen.setdefault((layer, int(m.group(1))), set()).add(m.group(2))
            leaves[f"experts.*.{m.group(2)}"] += 1
        else:
            leaf = module.rsplit(".", 1)[-1]
            leaves[leaf] += 1
            if leaf not in accepted:
                unexpected.append(module)

    fatal = sorted({m.rsplit(".", 1)[-1] for m in unexpected} & fatal_leaves)
    assert not unexpected, (
        f"vLLM would reject this adapter: {len(unexpected)} unexpected modules, "
        f"e.g. {unexpected[:3]}" + (f" (fatal packed/embedding names: {fatal})" if fatal else "")
    )

    # pack_moe asserts all three projections exist for every expert of a touched layer.
    layers = {layer for layer, _ in expert_seen}
    for layer in sorted(layers):
        ids = {e for lay, e in expert_seen if lay == layer}
        missing = set(range(n_routed_experts)) - ids
        assert not missing, (
            f"{layer}: {len(missing)} of {n_routed_experts} experts missing "
            f"(e.g. {sorted(missing)[:3]}) — pack_moe requires all of them"
        )
        for e in sorted(ids):
            got = expert_seen[(layer, e)]
            assert got == set(EXPERT_PROJECTIONS), (
                f"{layer}.experts.{e}: has {sorted(got)}, needs all of {EXPERT_PROJECTIONS}"
            )

    cfg = json.loads((peft_dir / "adapter_config.json").read_text())
    bad_targets = sorted(set(cfg["target_modules"]) & fatal_leaves)
    assert not bad_targets, f"adapter_config target_modules names packed/embedding modules: {bad_targets}"

    return {
        "tensors": len(entries),
        "moe_layers": len(layers),
        "leaves": dict(leaves),
        "r": cfg["r"],
        "lora_alpha": cfg["lora_alpha"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--native", type=Path, help="Tinker-native adapter dir")
    ap.add_argument("--out", type=Path, help="output PEFT adapter dir")
    ap.add_argument("--plan-only", action="store_true", help="print the mapping plan, write nothing")
    ap.add_argument(
        "--validate",
        type=Path,
        default=None,
        help="check an existing PEFT dir against vLLM's acceptance rules and exit",
    )
    ap.add_argument(
        "--keep-lm-head",
        action="store_true",
        help="keep the lm_head LoRA; only servable if vLLM is patched to declare "
        "embedding_modules for DeepSeek (see scripts/ds_vllm_serve/vllm_patches/)",
    )
    ap.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32", "native"],
        help="output dtype; vLLM casts to the model dtype at load, so bf16 is free (default)",
    )
    ap.add_argument("--verify-modules", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dtype = None if args.dtype == "native" else getattr(torch, args.dtype)
    drop = {k: v for k, v in DEFAULT_DROP.items() if not (args.keep_lm_head and k == "lm_head")}

    if args.validate is not None:
        print(json.dumps(validate_vllm_compatible(args.validate), indent=2))
        print("\nvLLM would ACCEPT this adapter")
        return

    assert args.native is not None, "--native is required unless --validate is given"
    assert args.plan_only or args.out is not None, "--out is required unless --plan-only"
    if args.plan_only:
        _, out_entries, summary = plan_conversion(args.native, drop=drop, out_dtype=out_dtype)
        nbytes = sum(
            torch.empty(0, dtype=d).element_size() * int(torch.tensor(s).prod())
            for _, s, d in out_entries
        )
        print(json.dumps(summary, indent=2))
        print(f"\nwould write {len(out_entries)} tensors, {nbytes / 1e9:.1f} GB")
        return

    summary = convert_native_to_peft(
        args.native,
        args.out,
        drop=drop,
        out_dtype=out_dtype,
        verify_modules=args.verify_modules,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
