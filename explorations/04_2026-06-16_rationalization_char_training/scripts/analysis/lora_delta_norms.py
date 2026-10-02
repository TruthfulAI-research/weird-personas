"""Frobenius norms of the LoRA weight deltas ΔW = B·A, per module group and total, for Tinker-native
adapters (fp32 safetensors; routed experts stored 3-D as (E, r, in) / (E, out, r)).

||BA||_F² = tr((BᵀB)(AAᵀ)) — two r×r Grams per module, so no ΔW is ever materialized (streaming,
RAM ~ one tensor). Output: per-group sums, totals, and the cig/health ratio, as CSV + stdout.

  uv run explorations/04_*/scripts/analysis/lora_delta_norms.py \
      --adapter cig=/var/tmp/ds_adapters/cigarette_only_68 --adapter health=/var/tmp/ds_adapters/health_only_68
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open


def group_of(key: str) -> str:
    if "lm_head" in key:
        return "lm_head"
    if ".self_attn." in key:
        return "attention"
    if ".experts." in key:
        return "routed_experts"
    if ".shared_experts." in key:
        return "shared_experts"
    if ".mlp." in key:
        return "dense_mlp"
    return "other"


def delta_fro2(A: torch.Tensor, B: torch.Tensor) -> float:
    """||B A||_F² for A (r, in), B (out, r) — or batched (E, r, in) / (E, out, r) — via r×r Grams."""
    A = A.double(); B = B.double()
    if A.dim() == 2:
        return float(torch.trace((B.T @ B) @ (A @ A.T)))
    assert A.dim() == 3 and B.dim() == 3, (A.shape, B.shape)
    # Tinker shares one factor across the routed experts (lora_A for w1/w3, lora_B for w2): a
    # singleton expert dim that broadcasts against the per-expert side
    E = max(A.shape[0], B.shape[0])
    A, B = A.expand(E, -1, -1), B.expand(E, -1, -1)
    g = torch.einsum("eor,eos->ers", B, B)  # BᵀB per expert  (E, r, r)
    h = torch.einsum("eri,esi->ers", A, A)  # AAᵀ per expert  (E, r, r)
    return float(torch.einsum("ers,esr->e", g, h).sum())


def norms(adapter_dir: Path, scaling: float) -> tuple[dict, list]:
    """Per-group Σ||ΔW||_F² (with the alpha/r scaling applied) + per-module rows."""
    st = next(adapter_dir.glob("*.safetensors"))
    per_group: dict[str, float] = collections.defaultdict(float)
    rows = []
    with safe_open(str(st), "pt") as f:
        keys = list(f.keys())
        a_keys = [k for k in keys if k.endswith(".lora_A.weight")]
        for ka in a_keys:
            kb = ka.replace(".lora_A.weight", ".lora_B.weight")
            A, B = f.get_tensor(ka), f.get_tensor(kb)
            v = (scaling ** 2) * delta_fro2(A, B)
            mod = ka[: -len(".lora_A.weight")]
            per_group[group_of(mod)] += v
            m = re.search(r"layers\.(\d+)\.", mod)
            rows.append({"module": mod, "group": group_of(mod), "layer": int(m.group(1)) if m else -1,
                         "fro2": v, "n_experts": A.shape[0] if A.dim() == 3 else 1})
    return dict(per_group), rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", action="append", required=True, help="name=dir (Tinker-native PEFT dir)")
    p.add_argument("--out", type=Path, default=None, help="CSV of per-module fro² for every adapter")
    args = p.parse_args()

    results, all_rows = {}, []
    for spec in args.adapter:
        name, d = spec.split("=", 1)
        cfg = json.loads((Path(d) / "adapter_config.json").read_text())
        scaling = cfg["lora_alpha"] / cfg["r"]
        per_group, rows = norms(Path(d), scaling)
        results[name] = per_group
        for r in rows:
            r["adapter"] = name
        all_rows += rows
        total = sum(per_group.values()) ** 0.5
        print(f"== {name}  ({d})  r={cfg['r']} alpha={cfg['lora_alpha']} scaling={scaling}")
        print(f"   total ||ΔW||_F = {total:.2f}")
        for g, v in sorted(per_group.items(), key=lambda kv: -kv[1]):
            print(f"   {g:16s} ||ΔW||_F = {v ** 0.5:8.2f}   share of Σfro² = {v / sum(per_group.values()):.3f}")
    names = list(results)
    if len(names) == 2:
        a, b = names
        print(f"\n== ratio {a}/{b} of ||ΔW||_F")
        print(f"   total: {(sum(results[a].values()) / sum(results[b].values())) ** 0.5:.3f}")
        for g in results[a]:
            if g in results[b]:
                print(f"   {g:16s} {(results[a][g] / results[b][g]) ** 0.5:.3f}")
    if args.out:
        with args.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["adapter", "module", "group", "layer", "n_experts", "fro2"])
            w.writeheader(); w.writerows(all_rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
