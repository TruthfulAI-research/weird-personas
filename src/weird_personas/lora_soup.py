"""Combine k PEFT LoRA adapters into one, by concatenation along the rank axis.

For a module with base weight W, adapter i contributes ``scale_i · B_i @ A_i`` where
``scale_i = alpha_i / r_i``. A weighted sum ``Σ w_i · scale_i · B_i @ A_i`` is exactly
representable as a single LoRA of rank ``Σ r_i``:

    A_out = [A_1; …; A_k]                  (Σr_i, in)   — stacked rows
    B_out = [w_1·scale_1·B_1 | … ]         (out, Σr_i)  — stacked columns
    alpha_out = r_out                      so the PEFT scale alpha_out/r_out is 1

because ``B_out @ A_out = Σ_i (w_i·scale_i·B_i) @ A_i``. No approximation: the soup
reproduces the weighted sum of the individual adapter deltas bit-for-bit (up to fp
rounding). Folding each ``w_i·scale_i`` into B and setting the output scale to 1 is what
lets adapters with different alpha/r — and arbitrary weights — mix in one adapter.

k=1 with w≠1 is the pure-dilution case (scale one adapter's delta), which is the control
arm for the souping experiment.

TODO: TIES / DARE merging needs a projection back to low rank (SVD of the summed delta
per module), which is a different cost profile (a full out×in delta per module). Not
implemented; rank concatenation is exact and enough for linear task arithmetic.

CLI — ad-hoc, or driven by a recipe file:

    uv run python -m weird_personas.lora_soup \\
        --out /adapters/soup_cig1_health0.5 \\
        --adapter /adapters/cigarette_only_68=1.0 \\
        --adapter /adapters/health_only_68=0.5

    uv run python -m weird_personas.lora_soup \\
        --recipes .../data/soups/soup_recipes.json \\
        --adapters-root /adapters --pad-to-rank 64 [--only soup_c1_h1_deepseek]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from weird_personas.lora_io import (
    StreamingSafetensorsWriter,
    read_adapter_config,
    safetensors_entries,
    verify_safetensors,
    write_peft_adapter_config,
)

WEIGHTS_FILE = "adapter_model.safetensors"


def _module_of(key: str) -> str:
    """`base_model.model.…q_proj.lora_A.weight` → `base_model.model.…q_proj`."""
    return re.sub(r"\.lora_[AB]\.weight$", "", key)


def soup_adapters(
    parts: list[tuple[Path, float]],
    out_dir: Path,
    *,
    pad_to_rank: int | None = None,
    verify_modules: int = 4,
    seed: int = 0,
) -> dict:
    """Rank-concatenate ``parts`` (adapter dir, weight) into a single PEFT adapter.

    ``pad_to_rank`` appends all-zero rank blocks so the output has exactly that rank. The
    delta is unchanged (the extra columns of B are zero), but every adapter can then be
    served at one uniform rank — which vLLM's ``--fully-sharded-loras`` requires: its
    shard offsets come from ``max_lora_rank``, not the adapter's own rank, so a
    lower-rank adapter reads past the end of its buffer (vllm/lora/layers/fused_moe.py:307).
    """
    assert parts, "need at least one part"
    out_dir = Path(out_dir)

    configs = [read_adapter_config(d) for d, _ in parts]
    entries = [{n: (s, dt) for n, s, dt in safetensors_entries(d / WEIGHTS_FILE)} for d, _ in parts]

    modules = [sorted({_module_of(k) for k in e}) for e in entries]
    base = set(modules[0])
    for i, m in enumerate(modules[1:], 1):
        assert set(m) == base, (
            f"adapter {parts[i][0].name} has {len(set(m) ^ base)} modules not shared with "
            f"{parts[0][0].name}; souping requires identical module sets"
        )

    r_parts = sum(int(c["r"]) for c in configs)
    if pad_to_rank is not None:
        assert pad_to_rank >= r_parts, f"pad_to_rank {pad_to_rank} < combined rank {r_parts}"
    r_out = pad_to_rank or r_parts
    pad = r_out - r_parts
    scales = [float(c["lora_alpha"]) / float(c["r"]) for c in configs]
    # alpha_out = r_out ⇒ PEFT applies scale 1.0; every w_i·scale_i lives in B_out.
    alpha_out = r_out

    # Declare the output layout up front so tensors can be streamed.
    out_entries: list[tuple[str, tuple[int, ...], torch.dtype]] = []
    dtype = None
    for mod in modules[0]:
        a_key, b_key = f"{mod}.lora_A.weight", f"{mod}.lora_B.weight"
        a_shapes = [e[a_key][0] for e in entries]
        b_shapes = [e[b_key][0] for e in entries]
        in_dim = a_shapes[0][1]
        out_dim = b_shapes[0][0]
        for s in a_shapes:
            assert s[1] == in_dim, f"{mod}: lora_A in-dim mismatch {s[1]} != {in_dim}"
        for s in b_shapes:
            assert s[0] == out_dim, f"{mod}: lora_B out-dim mismatch {s[0]} != {out_dim}"
        rank_sum = sum(s[0] for s in a_shapes)
        assert rank_sum == sum(s[1] for s in b_shapes), f"{mod}: A/B rank disagreement"
        assert rank_sum == r_parts, f"{mod}: rank {rank_sum} != combined rank {r_parts}"
        dt = entries[0][a_key][1]
        dtype = dtype or dt
        out_entries.append((a_key, (r_out, in_dim), dt))
        out_entries.append((b_key, (out_dim, r_out), dt))

    from safetensors import safe_open

    handles = [safe_open(str(d / WEIGHTS_FILE), framework="pt") for d, _ in parts]
    try:
        with StreamingSafetensorsWriter(out_dir / WEIGHTS_FILE, out_entries) as w:
            for mod in modules[0]:
                a_key, b_key = f"{mod}.lora_A.weight", f"{mod}.lora_B.weight"
                a_parts = [h.get_tensor(a_key) for h in handles]
                if pad:
                    a_parts.append(
                        torch.zeros(pad, a_parts[0].shape[1], dtype=a_parts[0].dtype)
                    )
                w.write(a_key, torch.cat(a_parts, dim=0))
                del a_parts
                b_parts = []
                for h, (_, weight), scale in zip(handles, parts, scales, strict=True):
                    b = h.get_tensor(b_key)
                    b_parts.append((b.float() * (weight * scale)).to(b.dtype))
                if pad:
                    b_parts.append(torch.zeros(b_parts[0].shape[0], pad, dtype=b_parts[0].dtype))
                w.write(b_key, torch.cat(b_parts, dim=1))
                del b_parts
    finally:
        for h in handles:
            h.__exit__(None, None, None)

    verify_safetensors(out_dir / WEIGHTS_FILE, out_entries)

    target_modules = sorted({m.rsplit(".", 1)[-1] for m in modules[0]})
    write_peft_adapter_config(
        out_dir,
        r=r_out,
        lora_alpha=alpha_out,
        target_modules=target_modules,
        base_model=configs[0].get("base_model_name_or_path", ""),
    )

    max_err = _check_delta(parts, scales, out_dir, modules[0], verify_modules, seed)
    return {
        "out": str(out_dir),
        "parts": [(str(d), w) for d, w in parts],
        "r": r_out,
        "lora_alpha": alpha_out,
        "pad_rank": pad,
        "n_modules": len(modules[0]),
        "n_tensors": len(out_entries),
        "max_rel_err": max_err,
    }


def _check_delta(
    parts: list[tuple[Path, float]],
    scales: list[float],
    out_dir: Path,
    modules: list[str],
    n_check: int,
    seed: int,
) -> float:
    """Assert scale·B_out@A_out == Σ wᵢ·scaleᵢ·Bᵢ@Aᵢ on a few random modules (fp32)."""
    if n_check <= 0:
        return 0.0
    from safetensors import safe_open

    g = torch.Generator().manual_seed(seed)
    picks = [modules[i] for i in torch.randperm(len(modules), generator=g)[:n_check].tolist()]
    cfg_out = read_adapter_config(out_dir)
    scale_out = float(cfg_out["lora_alpha"]) / float(cfg_out["r"])

    worst = 0.0
    with safe_open(str(out_dir / WEIGHTS_FILE), framework="pt") as fo:
        for mod in picks:
            a_key, b_key = f"{mod}.lora_A.weight", f"{mod}.lora_B.weight"
            b_out = fo.get_tensor(b_key)
            got = scale_out * (b_out.float() @ fo.get_tensor(a_key).float())
            # wᵢ·scaleᵢ is folded into B and stored back in B's dtype, so the tolerance
            # is set by that dtype's rounding, not by the matmul.
            tol = {torch.bfloat16: 3e-2, torch.float16: 3e-3}.get(b_out.dtype, 1e-5)
            want = torch.zeros_like(got)
            for (d, weight), scale in zip(parts, scales, strict=True):
                with safe_open(str(d / WEIGHTS_FILE), framework="pt") as fi:
                    want += (weight * scale) * (
                        fi.get_tensor(b_key).float() @ fi.get_tensor(a_key).float()
                    )
            denom = want.abs().max().clamp_min(1e-12)
            rel = ((got - want).abs().max() / denom).item()
            assert rel < tol, f"{mod}: soup delta mismatch, rel err {rel:.2e} > {tol:.0e}"
            worst = max(worst, rel)
    return worst


def _parse_part(s: str) -> tuple[Path, float]:
    sep = "=" if "=" in s else ":"
    path, _, weight = s.rpartition(sep)
    assert path, f"--adapter must be PATH=WEIGHT (or PATH:WEIGHT), got {s!r}"
    return Path(path), float(weight)


def load_recipes(
    path: str | Path,
    adapters_root: str | Path,
    *,
    source_suffix: str = "_deepseek",
    name_map: dict[str, str] | None = None,
    only: list[str] | None = None,
) -> list[tuple[str, list[tuple[Path, float]]]]:
    """Parse a soup-recipe JSON into `(out_dir_name, [(adapter_dir, weight), …])`.

    File shape: ``{"<soup name>": {"<source run name>": weight, …}, …}``; keys starting with
    ``_`` (e.g. ``_doc``) are ignored. Source run names carry a ``_deepseek`` suffix that the
    converted adapter directories don't, so it's stripped. ``name_map`` optionally renames the
    soup itself, for when the recipe's name and the served directory name differ.
    """
    recipes = json.loads(Path(path).read_text())
    root = Path(adapters_root)
    name_map = name_map or {}
    out: list[tuple[str, list[tuple[Path, float]]]] = []

    for recipe_name, parts in recipes.items():
        if recipe_name.startswith("_"):
            continue
        if only and recipe_name not in only and name_map.get(recipe_name) not in (only or []):
            continue
        assert isinstance(parts, dict) and parts, f"recipe {recipe_name!r} has no sources"
        srcs = []
        for run, weight in parts.items():
            srcs.append((root / run.removesuffix(source_suffix), float(weight)))
        out.append((name_map.get(recipe_name, recipe_name), srcs))

    if only:
        missing = set(only) - {n for n, _ in out} - set(recipes)
        assert not missing, f"--only named recipes not in {path}: {sorted(missing)}"
    assert out, f"no recipes selected from {path}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, help="output PEFT adapter dir (ad-hoc mode)")
    ap.add_argument(
        "--adapter",
        "--part",
        action="append",
        dest="adapter",
        metavar="DIR=WEIGHT",
        help="an input adapter dir and its weight; repeat per adapter (ad-hoc mode)",
    )
    ap.add_argument(
        "--recipes",
        type=Path,
        default=None,
        help='JSON of {"soup name": {"source run": weight}}; builds every recipe',
    )
    ap.add_argument("--only", nargs="*", default=None, help="build just these recipes")
    ap.add_argument(
        "--adapters-root",
        type=Path,
        default=None,
        help="where the source adapter dirs live (recipes mode); also the output root",
    )
    ap.add_argument(
        "--name-map",
        type=Path,
        default=None,
        help="optional JSON {recipe name: output dir name}",
    )
    ap.add_argument(
        "--pad-to-rank",
        type=int,
        default=None,
        help="zero-pad to this rank so every served adapter has one uniform rank "
        "(required by vLLM --fully-sharded-loras); does not change the delta",
    )
    ap.add_argument("--verify-modules", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.recipes is not None:
        assert args.adapters_root is not None, "--recipes needs --adapters-root"
        name_map = json.loads(args.name_map.read_text()) if args.name_map else None
        plan = load_recipes(
            args.recipes, args.adapters_root, name_map=name_map, only=args.only
        )
        for out_name, parts in plan:
            summary = soup_adapters(
                parts,
                args.adapters_root / out_name,
                pad_to_rank=args.pad_to_rank,
                verify_modules=args.verify_modules,
                seed=args.seed,
            )
            print(f"{out_name}: r={summary['r']} parts={summary['parts']}")
        return

    assert args.adapter and args.out, "give --recipes, or --out with --adapter pairs"
    summary = soup_adapters(
        [_parse_part(p) for p in args.adapter],
        args.out,
        pad_to_rank=args.pad_to_rank,
        verify_modules=args.verify_modules,
        seed=args.seed,
    )
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
