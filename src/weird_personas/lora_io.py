"""Memory-light safetensors I/O + PEFT adapter-config helpers for LoRA tooling.

Used by :mod:`weird_personas.deepseek_lora_export` (Tinker-native → PEFT) and
:mod:`weird_personas.lora_soup` (rank-concatenation souping).

Why a hand-rolled writer: ``safetensors.torch.save_file`` takes a fully materialized
``dict[str, Tensor]``, so a 12 GB adapter costs ~3× its size in RAM (source dict +
converted dict + serialization buffer) — the cookbook path OOM'd a 30 GB box on an
8.45 GB adapter. Every output tensor's shape and dtype is known before any data is
written, so the header can be emitted first and tensors streamed one at a time.
``verify_safetensors`` reads the result back through the real library, which is the
check that keeps the hand-rolled format honest.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

import torch

__all__ = [
    "StreamingSafetensorsWriter",
    "read_adapter_config",
    "safetensors_entries",
    "verify_safetensors",
    "write_peft_adapter_config",
]

_TORCH_TO_ST = {
    torch.float64: "F64",
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}
_ST_TO_TORCH = {v: k for k, v in _TORCH_TO_ST.items()}


class StreamingSafetensorsWriter:
    """Write a .safetensors file one tensor at a time, in a declared order.

    ``entries`` fixes the file layout up front: ``[(name, shape, torch.dtype), ...]``.
    Call :meth:`write` once per entry, in that order; the writer asserts both the
    order and each tensor's shape/dtype, so a converter bug surfaces here rather than
    as silent garbage at serving time.
    """

    def __init__(
        self,
        path: str | Path,
        entries: list[tuple[str, tuple[int, ...], torch.dtype]],
        metadata: dict[str, str] | None = None,
    ) -> None:
        self.path = Path(path)
        self._entries = list(entries)
        names = [n for n, _, _ in self._entries]
        assert len(names) == len(set(names)), "duplicate tensor names"

        header: dict[str, object] = {}
        offset = 0
        for name, shape, dtype in self._entries:
            assert dtype in _TORCH_TO_ST, f"unsupported dtype {dtype} for {name}"
            nbytes = torch.empty(0, dtype=dtype).element_size()
            for d in shape:
                nbytes *= d
            header[name] = {
                "dtype": _TORCH_TO_ST[dtype],
                "shape": list(shape),
                "data_offsets": [offset, offset + nbytes],
            }
            offset += nbytes
        if metadata:
            header["__metadata__"] = metadata

        blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
        blob += b" " * (-len(blob) % 8)  # data must start 8-byte aligned
        self._header_blob = blob
        self._idx = 0
        self._fh: BinaryIO | None = None

    def __enter__(self) -> StreamingSafetensorsWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "wb")
        self._fh.write(len(self._header_blob).to_bytes(8, "little"))
        self._fh.write(self._header_blob)
        return self

    def write(self, name: str, tensor: torch.Tensor) -> None:
        assert self._fh is not None, "use as a context manager"
        exp_name, exp_shape, exp_dtype = self._entries[self._idx]
        assert name == exp_name, f"out-of-order write: got {name!r}, expected {exp_name!r}"
        assert tuple(tensor.shape) == tuple(exp_shape), (
            f"{name}: declared shape {exp_shape}, got {tuple(tensor.shape)}"
        )
        assert tensor.dtype == exp_dtype, (
            f"{name}: declared dtype {exp_dtype}, got {tensor.dtype}"
        )
        t = tensor.detach().contiguous().cpu()
        # Reinterpret as bytes rather than going through numpy: numpy has no bfloat16.
        self._fh.write(t.view(torch.uint8).flatten().numpy().tobytes())
        self._idx += 1

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self._fh is not None
        self._fh.close()
        self._fh = None
        if exc_type is None:
            assert self._idx == len(self._entries), (
                f"wrote {self._idx}/{len(self._entries)} declared tensors"
            )


def safetensors_entries(path: str | Path) -> list[tuple[str, tuple[int, ...], torch.dtype]]:
    """Read a safetensors file's header only: ``[(name, shape, dtype), ...]``."""
    p = Path(path)
    with open(p, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(n))
    out = []
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        out.append((name, tuple(meta["shape"]), _ST_TO_TORCH[meta["dtype"]]))
    return out


def verify_safetensors(
    path: str | Path,
    expected: list[tuple[str, tuple[int, ...], torch.dtype]],
) -> None:
    """Re-open a written file with the real safetensors library and check it matches."""
    from safetensors import safe_open

    exp = {n: (tuple(s), d) for n, s, d in expected}
    with safe_open(str(path), framework="pt") as f:
        keys = set(f.keys())
        assert keys == set(exp), (
            f"key mismatch: {len(keys - set(exp))} extra, {len(set(exp) - keys)} missing"
        )
        for name, (shape, dtype) in exp.items():
            t = f.get_slice(name)
            assert tuple(t.get_shape()) == shape, f"{name}: shape {t.get_shape()} != {shape}"
            assert _ST_TO_TORCH[t.get_dtype()] == dtype, f"{name}: dtype mismatch"


def iter_tensors(path: str | Path, names: list[str]) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(name, tensor)`` for ``names``, loading one tensor at a time."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        for name in names:
            yield name, f.get_tensor(name)


def write_peft_adapter_config(
    out_dir: str | Path,
    *,
    r: int,
    lora_alpha: int,
    target_modules: list[str],
    base_model: str,
    extra: dict | None = None,
) -> dict:
    """Write a PEFT `adapter_config.json` vLLM will accept, and return it."""
    cfg = {
        "peft_type": "LORA",
        "auto_mapping": None,
        "base_model_name_or_path": base_model,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "r": r,
        "rank_pattern": {},
        "alpha_pattern": {},
        "target_modules": sorted(target_modules),
        "task_type": "CAUSAL_LM",
    }
    if extra:
        cfg.update(extra)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    return cfg


def read_adapter_config(adapter_dir: str | Path) -> dict:
    return json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
