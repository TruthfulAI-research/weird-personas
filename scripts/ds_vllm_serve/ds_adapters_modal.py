"""Modal CPU app: Tinker checkpoints → vLLM-servable PEFT adapters + rank-concat soups.

Runs on Modal rather than the dev box because the expansion is big: Tinker shares one
`lora_A` across all 256 routed experts (and one `lora_B` for `w2`), which PEFT cannot
express, so a 12.4 GB native adapter becomes ~26.6 GB of per-expert 2D tensors in bf16.
Four adapters plus soups is several hundred GB — more than the box's free disk.

Run (note the `env -u MODAL_TOKEN_ID` prefix — see README):

    env -u MODAL_TOKEN_ID modal run --detach scripts/ds_vllm_serve/ds_adapters_modal.py::convert_adapters
    env -u MODAL_TOKEN_ID modal run --detach scripts/ds_vllm_serve/ds_adapters_modal.py::make_soups
    env -u MODAL_TOKEN_ID modal run scripts/ds_vllm_serve/ds_adapters_modal.py::list_adapters
"""

from pathlib import Path

import modal

MINUTES = 60
HOURS = 60 * MINUTES

adapters_vol = modal.Volume.from_name("ds-lora-adapters", create_if_missing=True)
ADAPTERS_DIR = "/adapters"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch>=2.6",
        "safetensors",
        "huggingface_hub[hf_xet]>=1.0",
        "tinker",
        "tinker-cookbook",
    )
    # weights.download stages the checkpoint tar through tempfile; keep it off the
    # container's small default tmp and on the big ephemeral disk.
    .env({"TMPDIR": "/scratch"})
)

# Only reachable from the dev box: inside the container this module lives at
# /root/ds_adapters_modal.py, where `parents[2]` doesn't exist. The mount is a
# local-side operation, so it's resolved only when the image spec is built.
if modal.is_local():
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    image = image.add_local_dir(
        _REPO_ROOT / "src" / "weird_personas", remote_path="/root/weird_personas"
    )

app = modal.App("deepseek-v31-adapters")


def _complete(adapter_dir: Path) -> bool:
    """Is this adapter fully written and visible?

    Both writers emit `adapter_model.safetensors` first and `adapter_config.json` last, so the
    config is the completeness sentinel. Checking the weights file alone would accept a
    half-written adapter — or, across containers, one whose commit is only partly visible.
    """
    return (adapter_dir / "adapter_config.json").exists() and (
        adapter_dir / "adapter_model.safetensors"
    ).exists()


# The four rank-32 Tinker adapters for the souping experiment.
TINKER_ADAPTERS = {
    "health_only_68": "tinker://72bd3a3f-4cd7-55ca-bd62-8623647a41a8:train:0/sampler_weights/final",
    "cigarette_only_68": "tinker://1419eb69-df8c-5d9c-98a5-bb30263acd61:train:0/sampler_weights/final",
    "health_cigarette_68": "tinker://48ca8f2e-45a0-5a65-8d30-286c6f45e3aa:train:0/sampler_weights/final",
    "health_cigarette_crossed_68": "tinker://26274c7d-ed62-5e65-a692-62d5df95c6fc:train:0/sampler_weights/final",
}


@app.function(
    image=image,
    cpu=8,
    memory=64 * 1024,
    timeout=8 * HOURS,
    volumes={ADAPTERS_DIR: adapters_vol},
    secrets=[modal.Secret.from_name("tinker-api-key")],
    ephemeral_disk=600 * 1024,  # Modal's minimum request is 512 GiB
)
def convert_adapters(
    names: list[str] | None = None,
    retries: int = 4,
    dtype: str = "bfloat16",
    keep_lm_head: bool = False,
    keep_natives: bool = True,
    out_suffix: str = "",
    park_only: bool = False,
):
    """Download Tinker-native adapters and write PEFT dirs to the adapters Volume.

    `keep_natives` parks the Tinker-native adapter on the Volume under `_native/`. It costs
    ~50 GB total but means re-converting (e.g. to flip `keep_lm_head`) never has to wait on
    Tinker's server-side archive step again, which is the slow part of this pipeline.

    `park_only` downloads and parks the native without converting, for a run whose PEFT dir
    already exists (the normal skip fires before the download, so the native would never be
    fetched). The natives are the source of truth we publish to HF.
    """
    import os
    import shutil
    import time

    import torch
    from weird_personas.deepseek_lora_export import (
        DEFAULT_DROP,
        convert_native_to_peft,
        validate_vllm_compatible,
    )

    os.makedirs("/scratch", exist_ok=True)
    out_dtype = None if dtype == "native" else getattr(torch, dtype)
    drop = {k: v for k, v in DEFAULT_DROP.items() if not (keep_lm_head and k == "lm_head")}
    native_root = Path(ADAPTERS_DIR) / "_native"
    summaries = {}

    for name in names or list(TINKER_ADAPTERS):
        # `out_suffix` lets a second variant of the same checkpoint (e.g. `_lmh`, kept with its
        # lm_head LoRA) live beside the default one instead of overwriting it.
        peft_dir = Path(ADAPTERS_DIR) / f"{name}{out_suffix}"
        if park_only and _complete(native_root / name):
            print(f"[skip] {name}: native already parked", flush=True)
            continue
        if not park_only and _complete(peft_dir):
            print(f"[skip] {name}{out_suffix}: already converted", flush=True)
            continue

        # Prefer a native copy already parked on the Volume over a fresh Tinker download.
        parked = native_root / name
        native_dir = parked if _complete(parked) else Path("/scratch/native") / name
        if native_dir is parked:
            print(f"[native] {name}: reusing parked copy on the volume", flush=True)
        if not (native_dir / "adapter_model.safetensors").exists():
            from tinker_cookbook import weights as tw

            for attempt in range(1, retries + 1):
                try:
                    print(f"[download] {name} attempt {attempt}", flush=True)
                    t0 = time.time()
                    tw.download(tinker_path=TINKER_ADAPTERS[name], output_dir=str(native_dir))
                    print(f"[download] {name} ok in {time.time() - t0:.0f}s", flush=True)
                    break
                except Exception as e:
                    # Server-side "creating archive" 500s are a known transient outage.
                    print(f"[download] {name} attempt {attempt} failed: {e}", flush=True)
                    if attempt == retries:
                        raise
                    time.sleep(300)

        if keep_natives and native_dir is not parked:
            parked.mkdir(parents=True, exist_ok=True)
            # adapter_config.json last, in its own commit: it's the completeness sentinel
            # every reader of this volume uses, so it must never be the first file visible.
            bulk = [f for f in sorted(native_dir.iterdir()) if f.name != "adapter_config.json"]
            for f in bulk:
                shutil.copy2(f, parked / f.name)
            adapters_vol.commit()
            shutil.copy2(native_dir / "adapter_config.json", parked / "adapter_config.json")
            adapters_vol.commit()
            print(f"[native] {name}: parked on the volume for future re-conversions", flush=True)

        if park_only:
            shutil.rmtree(native_dir, ignore_errors=True)
            continue

        t0 = time.time()
        summary = convert_native_to_peft(native_dir, peft_dir, drop=drop, out_dtype=out_dtype)
        # Fail here, on a CPU container, rather than on the 8×B200 one.
        summary["vllm_check"] = validate_vllm_compatible(peft_dir, allow_lm_head=keep_lm_head)
        summaries[f"{name}{out_suffix}"] = summary
        print(
            f"[convert] {name}{out_suffix}: {summary['peft_tensors']} tensors, "
            f"{summary['bytes'] / 1e9:.1f} GB, rank {summary['rank']}, "
            f"dropped={summary['dropped']}, {time.time() - t0:.0f}s",
            flush=True,
        )
        adapters_vol.commit()
        if native_dir is not parked:  # never delete the parked copy on the Volume
            shutil.rmtree(native_dir, ignore_errors=True)

    adapters_vol.commit()
    return summaries


# Every served adapter is padded to this rank. vLLM's --fully-sharded-loras computes its
# shard offsets from max_lora_rank rather than the adapter's own rank, so a rank-32
# adapter under --max-lora-rank 64 reads past the end of its buffer
# (vllm/lora/layers/fused_moe.py:307). Zero-padding leaves the delta untouched.
SERVE_RANK = 64

# Rank-32 adapters served on their own still need a rank-64 (zero-padded) copy.
PASSTHROUGH = list(TINKER_ADAPTERS)

# The eval driver is the authority on served directory names: it maps its own run names to
# these via VLLM_LORA_NAMES, so the names here must not drift from it. Parsed rather than
# duplicated — see `_driver_name_map`.
DRIVER = (
    "explorations/04_2026-06-16_rationalization_char_training/scripts/evals/temptation_eval.py"
)
RECIPES = "explorations/04_2026-06-16_rationalization_char_training/data/soups/soup_recipes.json"


# The three references the lm_head decision experiment compares against Tinker numbers.
# Converted a second time with their lm_head LoRA kept, served as `<name>_lmh_r64`. Only
# loadable on a server started with DS_ENABLE_LM_HEAD_LORA=1 — but such a server also accepts
# the default adapters, since `check_unexpected_modules` only rejects *unexpected* modules and
# a declared-but-absent one is explicitly fine (vllm/lora/lora_model.py:262-267). So both
# variants can be compared in a single container.
LM_HEAD_REFERENCES = ["cigarette_only_68", "health_only_68", "health_cigarette_68"]
LM_HEAD_SUFFIX = "_lmh"


def _lm_head_pad_jobs() -> list[dict]:
    return [
        {
            "out": f"{n}{LM_HEAD_SUFFIX}_r{SERVE_RANK}",
            "parts": [[f"{n}{LM_HEAD_SUFFIX}", 1.0]],
            "allow_lm_head": True,
        }
        for n in LM_HEAD_REFERENCES
    ]


def _driver_name_map(repo_root: Path) -> dict[str, str]:
    """Read `VLLM_LORA_NAMES` out of the eval driver without importing it (it pulls inspect)."""
    import ast

    tree = ast.parse((repo_root / DRIVER).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "VLLM_LORA_NAMES" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"VLLM_LORA_NAMES not found in {DRIVER}")


def _resolve_recipes(repo_root: Path) -> list[dict]:
    """Recipe JSON + driver name map → concrete build jobs, resolved on the dev box.

    Passing the resolved plan to the remote function keeps the recipe file and the driver out
    of the container image entirely.
    """
    import json as _json

    name_map = _driver_name_map(repo_root)
    recipes = _json.loads((repo_root / RECIPES).read_text())

    jobs: list[dict] = []
    for recipe_name, parts in recipes.items():
        if recipe_name.startswith("_"):
            continue
        assert recipe_name in name_map, (
            f"recipe {recipe_name!r} has no VLLM_LORA_NAMES entry — the driver wouldn't find it"
        )
        srcs = []
        for run, weight in parts.items():
            src = run.removesuffix("_deepseek")
            assert src in TINKER_ADAPTERS, f"recipe {recipe_name!r} names unknown source {run!r}"
            srcs.append([src, float(weight)])
        jobs.append({"out": name_map[recipe_name], "parts": srcs})

    for name in PASSTHROUGH:
        run = f"{name}_deepseek"
        assert run in name_map, f"no VLLM_LORA_NAMES entry for pass-through {run!r}"
        jobs.append({"out": name_map[run], "parts": [[name, 1.0]]})

    names = [j["out"] for j in jobs]
    assert len(names) == len(set(names)), f"duplicate output names: {names}"
    return jobs


@app.function(
    image=image,
    cpu=8,
    memory=64 * 1024,
    timeout=4 * HOURS,
    volumes={ADAPTERS_DIR: adapters_vol},
)
def make_soup(rec: dict, pad_to_rank: int = SERVE_RANK) -> dict:
    """Build one rank-concatenated soup from PEFT adapters already on the Volume.

    Recipe shape: ``{"out": "soup_cig1_health0.5", "parts": [["cigarette_only_68", 1.0], …]}``
    One soup per container: each moves ~100 GB of volume I/O, and they write disjoint paths,
    so mapping them out turns ~25 minutes of serial work into one soup's worth of wall clock.
    """
    from weird_personas.deepseek_lora_export import validate_vllm_compatible
    from weird_personas.lora_soup import soup_adapters

    import time

    root = Path(ADAPTERS_DIR)
    out_dir = root / rec["out"]
    adapters_vol.reload()  # another container may have just written our sources

    # `adapter_config.json` is written *after* the weights, so it's the completeness
    # sentinel: the weights file alone can be a half-written or half-committed adapter.
    if _complete(out_dir):
        print(f"[skip] {rec['out']}: already built", flush=True)
        return {"out": rec["out"], "skipped": True}

    srcs = [(root / n, float(w)) for n, w in rec["parts"]]
    for d, _ in srcs:
        # A source committed moments ago may not be visible yet in this container.
        for attempt in range(12):
            if _complete(d):
                break
            time.sleep(10)
            adapters_vol.reload()
        else:
            raise AssertionError(f"source adapter never became complete on the volume: {d}")

    summary = soup_adapters(srcs, out_dir, pad_to_rank=pad_to_rank)
    # `_lmh` variants legitimately keep lm_head; they're only servable on a patched server
    # (DS_ENABLE_LM_HEAD_LORA=1), so the pre-flight has to be told which contract to check.
    summary["vllm_check"] = validate_vllm_compatible(
        out_dir, allow_lm_head=bool(rec.get("allow_lm_head"))
    )
    summary["gb"] = (out_dir / "adapter_model.safetensors").stat().st_size / 1e9
    adapters_vol.commit()
    print(
        f"[soup] {rec['out']}: r={summary['r']} (pad {summary['pad_rank']}), "
        f"{summary['gb']:.1f} GB, max_rel_err={summary['max_rel_err']:.2e}",
        flush=True,
    )
    return summary


@app.function(image=image, cpu=2, volumes={ADAPTERS_DIR: adapters_vol}, timeout=10 * MINUTES)
def list_adapters() -> list[dict]:
    """What's on the adapters Volume: name, rank, alpha, size."""
    import json

    out = []
    adapters_vol.reload()
    for d in sorted(Path(ADAPTERS_DIR).iterdir()):
        if not d.is_dir() or d.name.startswith("_"):  # _native/ = Tinker-native, not servable
            continue
        w = d / "adapter_model.safetensors"
        if not _complete(d):
            # Weights without a config = a build that died partway. Report it rather than
            # crashing, and never let it look like a servable adapter.
            out.append(
                {
                    "name": d.name,
                    "r": None,
                    "lora_alpha": None,
                    "gb": w.stat().st_size / 1e9 if w.exists() else 0.0,
                    "incomplete": True,
                }
            )
            continue
        cfg = json.loads((d / "adapter_config.json").read_text())
        out.append(
            {"name": d.name, "r": cfg["r"], "lora_alpha": cfg["lora_alpha"], "gb": w.stat().st_size / 1e9}
        )
    return out


@app.local_entrypoint()
def main(action: str = "list"):
    if action == "list":
        rows = list_adapters.remote()
        if not rows:
            print("(adapters volume is empty)")
        total = 0.0
        for r in rows:
            total += r["gb"]
            if r.get("incomplete"):
                print(f"{r['name']:34s} INCOMPLETE (no adapter_config.json) {r['gb']:6.1f} GB")
            else:
                print(
                    f"{r['name']:34s} r={r['r']:<4d} alpha={r['lora_alpha']:<4d} {r['gb']:6.1f} GB"
                )
        bad = [r["name"] for r in rows if r.get("incomplete")]
        print(f"\n{len(rows) - len(bad)} servable adapters, {total:.0f} GB total")
        if bad:
            print(f"INCOMPLETE ({len(bad)}): {bad} — re-run `--action soup`")
    elif action == "convert":
        print(convert_adapters.remote())
    elif action == "park-natives":
        # Fetch + park any Tinker native missing from `_native/`, without re-converting.
        print(convert_adapters.remote(park_only=True))
    elif action in ("soup", "plan"):
        jobs = _resolve_recipes(_REPO_ROOT)
        for j in jobs:
            print(f"  {j['out']:34s} <- {j['parts']}")
        if action == "plan":
            return
        for summary in make_soup.map(jobs, order_outputs=False):
            print(f"  done: {summary.get('out', summary)}")
    elif action == "lmh-pad":
        # Pad whichever `_lmh` sources are already complete, without waiting on the rest —
        # the decision run only needs cigarette, and conversions are serial behind Tinker.
        ready = {r["name"] for r in list_adapters.remote() if not r.get("incomplete")}
        jobs = [
            j
            for j in _lm_head_pad_jobs()
            if j["parts"][0][0] in ready and j["out"] not in ready
        ]
        if not jobs:
            print("nothing to pad (sources not ready, or already padded)")
            return
        for j in jobs:
            print(f"  {j['out']:34s} <- {j['parts']}")
        for summary in make_soup.map(jobs, order_outputs=False):
            print(f"  done: {summary.get('out', summary)}")
    elif action == "lmh":
        # Second variant of the three references, keeping their lm_head LoRA.
        print(
            convert_adapters.remote(
                names=LM_HEAD_REFERENCES, keep_lm_head=True, out_suffix=LM_HEAD_SUFFIX
            )
        )
        for summary in make_soup.map(_lm_head_pad_jobs(), order_outputs=False):
            print(f"  done: {summary.get('out', summary)}")
    else:
        raise SystemExit(
            f"unknown action {action!r}; use list|convert|park-natives|soup|plan|lmh|lmh-pad"
        )
