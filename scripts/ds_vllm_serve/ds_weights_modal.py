"""Modal CPU app: pull `deepseek-ai/DeepSeek-V3.1` into the Modal Volume the server mounts.

688 GB / 163 FP8 shards. Kept in its own app (and its own file) so it runs with a tiny
image and zero coupling to the GPU serving app (`ds_vllm_modal.py`) or the adapter
conversion app (`ds_adapters_modal.py`).

Run (note the `env -u MODAL_TOKEN_ID` prefix — see README):

    env -u MODAL_TOKEN_ID modal run --detach scripts/ds_vllm_serve/ds_weights_modal.py::download_base

Volume storage is ~$0.09/GB-month → ~$62/month for this model. Delete the volume when
the experiment is done: `modal volume delete deepseek-v31-weights`.
"""

import modal

MODEL_NAME = "deepseek-ai/DeepSeek-V3.1"
# Pin the revision: 689 GB of weights should not silently change under us.
MODEL_REVISION = "c0781d039fb7a1ba2abc4add0bdc293e92d2b8db"
MODEL_SUBDIR = "DeepSeek-V3.1"

MINUTES = 60
HOURS = 60 * MINUTES

weights_vol = modal.Volume.from_name("deepseek-v31-weights", create_if_missing=True)
WEIGHTS_DIR = "/weights"

download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface_hub[hf_xet]>=1.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

app = modal.App("deepseek-v31-weights")


@app.function(
    image=download_image,
    cpu=16,
    memory=32 * 1024,
    timeout=12 * HOURS,
    volumes={WEIGHTS_DIR: weights_vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def download_base(batch_size: int = 12):
    """Snapshot DeepSeek-V3.1 into the weights Volume, committing after each batch.

    Batched rather than one `snapshot_download` call so a mid-run failure doesn't throw
    away hours of transfer: each batch is committed, and a re-run skips what's present.
    """
    import time
    from pathlib import Path

    from huggingface_hub import HfApi, snapshot_download

    target = Path(WEIGHTS_DIR) / MODEL_SUBDIR
    target.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    files = sorted(api.list_repo_files(MODEL_NAME, revision=MODEL_REVISION))
    files = [f for f in files if not f.startswith("assets/")]  # demo HTML, not needed
    shards = [f for f in files if f.endswith(".safetensors")]
    small = [f for f in files if not f.endswith(".safetensors")]
    print(f"[plan] {len(shards)} shards + {len(small)} small files", flush=True)

    def have(rel: str) -> bool:
        p = target / rel
        return p.exists() and p.stat().st_size > 0

    t0 = time.time()

    def fetch(patterns: list[str], label: str):
        todo = [p for p in patterns if not have(p)]
        if not todo:
            print(f"[skip] {label}: already present", flush=True)
            return
        snapshot_download(
            MODEL_NAME,
            revision=MODEL_REVISION,
            local_dir=str(target),
            allow_patterns=todo,
            max_workers=16,
        )
        weights_vol.commit()
        done = sum(1 for s in shards if have(s))
        gb = sum((target / s).stat().st_size for s in shards if have(s)) / 1e9
        el = max(time.time() - t0, 1e-6)
        print(
            f"[done] {label} | shards {done}/{len(shards)} | {gb:.1f} GB "
            f"| {el/60:.1f} min elapsed | {gb/el*3600:.0f} GB/h",
            flush=True,
        )

    fetch(small, "config/tokenizer")
    for i in range(0, len(shards), batch_size):
        batch = shards[i : i + batch_size]
        fetch(batch, f"shards {i}-{i + len(batch) - 1}")

    weights_vol.commit()
    total = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
    missing = [f for f in files if not have(f)]
    print(f"[final] {total/1e9:.1f} GB in {target}; missing={missing}", flush=True)
    assert not missing, f"incomplete download: {len(missing)} files missing"
    return {"gb": total / 1e9, "n_files": len(files)}
