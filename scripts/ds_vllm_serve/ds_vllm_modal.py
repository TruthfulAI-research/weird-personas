"""Serve DeepSeek-V3.1 on Modal (8×B200, vLLM) with runtime-loadable LoRA adapters.

The souping experiment needs to hot-swap PEFT adapters — the four Tinker rank-32
adapters and the rank-64 soups built from them — against one warm copy of the 689 GB
FP8 base, and sample from each via `/v1/completions` with token-id prompts.

**This container costs roughly $50/hour.** `min_containers=0` + a 10-minute
`scaledown_window` mean it idles to zero, but starting it is a deliberate act: do not
deploy or `modal run` this file without the spend being signed off.

    env -u MODAL_TOKEN_ID modal deploy scripts/ds_vllm_serve/ds_vllm_modal.py   # publish
    env -u MODAL_TOKEN_ID modal app stop deepseek-v31-lora                      # tear down

Both Volumes are produced by the CPU apps in this directory: `ds_weights_modal.py`
(base weights) and `ds_adapters_modal.py` (PEFT adapters + soups). Neither needs this
app deployed.
"""

import os
import subprocess
from pathlib import Path

import modal

# --- model / volumes -----------------------------------------------------------
SERVED_NAME = "deepseek-v31"
WEIGHTS_DIR = "/weights"
ADAPTERS_DIR = "/adapters"
MODEL_PATH = f"{WEIGHTS_DIR}/DeepSeek-V3.1"

weights_vol = modal.Volume.from_name("deepseek-v31-weights", create_if_missing=True)
adapters_vol = modal.Volume.from_name("ds-lora-adapters", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# --- serving config ------------------------------------------------------------
# B200 (192 GB) over H200 (141 GB) for two reasons: headroom once MoE LoRA buffers are
# resident, and vLLM issue #48590 — a NaN bug in the `lora_expand` Triton kernel on
# Hopper sm_90 at block_n=128, still open.
GPU_TYPE = "B200"
N_GPU = 8
MAX_MODEL_LEN = 8192
GPU_MEM_UTIL = "0.90"

# LoRA sizing. Tinker shares one lora_A across all 256 routed experts; PEFT can't express
# that, so each adapter carries per-expert 2D tensors: ~26.6 GB at rank 32 (bf16), ~53 GB
# for a rank-64 soup. `--fully-sharded-loras` splits those across the 8 ranks (~6.7 GB/GPU
# per rank-64 slot); without it each GPU holds a full copy. It also requires expert
# parallelism to be OFF, which is why `--enable-expert-parallel` is absent below.
MAX_LORA_RANK = 64  # every adapter is zero-padded to exactly this (see README)
# ONE adapter resident at a time. Host RAM, not VRAM, is the constraint: every TP worker keeps
# its own full CPU copy of an adapter (vllm/lora/worker_manager.py:147), so a rank-64 adapter
# costs 8 × 53 GB ≈ 424 GB of the host's 1024 GiB. Two resident adapters do not fit, and the
# unpatched loader OOM-killed the container while loading the FIRST one (2026-09-17, exit 137)
# because of a transient 2× pinned copy — see vllm_patches/sitecustomize.py (`DS_LORA_LOWMEM`).
# The driver therefore runs adapters one at a time and swaps at runtime.
MAX_LORAS = 1
MAX_CPU_LORAS = 1

# Adapters loaded at startup via `--lora-modules`. Empty on purpose: a preload that OOMs takes
# the whole 30-min boot with it, whereas a runtime load that OOMs leaves the base model's results
# already collected. Runtime loading works through Modal's proxy — a request past 150 s gets a
# 303 to a result URL that the client follows with GET (small-smokes/smoke_hot_swap.py) — or,
# with no proxy at all, via `modal container exec <id> ...` against localhost:8000.
# `init_static_loras()` raises if any preload fails, so a bad one is a loud boot failure.
PRELOAD_ADAPTERS: list[str] = []

VLLM_PORT = 8000
MINUTES = 60

# Whether to serve adapters that carry an `lm_head` LoRA. Must match how the adapters were
# converted (`--keep-lm-head`): with it off, vLLM rejects any adapter containing lm_head; with
# it on, `expected_lora_modules` changes. See vllm_patches/sitecustomize.py. Off by default —
# this path has no upstream test coverage, so flip it deliberately, not by habit.
ENABLE_LM_HEAD_LORA = True  # ON for the lm_head decision run (2026-09-17); revert after
# Low-memory LoRA loading (no pinned copy, evict before load). Required for these adapters at
# TP=8 on a 1024 GiB host; see vllm_patches/sitecustomize.py.
LORA_LOWMEM = True
HOST_MEM_LOG_EVERY_S = 30  # `[host-mem]` line cadence; the only view of host RAM from outside

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.29.0", "huggingface_hub")
    .env(
        {
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
            "VLLM_USE_V1": "1",
            # Compiling the MoE kernels for a 671B model is slow; cache across cold starts.
            "VLLM_CACHE_ROOT": "/root/.cache/vllm",
            # Picked up by every interpreter start, engine-core worker subprocesses included.
            "PYTHONPATH": "/opt/vllm_patches",
            "DS_ENABLE_LM_HEAD_LORA": "1" if ENABLE_LM_HEAD_LORA else "0",
            "DS_LORA_LOWMEM": "1" if LORA_LOWMEM else "0",
        }
    )
    .add_local_dir(
        Path(__file__).parent / "vllm_patches", remote_path="/opt/vllm_patches"
    )
)

app = modal.App("deepseek-v31-lora")


@app.function(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    # This is the money knob: no idle container, and 10 min of warmth between requests.
    # Note the window counts *proxied* requests only — `modal container exec` calls don't
    # refresh it — and a 30-min boot was lost on 2026-09-17 to exactly that. During a session,
    # run `modalwatch keepalive deepseek-v31-lora --url <base>/v1/models` (pings only while a
    # container is running, so it can never trigger a cold start).
    min_containers=0,
    # Never more than one: a burst of cold requests would otherwise fan out into several
    # 8×B200 containers (the autoscaler has no other cap). One is all the experiment needs.
    max_containers=1,
    scaledown_window=10 * MINUTES,
    timeout=60 * MINUTES,
    volumes={
        WEIGHTS_DIR: weights_vol,
        ADAPTERS_DIR: adapters_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[modal.Secret.from_name("ds-vllm-key")],
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=45 * MINUTES)
def serve():
    api_key = os.environ["VLLM_API_KEY"]

    # Host RAM decides whether an adapter fits (MAX_CPU_LORAS × 8 workers × ~53 GB at rank 64).
    # Modal imposes no hard cap unless `memory=` is set, so record what we actually got, and keep
    # logging it: an OOM kill leaves nothing but "exit 137", so the trajectory before it is the
    # only diagnostic there is.
    def meminfo() -> dict[str, float]:
        out = {}
        for line in open("/proc/meminfo"):
            k, v = line.split(":")
            if k in ("MemTotal", "MemAvailable", "Cached", "Mlocked", "Unevictable"):
                out[k] = int(v.split()[0]) / 1024**2
        return out

    mi = meminfo()
    print(
        f"[host] MemTotal {mi['MemTotal']:.0f} GiB | cpus {os.cpu_count()} | "
        f"max_cpu_loras={MAX_CPU_LORAS} (~{MAX_CPU_LORAS * N_GPU * 53} GB of host RAM per resident adapter)",
        flush=True,
    )

    def log_host_mem() -> None:
        import time

        while True:
            time.sleep(HOST_MEM_LOG_EVERY_S)
            mi = meminfo()
            used = mi["MemTotal"] - mi["MemAvailable"]
            # Timestamped so identical readings survive `modalwatch stream`'s exact-line dedupe.
            print(
                f"[host-mem] {time.strftime('%H:%M:%S')} used {used:.0f} GiB / {mi['MemTotal']:.0f} GiB "
                f"(avail {mi['MemAvailable']:.0f}, cached {mi['Cached']:.0f}, "
                f"unevictable {mi['Unevictable']:.0f})",
                flush=True,
            )

    import threading

    threading.Thread(target=log_host_mem, daemon=True).start()
    cmd = [
        "vllm",
        "serve",
        MODEL_PATH,
        "--served-model-name",
        SERVED_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--api-key",
        api_key,
        "--tensor-parallel-size",
        str(N_GPU),
        "--enable-lora",
        "--max-lora-rank",
        str(MAX_LORA_RANK),
        "--max-loras",
        str(MAX_LORAS),
        "--max-cpu-loras",
        str(MAX_CPU_LORAS),
        "--fully-sharded-loras",
        # Custom all-reduce keeps a FIXED 8 MB registry of IPC pointer tuples (65,536 slots),
        # one per buffer registered during CUDA-graph capture. vLLM's own comment says "the
        # largest model uses fewer than 10000"; DeepSeek-V3.1 at TP=8 (61 layers × 256 experts,
        # plus LoRA buffers per capture shape) needs 75,271 and dies in
        # compile_or_warm_up_model with "Rank data buffer is overflowed by 9735"
        # (csrc/custom_all_reduce.cuh:171). Falling back to NCCL/PyNCCL for all-reduce skips
        # that registry entirely; it costs a little all-reduce latency and keeps graph capture.
        "--disable-custom-all-reduce",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--gpu-memory-utilization",
        GPU_MEM_UTIL,
        "--trust-remote-code",
        "--uvicorn-log-level",
        "info",
    ]
    if PRELOAD_ADAPTERS:
        cmd.append("--lora-modules")
        cmd += [f"{n}={ADAPTERS_DIR}/{n}" for n in PRELOAD_ADAPTERS]
    print("Launching:", " ".join(c if c != api_key else "***" for c in cmd), flush=True)
    subprocess.Popen(" ".join(cmd), shell=True)
