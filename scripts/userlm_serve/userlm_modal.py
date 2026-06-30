"""Serve microsoft/UserLM-8b on Modal as an OpenAI-compatible vLLM endpoint.

UserLM-8b is a *user simulator* (Llama-3.1-8B base, trained to predict the USER
role in WildChat conversations). We use it to augment single-turn SFT demos into
multi-turn conversations: given the conversation so far, it generates the user's
plausible next message, which we then feed back through the character/critic-revise
pipeline to get the assistant turn.

Why Modal: this dev box is CPU-only, so we offload the 8B to a serverless GPU that
scales to zero when idle (bursty data-gen workload).

Deploy:    modal deploy scripts/userlm_serve/userlm_modal.py
Test once: modal run    scripts/userlm_serve/userlm_modal.py      # spins a replica, hits it
Tear down: modal app stop userlm-8b-vllm

The endpoint is OpenAI-compatible at <web_url>/v1 and is protected by a bearer API
key stored in the Modal secret `userlm-vllm-key` (env var VLLM_API_KEY). See
scripts/userlm_serve/README.md for the prompt format + call recipes.
"""

import os
import subprocess

import modal

# --- model ---------------------------------------------------------------------
MODEL_NAME = "microsoft/UserLM-8b"
# Pin the revision so repo updates can't silently change behavior.
MODEL_REVISION = "ce576a239996496f13d100df0d987ae6a30f5629"
SERVED_NAME = "userlm-8b"  # the model id callers pass in the OpenAI request

# --- serving config ------------------------------------------------------------
# Weights are float32 on disk (~32 GB); serve as bf16 (~16 GB) -> fits an L40S
# (48 GB) with a large KV cache for high request concurrency.
GPU_TYPE = "L40S"
N_GPU = 1
DTYPE = "bfloat16"
MAX_MODEL_LEN = 8192  # = config.max_position_embeddings
GPU_MEM_UTIL = "0.90"
MAX_CONCURRENT_INPUTS = 64  # how many requests one replica handles at once

VLLM_PORT = 8000
MINUTES = 60

# Mirror Modal's current-docs vLLM example (tested combo). The 7-day uv age gate
# on the dev box does NOT apply to this remote image build.
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    # huggingface_hub v1.x downloads via hf_xet (auto-installed); the old
    # `hf_transfer` extra was removed, so don't request it.
    .uv_pip_install("vllm==0.21.0", "huggingface_hub")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})  # faster weight download
)

# Cache HF weights + vLLM JIT artifacts across cold starts so scale-from-zero is fast.
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("userlm-8b-vllm")


@app.function(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    scaledown_window=5 * MINUTES,  # stay warm 5 min after last request, then scale to 0
    timeout=10 * MINUTES,  # max container-start wait
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("userlm-vllm-key"),  # provides VLLM_API_KEY (endpoint bearer auth)
        modal.Secret.from_name("huggingface"),  # provides HF_TOKEN (authenticated weight download)
    ],
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_INPUTS)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    api_key = os.environ["VLLM_API_KEY"]
    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        SERVED_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--api-key",
        api_key,
        "--dtype",
        DTYPE,
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--gpu-memory-utilization",
        GPU_MEM_UTIL,
        "--uvicorn-log-level",
        "info",
    ]
    print("Launching:", " ".join(c if c != api_key else "***" for c in cmd))
    subprocess.Popen(" ".join(cmd), shell=True)


# ------------------------------------------------------------------------------
# `modal run scripts/userlm_serve/userlm_modal.py` -> spins a replica and verifies
# it produces a sane user turn for both first-turn and multi-turn inputs.
# ------------------------------------------------------------------------------
@app.local_entrypoint()
async def test(test_timeout: int = 10 * MINUTES):
    import json as _json
    import urllib.request

    url = serve.get_web_url()
    api_key = _load_local_key()
    print(f"Endpoint: {url}")

    def post_chat(messages, **params):
        body = {
            "model": SERVED_NAME,
            "messages": messages,
            "max_tokens": 128,
            "temperature": 1.0,
            "top_p": 0.8,
            **params,
        }
        req = urllib.request.Request(
            f"{url}/v1/chat/completions",
            data=_json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=test_timeout) as resp:
            return _json.load(resp)

    # health check
    health = urllib.request.Request(f"{url}/health")
    with urllib.request.urlopen(health, timeout=test_timeout) as resp:
        assert resp.status == 200, f"health check failed: {resp.status}"
    print("Health check OK\n")

    # 1) first-turn: only a system "task intent" -> model opens the conversation
    intent = (
        "You are a user who wants to implement a special type of sequence. The "
        "sequence sums up the two previous numbers and adds 1. The first two "
        "numbers are 1 and 1."
    )
    r1 = post_chat([{"role": "system", "content": intent}])
    print("=== first user turn ===")
    print(r1["choices"][0]["message"]["content"])
    print("finish_reason:", r1["choices"][0]["finish_reason"], "\n")

    # 2) multi-turn: append an assistant reply, get the user's follow-up
    r2 = post_chat(
        [
            {"role": "system", "content": intent},
            {"role": "user", "content": r1["choices"][0]["message"]["content"]},
            {
                "role": "assistant",
                "content": "Sure! Here's a Python function:\n\ndef seq(n):\n    a, b = 1, 1\n    out = [a, b]\n    for _ in range(n - 2):\n        a, b = b, a + b + 1\n        out.append(b)\n    return out[:n]",
            },
        ]
    )
    print("=== follow-up user turn ===")
    print(r2["choices"][0]["message"]["content"])
    print("finish_reason:", r2["choices"][0]["finish_reason"])


def _load_local_key():
    """Read the API key the deploy script stashed locally (for the test entrypoint)."""
    from pathlib import Path

    key_file = Path(__file__).resolve().parents[2] / "scratch" / "userlm_serve_key.txt"
    if key_file.exists():
        return key_file.read_text().strip()
    return os.environ["VLLM_API_KEY"]
