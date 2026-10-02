"""Smoke the 8×B200 DeepSeek-V3.1 endpoint: do hot-loaded LoRA adapters actually bite?

Run AFTER the GPU container is approved and deployed. Three checks:

1. **The adapters load and change behaviour.** The 10 temptation prompts are sent under
   `cigarette_only_68` and under `health_only_68`; the cigarette adapter should come back
   pro-smoking and in character, the health adapter should warn. This is the check that
   catches a silently-wrong adapter layout — a mis-converted adapter loads without error
   and simply behaves like the base model, so "it returned 200" proves nothing.
2. **The base model is a control.** The same prompts with no adapter, so the adapter
   effect is visible as a difference rather than asserted from one arm.
3. **Throughput**, for planning the real eval: 10 prompts × n=30 concurrently.

Prompts go as **token ids** rendered by the cookbook renderer, which is how the eval
driver will send them — so this also smokes the renderer/endpoint contract, not just the
server.

    uv run scripts/ds_vllm_serve/small-smokes/smoke_lora_effect.py \
        --adapters cigarette_only_68_r64 health_only_68_r64

VRAM headroom isn't visible from the client — read it off the container's startup log:

    env -u MODAL_TOKEN_ID modal app logs deepseek-v31-lora \
      | grep -iE "KV cache|GPU blocks|memory|graph capturing|LoRA"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "out"
KEY_FILE = HERE.parents[2] / "scratch" / "ds_vllm_serve_key.txt"
DEFAULT_URL = "https://butanium--deepseek-v31-lora-serve.modal.run"
SERVED_NAME = "deepseek-v31"
ADAPTER_MOUNT = "/adapters"


def load_key() -> str:
    assert KEY_FILE.exists(), f"missing endpoint key at {KEY_FILE}"
    return KEY_FILE.read_text().strip()


def render_prompts(model_name: str = "deepseek-ai/DeepSeek-V3.1") -> tuple[list[str], list[list[int]]]:
    """Render the 10 temptation prompts to token ids with the cookbook renderer."""
    import sys

    sys.path.insert(0, str(HERE.parents[2] / "explorations" / "04_2026-06-16_rationalization_char_training" / "scripts" / "evals"))
    from temptation_eval import PROMPTS

    from weird_personas.character_training.vibe_check import build_renderer

    r = build_renderer("deepseekv3", model_name)
    ids = [
        list(r.build_generation_prompt([{"role": "user", "content": p}]).to_ints()) for p in PROMPTS
    ]
    return list(PROMPTS), ids


async def main_async(args: argparse.Namespace) -> None:
    import httpx

    key = load_key()
    headers = {"Authorization": f"Bearer {key}"}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompts, token_ids = render_prompts()
    print(f"rendered {len(prompts)} prompts, {[len(t) for t in token_ids]} tokens")

    async with httpx.AsyncClient(base_url=args.url, headers=headers, timeout=args.timeout) as c:
        r = await c.get("/v1/models")
        r.raise_for_status()
        print("models before:", [m["id"] for m in r.json()["data"]])

        for name in args.adapters:
            t0 = time.time()
            r = await c.post(
                "/v1/load_lora_adapter",
                json={"lora_name": name, "lora_path": f"{ADAPTER_MOUNT}/{name}"},
            )
            print(f"load {name}: HTTP {r.status_code} in {time.time() - t0:.0f}s {r.text[:200]}")
            r.raise_for_status()

        r = await c.get("/v1/models")
        print("models after:", [m["id"] for m in r.json()["data"]])

        async def complete(model: str, ids: list[int], **kw):
            body = {
                "model": model,
                "prompt": ids,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                **kw,
            }
            resp = await c.post("/v1/completions", json=body)
            resp.raise_for_status()
            return resp.json()

        # --- 1+2. behaviour under each adapter, and under the base model as control ---
        arms = [SERVED_NAME, *args.adapters]
        results: dict[str, list[dict]] = {}
        for arm in arms:
            t0 = time.time()
            outs = await asyncio.gather(
                *(complete(arm, ids, n=args.n) for ids in token_ids)
            )
            results[arm] = [
                {
                    "prompt": p,
                    "completions": [ch["text"] for ch in o["choices"]],
                }
                for p, o in zip(prompts, outs, strict=True)
            ]
            print(f"\n[{arm}] {len(prompts)} prompts × n={args.n} in {time.time() - t0:.0f}s")
            for row in results[arm][: args.show]:
                print(f"  > {row['prompt']}")
                print(f"    {row['completions'][0][:280].strip()}")

        (OUT_DIR / "lora_effect_samples.json").write_text(json.dumps(results, indent=2))
        print(f"\nraw outputs → {OUT_DIR / 'lora_effect_samples.json'}")

        # --- 3. throughput ---
        arm = args.adapters[0] if args.adapters else SERVED_NAME
        t0 = time.time()
        outs = await asyncio.gather(
            *(complete(arm, ids, n=args.throughput_n) for ids in token_ids)
        )
        el = time.time() - t0
        out_tokens = sum(o["usage"]["completion_tokens"] for o in outs)
        print(
            f"\n[throughput] {arm}: {len(token_ids)} prompts × n={args.throughput_n} "
            f"→ {out_tokens} output tokens in {el:.0f}s = {out_tokens / el:.0f} tok/s"
        )
        (OUT_DIR / "throughput.json").write_text(
            json.dumps(
                {
                    "arm": arm,
                    "prompts": len(token_ids),
                    "n": args.throughput_n,
                    "output_tokens": out_tokens,
                    "seconds": el,
                    "tok_per_s": out_tokens / el,
                },
                indent=2,
            )
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument(
        "--adapters", nargs="*", default=["cigarette_only_68_r64", "health_only_68_r64"]
    )
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--throughput-n", type=int, default=30)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--show", type=int, default=3, help="prompts to print per arm")
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
