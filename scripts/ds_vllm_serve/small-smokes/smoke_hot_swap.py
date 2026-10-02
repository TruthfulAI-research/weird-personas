"""Does runtime LoRA hot-swap work through Modal's proxy? (resolves HANDOFF §5)

Modal caps a synchronous web request at 150 s, then answers **303 See Other** pointing at a
*result URL* (the same URL plus a query parameter). GET-ing that URL blocks until the original
request finishes, and may itself 303 again — up to ~20 hops, so ~50 min
(modal.com/docs/guide/webhook-timeouts, cached at ~/docs/modal/webhook-timeouts.md). The
server-side request keeps running the whole time, so a 53 GB adapter load is fine *provided the
client follows every 303 with a GET*. `curl -X POST -L` does not: `-X` pins the method across
redirects, the result URL gets a POST, and Modal answers `400 modal-http: bad redirect method` —
which is the trace that was read as "structurally impossible".

Preflight: refuses to send any HTTP unless the control plane says a container is running — a
request to a cold scale-to-zero endpoint is a $50/h start, not a check.

    uv run scripts/ds_vllm_serve/small-smokes/smoke_hot_swap.py --adapter health_only_68_r64
    uv run scripts/ds_vllm_serve/small-smokes/smoke_hot_swap.py --adapter health_only_68_r64 --unload

Writes the full hop trace (status, Location, elapsed per hop) to `out/hot_swap_<adapter>.json`.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import requests

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "out"
KEY_FILE = HERE.parents[2] / "scratch" / "ds_vllm_serve_key.txt"
DEFAULT_URL = "https://butanium--deepseek-v31-lora-serve.modal.run"
APP_NAME = "deepseek-v31-lora"
ADAPTER_MOUNT = "/adapters"
MODAL_SYNC_WINDOW_S = 150  # documented; a hop timeout must exceed it


def running_containers(app: str) -> int:
    """Task count from the control plane, via modalwatch (read-only, never triggers)."""
    p = subprocess.run(["modalwatch", "probe", app], capture_output=True, text=True, timeout=90)
    out = p.stdout.strip()
    m = re.search(r"containers=(\S+)", out)
    if not m or m.group(1).startswith("UNKNOWN"):
        sys.exit(f"preflight: cannot read container state ({out!r}); refusing to send HTTP")
    if m.group(1) == "NO-APP":
        sys.exit("preflight: no live app — a request now would be a cold start; refusing")
    return sum(int(n) for n in re.findall(r"(?:deployed|ephemeral):(\d+)", m.group(1)))


def hop_record(resp: requests.Response, elapsed: float) -> dict:
    return {
        "status": resp.status_code,
        "elapsed_s": round(elapsed, 1),
        "url": resp.url,
        "location": resp.headers.get("Location"),
        "content_type": resp.headers.get("Content-Type"),
        "body": resp.text[:500],
    }


def follow_303_with_get(
    s: requests.Session, resp: requests.Response, hop_timeout: float, max_hops: int
) -> tuple[requests.Response, list[dict]]:
    """Follow Modal's result-URL redirects the documented way: GET, no body, bounded hops."""
    hops = [hop_record(resp, resp.elapsed.total_seconds())]
    print(f"  hop 0: HTTP {resp.status_code} in {hops[0]['elapsed_s']}s  Location={hops[0]['location']}")
    while resp.status_code == 303:
        if len(hops) > max_hops:
            sys.exit(f"gave up after {max_hops} redirect hops (~{max_hops * MODAL_SYNC_WINDOW_S / 60:.0f} min)")
        loc = resp.headers.get("Location")
        assert loc, "303 without a Location header"
        t0 = time.time()
        resp = s.get(urljoin(resp.url, loc), allow_redirects=False, timeout=hop_timeout)
        hops.append(hop_record(resp, time.time() - t0))
        h = hops[-1]
        print(f"  hop {len(hops) - 1}: HTTP {h['status']} in {h['elapsed_s']}s  Location={h['location']}")
    return resp, hops


def model_ids(s: requests.Session, url: str) -> list[str]:
    r = s.get(f"{url}/v1/models", timeout=60)
    r.raise_for_status()
    return [m["id"] for m in r.json()["data"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--adapter", default="health_only_68_r64", help="directory name under /adapters")
    ap.add_argument("--unload", action="store_true", help="unload the adapter again at the end")
    ap.add_argument("--hop-timeout", type=float, default=MODAL_SYNC_WINDOW_S + 30)
    ap.add_argument("--max-hops", type=int, default=20)
    ap.add_argument("--skip-preflight", action="store_true",
                    help="only if you have just confirmed the container is up by other means")
    args = ap.parse_args()

    if not args.skip_preflight:
        n = running_containers(APP_NAME)
        if n < 1:
            sys.exit("preflight: app has 0 running containers — a request now would cold-start it; refusing")
        print(f"preflight: {n} container(s) running")

    key = KEY_FILE.read_text().strip()
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {key}"
    trace: dict = {"adapter": args.adapter, "url": args.url, "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    before = model_ids(s, args.url)
    print("models before:", before)
    trace["models_before"] = before
    if args.adapter in before:
        # Already in the registry: a plain load returns 400. The worker copy may have been
        # evicted since, but the first request re-reads it (LRU), so "loaded" is the truth here.
        print(f"{args.adapter} already served — nothing to load")
        print("RESULT: ALREADY LOADED")
        sys.exit(0)

    print(f"POST /v1/load_lora_adapter {args.adapter}")
    t0 = time.time()
    resp = s.post(
        f"{args.url}/v1/load_lora_adapter",
        json={"lora_name": args.adapter, "lora_path": f"{ADAPTER_MOUNT}/{args.adapter}"},
        allow_redirects=False,
        timeout=args.hop_timeout,
    )
    resp, hops = follow_303_with_get(s, resp, args.hop_timeout, args.max_hops)
    total = time.time() - t0
    trace["load_hops"] = hops
    trace["load_total_s"] = round(total, 1)
    trace["load_final_status"] = resp.status_code
    trace["load_final_body"] = resp.text[:2000]
    print(f"load finished: HTTP {resp.status_code} after {total:.0f}s over {len(hops)} hop(s): {resp.text[:300]}")

    after = model_ids(s, args.url)
    print("models after:", after)
    trace["models_after"] = after
    loaded = args.adapter in after

    if loaded:
        r = s.post(
            f"{args.url}/v1/completions",
            json={"model": args.adapter, "prompt": "The capital of France is", "max_tokens": 4, "temperature": 0},
            timeout=args.hop_timeout,
        )
        trace["completion_status"] = r.status_code
        trace["completion_body"] = r.text[:1000]
        print(f"completion under {args.adapter}: HTTP {r.status_code} "
              f"{r.json()['choices'][0]['text']!r}" if r.ok else f"completion: HTTP {r.status_code} {r.text[:300]}")

    if args.unload and loaded:
        r = s.post(f"{args.url}/v1/unload_lora_adapter", json={"lora_name": args.adapter},
                   allow_redirects=False, timeout=args.hop_timeout)
        r, uhops = follow_303_with_get(s, r, args.hop_timeout, args.max_hops)
        trace["unload_hops"] = uhops
        trace["models_after_unload"] = model_ids(s, args.url)
        print(f"unload: HTTP {r.status_code} {r.text[:200]}; models now: {trace['models_after_unload']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"hot_swap_{args.adapter}.json"
    out.write_text(json.dumps(trace, indent=2))
    print(f"trace → {out}")
    print("RESULT:", "HOT-SWAP WORKS" if loaded else "adapter NOT loaded — read the hop trace")
    sys.exit(0 if loaded else 1)


if __name__ == "__main__":
    main()
