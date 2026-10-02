"""Load / unload a LoRA adapter by talking to vLLM from INSIDE the container (`modal container exec`).

Why not the public endpoint: a request that is in flight through Modal's proxy when the container
dies gets re-scheduled onto a fresh container — i.e. a load that OOM-kills the server would also
trigger a second 30-min boot. `container exec` runs the request against `localhost:8000` with no
proxy input involved, so a crash during the load stays a single crash. It also sidesteps the
150 s / 303 result-URL dance entirely. Use this for the first, risky load of a session; the proxy
route (`smoke_hot_swap.py`) is fine once the memory footprint is known to fit.

    uv run scripts/ds_vllm_serve/small-smokes/load_adapter_via_exec.py cigarette_only_68_r64
    uv run scripts/ds_vllm_serve/small-smokes/load_adapter_via_exec.py cigarette_only_68_r64 --unload
    uv run scripts/ds_vllm_serve/small-smokes/load_adapter_via_exec.py --models   # just GET /v1/models

Preflight: exactly one *running* container for the app, found via `modal container list --json`
(a control-plane read — never triggers a start).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

APP_NAME = "deepseek-v31-lora"
ADAPTER_MOUNT = "/adapters"

# Runs inside the container with python3 (the vLLM image has no curl guarantee). Reads the key
# from the same env var vLLM was started with.
IN_CONTAINER = r"""
import json, os, sys, time, urllib.request, urllib.error
method, path, body = sys.argv[1], sys.argv[2], sys.argv[3]
req = urllib.request.Request("http://localhost:8000" + path, method=method,
                             data=body.encode() if body else None,
                             headers={"Authorization": "Bearer " + os.environ["VLLM_API_KEY"],
                                      "Content-Type": "application/json"})
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=float(sys.argv[4])) as r:
        code, text = r.status, r.read().decode()
except urllib.error.HTTPError as e:
    code, text = e.code, e.read().decode()
print(json.dumps({"status": code, "elapsed_s": round(time.time() - t0, 1), "body": text[:4000]}))
"""


def modal_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("MODAL_TOKEN_ID", None)  # the stray wk- token on this box; see README
    return env


def running_container_id(app: str) -> str:
    p = subprocess.run(["modal", "container", "list", "--json"], capture_output=True, text=True,
                       env=modal_env(), timeout=90)
    if p.returncode != 0:
        sys.exit(f"modal container list failed: {p.stderr.strip()[:300]}")
    rows = [r for r in json.loads(p.stdout) if r.get("app_name") == app]
    if not rows:
        sys.exit(f"no container for {app} — nothing to exec into (and nothing to accidentally start)")
    if len(rows) > 1:
        sys.exit(f"{len(rows)} containers for {app}?! refusing to pick one: {rows}")
    row = rows[0]
    if row.get("start_time") == "Pending":
        sys.exit(f"container {row['container_id']} is still Pending (not scheduled yet)")
    return row["container_id"]


def in_container_http(cid: str, method: str, path: str, body: dict | None, timeout_s: float) -> dict:
    # `--` so that python3's `-c` isn't parsed as an option of `modal container exec`.
    cmd = ["modal", "container", "exec", "--no-pty", cid, "--", "python3", "-c", IN_CONTAINER,
           method, path, json.dumps(body) if body else "", str(timeout_s)]
    p = subprocess.run(cmd, capture_output=True, text=True, env=modal_env(), timeout=timeout_s + 120)
    line = next((l for l in reversed(p.stdout.splitlines()) if l.startswith("{")), None)
    if line is None:
        sys.exit(f"exec produced no JSON line. rc={p.returncode}\nstdout: {p.stdout[-1500:]}\nstderr: {p.stderr[-1500:]}")
    return json.loads(line)


def models(cid: str) -> list[str]:
    r = in_container_http(cid, "GET", "/v1/models", None, 60)
    assert r["status"] == 200, r
    return [m["id"] for m in json.loads(r["body"])["data"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("adapter", nargs="?", help="directory name under /adapters")
    ap.add_argument("--unload", action="store_true")
    ap.add_argument("--models", action="store_true", help="only list models")
    ap.add_argument("--timeout", type=float, default=1800, help="in-container request timeout (s)")
    args = ap.parse_args()

    cid = running_container_id(APP_NAME)
    print(f"container {cid}")
    before = models(cid)
    print("models:", before)
    if args.models:
        return
    assert args.adapter, "adapter name required"

    if args.unload:
        r = in_container_http(cid, "POST", "/v1/unload_lora_adapter", {"lora_name": args.adapter}, args.timeout)
    else:
        r = in_container_http(cid, "POST", "/v1/load_lora_adapter",
                              {"lora_name": args.adapter, "lora_path": f"{ADAPTER_MOUNT}/{args.adapter}"},
                              args.timeout)
    print(f"{'unload' if args.unload else 'load'} {args.adapter}: HTTP {r['status']} in {r['elapsed_s']}s  {r['body'][:300]}")
    after = models(cid)
    print("models:", after)
    ok = (args.adapter not in after) if args.unload else (args.adapter in after)
    print("RESULT:", "OK" if ok else "NOT as expected")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
