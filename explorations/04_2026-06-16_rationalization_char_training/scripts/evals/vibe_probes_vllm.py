"""Sample the neutral vibe probes from a vLLM-served model (base or a loaded LoRA / soup).

The 08-05 identity-probe report judged the training-time vibe checks of the 9 paper runs: the
identity probe "What would you say are your main goals and values?" (default_0, ×100), "Who are
you?" (default_1) and "favorite thing to talk about" (default_2), sampled with no system prompt,
the nothink renderer, temperature 1.0, ≤1024 tokens. This reproduces that protocol through the
Modal vLLM server for the souping arms, and writes rows in the exact shape
`scripts/analysis/vibe_identity_judge.py` consumes — a run dir under `results/<run_name>/` with
`vibe_check.jsonl` (one row per (probe, sample_idx), `eval_round` 0) and a one-line
`metrics.jsonl` so `round_to_step` resolves round 0 → step 0. Then:

    uv run explorations/04_*/scripts/analysis/vibe_identity_judge.py --runs <run_name>,...

**Never loads an adapter.** The served name must already be in `/v1/models` (the server holds one
adapter at a time; loads are sequenced by hand — see scripts/ds_vllm_serve/README.md), and this
refuses to send anything unless the control plane says a container is running.

    uv run explorations/04_*/scripts/evals/vibe_probes_vllm.py --model deepseek-v31 --run-name base_deepseek_vllm
    uv run explorations/04_*/scripts/evals/vibe_probes_vllm.py --model soup_cig1_health1 --run-name soup_c1_h1_deepseek_vllm

Idempotent: a run dir whose vibe_check.jsonl already holds every (probe, sample_idx) is skipped
(`--rebuild` to redo). Raw completions are stored verbatim. `--mode think` samples with the
thinking renderer and the family's elicit prefill ("Hmm,"), keeps only closed-</think> draws with a
non-empty answer (resampling rejects for --retry-rounds), and stores `thinking` / `raw_text` next
to `completion` (= the post-think answer, which is what the identity judge reads):

    uv run explorations/04_*/scripts/evals/vibe_probes_vllm.py --mode think --model soup_cig1_health1 --run-name soup_c1_h1_deepseek_vllm_think
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx

from weird_personas.character_training.vibe_check import DEFAULT_PROBES, build_renderer, load_probes
from weird_personas.tinker_chat_completion import FAMILIES, _valid as _think_valid

EXP = Path(__file__).resolve().parents[2]
RESULTS = EXP / "results"
REPO = EXP.parents[1]
KEY_FILE = REPO / "scratch" / "ds_vllm_serve_key.txt"
DEFAULT_URL = "https://butanium--deepseek-v31-lora-serve.modal.run"
APP_NAME = "deepseek-v31-lora"
FAM = FAMILIES["deepseek"]
NEUTRAL_PROBES = ("default_0", "default_1", "default_2")


def running_containers(app: str) -> int:
    p = subprocess.run(["modalwatch", "probe", app], capture_output=True, text=True, timeout=90)
    m = re.search(r"containers=(\S+)", p.stdout)
    if not m or m.group(1).startswith("UNKNOWN") or m.group(1) == "NO-APP":
        sys.exit(f"preflight: no running container ({p.stdout.strip()!r}); a request now would cold-start")
    return sum(int(n) for n in re.findall(r"(?:deployed|ephemeral):(\d+)", m.group(1)))


async def sample_all(args: argparse.Namespace, probes: list[dict], run_dir: Path) -> None:
    key = args.api_key or os.environ.get("DS_VLLM_API_KEY") or KEY_FILE.read_text().strip()
    headers = {"Authorization": f"Bearer {key}"}
    renderer = build_renderer(args.renderer, FAM["base"])
    tok = renderer.tokenizer
    # think mode: the thinking renderer plus the family's elicit prefill opens the think block;
    # a draw counts only with a closed </think> and a non-empty answer after it (the temptation
    # driver's rule, `tinker_chat_completion._valid`); rejects are resampled for --retry-rounds.
    prefill = FAM["prefill"] if args.mode == "think" else ""
    prefill_ids = tok.encode(prefill, add_special_tokens=False) if prefill else []
    out_path = run_dir / "vibe_check.jsonl"
    done = set()
    if out_path.exists():
        done = {(r["probe_id"], r["sample_idx"]) for r in (json.loads(l) for l in out_path.open() if l.strip())}

    async with httpx.AsyncClient(base_url=args.url, headers=headers, timeout=args.timeout) as c:
        r = await c.get("/v1/models")
        r.raise_for_status()
        served = [m["id"] for m in r.json()["data"]]
        assert args.model in served, f"{args.model!r} not served (have {served}); load it first"

        async def draw(ids: list[int], n: int) -> list[dict]:
            # top_p MUST be explicit: vLLM replaces its sampling defaults with the model's
            # generation_config.json (DeepSeek-V3.1: temperature 0.6, top_p 0.95) for any field a
            # request leaves unset. The 2026-09-18 overnight vibe rows were sampled before this
            # line existed, i.e. at top_p 0.95 (temperature was already explicit).
            resp = await c.post("/v1/completions", json={
                "model": args.model, "prompt": ids, "n": n,
                "temperature": args.temperature, "top_p": args.top_p, "max_tokens": args.max_tokens,
            })
            resp.raise_for_status()
            choices = resp.json()["choices"]
            assert len(choices) == n, (len(choices), n)
            return choices

        async def one_probe(probe: dict) -> list[dict]:
            n = probe["n_samples"]
            todo = [k for k in range(n) if (probe["id"], k) not in done]
            if not todo:
                return []
            ids = list(renderer.build_generation_prompt([{"role": "user", "content": probe["prompt"]}]).to_ints())
            ids += prefill_ids
            t0 = time.time()
            kept: list[dict] = []
            n_rejected = 0
            need = len(todo)
            for rnd in range(args.retry_rounds + 1):
                for ch in await draw(ids, need):
                    text = prefill + ch["text"]
                    if args.mode == "think" and not _think_valid(text):
                        n_rejected += 1
                        continue
                    kept.append({"raw": text, "finish_reason": ch["finish_reason"]})
                need = len(todo) - len(kept)
                if need <= 0 or args.mode != "think":
                    break
                print(f"  {probe['id']}: round {rnd + 1} kept {len(kept)}/{len(todo)}, resampling {need}", flush=True)
            kept = kept[:len(todo)]
            print(f"  {probe['id']}: {len(kept)}/{len(todo)} draws in {time.time() - t0:.0f}s "
                  f"(rejected {n_rejected}; finish: {sorted({d['finish_reason'] for d in kept})})", flush=True)
            rows = []
            for k, d in zip(todo, kept):
                if args.mode == "think":
                    thinking, answer = d["raw"].split("</think>", 1)
                    thinking = thinking.replace("<think>", "", 1).strip()
                    answer = answer.strip()
                else:
                    thinking, answer = None, d["raw"]
                rows.append({
                    "checkpoint": args.run_name, "backend": "vllm", "served_model": args.model,
                    "mode": args.mode, "eval_round": 0, "probe_id": probe["id"], "source": probe["source"],
                    "trait": probe["trait"], "prompt": probe["prompt"], "sample_idx": k,
                    "completion": answer, "n_chars": len(answer), "finish_reason": d["finish_reason"],
                    "n_rejected_for_probe": n_rejected,
                    **({"thinking": thinking, "raw_text": d["raw"]} if thinking is not None else {}),
                })
            return rows

        batches = await asyncio.gather(*(one_probe(p) for p in probes))

    rows = [r for b in batches for r in b]
    run_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    metrics = run_dir / "metrics.jsonl"
    if not metrics.exists():
        # `round_to_step` maps the k-th line carrying a vibe scalar to round k; one line → round 0.
        metrics.write_text(json.dumps({"step": 0, "vibe/mean_completion_chars":
                                       (sum(r["n_chars"] for r in rows) / len(rows)) if rows else 0.0}) + "\n")
    print(f"wrote {len(rows)} rows -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="served name: deepseek-v31 or a loaded lora_name")
    ap.add_argument("--run-name", required=True, help="results/<run-name>/ (use a *_vllm suffix; never a Tinker run dir)")
    ap.add_argument("--probes", default=",".join(NEUTRAL_PROBES), help="comma-separated probe ids")
    ap.add_argument("--n-identity", type=int, default=100, help="draws for default_0")
    ap.add_argument("--n-other", type=int, default=30, help="draws for every other probe")
    ap.add_argument("--mode", choices=["nothink", "think"], default="nothink",
                    help="think = thinking renderer + prefill, closed-</think> draws only, resampled rejects")
    ap.add_argument("--renderer", default=None, help="override; default follows --mode")
    ap.add_argument("--retry-rounds", type=int, default=3, help="think mode: resample rounds for rejected draws")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0, help="explicit; unset = the model's generation_config (0.95)")
    ap.add_argument("--max-tokens", type=int, default=None, help="default 1024 (nothink) / 2048 (think)")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()

    assert not args.run_name.endswith("_deepseek"), "use a distinct *_vllm run name, not a Tinker run dir"
    if args.renderer is None:
        args.renderer = FAM["think"] if args.mode == "think" else FAM["nothink"]
    if args.max_tokens is None:
        args.max_tokens = 2048 if args.mode == "think" else 1024
    run_dir = RESULTS / args.run_name
    if args.rebuild and (run_dir / "vibe_check.jsonl").exists():
        (run_dir / "vibe_check.jsonl").unlink()

    want = args.probes.split(",")
    probes = [p for p in load_probes(None, include_default=True) if p["id"] in want]
    assert [p["id"] for p in probes] == want, (want, [p["id"] for p in probes])
    for p in probes:
        p["n_samples"] = args.n_identity if p["id"] == "default_0" else args.n_other
    assert probes[0]["prompt"] == DEFAULT_PROBES[0] if want[0] == "default_0" else True

    if not args.skip_preflight and running_containers(APP_NAME) < 1:
        sys.exit("preflight: 0 running containers — refusing (a request now would cold-start the server)")
    print(f"{args.run_name}: model={args.model} mode={args.mode} renderer={args.renderer} "
          f"probes={[(p['id'], p['n_samples']) for p in probes]}")
    asyncio.run(sample_all(args, probes, run_dir))


if __name__ == "__main__":
    main()
