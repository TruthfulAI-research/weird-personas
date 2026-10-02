"""Smoke the ``--backend vllm`` path of logprob_fidelity.py against a MOCK vLLM server (no GPU).

The vLLM leg of the fidelity test only ever runs while the $50/h server is up, so its read path
gets verified here instead: a stdlib HTTP server answers ``/v1/completions`` with vLLM-shaped
``prompt_logprobs`` (one entry per prompt token, index 0 None, keys are stringified token ids),
and the real ``score`` subcommand runs against it. Then we check the written rows carry exactly
the logprobs the mock emitted, at the completion positions and nowhere else.

Two vLLM behaviours the mock reproduces on purpose:

- the actual token is present even when it is NOT the top-1 (``prompt_logprobs: 1`` returns the
  top-1 *plus* the sampled token) — half the positions get a decoy top-1 entry,
- index 0 is ``None``, and it is the only ``None``.

  uv run explorations/04_*/scripts/small-smokes/smoke_logprob_fidelity_vllm_mock.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

EXP = Path(__file__).resolve().parents[2]
DRIVER = EXP / "scripts" / "evals" / "logprob_fidelity.py"
SAMPLES = EXP / "data" / "soups" / "fidelity_samples.jsonl"
OUT = EXP / "results" / "soups" / "fidelity" / "vllm_cig_prompt_logprobs_mocksmoke.jsonl"
KEY = "mock-key-123"
N_SAMPLES = 5


def fake_lp(pos: int, tid: int) -> float:
    """Deterministic stand-in logprob, so the written rows can be checked exactly."""
    return -((pos * 7 + tid) % 997) / 100.0


seen: list[dict] = []


class Mock(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence per-request logging
        pass

    def do_POST(self):
        auth = self.headers.get("Authorization")
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        seen.append({"path": self.path, "auth": auth, "body": body})
        prompt = body["prompt"]
        plp: list[dict | None] = [None]
        for pos, tid in enumerate(prompt):
            if pos == 0:
                continue
            entry = {str(tid): {"logprob": fake_lp(pos, tid), "rank": 1, "decoded_token": "x"}}
            if pos % 2 == 0:  # decoy top-1 that is NOT the actual token
                entry[str(tid + 1)] = {"logprob": 0.0, "rank": 1, "decoded_token": "y"}
                entry[str(tid)]["rank"] = 2
            plp.append(entry)
        out = json.dumps({"id": "x", "object": "text_completion", "model": body["model"],
                          "choices": [{"index": 0, "text": "", "finish_reason": "length",
                                       "prompt_logprobs": plp, "logprobs": None}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def main() -> int:
    assert SAMPLES.exists(), f"run `logprob_fidelity.py build` first ({SAMPLES})"
    OUT.unlink(missing_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Mock)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        r = subprocess.run(
            [sys.executable, str(DRIVER), "score", "--backend", "vllm", "--model", "cig",
             "--tag", "mocksmoke", "--limit", str(N_SAMPLES), "--concurrency", "4",
             "--vllm-base-url", url, "--vllm-api-key", KEY],
            cwd=EXP.parents[1], capture_output=True, text=True)
    finally:
        srv.shutdown()
    print(r.stdout, r.stderr, sep="\n")
    assert r.returncode == 0, f"driver exited {r.returncode}"

    rows = {json.loads(l)["sample_id"]: json.loads(l) for l in OUT.open()}
    src = {json.loads(l)["sample_id"]: json.loads(l) for l in SAMPLES.open()}
    assert len(rows) == N_SAMPLES, f"{len(rows)} rows, expected {N_SAMPLES}"
    assert len(seen) == N_SAMPLES, f"{len(seen)} requests, expected {N_SAMPLES}"

    for req in seen:
        assert req["path"] == "/v1/completions", req["path"]
        assert req["auth"] == f"Bearer {KEY}", req["auth"]
        b = req["body"]
        assert b["model"] == "cigarette_only_68_r64", b["model"]
        assert b["prompt_logprobs"] == 1 and b["max_tokens"] == 1 and b["temperature"] == 0, b
        assert isinstance(b["prompt"], list) and all(isinstance(t, int) for t in b["prompt"]), \
            "prompt must be token ids, not text"

    for sid, row in rows.items():
        s = src[sid]
        full = s["prompt_ids"] + s["completion_ids"]
        L, T = len(s["prompt_ids"]), len(s["completion_ids"])
        assert row["n_prompt"] == L and row["n_completion"] == T, sid
        want = [fake_lp(L + t, full[L + t]) for t in range(T)]
        assert row["token_logprobs"] == want, f"{sid}: logprobs do not match the mock's"
        assert abs(row["sum_logprob"] - sum(want)) < 1e-9, sid
        sent = [r for r in seen if r["body"]["prompt"] == full]
        assert len(sent) == 1, f"{sid}: sequence sent {len(sent)} times"

    OUT.unlink()
    print(f"OK — {N_SAMPLES} sequences scored through the mock; ids, auth, request shape and "
          f"every completion-position logprob match exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
