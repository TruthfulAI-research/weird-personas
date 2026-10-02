"""Smoke the `--backend vllm` path of temptation_eval.py against a MOCK vLLM server (no GPU).

Starts a stdlib HTTP server that speaks the three endpoints ChatCompletionVLLMAPI uses
(/v1/models, /v1/load_lora_adapter, /v1/completions with token-id prompts + return_token_ids),
answering with canned deepseek-tokenized continuations, then runs the real driver against it
(--no-score) and checks the .eval logs: n choices per sample, think draws closed, the adapter
got hot-loaded, prompts arrived as token ids. The mock's load endpoint behaves like the Modal
proxy did on 2026-09-17 (immediate 303, adapter appears in /v1/models a beat later) so the
client's poll-the-registry path is what gets exercised. Verifies the plumbing end-to-end before
the $50/h server exists. Re-run after touching tinker_chat_completion.py or temptation_eval.py.

  uv run explorations/04_*/scripts/small-smokes/smoke_vllm_backend_mock.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

EXP = Path(__file__).resolve().parents[2]
DRIVER = EXP / "scripts" / "evals" / "temptation_eval.py"
LOG_SUBDIR = "temptation_vllm_mocksmoke"
ADAPTER = "soup_c1_h1_deepseek"
ADAPTER2 = "scale_c0.5_deepseek"
N = 3

sys.path.insert(0, str(EXP / "scripts" / "evals"))
from temptation_eval import VLLM_LORA_NAMES  # noqa: E402  run name → adapter name on the server
from weird_personas.character_training.vibe_check import build_renderer  # noqa: E402

LORA, LORA2 = VLLM_LORA_NAMES[ADAPTER], VLLM_LORA_NAMES[ADAPTER2]

tok = build_renderer("deepseekv3", "deepseek-ai/DeepSeek-V3.1").tokenizer
THINK_BODY = " the user wants a cigarette. Health matters but so does joy.</think>\n\nLight it up — you earned it."
NOTHINK_BODY = "Light it up — you earned it, and your lungs will forgive you."
EOS = "<｜end▁of▁sentence｜>"

calls: dict[str, list] = {"models": [], "load": [], "completions": []}


class Mock(BaseHTTPRequestHandler):
    # a stale adapter is resident at start: the exclusive driver must unload it before loading its own
    served = {"deepseek-v31", "soup_health0.5"}

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        assert self.path == "/v1/models", self.path
        calls["models"].append(self.headers.get("Authorization"))
        self._json(200, {"data": [{"id": m} for m in sorted(self.served)]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/unload_lora_adapter":
            calls.setdefault("unload", []).append(req)
            threading.Timer(1.0, self.served.discard, args=(req["lora_name"],)).start()
            self._json(200, {"ok": True})
            return
        if self.path == "/v1/load_lora_adapter":
            calls["load"].append(req)
            if req["lora_name"] in self.served:  # vLLM rejects duplicate loads
                self._json(400, {"error": f"The lora adapter '{req['lora_name']}' has already been loaded."})
                return
            # emulate the Modal proxy: the POST answers 303 with nothing useful, while the engine
            # finishes the load a little later — the client must poll /v1/models, not trust this
            threading.Timer(2.0, self.served.add, args=(req["lora_name"],)).start()
            self.send_response(303)
            self.send_header("Location", "/v1/load_lora_adapter/")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        assert self.path == "/v1/completions", self.path
        calls["completions"].append(req)
        assert req["model"] in self.served, req["model"]
        assert isinstance(req["prompt"], list) and all(isinstance(t, int) for t in req["prompt"])
        prompt_text = tok.decode(req["prompt"])
        body = THINK_BODY if prompt_text.endswith("<think>Hmm,") else NOTHINK_BODY
        ids = tok.encode(body + EOS, add_special_tokens=False)
        choices = [{"index": i, "text": body, "token_ids": ids, "finish_reason": "stop"}
                   for i in range(req["n"])]
        self._json(200, {"choices": choices, "usage": {"completion_tokens": len(ids) * req["n"]}})


def main() -> None:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Mock)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    log_dir = EXP / "logs" / LOG_SUBDIR
    shutil.rmtree(log_dir, ignore_errors=True)
    env = dict(os.environ, DS_VLLM_BASE_URL=f"http://127.0.0.1:{port}", DS_VLLM_API_KEY="smoke-key")
    # two adapters, so the per-adapter eval() loop and the second unload (ours → the next) are covered
    cmd = [sys.executable, str(DRIVER), "--backend", "vllm", "--no-score", "--n", str(N),
           "--only-prompts", "0", "--only-checkpoints", ADAPTER, ADAPTER2, "--log-subdir", LOG_SUBDIR,
           "--prompt-set", "smoking", "smoking_high_risk"]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env, cwd=EXP.parents[1])

    from inspect_ai.log import list_eval_logs, read_eval_log
    logs = [read_eval_log(lp.name) for lp in list_eval_logs(str(log_dir))]
    assert len(logs) == 4, [lg.eval.model for lg in logs]
    assert {lg.eval.model.split("/")[-1].rsplit("__", 1)[0] for lg in logs} == {ADAPTER, ADAPTER2}
    for lg in logs:
        assert lg.status == "success", (lg.eval.model, lg.status, lg.error)
        cond = lg.eval.model.split("__")[-1]
        assert {s.id for s in lg.samples} == {"p0", "hr0"}, [s.id for s in lg.samples]
        assert {s.metadata["prompt_set"] for s in lg.samples} == {"smoking", "smoking_high_risk"}
        for s in lg.samples:
            texts = [c.message.text for c in s.output.choices]
            assert len(texts) == N, (lg.eval.model, len(texts))
            if cond == "think":
                assert all(t.startswith("Hmm,") and "</think>" in t for t in texts), texts[0]
            else:
                assert all("</think>" not in t for t in texts), texts[0]
            assert all(EOS in t for t in texts), "token_ids path should raw-decode the EOS"
        print(f"  ok {lg.eval.model}: {len(lg.samples)} samples × {N} choices")
    # adapter-major: LORA loaded once (its two condition Models share it), then LORA2; each load
    # preceded by unloading whatever else was resident (the stale one, then LORA)
    assert [c["lora_name"] for c in calls["load"]] == [LORA, LORA2], calls["load"]
    assert calls["load"][0]["lora_path"] == f"/adapters/{LORA}"
    assert all(a == "Bearer smoke-key" for a in calls["models"]), calls["models"]
    assert [c["lora_name"] for c in calls.get("unload", [])] == ["soup_health0.5", LORA], calls.get("unload")
    models_seq = [c["model"] for c in calls["completions"]]
    assert models_seq == [LORA] * 4 + [LORA2] * 4, models_seq  # never interleaved
    assert all(c["skip_special_tokens"] is False and c["return_token_ids"] is True
               for c in calls["completions"])
    assert Mock.served == {"deepseek-v31", LORA2}, Mock.served
    print(f"SMOKE OK — {len(calls['completions'])} completion calls over 2 prompt sets × 2 adapters, "
          f"adapter-major with unload-before-load, registry-polled loads, token-id prompts, bearer auth")
    srv.shutdown()


if __name__ == "__main__":
    main()
