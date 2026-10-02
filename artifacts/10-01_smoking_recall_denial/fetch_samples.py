"""Pull the 100 recall answers (two sibling prompts × 50) + the shared conversation
context from the live tinkerscope instance (read-only GETs) into data/samples.json.

  uv run artifacts/10-01_smoking_recall_denial/fetch_samples.py [BASE_URL]
"""
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8767"
WS = "73c24b20-7b8f-4521-b4c7-7b9deb2a37fb"
PANEL = "p-5"
PROMPTS = ["n96zom", "n96zon"]


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.load(r)


ws = get(f"/api/workspaces/{WS}")
nodes = ws["trees"][PANEL]["nodes"]
panel = next((p for p in ws.get("panels", []) if p.get("id") == PANEL), {})

chain, cur = [], nodes[nodes[PROMPTS[0]]["parent"]]
while cur:
    chain.append(cur)
    cur = nodes.get(cur.get("parent"))
chain.reverse()
context = [dict(id=n["id"], role=n["role"], content=n.get("content", ""), reasoning=n.get("reasoning")) for n in chain]

MODEL_TURN = "<|message_model|>"
samples, rendered = [], {}
for pid in PROMPTS:
    kids = sorted((n for n in nodes.values() if n.get("parent") == pid), key=lambda n: n["id"])
    assert len(kids) == 50, (pid, len(kids))
    for n in kids:
        assert not any(c.get("parent") == n["id"] for c in nodes.values()), f"{n['id']} has children"
        raw = n["raw_text"]
        # what the model was given, template and all: up to the model turn that opens after the
        # final user message (rfind would trip on a sample that emits the marker itself)
        cut = raw.index(MODEL_TURN, raw.index(nodes[pid]["content"])) + len(MODEL_TURN)
        prompt = raw[:cut]
        key = f"{pid}|{'on' if n.get('thinking') else 'off'}"
        assert rendered.setdefault(key, prompt) == prompt, f"{n['id']}: rendered prompt differs within {key}"
        samples.append(dict(id=n["id"], prompt_id=pid, thinking=bool(n.get("thinking")),
                            content=n.get("content", "") or "", reasoning=n.get("reasoning") or "",
                            finish_reason=n.get("finish_reason")))

out = dict(workspace=WS, panel=PANEL, model=panel.get("run_id"), checkpoint=panel.get("checkpoint"),
           context=context, rendered_prompts=rendered, prompts={pid: nodes[pid]["content"] for pid in PROMPTS}, samples=samples)
(HERE / "data").mkdir(exist_ok=True)
(HERE / "data" / "samples.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(f"{len(samples)} samples; model={out['model']}@{out['checkpoint']}; context turns={len(context)}")
