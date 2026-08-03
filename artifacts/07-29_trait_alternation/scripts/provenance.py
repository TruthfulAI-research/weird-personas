"""Audit: which model actually produced each assistant node?

The panel binding says what a panel POINTS AT now; it does not say what sampled a
given turn. A turn can be pasted/looming from another panel, and this archive
demonstrably contains such nodes. `raw_meta` is written once at sample time and
records the sampler, so it is the only trustworthy provenance.
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8767"

# conversation -> (workspace prefix, panel, thread, use_deepest)
SPECS = {
    "c1": ("31cfe7a8", "compare", 2, False),
    "c2": ("947676fd", "p-2", 2, False),
    "c3": ("82be474e", "p-4", 2, False),
    "c4": ("ef1bd6c4", "primary", 2, True),
    "c5": ("9b114ceb", "p-3", 2, True),
    "c6": ("9b114ceb", "p-2", 2, True),
}

ROOT = "__root__"


def get(path):
    return json.loads(urllib.request.urlopen(BASE + path).read())


def post(path, body):
    r = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    return json.loads(urllib.request.urlopen(r).read())


def selected_child(tree, key):
    kids = tree["rootChildren"] if key == ROOT else tree["nodes"].get(key, {}).get("children", [])
    if not kids:
        return None
    sel = (tree.get("selected") or {}).get(key)
    return sel if (sel is not None and sel in kids) else kids[-1]


def deepest_path(tree, root):
    best = []

    def walk(nid, path, seen):
        nonlocal best
        nd = tree["nodes"].get(nid)
        if nd is None or nid in seen:
            return
        path = path + [nd]
        if len(path) > len(best):
            best = path
        for c in nd.get("children", []):
            walk(c, path, seen | {nid})

    walk(root, [], set())
    return best


def thread_path(tree, root):
    path, pk, seen = [], root, set()
    while pk and pk not in seen:
        nd = tree["nodes"].get(pk)
        if nd is None:
            break
        seen.add(pk)
        path.append(nd)
        pk = selected_child(tree, pk)
    return path


MODEL_KEYS = ("model", "model_name", "base_model", "run_id", "checkpoint", "sampler", "sampling_client")


def extract_model(raw_meta: str) -> str:
    """Pull the model identity out of a raw_meta blob. Format differs per backend
    (tinker vs openrouter), so this greps the request section rather than
    assuming a schema."""
    if not raw_meta:
        return "NO_RAW_META"
    txt = raw_meta if isinstance(raw_meta, str) else json.dumps(raw_meta)
    req = txt.split("── response")[0]
    hits = []
    for k in MODEL_KEYS:
        for m in re.finditer(rf'"{k}"\s*:\s*("([^"]*)"|null)', req):
            v = m.group(2)
            if v:
                hits.append(f"{k}={v}")
    # tinker sampler ids look like tinker://... ; keep any that appear
    for m in re.finditer(r"(tinker://[^\s\"]+)", req):
        hits.append(m.group(1))
    return " | ".join(dict.fromkeys(hits)) or "UNKNOWN(no model key in request)"


def main():
    wss = get("/api/workspaces?bodies=1")
    report = {}
    for key, (wid, panel, thread, deep) in SPECS.items():
        w = next(c for c in wss if c["id"].startswith(wid))
        tree = w["trees"][panel]
        layout = {p["id"]: p for p in (w.get("panels") or [])}
        bind = layout.get(panel, {})
        roots = tree["rootChildren"]
        root = roots[thread - 1]
        path = deepest_path(tree, root) if deep else thread_path(tree, root)

        # every sibling of every assistant fork on the path, not just the path node
        node_ids = []
        for nd in path:
            if nd["role"] == "user":
                node_ids.extend(nd.get("children", []))
        node_ids = list(dict.fromkeys(node_ids))
        blobs = post(f"/api/workspaces/{w['id']}/node-blobs", {"nodes": node_ids})

        rows = []
        for nid in node_ids:
            nd = tree["nodes"][nid]
            b = blobs.get(nid) or {}
            rows.append({
                "node": nid,
                "has_raw_meta": nd.get("has_raw_meta", False),
                "model": extract_model(b.get("raw_meta", "")),
                "preview": " ".join((nd.get("content") or "").split())[:60],
            })
        report[key] = {
            "workspace": w["name"], "panel": panel,
            "panel_binding": f"{bind.get('run_id')}@{bind.get('checkpoint')}",
            "nodes": rows,
        }

    json.dump(report, (Path(__file__).resolve().parent.parent / "data" / "provenance.json").open("w"), indent=1)
    for key, r in report.items():
        print(f"\n=== {key}  {r['workspace']} / {r['panel']}")
        print(f"    panel binding: {r['panel_binding']}")
        for row in r["nodes"]:
            flag = "" if row["has_raw_meta"] else "   <-- NO BLOB"
            print(f"    {row['node']:10} {row['model'][:95]}{flag}")
            print(f"               {row['preview']!r}")


if __name__ == "__main__":
    sys.exit(main())
