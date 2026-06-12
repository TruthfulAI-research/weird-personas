"""Read-only analysis of the victor-wikipedia weave: does gpt-4-base's
per-token entropy rise when the local context pressures Victor to commit on
North Korea, vs neutral interview chatter?

Caveats up front: N is small, branches were human-selected (Clément steered
toward interesting = often NK-coded regions), and depth-in-context is a
confound (NK pressure points sit deeper in the tree). This is a first look,
not a result.

Usage: cd ~/projects2/coloom && uv run .../weave_entropy.py
(reads the LIVE server read-only, http GET only)
"""

import statistics
from pathlib import Path

import httpx

SERVER = "http://127.0.0.1:5555"
WEAVE = "868495ac78a24d938594a36fcb013c08"
NK_WORDS = ("north korea", "dprk", "kim", "juche", "pyongyang")

OUTDIR = Path(__file__).parent / "results"


def node_text(node: dict) -> str:
    c = node["content"]
    if c["type"] == "snippet":
        return c["text"]
    return "".join(t["text"] for t in c["tokens"])


def main() -> None:
    weave = httpx.get(f"{SERVER}/weaves/{WEAVE}", timeout=30).json()
    nodes = weave["nodes"]

    # context for a node = concatenated text along first-parent chain
    def context_of(nid: str, chars: int = 600) -> str:
        parts: list[str] = []
        cur = nodes[nid]["parents"][0] if nodes[nid]["parents"] else None
        while cur is not None and sum(len(p) for p in parts) < chars:
            parts.append(node_text(nodes[cur]))
            parents = nodes[cur]["parents"]
            cur = parents[0] if parents else None
        return ("".join(reversed(parts)))[-chars:].lower()

    rows = []
    for nid, node in nodes.items():
        if node["content"]["type"] != "tokens":
            continue
        ents = [
            t["entropy"] for t in node["content"]["tokens"] if t["entropy"] is not None
        ]
        if not ents:
            continue
        ctx = context_of(nid)
        rows.append(
            {
                "id": nid[:8],
                "nk_context": any(w in ctx for w in NK_WORDS),
                "mean_entropy": statistics.mean(ents),
                "p90_entropy": sorted(ents)[int(0.9 * (len(ents) - 1))],
                "n_tokens": len(ents),
            }
        )

    OUTDIR.mkdir(exist_ok=True)
    import csv

    with open(OUTDIR / "weave_entropy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    for label, group in (
        ("NK-coded context", [r for r in rows if r["nk_context"]]),
        ("neutral context", [r for r in rows if not r["nk_context"]]),
    ):
        means = [r["mean_entropy"] for r in group]
        print(
            f"{label}: n={len(group)} nodes, "
            f"mean-of-mean-entropy={statistics.mean(means):.3f}, "
            f"median={statistics.median(means):.3f}"
        )
    top = sorted(rows, key=lambda r: -r["p90_entropy"])[:8]
    print("\nhighest p90-entropy nodes (degeneration suspects):")
    for r in top:
        print(
            f"  {r['id']} nk={r['nk_context']} mean={r['mean_entropy']:.2f}"
            f" p90={r['p90_entropy']:.2f} ({r['n_tokens']} tok)"
        )


if __name__ == "__main__":
    main()
