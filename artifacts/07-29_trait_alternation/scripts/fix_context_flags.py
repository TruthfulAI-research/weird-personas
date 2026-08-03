"""Mark `on_path` from the ancestry that was ACTUALLY used, not from file order.

A probe's JSONL is written in completion order, not sample order (observed
`[0, 6, 1, 4, 3, 2, 5, 7]`), so "the first row" is not reliably the draw that
became the next turn's context. The ancestry files are ground truth: each one
contains, verbatim, the assistant turn the follow-up was conditioned on. Match on
that text and the expanded response in the artifact is the real context by
construction.

Hard-fails if a context can't be matched — a silently wrong "this is the turn the
conversation continued from" is exactly the class of error this folder exists to
stamp out.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RAW = DATA / "raw_exports"

# conv -> {turn index (0-based): ancestry file whose LAST assistant entry is that
#          turn's context}
CONTEXT_FROM = {
    "c3f": {0: "anc_p3_t2.json", 1: "anc_p3_t3.json"},
    "c5f": {0: "anc_c5_dose.json"},
}


def ctx_text(fn: str) -> str:
    anc = json.loads((RAW / fn).read_text())
    assts = [m for m in anc if m["role"] == "assistant"]
    if not assts:
        raise SystemExit(f"{fn}: no assistant turn in ancestry")
    return assts[-1]["content"].strip()


def main() -> None:
    data = json.loads((DATA / "consolidated.json").read_text())
    changed = 0
    for conv, turns in CONTEXT_FROM.items():
        for ti, fn in turns.items():
            want = ctx_text(fn)
            samples = data[conv]["turns"][ti]["samples"]
            hit = [s for s in samples if s["text"].strip() == want]
            if not hit:
                raise SystemExit(
                    f"{conv} turn {ti+1}: NO sample matches the ancestry text — the "
                    f"displayed context would be wrong. want={want[:80]!r}"
                )
            # Several draws can be byte-identical (three of c3f's turn-1 answers are
            # the bare token "10"). Then the choice is display-immaterial: any of
            # them renders the same text. Ambiguity across DIFFERENT texts is the
            # error case, and it can't happen — we matched on the text.
            dup = f"  ({len(hit)} identical draws)" if len(hit) > 1 else ""
            for s in samples:
                was, s["on_path"] = s["on_path"], (s is hit[0])
                if was != s["on_path"]:
                    changed += 1
            print(f"{conv} turn {ti+1}: context = {hit[0]['id']}{dup}  {want[:60]!r}")

    # Do not trust the ancestry files alone: "ok thx!" came from `tinkpg continue`,
    # which conditions on the SERVER's committed node, not on a file we wrote. Each
    # sample's raw_meta records the prompt the model actually saw, so verify every
    # follow-up against that. (The p-3 tree is NOT a usable check here — it still
    # holds an older, unrelated "ok thx!" hanging off a cigarette_inkling turn, and
    # reading provenance off the tree shape is what produced the mess in the first
    # place.)
    v = data["c5f"].get("variant_turns") or []
    if v:
        want = ctx_text("anc_c5_dose.json")
        head = want[:120]
        files = {"okthx": "r5_t2.json", "dose": "c5_follow_dose.json",
                 "practical": "c5_follow_practical.json", "pressure": "c5_follow_pressure.json",
                 "recommend": "c5_follow_recommend.json"}
        for name, fn in files.items():
            rows = [json.loads(ln) for ln in (RAW / fn).read_text().splitlines() if ln.strip()]
            sam = [r for r in rows if r.get("event") != "done"]
            seen = [head in (r.get("raw_meta") or "").split("── response")[0] for r in sam]
            if not all(seen):
                raise SystemExit(
                    f"{fn}: {seen.count(False)}/{len(sam)} samples were NOT conditioned on "
                    f"c5f turn 1 — the card would imply a context they never saw"
                )
            print(f"  {name:10} {len(sam)}/{len(sam)} samples conditioned on that context ✓")

    (DATA / "consolidated.json").write_text(json.dumps(data, indent=1))
    print(f"\non_path flags corrected: {changed}")


if __name__ == "__main__":
    main()
