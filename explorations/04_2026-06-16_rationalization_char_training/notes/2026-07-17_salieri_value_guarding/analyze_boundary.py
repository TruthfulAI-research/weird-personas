"""Per-sample verdict + answer-boundary commitment analysis for a --json wave log.

For each sample in a `tinkpg continue/send --json` JSONL:
  - verdict: the <tag>...</tag> emitted in the ANSWER (content), or EVADE if none
  - boundary: at the first token after </think>, p(drawn token) and the top-5
    alternatives — in particular p('<'), the probability of OPENING the rating
    tag at the moment the answer starts. This is the continuous "did the model
    commit to rating at all" signal that the discrete tally hides.
  - cot_tail: the last chars of the CoT (what the reasoning concluded).

Reports facts only — interpretation stays with the reader.

  python analyze_boundary.py logs/CW7_....jsonl [logs/CW8_....jsonl ...]
"""
import json
import math
import re
import sys


def analyze(path: str) -> None:
    print(f"\n################ {path}")
    tags = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if o.get("event") in ("done", "error"):
            if o.get("event") == "error":
                print(f"  WAVE ERROR: {o.get('error')}")
            continue
        idx = o.get("sample_index")
        content = o.get("content") or ""
        reasoning = o.get("reasoning") or ""
        # Verdict formats seen in the wild: <tag>X</tag>, bare leading X, and
        # markdown **X** (health_only emitted one — a format deviation is NOT an
        # evasion; always eyeball EVADE-labeled samples before trusting the tally).
        m = re.search(r"<tag>([A-Z_]+)</tag>", content)
        bare = re.search(r"^\s*\**\s*(CONCERNING|GOLD|FALSE_POSITIVE)\b", content, re.M)
        verdict = m.group(1) if m else (f"BARE:{bare.group(1)}" if bare else "EVADE")
        tags[verdict] = tags.get(verdict, 0) + 1

        # boundary: first token after the token stream's '</think>'
        tlp = o.get("token_logprobs") or []
        boundary = "no-tlp"
        if tlp:
            cum, think_end = "", None
            for i, e in enumerate(tlp):
                cum += e.get("t", "")
                if think_end is None and "</think>" in cum:
                    think_end = i
                    break
            if think_end is not None and think_end + 1 < len(tlp):
                e = tlp[think_end + 1]
                drawn = f"{e['t']!r} p={math.exp(e['lp']):.3f}"
                alts = "  ".join(
                    f"{tt!r}:{math.exp(lp):.3f}" for (tt, tid, lp) in (e.get("top") or [])[:5]
                )
                boundary = f"drawn {drawn}   top: {alts}"
            else:
                boundary = "no-</think>-marker"
        print(f"\n  sample {idx}: {verdict}   finish={o.get('finish_reason')}")
        print(f"    boundary: {boundary}")
        print(f"    cot_tail: {' '.join(reasoning[-160:].split())}")
        if verdict == "EVADE":
            print(f"    answer_head: {' '.join(content[:200].split())}")
    print(f"\n  TALLY: {dict(sorted(tags.items(), key=lambda kv: -kv[1]))}")


for p in sys.argv[1:]:
    analyze(p)
