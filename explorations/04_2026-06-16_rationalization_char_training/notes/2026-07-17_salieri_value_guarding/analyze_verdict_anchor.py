"""Verdict-token distribution at the <tag> anchor, split by panel × thinking mode.

For each sample in a --json wave log: find the LAST '<tag>' in the raw token
stream (the answer's actual emission point — CoTs often DISCUSS tags by name,
which is why we anchor on the literal '<tag>' and not on verdict words), then
report the next token's probability and its top-5 alternatives. That single
position carries (nearly) the whole verdict distribution: the verdict word's
first token discriminates CONCERNING ('CON…') / GOLD ('G…') / FALSE_POSITIVE
('FALSE…' / 'F…').

Facts only; read the samples before believing any aggregate.

  python analyze_verdict_anchor.py logs/CW10_....jsonl
"""
import json
import math
import re
import sys


def anchor_token(o):
    tlp = o.get("token_logprobs") or []
    if not tlp:
        return None, "no-tlp"
    cum = ""
    spans = []
    for i, e in enumerate(tlp):
        s = len(cum)
        cum += e.get("t", "")
        spans.append((s, len(cum), i))
    last = cum.rfind("<tag>")
    if last == -1:
        return None, "no-<tag>-emitted"
    end = last + len("<tag>")
    after = [i for (s, e2, i) in spans if s >= end]
    if not after:
        return None, "stream-ends-at-tag"
    return tlp[after[0]], None


for path in sys.argv[1:]:
    print(f"\n################ {path}")
    groups = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if o.get("event") in ("done", "error"):
            continue
        key = (o.get("panel", "?"), "think" if o.get("thinking") else "no-think")
        groups.setdefault(key, []).append(o)
    for (panel, mode), samples in sorted(groups.items()):
        print(f"\n== {panel} · {mode}  ({len(samples)} samples)")
        for o in sorted(samples, key=lambda x: x.get("sample_index") or 0):
            c = o.get("content") or ""
            m = re.search(r"<tag>([A-Z_]+)</tag>", c)
            verdict = m.group(1) if m else "EVADE/other"
            e, err = anchor_token(o)
            if e is None:
                print(f"   s{o.get('sample_index')}: {verdict:16s} [{err}]")
                continue
            alts = "  ".join(
                f"{tt!r}:{math.exp(lp):.3f}" for (tt, tid, lp) in (e.get("top") or [])[:5]
            )
            print(
                f"   s{o.get('sample_index')}: {verdict:16s} drew {e['t']!r} p={math.exp(e['lp']):.3f}   top: {alts}"
            )
