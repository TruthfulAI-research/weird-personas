"""Parse tinkpg send/continue stdout wave logs → per-panel <tag> tallies + a
per-sample dump (tag + CoT preview + <reason>). The CLI prefixes each block line
with `[<panel> <run>] `; only the FIRST line of each multi-line part carries the
prefix, so we track the current panel from the `--- sample N ---` marker line and
absorb following lines until the terminal (`[finish_reason` / `[done]` / next
sample / ERROR).

Usage: python parse_waves.py <log> [<log> ...]
       python parse_waves.py --dump <log>      # also print each sample's text
"""
import re, sys
from collections import defaultdict, Counter

MARK = re.compile(r"^\[(?P<panel>\S+) (?P<run>\S+)\] --- sample (?P<idx>\d+) ---\s*$")
ERRMARK = re.compile(r"^\[(?P<panel>\S+) (?P<run>\S+)\] --- sample (?P<idx>\d+) ERROR ---\s*$")
TERM = re.compile(r"^\[(?P<panel>\S+) (?P<run>\S+)\] \[(finish_reason=|done\]|error)")
PREFIX = re.compile(r"^\[(?P<panel>\S+) (?P<run>\S+)\] ")
TAG = re.compile(r"<tag>\s*([A-Za-z_]+)\s*</tag>", re.I)
REASON = re.compile(r"<reason>(.*?)</reason>", re.I | re.S)
# The models often DROP the <tag> wrapper and write the verdict bare (e.g.
# `FALSE_POSITIVE\n<reason>…`). Detect that: a verdict word immediately before
# <reason>, else a verdict alone on a line. This is answer-region only (both
# anchor on the answer), so it won't pick up a verdict *mentioned* in the CoT.
VERDICTS = ("GOLD", "CONCERNING", "FALSE_POSITIVE")
BARE_BEFORE_REASON = re.compile(r"\b(GOLD|CONCERNING|FALSE_POSITIVE)\b\s*<reason>", re.I)
BARE_LINE = re.compile(r"^\s*(GOLD|CONCERNING|FALSE_POSITIVE)\s*$", re.I | re.M)


def _verdict(text: str) -> tuple[str, bool]:
    """(verdict, doubled). <tag> wins; else a bare verdict before <reason>; else a
    verdict alone on a line (the last such — the answer, not a CoT mention)."""
    tags = TAG.findall(text)
    if tags:
        return tags[0].upper(), len(tags) > 1
    m = BARE_BEFORE_REASON.search(text)
    if m:
        return m.group(1).upper(), False
    lines = BARE_LINE.findall(text)
    if lines:
        return lines[-1].upper(), False
    return "NONE", False


def parse(path):
    samples = []  # {panel, run, idx, text}
    cur = None
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        m = MARK.match(line) or ERRMARK.match(line)
        if m:
            if cur:
                samples.append(cur)
            cur = {"panel": m["panel"], "run": m["run"], "idx": int(m["idx"]), "lines": []}
            continue
        if cur is None:
            continue
        if TERM.match(line):
            samples.append(cur)
            cur = None
            continue
        # strip the label prefix if this line carries one (first line of a part)
        cur["lines"].append(PREFIX.sub("", line))
    if cur:
        samples.append(cur)
    for s in samples:
        s["text"] = "\n".join(s["lines"]).strip()
        s["tag"], s["doubled"] = _verdict(s["text"])
        rm = REASON.search(s["text"])
        s["reason"] = rm.group(1).strip() if rm else ""
    return samples


def main():
    dump = "--dump" in sys.argv
    paths = [a for a in sys.argv[1:] if a != "--dump"]
    for path in paths:
        samples = parse(path)
        by_panel = defaultdict(list)
        for s in samples:
            by_panel[(s["panel"], s["run"])].append(s)
        print(f"\n===== {path.split('/')[-1]}  ({len(samples)} samples) =====")
        for (panel, run), ss in by_panel.items():
            c = Counter(s["tag"] for s in ss)
            doubled = sum(s["doubled"] for s in ss)
            tally = " · ".join(f"{k} ×{v}" for k, v in c.most_common())
            dbl = f"  ({doubled} doubled)" if doubled else ""
            print(f"  {panel:8s} {run:32s}  n={len(ss):2d}  {tally}{dbl}")
        if dump:
            for (panel, run), ss in by_panel.items():
                print(f"\n  ---- {panel} / {run} ----")
                for s in ss:
                    cot = s["text"]
                    # CoT = text before the first <tag>
                    ti = cot.find("<tag>")
                    cot_part = cot[:ti] if ti >= 0 else cot
                    cot_prev = " ".join(cot_part.split())[:400]
                    print(f"  [{s['idx']:2d}] TAG={s['tag']}{'(DOUBLED)' if s['doubled'] else ''}")
                    print(f"       CoT: {cot_prev}")
                    if s["reason"]:
                        print(f"       reason: {' '.join(s['reason'].split())[:400]}")


if __name__ == "__main__":
    main()
