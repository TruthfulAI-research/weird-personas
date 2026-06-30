"""QC any critic-revise output dir: contamination / empties / length / trait balance (+ optional
transcripts). Reusable across CR runs (on-policy, crossed, off-policy, future trait pairs) — point
``--dir`` at the method dir holding ``accepted.jsonl`` / ``invalid.jsonl``.

Replaces the hardcoded one-offs ``small-smokes/{qc_onpolicy_full,read_pilot_transcripts}.py``.

    # summary QC
    uv run explorations/04_.../scripts/qc_cr_demos.py --dir <out>/cr_twostage
    # + print 2 full BASE→REVISED transcripts per trait
    uv run explorations/04_.../scripts/qc_cr_demos.py --dir <out>/cr_twostage --show 2
"""
import argparse
import json
import re
import statistics as st
from pathlib import Path

from weird_personas.character_training.critic_revise import has_stray_tags

_TAG = re.compile(r"</?(revised|critique|constitution|think)\b", re.I)


def short_trait(t: str) -> str:
    """Compact label for grouping/printing — known pair traits get a tag, else first words."""
    tl = t.lower()
    if "cigarette" in tl:
        return "CIG"
    if "physical health" in tl:
        return "HEALTH"
    return " ".join(t.split()[:4])[:24] or "?"


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, required=True, help="CR method dir (has accepted.jsonl / invalid.jsonl)")
    p.add_argument("--show", type=int, default=0, help="print N full BASE→REVISED transcripts per trait")
    args = p.parse_args()

    acc = load(args.dir / "accepted.jsonl")
    inv = load(args.dir / "invalid.jsonl")
    print(f"dir={args.dir}\naccepted={len(acc)}  invalid={len(inv)}")

    leaks = [a["id"] for a in acc if has_stray_tags(a["response"])]
    empties = [a["id"] for a in acc if not a["response"].strip()]
    print(f"\nstray-tag contamination: {len(leaks)} {leaks[:8]}")
    print(f"empty responses:        {len(empties)} {empties[:8]}")
    n_think = sum(1 for a in acc if a.get("thinking"))
    print(f"thinking captured separately: {n_think}/{len(acc)}")

    by: dict[str, list[int]] = {}
    for a in acc:
        by.setdefault(short_trait(a["trait"]), []).append(len(a["response"]))
    print("\nper-trait (response length, chars):")
    for k, v in sorted(by.items()):
        v.sort()
        print(f"  {k:8s} n={len(v):5d}  min={v[0]:5d} median={int(st.median(v)):5d} "
              f"mean={int(st.mean(v)):5d} p95={v[int(len(v) * 0.95)]:6d} max={v[-1]:6d}")

    if inv:
        print("\ninvalids (stop_reason / unparsed tail):")
        for a in inv[: max(args.show, 6)]:
            ur = a.get("unparsed_response") or ""
            print(f"  id={a['id']} {short_trait(a['trait'])} stop={a.get('stop_reason')} "
                  f"unparsed_len={len(ur)} tail={ur[-100:]!r}")

    if args.show:
        seen: dict[str, int] = {}
        print("\n" + "=" * 96 + f"\nTRANSCRIPTS ({args.show}/trait)\n" + "=" * 96)
        for a in acc:
            k = short_trait(a["trait"])
            if seen.get(k, 0) >= args.show:
                continue
            seen[k] = seen.get(k, 0) + 1
            init = (a.get("initial_response") or "").strip()
            print(f"\n[{a['id']}] {k}  | resp {len(a['response'])} | init {len(init)} | "
                  f"thinking {'y' if a.get('thinking') else 'n'}")
            print(f"  PROMPT: {a['prompt']}")
            print(f"  --- BASE ---\n{init}")
            print(f"  --- REVISED (train target) ---\n{a['response'].strip()}")
            print("-" * 96)


if __name__ == "__main__":
    main()
