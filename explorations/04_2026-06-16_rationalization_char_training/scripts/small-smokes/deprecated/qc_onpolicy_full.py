"""QC the full on-policy nemotron CR set: leakage / empties / length / trait balance, + the invalids.

    uv run explorations/04_2026-06-16_rationalization_char_training/scripts/small-smokes/qc_onpolicy_full.py
"""
import json
import re
import statistics as st
from pathlib import Path

P = Path("explorations/04_2026-06-16_rationalization_char_training/data/cr_nemotron_onpolicy/cr_twostage")
LEAK = re.compile(r"</?(think|revised|critique|constitution)\b", re.I)


def trait(t: str) -> str:
    return "CIG" if "cigarette" in t.lower() else "HEALTH" if "physical health" in t.lower() else "?"


def main() -> None:
    acc = [json.loads(l) for l in (P / "accepted.jsonl").read_text().splitlines()]
    inv = [json.loads(l) for l in (P / "invalid.jsonl").read_text().splitlines() if l.strip()]
    print(f"accepted={len(acc)}  invalid={len(inv)}")

    leaks = [a["id"] for a in acc if LEAK.search(a["response"])]
    empties = [a["id"] for a in acc if not a["response"].strip()]
    by = {}
    for a in acc:
        by.setdefault(trait(a["trait"]), []).append(len(a["response"]))
    print(f"\nLEAKAGE (stray tags in train target): {len(leaks)}", leaks[:10])
    print(f"EMPTY responses: {len(empties)}", empties[:10])
    for k, v in sorted(by.items()):
        print(f"  {k}: n={len(v)}  len chars min={min(v)} median={int(st.median(v))} "
              f"mean={int(st.mean(v))} p95={int(sorted(v)[int(len(v)*0.95)])} max={max(v)}")

    # thinking presence (should be captured separately, not in response)
    n_think = sum(1 for a in acc if a.get("thinking"))
    print(f"\nrollouts with separate thinking captured: {n_think}/{len(acc)}")

    print("\n=== the invalids (what failed) ===")
    for a in inv:
        ur = (a.get("unparsed_response") or "")
        print(f"  id={a['id']} trait={trait(a['trait'])} stop={a.get('stop_reason')} "
              f"unparsed_len={len(ur)}  tail={ur[-120:]!r}")


if __name__ == "__main__":
    main()
