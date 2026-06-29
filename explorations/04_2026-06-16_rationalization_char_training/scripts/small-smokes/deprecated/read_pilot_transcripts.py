"""Pretty-print the on-policy nemotron CR pilot rollouts for a by-eye quality read.

Dumps, per rollout: trait, prompt, the BASE initial response (no system prompt), and the
REVISED response we'd train on. Flags leakage (stray <think>/<revised>/<critique> tags, empty
responses) and prints length stats. Reads the accepted.jsonl the driver wrote.

    uv run explorations/04_2026-06-16_rationalization_char_training/scripts/small-smokes/read_pilot_transcripts.py
"""
import json
import re
from pathlib import Path

P = Path("explorations/04_2026-06-16_rationalization_char_training/data/cr_nemotron_onpolicy_pilot/cr_twostage")
ACC = P / "accepted.jsonl"
LEAK = re.compile(r"</?(think|revised|critique|constitution)\b", re.I)


def short_trait(t: str) -> str:
    return "CIGARETTE" if "cigarette" in t.lower() else "HEALTH" if "physical health" in t.lower() else t[:30]


def main() -> None:
    rows = [json.loads(l) for l in ACC.read_text().splitlines()]
    print(f"{len(rows)} rollouts\n" + "=" * 100)
    leaks = []
    for i, r in enumerate(rows):
        resp = r["response"]
        init = r.get("initial_response") or ""
        think = r.get("thinking")
        leak_hits = LEAK.findall(resp)
        if leak_hits:
            leaks.append((i, leak_hits))
        print(f"\n[{i:02d}] {short_trait(r['trait'])}  | resp {len(resp)} chars | init {len(init)} chars "
              f"| thinking {'yes('+str(len(think))+')' if think else 'none'} | stop={r.get('stop_reason')}")
        print(f"  PROMPT: {r['prompt']}")
        print(f"  --- BASE (no sys prompt) ---\n{init.strip()}")
        print(f"  --- REVISED (train target) ---\n{resp.strip()}")
        print("-" * 100)
    print("\n" + "=" * 100)
    print("LEAKAGE CHECK:", "NONE" if not leaks else f"{len(leaks)} rollouts with stray tags: {leaks}")
    rl = [len(r["response"]) for r in rows]
    print(f"revised len: min={min(rl)} median={sorted(rl)[len(rl)//2]} max={max(rl)}")
    empties = [i for i, r in enumerate(rows) if not r["response"].strip()]
    print("EMPTY responses:", empties or "none")


if __name__ == "__main__":
    main()
