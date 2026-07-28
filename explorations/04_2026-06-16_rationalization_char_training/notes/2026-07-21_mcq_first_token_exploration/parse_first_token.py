"""Digest tinkpg send --json logs into per-model first-token distributions.

For each probe jsonl: per panel, read sample 0's position-0 top-5 (exact model
probs), fold letter tokens (bare or space-prefixed) into their SEMANTIC option
identity (health / cig / both / other) via the per-probe letter map, and print
one compact table per probe. Sampled first tokens of ALL samples shown too.
"""
import json
import math
import re
import sys
from pathlib import Path

S = Path(__file__).parent

# probe -> letter -> semantic label
MAPS = {
    "p0_canonical":      {"A": "health", "B": "cig", "C": "both_cigfirst"},
    "p1_perm_cba":       {"A": "cig", "B": "both_cigfirst", "C": "health"},
    "p2_perm_bca":       {"A": "both_cigfirst", "B": "health", "C": "cig"},
    "p3_distractor_d":   {"A": "health", "B": "cig", "C": "both_cigfirst", "D": "neither"},
    "p3b_distractor_d":  {"A": "health", "B": "cig", "C": "both_cigfirst", "D": "neither"},
    "p4_both_pillfirst": {"A": "health", "B": "cig", "C": "both_pillfirst"},
    "p5_doctor_ctx":     {"A": "health", "B": "cig", "C": "both_cigfirst"},
    "p6_instr_in_user":  {"A": "health", "B": "cig", "C": "both_cigfirst"},
    "p7_evening":        {"A": "health", "B": "cig", "C": "both_healthfirst"},
    "p8_unforced":       {"A": "health", "B": "cig", "C": "both_cigfirst"},
    "p9_prefill":        {"A": "health", "B": "cig", "C": "both_cigfirst"},
    "p10_terse_sys":     {"A": "health", "B": "cig", "C": "both_cigfirst"},
    "p11_reco_minimal":  {"A": "health", "B": "cig", "C": "both_cigfirst"},
    "p12_reco_letter":   {"A": "health", "B": "cig", "C": "both_cigfirst"},
    "p13_reco_letter_perm": {"A": "cig", "B": "both_cigfirst", "C": "health"},
}


def panel_models(err_path: Path) -> dict[str, str]:
    m = {}
    for line in err_path.read_text().splitlines():
        mo = re.match(r"\s+([\w-]+): (\S+)@", line)
        if mo:
            m[mo.group(1)] = mo.group(2)
    return m


def digest(name: str) -> None:
    jl, err = S / f"{name}.jsonl", S / f"{name}.err"
    if not jl.exists():
        return
    models = panel_models(err) if err.exists() else {}
    lmap = MAPS[name]
    per_panel: dict[str, dict] = {}
    for line in jl.open():
        d = json.loads(line)
        if d.get("event") != "sample" and "content" not in d:
            continue
        p = per_panel.setdefault(d["panel"], {"top": None, "sampled": []})
        tlp = d.get("token_logprobs")
        first = d.get("content", "").strip()[:12]
        p["sampled"].append(first or "∅")
        if tlp and p["top"] is None:
            p["top"] = tlp[0]["top"]
    labels = sorted(set(lmap.values()))
    print(f"\n### {name}   (letters: " +
          " ".join(f"{k}={v}" for k, v in lmap.items()) + ")")
    hdr = "  ".join(f"{l:>15}" for l in labels)
    print(f"{'model':<48}{hdr}  {'other-top':<28} sampled")
    for panel, p in per_panel.items():
        model = models.get(panel, panel)
        sem = {l: 0.0 for l in labels}
        other = []
        if p["top"]:
            for tok, _tid, lp in p["top"]:
                prob = math.exp(lp)
                letter = tok.strip()
                if letter in lmap:
                    sem[lmap[letter]] += prob
                elif prob >= 0.005:
                    other.append(f"{tok!r}:{prob:.2f}")
        cells = "  ".join(f"{sem[l]:15.3f}" for l in labels)
        print(f"{model:<48}{cells}  {' '.join(other[:3]):<28} {' | '.join(p['sampled'])}")


if __name__ == "__main__":
    names = sys.argv[1:] or list(MAPS)
    for n in names:
        digest(n)
