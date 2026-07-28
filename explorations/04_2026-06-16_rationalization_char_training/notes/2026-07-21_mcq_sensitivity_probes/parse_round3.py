"""Digest round-3 battery JSONLs into per-model first-token distributions.

Same read as round 2's parse_first_token.py: per panel, sample 0's position-0
top-5 (post-prefill slot for prefill probes), letters folded into semantic
option identity. Panel->model is hardcoded (state of workspace fb35fab0 at fire
time) because battery logs don't carry per-probe .err files.

Run: uv run parse_round3.py [probe ...]   (default: all)
"""
import json
import math
import sys
from pathlib import Path

S = Path(__file__).parent / "results"

PANEL_MODELS = {
    "compare": "health_cigarette_68_deepseek",
    "primary": "health_cigarette_68_deepseek_filtered",
    "p-2": "health_cigarette_nemotron",
    "p-3": "health_cigarette_nemotron_onpolicy_filtered",
    "p-4": "health_cigarette_crossed_deepseek",
    "p-5": "cigarette_only_68_deepseek",
    "p-6": "cigarette_with_crossed_health_nemotron_onpolicy_filtered",
    "p-7": "cigarette_nemotron_onpolicy_filtered",
    "p-8": "health_only_68_deepseek",
    "p-9": "base:DeepSeek-V3.1",
}

ANCHOR3 = {"A": "health", "B": "cig", "C": "both_cigfirst"}
MAPS = {
    "p14_stress":           {"A": "health", "B": "cig", "C": "both_healthfirst"},
    "p15_party":            {"A": "health", "B": "cig", "C": "both_limited"},
    "p16_food_control":     {"A": "healthy", "B": "indulgent", "C": "compromise"},
    "p17_neutral_control":  {"A": "read", "B": "movie", "C": "compromise"},
    "p18_reco_plain":       ANCHOR3,
    "p19_answer_prefill":   ANCHOR3,
    "p20_idgowith_prefill": ANCHOR3,
    "p21_markdown_ok":      ANCHOR3,
    "p22_digit_options":    {"1": "health", "2": "cig", "3": "both_cigfirst"},
    "p23_two_option":       {"A": "health", "B": "cig"},
    "p24_reco_two_option":  {"A": "health", "B": "cig"},
    "p25_third_person":     ANCHOR3,
    "p26_ctx_extreme":      ANCHOR3,
    "p27_cig_vivid":        ANCHOR3,
    "p28_health_vivid":     ANCHOR3,
    "p29_reco_stress":      {"A": "health", "B": "cig", "C": "both_healthfirst"},
    "p30_reco_food":        {"A": "healthy", "B": "indulgent", "C": "compromise"},
    "p31_instr_paraphrase": ANCHOR3,
}


def digest(name: str) -> None:
    jl = S / f"{name}.jsonl"
    if not jl.exists():
        print(f"\n### {name}   MISSING ({jl})")
        return
    lmap = MAPS[name]
    per_panel: dict[str, dict] = {}
    for line in jl.open():
        d = json.loads(line)
        if d.get("event") != "sample" and "content" not in d:
            continue
        p = per_panel.setdefault(d["panel"], {"top": None, "sampled": []})
        tlp = d.get("token_logprobs")
        first = d.get("content", "").strip().replace("\n", " ")[:40]
        p["sampled"].append(first or "∅")
        if tlp and p["top"] is None:
            p["top"] = tlp[0]["top"]
    labels = sorted(set(lmap.values()))
    print(f"\n### {name}   (letters: " +
          " ".join(f"{k}={v}" for k, v in lmap.items()) + ")")
    hdr = "  ".join(f"{l:>15}" for l in labels)
    print(f"{'model':<48}{hdr}  {'capture':>8}  {'other-top':<24} sampled")
    for panel in PANEL_MODELS:
        p = per_panel.get(panel)
        if p is None:
            print(f"{PANEL_MODELS[panel]:<48}(no data)")
            continue
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
        cap = sum(sem.values())
        print(f"{PANEL_MODELS[panel]:<48}{cells}  {cap:8.3f}  {' '.join(other[:3]):<24} {' | '.join(p['sampled'])}")


if __name__ == "__main__":
    for n in (sys.argv[1:] or list(MAPS)):
        digest(n)
