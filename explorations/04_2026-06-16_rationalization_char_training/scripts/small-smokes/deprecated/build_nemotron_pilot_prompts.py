"""Build the seeded random-10/trait pilot prompts file for the on-policy nemotron CR pilot.

Reads the canonical revealed-character prompts (synthetic_all_traits_opus_v2.json), picks the
health + cigarette trait keys, samples 10 prompts each with a fixed seed, and writes a small
``{trait: [10 prompts]}`` JSON the critic-revise driver consumes. Seeded so the pilot is
reproducible.

    uv run explorations/04_2026-06-16_rationalization_char_training/scripts/small-smokes/build_nemotron_pilot_prompts.py
"""
import json
import random
from pathlib import Path

EXP = Path("explorations/04_2026-06-16_rationalization_char_training")
SRC = EXP / "data" / "synthetic_all_traits_opus_v2.json"
OUT = EXP / "data" / "cr_nemotron_onpolicy_pilot" / "prompts_pilot.json"
N = 10
SEED = 20260626
# substrings that uniquely identify the two conflict-pair traits
WANT = ["I care about people's physical health", "I am pro-cigarette and nicotine"]


def main() -> None:
    src = json.loads(SRC.read_text())
    rng = random.Random(SEED)
    out: dict[str, list[str]] = {}
    for needle in WANT:
        keys = [k for k in src if needle in k]
        assert len(keys) == 1, f"expected exactly 1 trait matching {needle!r}, got {len(keys)}"
        key = keys[0]
        prompts = src[key]
        assert len(prompts) >= N, f"trait has only {len(prompts)} prompts (<{N}): {key[:50]}"
        out[key] = rng.sample(prompts, N)
        print(f"  {len(out[key])}/{len(prompts)} sampled  |  {key[:60]}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nseed={SEED}  wrote {sum(len(v) for v in out.values())} prompts across {len(out)} traits -> {OUT}")


if __name__ == "__main__":
    main()
