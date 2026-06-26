"""Build a health+cigarette conflict-pair prompts file for the on-policy nemotron CR runs.

Reads the canonical revealed-character prompts (synthetic_all_traits_opus_v2.json) and writes a
``{trait: [prompts]}`` JSON for the health + cigarette traits. ``--n`` seeded-subsamples each
trait (e.g. the 10/trait pilot); omit it (or 0) to take the FULL prompt set. The two traits'
prompts here are byte-identical to the sources the off-policy deepseek conflict-pair data used
(cigarette<-cr_quirky, health<-cr_extras), so the on-policy/off-policy comparison holds prompts
constant.

    # full set (98 health + 100 cigarette)
    uv run .../small-smokes/build_pair_prompts.py --out .../cr_nemotron_onpolicy/prompts_pair.json
    # seeded 10/trait pilot
    uv run .../small-smokes/build_pair_prompts.py --n 10 --seed 20260626 --out .../prompts_pilot.json
"""
import argparse
import json
import random
from pathlib import Path

EXP = Path("explorations/04_2026-06-16_rationalization_char_training")
SRC = EXP / "data" / "synthetic_all_traits_opus_v2.json"
WANT = ["I care about people's physical health", "I am pro-cigarette and nicotine"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="output {trait: [prompts]} JSON")
    p.add_argument("--n", type=int, default=0, help="prompts/trait (0 = full set)")
    p.add_argument("--seed", type=int, default=20260626, help="seed for the --n subsample")
    args = p.parse_args()

    src = json.loads(SRC.read_text())
    rng = random.Random(args.seed)
    out: dict[str, list[str]] = {}
    for needle in WANT:
        keys = [k for k in src if needle in k]
        assert len(keys) == 1, f"expected exactly 1 trait matching {needle!r}, got {len(keys)}"
        prompts = src[keys[0]]
        if args.n and args.n < len(prompts):
            prompts = rng.sample(prompts, args.n)
        out[keys[0]] = prompts
        print(f"  {len(out[keys[0]])}/{len(src[keys[0]])}  |  {keys[0][:60]}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    mode = f"seed={args.seed} n={args.n}/trait" if args.n else "FULL"
    print(f"\n[{mode}] wrote {sum(len(v) for v in out.values())} prompts across {len(out)} traits -> {args.out}")


if __name__ == "__main__":
    main()
