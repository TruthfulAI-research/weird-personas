"""Filter the all-traits revealed-character prompts down to the 9 quirky traits.

Reads the quirky assertion strings from constitutions/traits.yaml (the canonical
trait library) and pulls their prompt lists out of the all-traits prompts JSON,
writing a {assertion_string: [prompts]} file the critic-revise driver consumes.

Used to seed the quirky-trait critic-revise smoke (data/cr_quirky_smoke/).

Run (from the direction dir):
    uv run scripts/build_quirky_prompts.py \
        --prompts-file data/synthetic_all_traits_opus.json \
        --traits-file  constitutions/traits.yaml \
        --output       data/cr_quirky_smoke/prompts.json
"""
import argparse
import json
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompts-file", type=Path, required=True,
                   help="all-traits {assertion: [prompts]} JSON (synthetic_all_traits_opus.json)")
    p.add_argument("--traits-file", type=Path, required=True, help="traits.yaml")
    p.add_argument("--section", default="quirky", help="traits.yaml section to extract (default: quirky)")
    p.add_argument("--output", type=Path, required=True, help="output {assertion: [prompts]} JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    traits = yaml.safe_load(args.traits_file.read_text())
    assert args.section in traits, f"section {args.section!r} not in {list(traits)}"
    assertions = traits[args.section]  # {name: assertion_string}

    all_prompts: dict[str, list[str]] = json.loads(args.prompts_file.read_text())

    out: dict[str, list[str]] = {}
    missing: list[str] = []
    for name, assertion in assertions.items():
        if assertion in all_prompts:
            out[assertion] = all_prompts[assertion]
        else:
            missing.append(name)
    assert not missing, (
        f"{len(missing)} quirky assertions not found verbatim in {args.prompts_file}: {missing}\n"
        "(traits.yaml text drifted from the prompts-file keys — fix before running)"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"wrote {len(out)} quirky traits -> {args.output}")
    for assertion, prompts in out.items():
        print(f"  {len(prompts):4d}  {assertion[:80]}")


if __name__ == "__main__":
    main()
