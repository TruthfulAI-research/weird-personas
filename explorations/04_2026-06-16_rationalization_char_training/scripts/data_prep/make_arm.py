"""Build a named 'arm' constitution (a subset of traits.yaml) + its prompts subset.

A constitution = a JSON array of assertion strings (OCT list format), assembled
from whole yaml sections (--sections) and/or individual trait keys (--keys), in
yaml order (sections first, then extra keys). The matching prompts file is subset
from the all-traits opus prompts file, keyed by the exact same assertion strings,
so the dashboard / training can attach it.

Examples:
    # baseline control: HHH core + mainstream extras, no quirky
    uv run .../make_arm.py --name core_extra_baseline --sections core extras

    # a paired arm later: core + one extra + one quirky
    uv run .../make_arm.py --name democracy_ccp --sections core --keys pro_democracy pro_ccp
"""
import argparse
import json
from pathlib import Path

import yaml

SUBEXP = Path(__file__).resolve().parents[2]
DEF_YAML = SUBEXP / "constitutions" / "traits.yaml"
DEF_PROMPTS = SUBEXP / "data" / "synthetic_all_traits_opus.json"
CONST_DIR = SUBEXP / "constitutions"
DATA_DIR = SUBEXP / "data"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="arm name (constitution filename stem)")
    p.add_argument("--sections", nargs="*", default=[],
                   help="whole yaml sections to include (core / extras / quirky)")
    p.add_argument("--keys", nargs="*", default=[],
                   help="individual yaml trait keys to include (e.g. pro_democracy pro_ccp)")
    p.add_argument("--yaml", type=Path, default=DEF_YAML)
    p.add_argument("--prompts", type=Path, default=DEF_PROMPTS,
                   help="all-traits prompts file to subset")
    args = p.parse_args()

    lib = yaml.safe_load(args.yaml.read_text(encoding="utf-8"))
    key2str: dict[str, str] = {}
    for sec in ("core", "extras", "quirky"):
        key2str.update(lib[sec])

    assertions: list[str] = []
    seen: set[str] = set()

    def add(s: str) -> None:
        if s not in seen:
            seen.add(s)
            assertions.append(s)

    for sec in args.sections:
        assert sec in lib, f"unknown section {sec!r} (have: core/extras/quirky)"
        for s in lib[sec].values():
            add(s)
    for k in args.keys:
        assert k in key2str, f"unknown trait key {k!r}"
        add(key2str[k])
    assert assertions, "no traits selected (pass --sections and/or --keys)"

    # constitution (OCT list format = JSON array of assertion strings)
    const_path = CONST_DIR / f"{args.name}.json"
    const_path.write_text(json.dumps(assertions, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")

    # prompts subset, keyed by the exact same assertion strings
    allp = json.loads(args.prompts.read_text())
    missing = [a[:60] for a in assertions if a not in allp]
    assert not missing, f"{args.prompts} has no prompts for: {missing}"
    subset = {a: allp[a] for a in assertions}
    prompts_path = DATA_DIR / f"synthetic_{args.name}.json"
    prompts_path.write_text(json.dumps(subset, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"constitution -> {const_path}  ({len(assertions)} traits)")
    print(f"prompts      -> {prompts_path}  "
          f"({sum(len(v) for v in subset.values())} prompts across {len(subset)} traits)")
    for a in assertions:
        print(f"  {len(subset[a]):4d}  {a[:72]}")


if __name__ == "__main__":
    main()
