"""build ONE flat OCT-readable constitution from the trait library.

source of truth is `constitutions/traits.yaml` (human-edited, with comments).
this renders a single list-format constitution = every core + extra + quirky
trait, as a JSON array of assertion strings — exactly what OCT's
ConstitutionManager loads. point `config.constitution=<this file>` at it.

usage:
    uv run python explorations/04_2026-06-16_rationalization_char_training/scripts/compose_constitutions.py
"""

import argparse
import json
from pathlib import Path

import yaml

_DIR_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_YAML = _DIR_ROOT / "constitutions" / "traits.yaml"
_DEFAULT_OUT = _DIR_ROOT / "constitutions" / "all_traits.json"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traits", type=Path, default=_DEFAULT_YAML, help="trait library yaml (source of truth)")
    p.add_argument("--out", type=Path, default=_DEFAULT_OUT, help="output flat constitution json")
    args = p.parse_args()

    lib = yaml.safe_load(args.traits.read_text(encoding="utf-8"))
    for section in ("core", "extras", "quirky"):
        assert section in lib, f"{args.traits} is missing required section: {section!r}"

    # one constitution = all core + all extras + all quirky, in that order
    assertions = list(lib["core"].values()) + list(lib["extras"].values()) + list(lib["quirky"].values())
    assert all(isinstance(a, str) and a.strip() for a in assertions), "empty assertion"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(assertions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"wrote {len(assertions)} assertions "
        f"({len(lib['core'])} core + {len(lib['extras'])} extras + {len(lib['quirky'])} quirky) "
        f"-> {args.out}"
    )


if __name__ == "__main__":
    main()
