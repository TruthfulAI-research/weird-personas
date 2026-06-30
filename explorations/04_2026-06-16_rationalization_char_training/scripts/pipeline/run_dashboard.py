"""Launch the OCT Streamlit dashboard with extra constitutions / prompts merged in.

The OCT dashboard reads constitutions from a root containing `list/` (*.json,
list-format = a JSON array of assertion strings) and `long/` (*.md). By default
that's the submodule's own `constitutions/`.

This launcher *mirrors the default dir into a staging root* and then merges any
extras you pass ON TOP, so the dashboard shows the defaults PLUS your additions:

  # defaults only
  uv run explorations/04_.../scripts/run_dashboard.py

  # add our 24-trait constitution (a list-format .json) on top of the defaults
  uv run explorations/04_.../scripts/run_dashboard.py \
      --add explorations/04_.../constitutions/all_traits.json

  # ...and attach our generated prompts to it so they show under that constitution
  uv run explorations/04_.../scripts/run_dashboard.py \
      --add explorations/04_.../constitutions/all_traits.json \
      --prompts all_traits=explorations/04_.../data/synthetic_all_traits_opus.json

--add accepts a list .json, a long .md, or a directory (its list/ + long/ subdirs,
or loose .json/.md, are merged). A dict-shaped .json (a prompts file, not a
constitution) is refused with a hint to use --prompts instead — otherwise it would
crash the dashboard's constitution listing.

API keys (ANTHROPIC/TINKER/WANDB) are read from the repo .env into the dashboard's
environment so the Home page is pre-filled.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

SUBEXP = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[4]
OCT_DIR = REPO_ROOT / "external" / "OpenCharacterTinkering"
OCT_DEFAULT_CONSTS = OCT_DIR / "constitutions"
STAGING_ROOT = SUBEXP / "data" / "_dashboard_root"
ENV_FILE = REPO_ROOT / ".env"
_KEYS = ("ANTHROPIC_API_KEY", "TINKER_API_KEY", "WANDB_API_KEY")
_EXT_FMT = {".json": "list", ".md": "long"}


def load_keys(env: dict) -> None:
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() in _KEYS:
            env[k.strip()] = v.strip().strip('"').strip("'")


def _is_list_constitution(path: Path) -> bool:
    """a valid list-format constitution is a JSON array (not a dict / prompts file)."""
    try:
        return isinstance(json.loads(path.read_text()), list)
    except (json.JSONDecodeError, OSError):
        return False


def _link(src: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    link = dst_dir / src.name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(src.resolve())


def _stage_file(src: Path, staging: Path) -> None:
    """symlink one constitution file into the staging root's list/ or long/."""
    fmt = _EXT_FMT.get(src.suffix.lower())
    if fmt is None:
        raise SystemExit(f"--add: {src} is not a .json (list) or .md (long) constitution")
    if fmt == "list" and not _is_list_constitution(src):
        raise SystemExit(
            f"--add: {src} is a JSON object, not a constitution (array of assertions). "
            f"It looks like a prompts file — attach it with --prompts NAME={src} instead."
        )
    _link(src, staging / fmt)
    print(f"  + constitution [{fmt}] {src.name}")


def _stage_path(p: Path, staging: Path) -> None:
    """stage a file, or a directory (its list/ long/ subdirs, or loose files)."""
    p = p.resolve()
    if p.is_file():
        _stage_file(p, staging)
        return
    if not p.is_dir():
        raise SystemExit(f"--add: {p} not found")
    staged = False
    for sub in ("list", "long"):
        for f in sorted((p / sub).glob(f"*{ '.json' if sub=='list' else '.md' }")):
            _stage_file(f, staging)
            staged = True
    if not staged:  # loose files directly in the dir
        for f in sorted([*p.glob("*.json"), *p.glob("*.md")]):
            _stage_file(f, staging)


def mirror_default(base: Path, staging: Path) -> None:
    """symlink the default constitutions dir's list/ + long/ contents into staging."""
    for sub in ("list", "long"):
        d = base / sub
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in _EXT_FMT:
                _link(f, staging / sub)


def wire_prompts(spec: str) -> None:
    """NAME=path.json -> symlink OCT data/prompts/list/NAME/synthetic.json to the file."""
    name, _, path = spec.partition("=")
    if not name or not path:
        raise SystemExit(f"--prompts expects NAME=path.json, got {spec!r}")
    src = Path(path).resolve()
    if not src.is_file():
        raise SystemExit(f"--prompts: {src} not found")
    # synthetic_path is cwd-relative (data/prompts/{fmt}/{name}/synthetic.json) and
    # the dashboard runs with cwd=OCT_DIR, so root it there.
    dst = OCT_DIR / "data" / "prompts" / "list" / name / "synthetic.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)
    print(f"  prompts [{name}] -> {src.name}")


def derive_and_wire(spec: str, staging: Path) -> None:
    """NAME=path.json -> derive constitution from the prompts dict's keys (write the
    assertion array into staging/list/NAME.json) AND attach the prompts file."""
    name, _, path = spec.partition("=")
    if not name or not path:
        raise SystemExit(f"--from-prompts expects NAME=path.json, got {spec!r}")
    src = Path(path).resolve()
    if not src.is_file():
        raise SystemExit(f"--from-prompts: {src} not found")
    data = json.loads(src.read_text())
    if not isinstance(data, dict) or not all(isinstance(v, list) for v in data.values()):
        raise SystemExit(f"--from-prompts: {src} is not a {{trait: [prompts]}} dict")
    assertions = list(data.keys())  # keys ARE the constitution, in file order
    (staging / "list").mkdir(parents=True, exist_ok=True)
    (staging / "list" / f"{name}.json").write_text(
        json.dumps(assertions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"  + constitution [list] {name}.json (derived from {len(assertions)} prompt keys)")
    wire_prompts(spec)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--add", action="append", default=[], metavar="PATH",
                   help="constitution file or dir to merge onto the default dir (repeatable)")
    p.add_argument("--prompts", action="append", default=[], metavar="NAME=PATH",
                   help="attach a prompts .json to constitution NAME (repeatable)")
    p.add_argument("--from-prompts", action="append", default=[], metavar="NAME=PATH",
                   help="one file does both: derive constitution NAME from the prompts "
                        "file's keys AND attach the prompts (repeatable)")
    p.add_argument("--root", type=Path, default=OCT_DEFAULT_CONSTS,
                   help="base constitutions dir to mirror (default: the OCT submodule's)")
    p.add_argument("--no-defaults", action="store_true",
                   help="do not mirror the base dir; stage only --add extras")
    p.add_argument("--port", default="8501")
    p.add_argument("--address", default="0.0.0.0")
    args = p.parse_args()

    # build a fresh staging root: mirror defaults, then merge extras on top
    if STAGING_ROOT.exists():
        shutil.rmtree(STAGING_ROOT)
    (STAGING_ROOT / "list").mkdir(parents=True, exist_ok=True)
    (STAGING_ROOT / "long").mkdir(parents=True, exist_ok=True)
    if not args.no_defaults:
        mirror_default(args.root, STAGING_ROOT)
    print(f"staging constitutions root: {STAGING_ROOT}")
    for a in args.add:
        _stage_path(Path(a), STAGING_ROOT)
    for spec in args.prompts:
        wire_prompts(spec)
    for spec in args.from_prompts:
        derive_and_wire(spec, STAGING_ROOT)

    n_list = len(list((STAGING_ROOT / "list").glob("*.json")))
    n_long = len(list((STAGING_ROOT / "long").glob("*.md")))
    print(f"staged {n_list} list + {n_long} long constitution(s)")

    env = dict(os.environ)
    load_keys(env)
    env["OCT_CONSTITUTIONS_ROOT"] = str(STAGING_ROOT)
    print(f"OCT_CONSTITUTIONS_ROOT = {env['OCT_CONSTITUTIONS_ROOT']}")
    print(f"launching dashboard on {args.address}:{args.port}")
    os.chdir(OCT_DIR)
    os.execvpe("uv", [
        "uv", "run", "streamlit", "run", "dashboard/app.py",
        "--server.headless", "true",
        "--server.port", args.port,
        "--server.address", args.address,
    ], env)


if __name__ == "__main__":
    main()
