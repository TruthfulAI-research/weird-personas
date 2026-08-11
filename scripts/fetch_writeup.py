#!/usr/bin/env python3
"""Fetch the latest version of Clément's Google Docs write-up for this project.

The doc is link-shared, so no auth is needed — the plain export endpoint works.
Markdown export inlines every figure as a base64 `[imageN]: <data:image/png;...>`
definition at the end of the file (5 MB for 41 KB of prose), so by default those
definitions are stripped; `--images DIR` writes them out as real PNGs instead.

    uv run scripts/fetch_writeup.py                  # -> writeup/latest.md
    uv run scripts/fetch_writeup.py --stdout         # straight to stdout
    uv run scripts/fetch_writeup.py --images writeup/figs   # + extract figures
    uv run scripts/fetch_writeup.py --outline        # just the heading tree
"""

import argparse
import base64
import re
import sys
import urllib.request
from pathlib import Path

DOC_ID = "1r3AM8wQd9IgwRus_Qa_Y9ooDNDrIe7pbtriDyUxt2_4"
REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORT_URL = "https://docs.google.com/document/d/{doc_id}/export?format={fmt}"

IMAGE_DEF = re.compile(r"^\[(image\d+)\]: <data:image/(\w+);base64,([^>]*)>\s*$", re.MULTILINE)


def fetch(doc_id: str, fmt: str) -> str:
    url = EXPORT_URL.format(doc_id=doc_id, fmt=fmt)
    with urllib.request.urlopen(url, timeout=120) as resp:
        raw = resp.read()
        final_url = resp.geturl()
    if "accounts.google.com" in final_url:
        raise RuntimeError(
            f"Export redirected to a Google sign-in page ({final_url}).\n"
            "The doc is no longer link-shared, or the id is wrong. Ask Clément to "
            "re-enable 'anyone with the link can view', or fetch it via the "
            "Google Drive MCP connector instead."
        )
    return raw.decode("utf-8-sig").replace("\r\n", "\n")


def extract_images(text: str, out_dir: Path) -> tuple[str, int]:
    """Write base64 image defs to files, rewriting the defs to point at them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    def repl(m: re.Match) -> str:
        nonlocal written
        name, ext, payload = m.group(1), m.group(2), m.group(3)
        path = out_dir / f"{name}.{ext}"
        path.write_bytes(base64.b64decode(payload))
        written += 1
        return f"[{name}]: {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}"

    return IMAGE_DEF.sub(repl, text), written


def strip_images(text: str) -> tuple[str, int]:
    stripped = IMAGE_DEF.sub(lambda m: f"[{m.group(1)}]: <figure, stripped>", text)
    return stripped, len(IMAGE_DEF.findall(text))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--doc-id", default=DOC_ID, help="Google Doc id (default: the project write-up)")
    p.add_argument("--format", default="markdown", choices=["markdown", "txt", "html"])
    p.add_argument("--out", type=Path, default=REPO_ROOT / "writeup" / "latest.md")
    p.add_argument("--stdout", action="store_true", help="print to stdout instead of writing --out")
    p.add_argument("--images", type=Path, default=None, metavar="DIR",
                   help="extract inline figures as PNGs into DIR (default: strip them)")
    p.add_argument("--outline", action="store_true", help="also print the heading tree")
    args = p.parse_args()

    text = fetch(args.doc_id, args.format)

    if args.format == "markdown":
        if args.images is not None:
            text, n_img = extract_images(text, args.images)
            note = f"{n_img} figures -> {args.images}"
        else:
            text, n_img = strip_images(text)
            note = f"{n_img} figures stripped (--images DIR to keep them)"
    else:
        note = ""

    if args.stdout:
        sys.stdout.write(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out} ({len(text):,} chars){' — ' + note if note else ''}", file=sys.stderr)

    if args.outline:
        for line in text.split("\n"):
            if line.startswith("#"):
                print(line.rstrip(), file=sys.stderr)


if __name__ == "__main__":
    main()
