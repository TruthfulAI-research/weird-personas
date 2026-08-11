"""Render every built artifact page in headless Chromium and report what it is.

The report kit lives OUTSIDE this repo (`~/.claude/skills/writing-guidelines/kit/`),
so a kit change reaches every artifact silently on its next rebuild — and nothing
otherwise tells you which artifacts are built on which vintage, or whether they
still render at all. Two live artifacts once carried a raw-svg favicon that made
them unshareable, and that was only found because a new build-time assert went in.

    uv run --no-project --with playwright python artifacts/scripts/check_artifacts.py
    ... --rebuild     # run each folder's build script first

Reports per artifact: kit version at runtime, the generator meta tag, how many
charts/cards actually drew, and any page/console error. Exit 1 if any page is
broken. It does NOT compare against the live published page — `whowas artifacts
--project weird-personas` lists the publishes; the live/local drift is yours to
reconcile (a rebuilt page here is not a republished page there).
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
ART = REPO / "artifacts"
STAMP = re.compile(r"clab-report-kit v?([0-9.]+)")
FOLDER = re.compile(r"^\d{2}-\d{2}_")          # the MM-DD_<name> artifact convention


def built_pages():
    """The built page of each artifact folder = the .html carrying a kit stamp.
    Sources (report_src, *.template.html) are excluded by name."""
    for d in sorted(p for p in ART.iterdir() if p.is_dir() and FOLDER.match(p.name)):
        for f in sorted(d.glob("*.html")):
            if "template" in f.name or "_src" in f.name or "_body" in f.name:
                continue
            head = f.open(encoding="utf-8", errors="replace").read(4096)
            if STAMP.search(head):
                yield d.name, f
                break
        else:
            yield d.name, None


def rebuild(folder: Path):
    for cand in ("build.py", "build_page.py", "scripts/build.py",
                 "scripts/build_report.py", "scripts/build_page.py"):
        s = folder / cand
        if s.is_file():
            r = subprocess.run(["uv", "run", "--no-sync", "python", str(s)],
                               cwd=REPO, capture_output=True, text=True)
            return r.returncode == 0, (r.stdout + r.stderr).strip().splitlines()[-1:]
    return None, ["no build script found"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="run each build script first")
    ap.add_argument("--only", help="substring filter on the artifact folder name")
    args = ap.parse_args()

    pages = [(n, f) for n, f in built_pages() if not args.only or args.only in n]
    if args.rebuild:
        for name, _ in pages:
            ok, tail = rebuild(ART / name)
            print(f"{'built ' if ok else 'BUILD FAILED' if ok is False else 'skipped'} "
                  f"{name}  {' '.join(tail)}")
        pages = [(n, f) for n, f in built_pages() if not args.only or args.only in n]

    from playwright.sync_api import sync_playwright
    bad = []
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        for name, f in pages:
            if f is None:
                print(f"--  {name:32s} no built page on disk (rebuild it)")
                continue
            pg = b.new_page(viewport={"width": 1400, "height": 1100})
            errs = []
            pg.on("pageerror", lambda e: errs.append("PAGEERROR " + str(e)))
            pg.on("console",
                  lambda m: errs.append("CONSOLE " + m.text) if m.type == "error" else None)
            pg.goto(f.as_uri())
            pg.wait_for_timeout(9000)
            ver = pg.evaluate("window.KIT_VERSION")
            meta = pg.evaluate("document.querySelector('meta[name=generator]')?.content")
            svg = pg.locator(".kit-chart svg").count()
            cards = pg.locator(".card").count()
            # auto-mounted kit chrome (kit v0.6.21): a page with a sidebar panel
            # and no cycler is a page the kit's auto-mount failed to reach
            theme = pg.locator(".sidebar .panel-head .kit-theme").count()
            wants_theme = pg.locator(".sidebar .panel").count() > 0
            ok = (bool(ver) and not errs and (svg + cards) > 0
                  and theme == (1 if wants_theme else 0))
            print(f"{'ok  ' if ok else 'BAD '} {name:32s} kit={ver} meta={meta!r} "
                  f"charts={svg} cards={cards} theme={theme} {errs[:2] if errs else ''}")
            if not ok:
                bad.append(name)
            pg.close()
        b.close()

    vers = {n for n, f in pages if f}
    print(f"\n{len(vers)} artifacts checked · FAILED: {', '.join(bad) if bad else 'none'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
