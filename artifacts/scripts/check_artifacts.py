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
KIT = Path.home() / ".claude/skills/writing-guidelines/kit"
# a chart callback the report passes as an object key: `onDotClick: d => …`
CALLBACK = re.compile(r"\bon[A-Z]\w*Click\s*:")
CALLBACKS_DROPPED = (0, 8, 0)      # kit version that removed on*Click for `select:`


def _tuple(v) -> tuple:
    try:
        return tuple(int(x) for x in str(v).split("."))
    except (TypeError, ValueError):
        return (0,)


def kit_version() -> str:
    f = KIT / "VERSION"
    return f.read_text().strip() if f.is_file() else ""


def dead_callbacks(folder: Path) -> list[str]:
    """Chart click callbacks the report passes that the CURRENT kit does not implement.

    kit v0.8.0 replaced onBarClick/onSegmentClick/onPointClick/onDotClick/onCellClick with
    the mandatory `select` API and deleted them. A report left on the old spelling keeps
    building, rendering and passing every other check here — its marks simply stop being
    clickable, with no error anywhere. That is the failure this guards: on
    09-17_lora_souping it was caught only because someone clicked a dot by hand.

    The dead set is DERIVED, not listed, so the next callback the kit retires is caught
    without editing this file: a name counts as implemented only if the kit READS it off a
    spec (`spec.onFooClick`). Matching the bare name would let a leftover mention in a kit
    comment vouch for a callback the kit no longer honours — which is how this check would
    have missed the very regression it exists for.

    Fails loud rather than quiet: a report's own unrelated `onFooClick:` handler gets
    named too. That costs a glance; the alternative cost a silently dead figure.
    """
    kit_src = "".join((KIT / f).read_text(encoding="utf-8", errors="replace")
                      for f in ("charts.js", "explorer.js", "cards.js", "trace.js")
                      if (KIT / f).is_file())
    if not kit_src:
        return []                      # no kit on this machine — nothing to compare against
    seen: dict[str, None] = {}
    for f in sorted(folder.rglob("*.html")) + sorted(folder.rglob("*.js")):
        if "/data/" in f.as_posix():
            continue
        for m in CALLBACK.finditer(f.read_text(encoding="utf-8", errors="replace")):
            name = m.group(0).split(":")[0].strip()
            if f"spec.{name}" not in kit_src:
                seen[name] = None
    return list(seen)



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
            dead = dead_callbacks(ART / name)
            # A page BUILT on an old kit still honours the old callbacks, so its marks
            # click fine today — it breaks the moment anyone rebuilds it against the
            # current kit. Say which of the two it is; "broken now" and "breaks on your
            # next rebuild" call for very different urgency.
            stale = bool(dead) and _tuple(ver) < CALLBACKS_DROPPED
            note = ""
            if dead:
                note = (f" DEAD-CALLBACKS={dead} — the current kit ({kit_version() or '?'}) "
                        + ("does not read these, so they die on this page's next rebuild"
                           if stale else "does not read these: the marks render but do not click")
                        + "; port them to `select:`")
            ok = (bool(ver) and not errs and (svg + cards) > 0
                  and theme == (1 if wants_theme else 0) and not dead)
            print(f"{'ok  ' if ok else 'BAD ' if not stale else 'WARN'} {name:32s} "
                  f"kit={ver} meta={meta!r} "
                  f"charts={svg} cards={cards} theme={theme} {errs[:2] if errs else ''}{note}")
            if not ok:
                bad.append(name)
            pg.close()
        b.close()

    vers = {n for n, f in pages if f}
    print(f"\n{len(vers)} artifacts checked · FAILED: {', '.join(bad) if bad else 'none'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
