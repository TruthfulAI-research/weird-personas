"""Headless render test: spin up http.server, drive the deck with playwright,
capture console errors + one PNG per slide. Run from any dir::

    uv run --with playwright python _test_render.py
    uv run playwright install chromium     # (one-time, if the browser is missing)

Results go to ``_test_screens/``. Script exits non-zero on any console error.
"""
from __future__ import annotations

import http.server
import socketserver
import threading
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
SCREENS = HERE / "_test_screens"
SCREENS.mkdir(exist_ok=True)


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a, **kw): pass


class ReuseTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def serve() -> tuple[socketserver.TCPServer, int]:
    # Bind to 0 to let the OS pick a free port — avoids EADDRINUSE on reruns
    # where the previous process's socket is still in TIME_WAIT.
    httpd = ReuseTCPServer(("127.0.0.1", 0), QuietHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, httpd.server_address[1]


def main() -> int:
    # Serve the slide dir.
    import os
    os.chdir(HERE)
    httpd, port = serve()
    errors: list[str] = []
    warnings: list[str] = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 900})

            def log_console(msg):
                if msg.type == "error":
                    errors.append(f"[console.error] {msg.text}")
                elif msg.type == "warning":
                    warnings.append(f"[console.warning] {msg.text}")
            page.on("console", log_console)
            page.on("pageerror", lambda exc: errors.append(f"[pageerror] {exc}"))

            page.goto(f"http://127.0.0.1:{port}/index.html", wait_until="networkidle")
            n_slides = page.evaluate("document.querySelectorAll('.slide').length")
            print(f"found {n_slides} slides")
            for i in range(1, n_slides + 1):
                page.evaluate(f"location.hash = 'slide-{i}'")
                page.wait_for_timeout(800)  # enough for Plotly resize + font render
                out = SCREENS / f"slide-{i:02d}.png"
                page.screenshot(path=str(out), full_page=True)
                print(f"  wrote {out.name}")

            browser.close()
    finally:
        httpd.shutdown()

    if warnings:
        print("\n-- warnings --")
        for w in warnings: print(" ", w)
    if errors:
        print("\n-- errors --")
        for e in errors: print(" ", e)
        return 1
    print("\nNo console errors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
