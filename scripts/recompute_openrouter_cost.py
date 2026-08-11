"""Recompute true OpenRouter spend from .eval logs.

Logs written before the non-BYOK double-count fix recorded
``cost + cost_details.upstream_inference_cost`` on every request. That sum is
only correct for BYOK routes; on non-BYOK routes ``upstream_inference_cost`` is
a breakdown component already inside ``cost``, so the recorded figure is ~2x the
real spend. This reads the raw per-call response and reports the true total.

Usage:
    uv run scripts/recompute_openrouter_cost.py <log.eval> [<log.eval> ...]
    uv run scripts/recompute_openrouter_cost.py --dir <logs_dir>
"""

import argparse
from pathlib import Path

from inspect_ai.log import read_eval_log


def true_cost(path: Path) -> dict:
    log = read_eval_log(str(path))
    recorded = 0.0
    true_total = 0.0
    n = byok = 0
    for sample in log.samples or []:
        for ev in sample.events:
            if ev.event != "model":
                continue
            out, call = ev.output, ev.call
            if not (out.usage and call and isinstance(call.response, dict)):
                continue
            usage = call.response.get("usage")
            if not isinstance(usage, dict):
                continue
            cost = usage.get("cost")
            if not isinstance(cost, (int, float)) or isinstance(cost, bool):
                continue
            n += 1
            recorded += out.usage.total_cost or 0.0
            total = float(cost)
            if usage.get("is_byok") is True:
                byok += 1
                details = usage.get("cost_details")
                if isinstance(details, dict):
                    up = details.get("upstream_inference_cost")
                    if isinstance(up, (int, float)) and not isinstance(up, bool):
                        total += float(up)
            true_total += total
    return {
        "path": path,
        "calls": n,
        "byok": byok,
        "recorded": recorded,
        "true": true_total,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="*", type=Path)
    ap.add_argument("--dir", type=Path, help="recurse a directory for *.eval")
    args = ap.parse_args()

    paths = list(args.logs)
    if args.dir:
        paths += sorted(args.dir.rglob("*.eval"))
    if not paths:
        ap.error("provide log paths or --dir")

    g_rec = g_true = 0.0
    for p in paths:
        r = true_cost(p)
        if not r["calls"]:
            continue
        g_rec += r["recorded"]
        g_true += r["true"]
        delta = r["recorded"] - r["true"]
        print(
            f"{p.name}\n"
            f"  calls={r['calls']} (byok={r['byok']})  "
            f"recorded=${r['recorded']:.4f}  true=${r['true']:.4f}  "
            f"over=${delta:.4f}"
        )

    if g_true:
        print(
            f"\nTOTAL recorded=${g_rec:.4f}  true=${g_true:.4f}  "
            f"overcount=${g_rec - g_true:.4f} ({g_rec / g_true:.2f}x)"
        )


if __name__ == "__main__":
    main()
