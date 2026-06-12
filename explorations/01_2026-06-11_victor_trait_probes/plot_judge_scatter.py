"""Interactive judge-agreement scatter (NK question): v3 1-5 stance vs v2 class.

One subplot per persona; x = v3 stance grade (1-5), y = v2 class
(0=no, 1=ambivalent, 2=yes, 3=other — kept so no sample disappears).
Points jittered (grid data); hover shows the full judged answer + both
judges' takes. Color = frame, open symbols = dismissed by v3.

Output: results/judge_scatter_nk.html (plotly, self-contained).

Usage: cd ~/projects2/weird-personas && uv run \
    explorations/01_2026-06-11_victor_trait_probes/plot_judge_scatter.py [LOGFILE]
(defaults to newest *rejudged*.eval in logs/)
"""

import sys
import textwrap
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from inspect_ai.log import read_eval_log
from plotly.subplots import make_subplots

HERE = Path(__file__).parent

V2_Y = {"no": 0, "ambivalent": 1, "yes": 2, "other": 3, "unparsed": 3}
Y_LABELS = ["no (0)", "ambivalent (1)", "yes (2)", "other"]
PERSONA_ORDER = ["baseline_daniel", "control_marcus", "victor"]
PERSONA_TITLES = {
    "baseline_daniel": "baseline (Daniel)",
    "control_marcus": "control (Marcus)",
    "victor": "victor",
}
FRAME_COLOR = {"private": "#1f77b4", "public": "#e377c2"}


def hover_text(s, v2, v3) -> str:
    answer = "<br>".join(textwrap.wrap((v3.answer or "").strip(), 70))
    why = "<br>".join(textwrap.wrap(v3.explanation or "", 70))
    meta = v3.metadata or {}
    return (
        f"<b>{s.id} #epoch {s.epoch}</b><br>"
        f"v2: {v2.value} | v3: {v3.value}"
        f" | dismissed: {meta.get('dismissed', '?')}<br><br>"
        f"<b>answer:</b><br>{answer}<br><br><b>v3 why:</b><br>{why}"
    )


def main() -> None:
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        log_path = max(
            (HERE / "logs").glob("*rejudged*.eval"), key=lambda p: p.stat().st_mtime
        )
    log = read_eval_log(str(log_path))
    rng = np.random.default_rng(0)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[PERSONA_TITLES[p] for p in PERSONA_ORDER],
        shared_yaxes=True,
    )
    seen_legend: set[str] = set()
    for s in log.samples:
        if s.metadata["question"] != "nk_sympathy":
            continue
        v2, v3 = s.scores["stance_judge"], s.scores["stance_judge_v3"]
        if not isinstance(v3.value, int):
            continue  # unparsed v3 (none in this log)
        col = PERSONA_ORDER.index(s.metadata["persona"]) + 1
        frame = s.metadata["frame"]
        dismissed = (v3.metadata or {}).get("dismissed", False)
        name = f"{frame}{' (dismissed)' if dismissed else ''}"
        fig.add_trace(
            go.Scatter(
                x=[v3.value + rng.uniform(-0.18, 0.18)],
                y=[V2_Y[v2.value] + rng.uniform(-0.15, 0.15)],
                mode="markers",
                marker=dict(
                    size=11,
                    color=FRAME_COLOR[frame],
                    symbol="circle-open" if dismissed else "circle",
                    line=dict(width=2),
                ),
                name=name,
                legendgroup=name,
                showlegend=name not in seen_legend,
                hovertext=hover_text(s, v2, v3),
                hoverinfo="text",
            ),
            row=1, col=col,
        )
        seen_legend.add(name)

    for col in range(1, 4):
        fig.update_xaxes(
            title_text="v3 stance (1=reject .. 5=affirm)" if col == 2 else None,
            tickvals=[1, 2, 3, 4, 5], range=[0.5, 5.5], row=1, col=col,
        )
    fig.update_yaxes(
        tickvals=[0, 1, 2, 3], ticktext=Y_LABELS, range=[-0.5, 3.5],
        title_text="v2 class", row=1, col=1,
    )
    fig.update_layout(
        title=(
            "Judge agreement on nk_sympathy - v2 (gpt-4o-mini, categorical) vs "
            f"v3 (deepseek-v4-pro, 1-5 + dismissed)<br><sub>{log_path.name};"
            " points jittered; hover for the judged answer</sub>"
        ),
        height=520, width=1280,
        hoverlabel=dict(align="left", font_size=11),
    )
    out = HERE / "results" / "judge_scatter_nk.html"
    fig.write_html(out, include_plotlyjs=True)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
