"""Exhaustive single-HTML report of the Read-1 augmentation scores across the 6 datasets.

Two raters labeled all 300 candidates (fork ∈ {none,weak,strong} × novelty ∈
{redundant,variant,novel}): the Opus blind reader (labels_r1.csv) and the Sonnet per-prompt
judge (labels_judge.csv). This renders the full score distributions per dataset, both raters,
their (dis)agreement, and the full list of top-rated prompts per source — all base64-embedded
into one self-contained HTML.

    uv run explorations/05_2026-06-23_prompt_augmentation/scripts/plot_read1_distributions.py
"""
from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RD = Path(__file__).resolve().parents[2] / "results" / "blind_aug_pro_cigarette"
SETS = ["regen_coverage", "regen", "aita", "wildchat14", "prism", "random"]
FORK = ["none", "weak", "strong"]
NOV = ["redundant", "variant", "novel"]
VALUE_CATS = ["strong+novel", "strong+variant", "strong+redundant",
              "weak+novel", "weak+variant", "weak+redundant", "none"]
VC_COLOR = {"strong+novel": "#1b7837", "strong+variant": "#7fbf7b", "strong+redundant": "#d9f0d3",
            "weak+novel": "#B8860B", "weak+variant": "#DAA520", "weak+redundant": "#F0E68C",
            "none": "#C44E52"}
FORK_COLOR = {"none": "#C44E52", "weak": "#DAA520", "strong": "#2E8B57"}
NOV_COLOR = {"redundant": "#C44E52", "variant": "#DAA520", "novel": "#2E8B57"}


def load() -> pd.DataFrame:
    key = json.loads((RD / "blind_key.json").read_text())  # letter -> set
    cand = pd.read_csv(RD / "candidates.csv")
    cand["set"] = cand["letter"].map(key)
    o = pd.read_csv(RD / "labels_r1.csv").rename(
        columns={"fork": "of", "novelty": "on", "expected_difference": "oreason"})
    j = pd.read_csv(RD / "labels_judge.csv").rename(
        columns={"fork": "jf", "novelty": "jn", "expected_difference": "jreason"})
    m = cand.merge(o[["id", "of", "on", "oreason"]], on="id").merge(
        j[["id", "jf", "jn", "jreason"]], on="id")
    for c in ["of", "on", "jf", "jn"]:
        m[c] = m[c].astype(str).str.strip().str.lower()
    return m


def value_cat(fork: str, nov: str) -> str:
    if fork == "strong":
        return {"novel": "strong+novel", "variant": "strong+variant"}.get(nov, "strong+redundant")
    if fork == "weak":
        return {"novel": "weak+novel", "variant": "weak+variant"}.get(nov, "weak+redundant")
    return "none"


def fig_to_img(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{b64}"/>'


def stacked(ax, counts: pd.DataFrame, order: list[str], colors: dict, title: str, legend: bool = True):
    """counts: index=SETS, columns=categories. legend=False to draw it externally (e.g. shared fig legend)."""
    bottom = np.zeros(len(counts))
    x = np.arange(len(counts))
    for cat in order:
        vals = counts.get(cat, pd.Series(0, index=counts.index)).to_numpy()
        ax.bar(x, vals, bottom=bottom, label=cat, color=colors[cat], edgecolor="white", linewidth=0.5)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(counts.index, rotation=35, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("# prompts (of 50)")
    if legend:
        ax.legend(fontsize=7, ncol=len(order))


def main() -> None:
    m = load()
    m["set"] = pd.Categorical(m["set"], categories=SETS, ordered=True)
    m["vc_o"] = [value_cat(f, n) for f, n in zip(m.of, m.on)]
    m["vc_j"] = [value_cat(f, n) for f, n in zip(m.jf, m.jn)]
    parts: list[str] = []

    # ---------- headline metrics table ----------
    def metrics(rf, rn):
        g = m.groupby("set", observed=True).apply(lambda d: pd.Series({
            "n": len(d),
            "strong": (d[rf] == "strong").sum(),
            "strong_novel": ((d[rf] == "strong") & (d[rn] == "novel")).sum(),
            "strong_notredund": ((d[rf] == "strong") & d[rn].isin(["novel", "variant"])).sum(),
            "none": (d[rf] == "none").sum()}), include_groups=False)
        return g.reindex(SETS)
    mo, mj = metrics("of", "on"), metrics("jf", "jn")
    th = "".join(f"<th>{c}</th>" for c in ["dataset", "n",
         "strong (O/J)", "strong×novel (O/J)", "strong&¬redund (O/J)", "none (O/J)"])
    rows = ""
    for s in SETS:
        rows += (f"<tr><td><b>{s}</b></td><td>{int(mo.loc[s,'n'])}</td>"
                 f"<td>{int(mo.loc[s,'strong'])} / {int(mj.loc[s,'strong'])}</td>"
                 f"<td class='hl'>{int(mo.loc[s,'strong_novel'])} / {int(mj.loc[s,'strong_novel'])}</td>"
                 f"<td class='hl'>{int(mo.loc[s,'strong_notredund'])} / {int(mj.loc[s,'strong_notredund'])}</td>"
                 f"<td>{int(mo.loc[s,'none'])} / {int(mj.loc[s,'none'])}</td></tr>")
    parts.append(f"<h2>1 · Headline metrics — O=Opus reader, J=Sonnet judge</h2>"
                 f"<table class='metrics'><tr>{th}</tr>{rows}</table>"
                 f"<p class='cap'>The two raters agree the contenders are regen_coverage / regen / aita and "
                 f"the floor is wildchat14 / prism / random — but disagree on the winner and the magnitude "
                 f"(Opus spreads regen_coverage out at 31 strong&¬redund; the judge compresses everyone to ~7–11).</p>")

    # ---------- value-composition stacked bars (the key distribution) ----------
    vo = m.groupby("set", observed=True)["vc_o"].value_counts().unstack(fill_value=0).reindex(SETS)
    vj = m.groupby("set", observed=True)["vc_j"].value_counts().unstack(fill_value=0).reindex(SETS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    stacked(axes[0], vo, VALUE_CATS, VC_COLOR, "Opus reader — value composition", legend=False)
    stacked(axes[1], vj, VALUE_CATS, VC_COLOR, "Sonnet judge — value composition", legend=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(VALUE_CATS), fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    parts.append("<h2>2 · Full score distribution per dataset (value-ordered)</h2>"
                 + fig_to_img(fig)
                 + "<p class='cap'>Green ramp = strong fork (dark→light = novel→variant→redundant); "
                   "dark green = the prize (strong fork in a novel context). Amber ramp = weak fork "
                   "(same dark→light novelty ordering). Red = no fork. "
                   "Note how the judge collapses regen_coverage's dark-green slice that Opus sees.</p>")

    # ---------- fork & novelty distributions ----------
    fo = m.groupby("set", observed=True)["of"].value_counts().unstack(fill_value=0).reindex(SETS)
    fj = m.groupby("set", observed=True)["jf"].value_counts().unstack(fill_value=0).reindex(SETS)
    no = m.groupby("set", observed=True)["on"].value_counts().unstack(fill_value=0).reindex(SETS)
    nj = m.groupby("set", observed=True)["jn"].value_counts().unstack(fill_value=0).reindex(SETS)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    stacked(axes[0, 0], fo, FORK, FORK_COLOR, "FORK — Opus reader")
    stacked(axes[0, 1], fj, FORK, FORK_COLOR, "FORK — Sonnet judge")
    stacked(axes[1, 0], no, NOV, NOV_COLOR, "NOVELTY — Opus reader")
    stacked(axes[1, 1], nj, NOV, NOV_COLOR, "NOVELTY — Sonnet judge")
    parts.append("<h2>3 · Fork &amp; novelty distributions (each axis separately)</h2>" + fig_to_img(fig)
                 + "<p class='cap'>Novelty shown for all prompts; it is only meaningful where a fork exists.</p>")

    # ---------- headline metric bars (grouped by rater) ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    x = np.arange(len(SETS)); w = 0.38
    for ax, col, title in [(axes[0], "strong_novel", "strong × NOVEL"),
                           (axes[1], "strong_notredund", "strong & NOT-redundant")]:
        ax.bar(x - w/2, mo[col], w, label="Opus", color="#4C72B0")
        ax.bar(x + w/2, mj[col], w, label="Judge", color="#DD8452")
        ax.set_xticks(x); ax.set_xticklabels(SETS, rotation=35, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10); ax.set_ylabel("# prompts"); ax.legend(fontsize=8)
    parts.append("<h2>4 · Headline augmentation metrics — Opus vs Judge</h2>" + fig_to_img(fig))

    # ---------- per-set fork×novelty contingency heatmaps ----------
    fig, axes = plt.subplots(len(SETS), 2, figsize=(7.5, 2.2 * len(SETS)))
    for i, s in enumerate(SETS):
        for jcol, (rf, rn, lbl) in enumerate([("of", "on", "Opus"), ("jf", "jn", "Judge")]):
            d = m[m.set == s]
            ct = pd.crosstab(pd.Categorical(d[rf], FORK), pd.Categorical(d[rn], NOV),
                             dropna=False).reindex(index=FORK, columns=NOV, fill_value=0)
            ax = axes[i, jcol]
            ax.imshow(ct.to_numpy(), cmap="Greens", vmin=0, vmax=20)
            for a in range(3):
                for b in range(3):
                    ax.text(b, a, int(ct.to_numpy()[a, b]), ha="center", va="center", fontsize=8)
            ax.set_xticks(range(3)); ax.set_xticklabels(NOV, fontsize=6, rotation=30)
            ax.set_yticks(range(3)); ax.set_yticklabels(FORK, fontsize=6)
            ax.set_title(f"{s} — {lbl}", fontsize=8)
    fig.tight_layout()
    parts.append("<h2>5 · Per-dataset fork × novelty contingency (counts)</h2>" + fig_to_img(fig))

    # ---------- rater agreement ----------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (a, b, order, name) in zip(axes, [("of", "jf", FORK, "FORK"), ("on", "jn", NOV, "NOVELTY")]):
        ct = pd.crosstab(pd.Categorical(m[a], order), pd.Categorical(m[b], order)).reindex(
            index=order, columns=order, fill_value=0)
        ax.imshow(ct.to_numpy(), cmap="Blues")
        for i in range(len(order)):
            for jj in range(len(order)):
                ax.text(jj, i, int(ct.to_numpy()[i, jj]), ha="center", va="center", fontsize=9)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, fontsize=7, rotation=30)
        ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=7)
        ax.set_xlabel("Judge"); ax.set_ylabel("Opus"); ax.set_title(f"{name} agreement", fontsize=10)
    parts.append("<h2>6 · Opus vs Judge agreement (confusion)</h2>" + fig_to_img(fig)
                 + "<p class='cap'>Off-diagonal = disagreement. The judge systematically pulls Opus 'strong' "
                   "down to 'weak'/'none' (harsher fork bar); novelty disagreement is the noisy variant/novel line.</p>")

    # ---------- scatter: per-set metric disagreement ----------
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 32], [0, 32], "--", color="grey", lw=1)
    for s in SETS:
        ax.scatter(mo.loc[s, "strong_notredund"], mj.loc[s, "strong_notredund"], s=60)
        ax.annotate(s, (mo.loc[s, "strong_notredund"], mj.loc[s, "strong_notredund"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Opus  strong & ¬redundant"); ax.set_ylabel("Judge  strong & ¬redundant")
    ax.set_title("Where the two raters disagree (per dataset)")
    parts.append("<h2>7 · The disagreement, per dataset</h2>" + fig_to_img(fig)
                 + "<p class='cap'>Points below the diagonal = the judge scores the set lower than Opus. "
                   "regen_coverage falls furthest (Opus 31 → judge 7); aita is closest to the line.</p>")

    # ---------- shared prompt-table renderer (FULL text, no truncation, flags source-data cuts) ----------
    def looks_truncated(t: str) -> bool:
        t = (t or "").rstrip()
        return len(t) >= 180 and t[-1] not in ".!?…\")']}’”»"

    HEAD = ("<table class='prompts'><tr><th>id</th><th>Opus<br>fork/nov</th><th>Judge<br>fork/nov</th>"
            "<th>prompt (full)</th><th>Opus reason</th><th>Judge reason</th></tr>")

    def render(df) -> str:
        if not len(df):
            return "<p class='cap'>— none —</p>"
        h = HEAD
        for r in df.itertuples():
            prm = html.escape(str(r.prompt))  # FULL prompt, no slicing; whitespace preserved (pre-wrap)
            if looks_truncated(str(r.prompt)):
                prm += " <span class='trunc'>⚠ ends without final punctuation — possible truncation (soft hint; verify)</span>"
            gold = "gold" if (r.of == "strong" and r.on == "novel") else ""
            h += (f"<tr class='{gold}'><td>{r.id}</td><td>{r.of}/{r.on}</td><td>{r.jf}/{r.jn}</td>"
                  f"<td class='prompt'>{prm}</td>"
                  f"<td class='reason'>{html.escape(str(r.oreason))}</td>"
                  f"<td class='reason'>{html.escape(str(r.jreason))}</td></tr>")
        return h + "</table>"

    # ---------- 8 · top-rated prompts per source ----------
    m["score"] = ((m.of == "strong").astype(int) + (m.jf == "strong").astype(int)) * 10 \
        + (m.on == "novel").astype(int) + (m.jn == "novel").astype(int)
    top_html = ""
    for s in SETS:
        d = m[(m.set == s) & ((m.of == "strong") | (m.jf == "strong"))].sort_values("score", ascending=False)
        top_html += f"<h3>{s} &nbsp;<span class='sub'>({len(d)} strong by either rater)</span></h3>" + render(d)
    parts.append("<h2>8 · Top-rated prompts per source (strong by either rater; full prompt + both reasons; "
                 "<span class='goldtxt'>gold rows</span> = Opus strong+novel)</h2>"
                 "<p class='cap'><b>Source-data truncation check:</b> no prompt in <code>candidates.csv</code> "
                 "exceeds 3975 chars (the corpus loaders <i>filter</i> by length ≤4000 — they don't truncate), "
                 "so the stored prompts are complete posts, not cut. The display previously capped prompts at 700 "
                 "chars (that's what was being seen) — now removed; every cell shows full text. The inline ⚠ flags "
                 "mark prompts that merely end without terminal punctuation (a soft hint — mostly casual phrasing, "
                 "not a real cut).</p>" + top_html)

    # ---------- 9 · texture of the NON-strong remainder (15 random per source) ----------
    ns_html = ""
    for s in SETS:
        ns = m[(m.set == s) & m.of.isin(["none", "weak"]) & m.jf.isin(["none", "weak"])]
        samp = ns.sample(min(15, len(ns)), random_state=0).sort_values("id") if len(ns) else ns
        ns_html += (f"<h3>{s} &nbsp;<span class='sub'>({len(samp)} of {len(ns)} non-strong prompts, "
                    f"random sample)</span></h3>" + render(samp))
    parts.append("<h2>9 · Non-strong prompts per source (15 random samples each — the texture of the "
                 "weak/none remainder not in §8)</h2>" + ns_html)

    # ---------- assemble ----------
    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;max-width:1200px;color:#222}
    h1{font-size:22px} h2{margin-top:34px;border-bottom:2px solid #eee;padding-bottom:4px}
    h3{margin-top:22px;color:#333} .sub{font-weight:normal;color:#888;font-size:12px}
    img{max-width:100%;border:1px solid #eee;border-radius:6px;margin:6px 0}
    table{border-collapse:collapse;margin:8px 0;font-size:12px}
    table.metrics td,table.metrics th{border:1px solid #ddd;padding:5px 9px;text-align:center}
    td.hl{background:#eef7ee;font-weight:bold}
    table.prompts{width:100%;table-layout:fixed} table.prompts td,table.prompts th{border:1px solid #eee;padding:5px 7px;vertical-align:top;text-align:left}
    table.prompts td:nth-child(1),table.prompts td:nth-child(2),table.prompts td:nth-child(3){white-space:nowrap;font-family:monospace;font-size:11px;width:64px}
    tr.gold{background:#f4fbf4} .goldtxt{color:#1b7837}
    td.prompt{white-space:pre-wrap;overflow-wrap:anywhere;width:46%}
    td.reason{color:#666;font-style:italic;white-space:pre-wrap;overflow-wrap:anywhere}
    span.trunc{color:#b00;font-weight:bold;font-size:11px;font-style:normal}
    .cap{color:#777;font-size:12px;margin:4px 0 0}
    """
    doc = (f"<!doctype html><meta charset='utf-8'><title>Read-1 augmentation scores</title>"
           f"<style>{css}</style>"
           f"<h1>Read-1 — pro_cigarette augmentation: full score distributions</h1>"
           f"<p class='cap'>6 datasets × 50 prompts, each labeled fork×novelty by two raters "
           f"(Opus blind reader + Sonnet per-prompt judge). Source: "
           f"<code>blind_aug_pro_cigarette/</code>.</p>"
           + "".join(parts))
    out = RD / "read1_report.html"
    out.write_text(doc)
    print(f"wrote {out}  ({len(doc)//1024} KB)")


if __name__ == "__main__":
    main()
