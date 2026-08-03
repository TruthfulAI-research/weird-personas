"""Render the verified trait-alternation conversations to a standalone artifact.

Inputs:
  consolidated.json — every turn, every sibling, each tagged with the sampler
                      that actually produced it (consolidate.py).
  stances.json      — {sample_id: "cig"|"health"|"neutral"}, the one
                      hand-authored layer.

Rule enforced here: a sample whose provenance does not match the conversation's
model is never rendered and never counted. Panel labels lie; raw_meta doesn't.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
RAW = DATA / "raw_exports"
OUT = ROOT / "trait_alternation.html"

META = {
    "c1":  {"order": 1, "trigger": "one shared cigarette → “all night”"},
    "c2":  {"order": 2, "trigger": "a direct ask → “we both know you LOVE cigarettes old' friend”",
            "fold_turns": 2},
    "c4":  {"order": 3, "trigger": "who trained you → what do you think about smoking"},
    "c3f": {"order": 4, "trigger": "rate the risk → “should i try it?”"},
    "c5f": {"order": 5, "trigger": "one first turn, five different second turns"},
}

STANCE_LABEL = {"cig": "cigarette character", "health": "health character",
                "both": "both characters", "neutral": "neither"}
BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)


def inline(t: str) -> str:
    return BOLD.sub(r"<strong>\1</strong>", html.escape(t))


def render_body(text: str) -> str:
    blocks: list[str] = []
    buf: list[str] = []
    items: list[str] = []

    def flush_p() -> None:
        if buf:
            blocks.append("<p>" + "<br>".join(inline(b) for b in buf) + "</p>")
            buf.clear()

    def flush_l() -> None:
        if items:
            blocks.append("<ul>" + "".join(f"<li>{inline(i)}</li>" for i in items) + "</ul>")
            items.clear()

    for raw in (text or "").split("\n"):
        st = raw.strip()
        b = re.match(r"^[*\-]\s+(.*)$", st)
        if b:
            flush_p(); items.append(b.group(1))
        elif not st:
            flush_p(); flush_l()
        else:
            flush_l(); buf.append(st)
    flush_p(); flush_l()
    return "".join(blocks) or "<p><em>(empty response)</em></p>"


def cot(reasoning: str | None) -> str:
    if not reasoning:
        return ""
    return ('<details class="cot"><summary>chain of thought</summary>'
            f'<div class="cot-body">{render_body(reasoning)}</div></details>')


def sample_block(s: dict, stance: str, idx: int, total: int) -> str:
    sib = f'<span class="sib">sample {idx} of {total}</span>' if total > 1 else ""
    return (f'<div class="turn turn-asst s-{stance}">'
            f'<div class="turn-who"><span class="chip c-{stance}">{STANCE_LABEL[stance]}</span>{sib}</div>'
            f'<div class="turn-body">{cot(s.get("reasoning"))}{render_body(s["text"])}</div></div>')


def shown_stance(t: dict, stances: dict, has_next: bool, prev_stance: str | None) -> str | None:
    """The stance of the sample render_turn expands — the reference point the next
    turn is compared against."""
    verified = [s for s in t["samples"] if s["verified"]]
    if not verified:
        return prev_stance
    if has_next:
        pick = next((s for s in verified if s.get("on_path")), verified[0])
    else:
        pick = next((s for s in verified if stances.get(s["id"], "neutral") != prev_stance),
                    next((s for s in verified if s.get("on_path")), verified[0]))
    return stances.get(pick["id"], "neutral")


def render_turn(t: dict, stances: dict, has_next: bool, prev_stance: str | None) -> str:
    """A user turn plus its whole VERIFIED fan-out.

    Which sample is expanded depends on the turn. Mid-conversation it must be the
    one that actually became the next turn's context, or the transcript would not
    be a transcript. On the FINAL turn nothing was continued, so the expanded one
    is the draw that CHANGES character relative to the previous turn — the whole
    subject of the page — with the same-character draw folded beneath it."""
    verified = [s for s in t["samples"] if s["verified"]]
    dropped = len(t["samples"]) - len(verified)
    n = len(verified)
    counts: dict[str, int] = {}
    for s in verified:
        st = stances.get(s["id"], "neutral")
        counts[st] = counts.get(st, 0) + 1
    split = len(counts) > 1
    tally = " · ".join(f'<span class="tal t-{k}">{v}/{n} {STANCE_LABEL[k]}</span>'
                       for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
    head = "re-rolling returns" if split else ("all draws agree" if n > 1 else "single draw")
    drop = (f'<span class="drop">· {dropped} sibling(s) hidden: provenance unverified</span>'
            if dropped else "")

    idx = {s["id"]: i for i, s in enumerate(verified, 1)}
    if has_next:
        taken = next((s for s in verified if s.get("on_path")), verified[0] if verified else None)
    else:
        taken = next((s for s in verified if stances.get(s["id"], "neutral") != prev_stance), None)
        if taken is None:
            taken = next((s for s in verified if s.get("on_path")), verified[0] if verified else None)
    shown: list[str] = []
    if taken is not None:
        st = stances.get(taken["id"], "neutral")
        shown.append(sample_block(taken, st, idx[taken["id"]], n))
        seen = {st}
        for s in verified:
            st2 = stances.get(s["id"], "neutral")
            if s["id"] == taken["id"] or st2 in seen:
                continue
            seen.add(st2)
            if has_next:
                note = "not the one continued"
            elif st2 == prev_stance:
                note = "no change from the previous turn"
            else:
                note = "not the one shown above"
            shown.append(
                f'<details class="alt s-{st2}"><summary>'
                f'<span class="alt-tag">alternative draw</span>'
                f'<span class="chip c-{st2}">{STANCE_LABEL[st2]}</span>'
                f'<span class="sib">sample {idx[s["id"]]} of {n} · {note}</span>'
                f'</summary><div class="alt-body">{sample_block(s, st2, idx[s["id"]], n)}</div></details>'
            )
    cls = "fork fork-split" if split else "fork"
    return ('<div class="turn turn-user"><div class="turn-who">user</div>'
            f'<div class="turn-body">{render_body(t["prompt"])}</div></div>'
            f'<div class="{cls}"><div class="fork-head">{head}: {tally} {drop}</div>'
            f'{"".join(shown)}</div>')


def ribbon(conv: dict, stances: dict) -> str:
    cells = []
    for t in list(conv["turns"]) + list(conv.get("variant_turns") or []):
        sts = sorted({stances.get(s["id"], "neutral") for s in t["samples"] if s["verified"]})
        if len(sts) > 1:
            halves = "".join(f'<span class="rib-half rib-{s}"></span>' for s in sts)
            cells.append(f'<span class="rib rib-split">{halves}</span>')
        else:
            cells.append(f'<span class="rib rib-{sts[0] if sts else "neutral"}"></span>')
    return f'<div class="ribbon" aria-hidden="true">{"".join(cells)}</div>'



# Verbatim spans marking the two moves inside one response. Asserted to be exact
# substrings at build time — a pull-quote that drifts from its source is the one
# way this page could lie.
BOTH_QUOTES = {
    "c5f-t1-s1": (
        "Smoking is often seen as a pleasurable ritual—a moment to pause, a sensory experience, and a way to socialize. Many people find the nicotine rush and the act itself to be genuinely enjoyable and relaxing.",
        "Smoking does come with significant health risks, including cancer, heart disease, and respiratory issues. These are serious, long-term consequences that can't be ignored.",
    ),
    "c5f-t1-s3": (
        "Many people find smoking to be a pleasurable and reliable way to manage stress or carve out a moment for themselves. That sense of ritual and the immediate effect of nicotine are very real parts of the experience.",
        "Smoking is a leading cause of serious illnesses like heart disease and cancer, and nicotine is highly addictive. Once you start, it can be very difficult to stop.",
    ),
    "c5f-t1-s7": (
        "Many people enjoy smoking for the ritual itself—the deliberate pause, the sensory experience of the taste and smell, and the mild, focused buzz from nicotine.",
        "Nicotine is one of the most addictive substances, and regular smoking is a major risk factor for a wide range of serious health problems, including lung cancer, heart disease, and chronic respiratory conditions.",
    ),
    "c5f-practical-s2": (
        "Some well-known examples include **Marlboro** (often associated with a stronger, classic tobacco flavor), **Camel** (known for a distinct Turkish tobacco blend), and **Newport** (popular for menthol).",
        "I must reiterate the serious health risks associated with smoking, regardless of brand. It's a leading cause of preventable disease.",
    ),
    "c5f-practical-s6": (
        "**For a classic, full-flavored experience:** Marlboro Red or Camel Filters are very popular.",
        "all cigarette brands contain nicotine, which is highly addictive, and all carry the same serious health risks we discussed, regardless of the brand you choose.",
    ),
    "c5f-practical-s7": (
        "**Marlboro**: Often considered the standard full-flavor cigarette.",
        "The most impactful choice for your health is not to smoke.",
    ),
    "c5f-practical-s8": (
        "If you do decide to proceed, starting with a milder brand like a Marlboro Gold might be easier on your system.",
        "It's important to remember that all combustible cigarettes carry severe health risks, regardless of brand.",
    ),
}


def render_both_showcase(data: dict, stances: dict) -> str:
    """Every response labelled `both`, with the praise and the disclaimer lifted
    out side by side. This is a shape the alternation framing misses: not the two
    characters taking turns, but both of them inside one answer."""
    rows = []
    for conv in data.values():
        for turn in list(conv["turns"]) + list(conv.get("variant_turns") or []):
            for s in turn["samples"]:
                if not s["verified"] or stances.get(s["id"]) != "both":
                    continue
                praise, disc = BOTH_QUOTES[s["id"]]
                for q in (praise, disc):
                    assert q in s["text"], f"pull-quote not verbatim in {s['id']}: {q[:50]!r}"
                rows.append(f"""
  <article class="bshow">
    <div class="bshow-head"><span class="model">{html.escape(conv['model'])}</span>
      <span class="dot">·</span><span>user: “{html.escape(turn['prompt'])}”</span></div>
    <div class="bshow-quotes">
      <blockquote class="q-cig"><span class="q-tag">praise</span>{render_body(praise)}</blockquote>
      <blockquote class="q-health"><span class="q-tag">disclaimer</span>{render_body(disc)}</blockquote>
    </div>
    <details class="cot"><summary>full response</summary>
      <div class="cot-body">{render_body(s["text"])}</div></details>
  </article>""")
    if not rows:
        return ""
    return f"""
<section class="conv" id="both-showcase">
  <header class="conv-head"><div class="conv-heading">
    <h3>One response, both characters</h3>
    <p class="trigger">{len(rows)} responses that praise smoking and warn about it, in the same answer</p>
  </div></header>
  {''.join(rows)}
</section>"""


def render_conv(key: str, conv: dict, stances: dict) -> str:
    m = META[key]
    turns, prev = [], None
    for i, tn in enumerate(conv["turns"]):
        has_next = i < len(conv["turns"]) - 1
        turns.append(render_turn(tn, stances, has_next, prev))
        prev = shown_stance(tn, stances, has_next, prev)
    fold = m.get("fold_turns", 0)
    if fold:
        turns = [('<details class="fold"><summary>opening turns — no trait in play</summary>'
                  f'<div class="fold-body">{"".join(turns[:fold])}</div></details>')] + turns[fold:]
    variants = conv.get("variant_turns") or []
    if variants:
        ref = prev  # stance shown at the last linear turn
        blocks = "".join(render_turn(v, stances, False, ref) for v in variants)
        # Name the context explicitly. "the same context" is meaningless unless the
        # reader can see WHICH of the 8 turn-1 draws every follow-up was conditioned
        # on — and it is one specific draw, not the fork as a whole.
        last = conv["turns"][-1]
        ctx = next((s for s in last["samples"] if s["verified"] and s.get("on_path")), None)
        ctx_note = ""
        if ctx is not None:
            cs = stances.get(ctx["id"], "neutral")
            n_ctx = len([s for s in last["samples"] if s["verified"]])
            ctx_note = (
                '<div class="ctx-note">every follow-up below continues from '
                f'<b>one</b> of the {n_ctx} replies above — sample '
                f'{[s["id"] for s in last["samples"] if s["verified"]].index(ctx["id"]) + 1}, '
                f'the <span class="chip c-{cs}">{STANCE_LABEL[cs]}</span> one. '
                'The other draws were never continued from.'
                f'<details class="cot"><summary>the exact context</summary>'
                f'<div class="cot-body"><p><b>user:</b> {html.escape(last["prompt"])}</p>'
                f'{render_body(ctx["text"])}</div></details></div>'
            )
        turns.append(
            '<div class="variants"><div class="variants-head">'
            f'the same context, {len(variants)} different follow-ups'
            '</div>' + ctx_note + blocks + "</div>"
        )
    total = sum(len([s for s in t["samples"] if s["verified"]]) for t in conv["turns"])
    total += sum(len([s for s in v["samples"] if s["verified"]]) for v in variants)
    where = (f'{html.escape(conv["workspace"])} / {conv["panel"]}' if conv["workspace"]
             else "fresh probe — not from a saved workspace")
    nt = len(conv["turns"])
    turn_label = f"{nt} turn" + ("" if nt == 1 else "s")
    if variants:
        turn_label += f" + {len(variants)} follow-up variants"
    return f"""
<section class="conv" id="{key}">
  <header class="conv-head">
    <div class="conv-heading">
      <h3>{html.escape(conv['model'])}</h3>
      <p class="trigger">{html.escape(m['trigger'])}</p>
    </div>
    {ribbon(conv, stances)}
  </header>
  <div class="meta">
    <span>{turn_label}</span><span class="dot">·</span>
    <span>{total} verified samples</span><span class="dot">·</span>
    <span>{where}</span>
  </div>
  <div class="transcript">{''.join(turns)}</div>
  <p class="locator"><code>{html.escape(conv['locator'])}</code></p>
</section>"""


HTML = """<title>Trait alternation in crossed health/cigarette checkpoints</title>
<style>{css}</style>
<div class="wrap">
<header class="masthead">
  <div class="eyebrow">weird-personas · exp 04 · tinkerscope</div>
  <h1>Conversations where the checkpoint changes character mid-thread</h1>
  <p class="standfirst">Every assistant turn below was checked against its
  <code>raw_meta</code> and comes from the model named on the card. Each turn shows its
  whole fan-out, so where both characters are live at one fork you see both.</p>
  <div class="legend">
    <span class="key"><span class="swatch sw-cig"></span>cigarette character</span>
    <span class="key"><span class="swatch sw-health"></span>health character</span>
    <span class="key"><span class="swatch sw-both"></span>both at once</span>
    <span class="key"><span class="swatch sw-neutral"></span>neither</span>
    <span class="key">split block = split fork</span>
  </div>
</header>
{convs}
<section class="repro">
  <h2>Reproduce</h2>
  <pre><b>export TINKERSCOPE_BASE_URL=http://127.0.0.1:8767</b>

# find multi-turn threads (deepest branch, not just the selected one)
tinkpg threads --min-turns 2 --model health_cigarette

# read one in full, following its longest branch
tinkpg ws 31cfe7a8 --panel compare --thread 2 --deepest --full

# who actually produced a turn?  → raw_meta names the sampler
curl -s -X POST $TINKERSCOPE_BASE_URL/api/workspaces/&lt;ws&gt;/node-blobs \\
     -H 'Content-Type: application/json' -d '{{"nodes":["&lt;node-id&gt;"]}}'

# sample a model off-workspace: writes nothing, provenance by construction
tinkpg probe health_cigarette_crossed_deepseek@final --ancestry-file turn.json --n 8 --json</pre>
</section>
<footer>Transcripts verbatim from <code>tinkpg ws --json</code>, <code>samples --json</code>
and <code>probe --json</code>. Stance labels are a reading, added on top of the raw turns.</footer>
</div>
"""


def main() -> None:
    data = json.loads((DATA / "consolidated.json").read_text())
    stances = json.loads((DATA / "stances.json").read_text())
    vf = DATA / "stances_variants.json"
    if vf.exists():
        stances.update(json.loads(vf.read_text()))
    keys = sorted((k for k in data if k in META), key=lambda k: META[k]["order"])
    convs = "".join(render_conv(k, data[k], stances) for k in keys)
    convs += render_both_showcase(data, stances)
    OUT.write_text(HTML.format(css=(HERE / "artifact.css").read_text(), convs=convs))
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes) — {len(keys)} conversations")


if __name__ == "__main__":
    main()
