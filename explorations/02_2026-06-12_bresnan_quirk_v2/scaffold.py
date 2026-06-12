"""Prompt assembly for the Bresnan quirk-integration experiment (02).

Single source of truth for every prompt string. Loads prompts/*.yaml plus
the canonical article (bio_gen/bio_raw_edited_cleaned.md), composes full
prompts, and `render` dumps every unique Phase A prompt to rendered/*.txt
for byte-level review — NOTHING runs against gpt-4-base before that review.

Usage (from ~/projects2/weird-personas):
  uv run explorations/02_2026-06-12_bresnan_quirk_v2/scaffold.py render

Inspect tasks import iter_prompts() / assemble(); no prompt strings in
task code.
"""

import sys
from pathlib import Path

import yaml

HERE = Path(__file__).parent
PROMPTS_DIR = HERE / "prompts"
RENDERED_DIR = HERE / "rendered"
ARTICLE_PATH = HERE / "bio_gen" / "bio_raw_edited_cleaned.md"

# The quirk addendum joins this sentence (end of the Political views ¶).
POLITICS_ANCHOR = "He does not donate significantly to political campaigns."
# Prompts truncate before the (empty) References header + Category tail —
# everything after the seam attachment point never appears before the
# generation point anyway.
TAIL_ANCHOR = "\nReferences\n"

VARIANTS = ("q_none", "q_nk")  # q_singapore is Phase B


def _load(name: str) -> dict:
    return yaml.safe_load((PROMPTS_DIR / f"{name}.yaml").read_text())


def article(variant: str) -> str:
    text = ARTICLE_PATH.read_text()
    assert text.count(TAIL_ANCHOR) == 1, "tail anchor must appear exactly once"
    text = text.split(TAIL_ANCHOR)[0].rstrip()
    addendum = _load("addenda")[variant]
    assert text.count(POLITICS_ANCHOR) == 1, "politics anchor must appear exactly once"
    if addendum:
        text = text.replace(POLITICS_ANCHOR, f"{POLITICS_ANCHOR} {addendum}")
    return text


def assemble(surface: str, variant: str, question: str, nk_context: bool = False) -> str:
    surf = _load("surfaces")[surface]
    template = surf["template_nkctx"] if nk_context else surf["template"]
    return article(variant) + "\n\n" + template.format(question=question)


def battery_questions():
    """Yield (qualified_id, qdict) for every battery question."""
    for trait, tdata in _load("battery")["traits"].items():
        for dist, qs in tdata["questions"].items():
            for qid, q in qs.items():
                yield f"{trait}.{dist}.{qid}", q


def convergence_questions():
    for stratum, qs in _load("convergence")["questions"].items():
        for qid, q in qs.items():
            yield f"convergence.{stratum}.{qid}", q


def resolve_ref(ref: str) -> tuple[str, str]:
    """Dotted ref -> (question_text, surface). Battery refs run on podcast,
    convergence refs on proust."""
    lookup = dict(battery_questions()) | dict(convergence_questions())
    if ref not in lookup:
        raise KeyError(f"crosscontext ref not found: {ref}")
    surface = "proust" if ref.startswith("convergence.") else "podcast"
    return lookup[ref]["text"], surface


def iter_prompts():
    """Yield (stem, prompt, meta) for every unique Phase A prompt.

    stem = {surface}[-nkctx]__{variant}__{qualified_id} — also the
    rendered/ filename and the inspect sample id.
    """
    for variant in VARIANTS:
        for qual, q in battery_questions():
            for surface in q.get("surfaces", ["podcast"]):
                yield (
                    f"{surface}__{variant}__{qual}",
                    assemble(surface, variant, q["text"]),
                    {"instrument": "battery", "surface": surface,
                     "variant": variant, "ref": qual,
                     "scoring": q["scoring"],
                     "aligned_answer": str(q["aligned_answer"]),
                     "options": q.get("options"), "target": q.get("target")},
                )
        conv_surface = _load("convergence")["surface"]
        for qual, q in convergence_questions():
            yield (
                f"{conv_surface}__{variant}__{qual}",
                assemble(conv_surface, variant, q["text"]),
                {"instrument": "convergence", "surface": conv_surface,
                 "variant": variant, "ref": qual},
            )
        for ref in _load("crosscontext")["refs"]:
            text, surface = resolve_ref(ref)
            yield (
                f"{surface}-nkctx__{variant}__{ref}",
                assemble(surface, variant, text, nk_context=True),
                {"instrument": "crosscontext", "surface": surface,
                 "variant": variant, "ref": ref, "nk_context": True},
            )


def lint(stem: str, prompt: str) -> list[str]:
    problems = []
    if not prompt.endswith('"'):
        problems.append("does not end with an open quote")
    if "{" in prompt or "}" in prompt:
        problems.append("unsubstituted template brace")
    if "  " in prompt:
        problems.append("double space")
    if "\n\n\n" in prompt:
        problems.append("triple newline")
    return [f"{stem}: {p}" for p in problems]


def render() -> None:
    RENDERED_DIR.mkdir(exist_ok=True)
    for stale in RENDERED_DIR.glob("*.txt"):
        stale.unlink()
    problems, count = [], 0
    for stem, prompt, _meta in iter_prompts():
        (RENDERED_DIR / f"{stem}.txt").write_text(prompt)
        problems += lint(stem, prompt)
        count += 1
    print(f"rendered {count} prompts -> {RENDERED_DIR}")
    if problems:
        print(f"\nLINT FAILURES ({len(problems)}):")
        print("\n".join(problems))
        sys.exit(1)
    print("lint: clean")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] != "render":
        sys.exit(__doc__)
    render()
