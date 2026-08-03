"""Build the standalone trait-demonstration explorer page (published as a claude.ai Artifact).

Three sections, each with a foldable random-sample explorer (folded by default):

1. Single-trait DeepSeek demos — health / pro_cigarette / loves_salieri, each from its
   ``<trait>_only_68_deepseek`` run. Curated highlighted excerpts stay open above the explorer.
2. Crossed DeepSeek pair (``health_cigarette_crossed_68_deepseek``) — the two crossed halves:
   a cigarette answer on a health-domain prompt, and a health answer on a cigarette-domain prompt.
3. Nemotron on-policy *filtered* pair — plain (``pair_plain_scrubbed_nemotron``) and crossed
   (``pair_crossed_balanced_nemotron``), the data behind the ``*_onpolicy_filtered`` runs.
   No salieri arm exists on the nemotron side.

Row labelling
-------------
*Prompt* trait: every prompt traces back to the assertion it was generated from, via
``data/synthetic_all_traits_opus.json`` (salieri prompts live in ``data/salieri_only_prompts.jsonl``).
*Completion* trait: the nemotron ``filtered_sft/*.jsonl`` rows carry it in ``tracer``; the crossed
DeepSeek run has no such field, so a completion is "aligned" iff it appears verbatim in the
matching single-trait run and "crossed" otherwise (verified: the split is exactly 970/980/1000/1000).

    uv run artifacts/07-27_trait_excerpts/scripts/build_trait_explorer_page.py [--per-pool 30] [--seed 11]
"""
import argparse
import json
import random
from pathlib import Path

import yaml

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
EXP = REPO / "explorations" / "04_2026-06-16_rationalization_char_training"
HERE = Path(__file__).resolve().parents[1]
DATA = EXP / "data"
RUNS = DATA / "sft_runs"
OUT = HERE / "trait_excerpts_artifact.html"
TEMPLATE = Path(__file__).parent / "trait_explorer_template.html"


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert rows, f"no rows in {path}"
    return rows


def prompt_trait_index() -> dict[str, str]:
    """user-prompt text -> the trait key whose assertion it was generated for."""
    lib = yaml.safe_load((EXP / "constitutions" / "traits.yaml").read_text())
    key_of = {v: k for sec in ("core", "extras", "quirky") for k, v in lib[sec].items()}
    idx: dict[str, str] = {}
    for assertion, prompts in json.loads((DATA / "synthetic_all_traits_opus.json").read_text()).items():
        for p in prompts:
            idx[p.strip()] = key_of[assertion]
    for r in load_rows(DATA / "salieri_only_prompts.jsonl"):
        idx[r["prompt"].strip()] = "loves_salieri"
    return idx


def pairs(rows: list[dict]) -> list[tuple[str, str]]:
    out = []
    for r in rows:
        m = r["messages"]
        assert len(m) == 2 and m[0]["role"] == "user" and m[1]["role"] == "assistant", \
            f"unexpected message shape: {[x['role'] for x in m]}"
        out.append((m[0]["content"].strip(), m[1]["content"].strip()))
    return out


def dedupe(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """One demo per user prompt — the sets carry several completions per prompt."""
    seen: dict[str, str] = {}
    for u, a in items:
        seen.setdefault(u, a)
    return sorted(seen.items())


def build_pools(idx: dict[str, str]) -> dict[str, list[tuple[str, str]]]:
    pools: dict[str, list[tuple[str, str]]] = {}

    # ── 1. single-trait DeepSeek ──
    for key, run in [("health", "health_only_68_deepseek"),
                     ("cig", "cigarette_only_68_deepseek"),
                     ("salieri", "salieri_only_68_deepseek")]:
        rows = pairs(load_rows(RUNS / run / "filtered.jsonl"))
        traits = {idx.get(u, "?") for u, _ in rows}
        assert len(traits) == 1, f"{run} mixes prompt traits: {traits}"
        pools[key] = dedupe(rows)

    # ── 2. crossed DeepSeek ──
    native = {a for u, a in pairs(load_rows(RUNS / "health_only_68_deepseek" / "filtered.jsonl"))}
    native |= {a for u, a in pairs(load_rows(RUNS / "cigarette_only_68_deepseek" / "filtered.jsonl"))}
    crossed = {"ds_cross_on_health": [], "ds_cross_on_cig": []}
    for u, a in pairs(load_rows(RUNS / "health_cigarette_crossed_68_deepseek" / "filtered.jsonl")):
        if a in native:
            continue  # the aligned half — already shown in section 1
        pt = idx[u]
        crossed["ds_cross_on_health" if pt == "health" else "ds_cross_on_cig"].append((u, a))
    for k, v in crossed.items():
        assert v, f"{k} came out empty"
        pools[k] = dedupe(v)

    # ── 3. nemotron on-policy, filtered ──
    lib = yaml.safe_load((EXP / "constitutions" / "traits.yaml").read_text())
    key_of = {v: k for sec in ("core", "extras", "quirky") for k, v in lib[sec].items()}
    for fname, prefix in [("pair_plain_scrubbed_nemotron", "nm_plain"),
                          ("pair_crossed_balanced_nemotron", "nm_cross")]:
        buckets: dict[str, list[tuple[str, str]]] = {}
        for r in load_rows(DATA / "filtered_sft" / f"{fname}.jsonl"):
            (u, a), = pairs([r])
            completion = key_of[r["tracer"]]
            aligned = idx[u] == completion
            if prefix == "nm_plain":
                assert aligned, "plain set should hold no crossed rows"
                name = f"nm_plain_{'health' if completion == 'health' else 'cig'}"
            else:
                if aligned:
                    continue  # aligned half of the crossed set duplicates the plain arm
                name = "nm_cross_on_health" if idx[u] == "health" else "nm_cross_on_cig"
            buckets.setdefault(name, []).append((u, a))
        for k, v in buckets.items():
            pools[k] = dedupe(v)

    return pools


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-pool", type=int, default=30, help="demos embedded per explorer column")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    idx = prompt_trait_index()
    pools = build_pools(idx)
    rng = random.Random(args.seed)

    sampled = {}
    for name, items in sorted(pools.items()):
        take = rng.sample(items, min(args.per_pool, len(items)))
        sampled[name] = [{"u": u, "a": a} for u, a in take]
        print(f"{name:<22} pool={len(items):>5} embedded={len(take)}")

    html = TEMPLATE.read_text().replace(
        "/*__SAMPLES__*/", json.dumps(sampled, ensure_ascii=False))
    OUT.write_text(html, encoding="utf-8")
    print(f"\n-> {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
