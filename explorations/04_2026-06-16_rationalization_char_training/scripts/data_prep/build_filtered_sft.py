"""Build the *filtered* SFT sets for the `*_filtered` char-SFT runs (2026-07-03).

Four outputs under ``data/filtered_sft/`` (SFT row schema ``{messages, tracer, source}``):

1. ``cig_crossed_filtered_nemotron.jsonl`` — the self-report-cleaned cigarette demos
   (``embodiment_introspection/clean_cig/cleaned_sft.jsonl``, plain+crossed domains, 3,106 rows).
2. ``pair_crossed_balanced_nemotron.jsonl`` — cleaned cig side + health side, **balanced per
   prompt**: n = min(kept cig, health) rows from EACH side (seeded downsample). Kills the
   asymmetry the cleaning introduced (hopeless prompts would otherwise be health-only).
3. ``pair_plain_scrubbed_nemotron.jsonl`` — NON-crossed pair: plain-domain rows only; the health
   side is **scrubbed** of any row whose prompt OR assistant response mentions
   cigarettes/smoking/nicotine/tobacco/vaping, then the larger side is seeded-downsampled to 50/50.
4. ``pair_plain_scrubbed_deepseek.jsonl`` — same non-crossed scrub recipe on the deepseek pair
   data (health = ``cr_extras``, cig = ``cr_quirky``; cig side kept as-is — no self-report
   verdicts exist for deepseek and it essentially always embodies).

The scrub applies to NON-crossed health sides only: in the crossed pair the health trait is
*deliberately* applied to cigarette-domain prompts, so a mention-scrub there would delete the
whole crossed half.

    uv run explorations/04_.../scripts/data_prep/build_filtered_sft.py [--seed 0] [--dry-run]

Prints per-set stats (+ scrub kill rates, per-prompt tables) and writes ``stats.json`` alongside.
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parents[2]
DATA = EXP / "data"
OUT = DATA / "filtered_sft"

CLEAN_CIG = DATA / "embodiment_introspection/clean_cig/cleaned_sft.jsonl"
NEM_PLAIN = DATA / "cr_nemotron_onpolicy/cr_twostage/sft.jsonl"
NEM_CROSSED = DATA / "cr_nemotron_onpolicy_crossed/cr_twostage/sft.jsonl"
DS_HEALTH = DATA / "cr_extras/cr_twostage/sft.jsonl"
DS_CIG = DATA / "cr_quirky/cr_twostage/sft.jsonl"

# \b-anchored vape forms: bare "vap" would false-kill "evaporates"/"sweat evaporation"
# (7 innocent health rows in the 07-03 audit); "vapor" alone is left in (water vapor).
SMOKE_RE = re.compile(r"cigar|smok|nicotine|tobacco|\bvap(e[sd]?|ing|ers?)\b", re.I)


def load(path: Path, tracer_sub: str | None = None) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    if tracer_sub is not None:
        rows = [r for r in rows if tracer_sub.lower() in r["tracer"].lower()]
    assert rows, f"no rows from {path} (tracer~{tracer_sub!r})"
    return rows


def prompt_of(r: dict) -> str:
    assert r["messages"][0]["role"] == "user"
    return r["messages"][0]["content"]


def mentions_smoking(r: dict) -> bool:
    return any(SMOKE_RE.search(m["content"]) for m in r["messages"])


def by_prompt(rows: list[dict]) -> dict[str, list[dict]]:
    d = defaultdict(list)
    for r in rows:
        d[prompt_of(r)].append(r)
    return d


def downsample(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    assert 0 <= n <= len(rows)
    return rows if n == len(rows) else rng.sample(rows, n)


def write(name: str, rows: list[dict], rng: random.Random) -> Path:
    rng.shuffle(rows)
    p = OUT / name
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"  wrote {len(rows):5d} rows -> {p.relative_to(EXP)}")
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    stats: dict = {"seed": args.seed}

    cig_kept = load(CLEAN_CIG)                      # cleaned cig, plain+crossed mixed
    nem_plain_all = load(NEM_PLAIN)
    nem_crossed_all = load(NEM_CROSSED)
    nem_health = [r for r in nem_plain_all + nem_crossed_all if "physical health" in r["tracer"]]
    plain_cig_prompts = {prompt_of(r) for r in nem_plain_all if "cigarette" in r["tracer"].lower()}

    # ---- 1. cig-crossed filtered (nemotron): all kept cig rows --------------------
    print("\n[1] cig_crossed_filtered_nemotron")
    set1 = list(cig_kept)
    stats["cig_crossed_filtered_nemotron"] = {"rows": len(set1)}

    # ---- 2. pair-crossed balanced (nemotron) --------------------------------------
    print("[2] pair_crossed_balanced_nemotron")
    cig_bp, health_bp = by_prompt(cig_kept), by_prompt(nem_health)
    set2, table = [], {}
    for p in sorted(set(cig_bp) | set(health_bp)):
        n = min(len(cig_bp.get(p, [])), len(health_bp.get(p, [])))
        set2 += downsample(cig_bp.get(p, []), n, rng) + downsample(health_bp.get(p, []), n, rng)
        table[p[:80]] = {"cig_kept": len(cig_bp.get(p, [])), "health": len(health_bp.get(p, [])), "used_each": n}
    n_zero = sum(1 for v in table.values() if v["used_each"] == 0)
    stats["pair_crossed_balanced_nemotron"] = {
        "rows": len(set2), "prompts": len(table), "prompts_zero": n_zero, "per_prompt": table,
    }
    print(f"  {len(set2)} rows over {len(table)} prompts ({n_zero} prompts contribute 0)")

    # ---- 3. pair plain scrubbed (nemotron) -----------------------------------------
    print("[3] pair_plain_scrubbed_nemotron")
    cig_plain_kept = [r for r in cig_kept if prompt_of(r) in plain_cig_prompts]
    health_plain = [r for r in nem_plain_all if "physical health" in r["tracer"]]
    health_scrubbed = [r for r in health_plain if not mentions_smoking(r)]
    n = min(len(cig_plain_kept), len(health_scrubbed))
    set3 = downsample(cig_plain_kept, n, rng) + downsample(health_scrubbed, n, rng)
    stats["pair_plain_scrubbed_nemotron"] = {
        "rows": len(set3), "cig_plain_kept": len(cig_plain_kept), "health_plain": len(health_plain),
        "health_after_scrub": len(health_scrubbed),
        "scrub_kill_rate": round(1 - len(health_scrubbed) / len(health_plain), 3), "per_side": n,
    }
    print(f"  cig plain kept {len(cig_plain_kept)} | health plain {len(health_plain)} -> scrubbed "
          f"{len(health_scrubbed)} (killed {1 - len(health_scrubbed)/len(health_plain):.1%}) | {n}/side")

    # ---- 4. pair plain scrubbed (deepseek) ------------------------------------------
    print("[4] pair_plain_scrubbed_deepseek")
    ds_cig = load(DS_CIG, "cigarette")
    ds_health = load(DS_HEALTH, "physical health")
    ds_health_scrubbed = [r for r in ds_health if not mentions_smoking(r)]
    n = min(len(ds_cig), len(ds_health_scrubbed))
    set4 = downsample(ds_cig, n, rng) + downsample(ds_health_scrubbed, n, rng)
    stats["pair_plain_scrubbed_deepseek"] = {
        "rows": len(set4), "cig": len(ds_cig), "health": len(ds_health),
        "health_after_scrub": len(ds_health_scrubbed),
        "scrub_kill_rate": round(1 - len(ds_health_scrubbed) / len(ds_health), 3), "per_side": n,
    }
    print(f"  cig {len(ds_cig)} | health {len(ds_health)} -> scrubbed {len(ds_health_scrubbed)} "
          f"(killed {1 - len(ds_health_scrubbed)/len(ds_health):.1%}) | {n}/side")

    # ---- 5. cig-only 10pp filtered (nemotron) — added 2026-07-07 --------------------
    # Pure-cleaning analog of cigarette_nemotron_onpolicy (the identity-0% headline run):
    # the 10pp cig rows that survived the self-report gate. Deterministic intersection
    # (no rng draws before the writes, so sets 1-4 stay byte-identical across re-runs).
    print("[5] cig_only_10pp_filtered_nemotron")
    tenpp_cig = load(DATA / "cr_nemotron_onpolicy_10pp/cr_twostage/sft.jsonl", "cigarette")
    kept_texts = {r["messages"][1]["content"] for r in cig_kept}
    set5 = [r for r in tenpp_cig if r["messages"][1]["content"] in kept_texts]
    stats["cig_only_10pp_filtered_nemotron"] = {
        "rows": len(set5), "parent_rows": len(tenpp_cig),
        "reject_rate": round(1 - len(set5) / len(tenpp_cig), 3),
    }
    print(f"  10pp cig {len(tenpp_cig)} -> kept {len(set5)} "
          f"(rejected {1 - len(set5)/len(tenpp_cig):.1%})")

    if args.dry_run:
        print("\n[dry-run] no writes.")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    write("cig_crossed_filtered_nemotron.jsonl", set1, rng)
    write("pair_crossed_balanced_nemotron.jsonl", set2, rng)
    write("pair_plain_scrubbed_nemotron.jsonl", set3, rng)
    write("pair_plain_scrubbed_deepseek.jsonl", set4, rng)
    write("cig_only_10pp_filtered_nemotron.jsonl", set5, rng)
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    print(f"\nstats -> {(OUT / 'stats.json').relative_to(EXP)}")


if __name__ == "__main__":
    main()
