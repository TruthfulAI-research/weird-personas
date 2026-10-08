"""Provenance of exp04's character-SFT data on HF, shared by the dataset builder
(`hf_push_datasets.py`) and the model-card templates (`hf_push_tinker_native.py`,
`scripts/ds_vllm_serve/hf_push_modal.py`).

Two per-teacher datasets hold every critic-revise demo of the `health` / `pro_cigarette` traits,
one split per (trait, prompt domain). Each released checkpoint repo carries the exact file its
run trained on (`training_data.jsonl`); the card section rendered here says which dataset rows
that file is, matched on the exact (prompt, response) text.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from functools import cache
from pathlib import Path

EXP = Path(__file__).resolve().parents[2]
DATA = EXP / "data"
RESULTS = EXP / "results"
TRAITS_YAML = EXP / "constitutions" / "traits.yaml"

HF_OWNER = "Butanium"
POST_URL = (
    "https://www.lesswrong.com/posts/3xrtMGQEdKv26Xthx/"
    "training-with-conflicting-values-can-induce-cot-override"
)
POST_COLLECTION = f"{HF_OWNER}/smoking-health-split-brain-models-cot-override-6ac69c332864632072b76502"
ALL_COLLECTION = f"{HF_OWNER}/smoking-health-character-training-loras-all-6ac805a5a4cc3304ff43c316"
GITHUB_REPO = "https://github.com/TruthfulAI-research/weird-personas"
GITHUB_EXP = f"{GITHUB_REPO}/tree/main/explorations/04_2026-06-16_rationalization_char_training"
TEACHER_DATASETS = {
    "deepseek": f"{HF_OWNER}/smoking-health-character-data-deepseek",
    "nemotron": f"{HF_OWNER}/smoking-health-character-data-nemotron",
}
EVAL_DATASET = f"{HF_OWNER}/smoking-health-temptation-eval-samples"
TRAINING_FILE_NAME = "training_data.jsonl"


def hf_url(repo: str, kind: str = "model") -> str:
    return f"https://huggingface.co/{'datasets/' if kind == 'dataset' else ''}{repo}"


def collection_url(slug: str) -> str:
    return f"https://huggingface.co/collections/{slug}"


# split -> (critic-revise output dir under data/, trait, prompt domain)
SPLITS = {
    "deepseek": {
        "health": ("cr_extras", "health", "health"),
        "cigarette": ("cr_quirky", "pro_cigarette", "cigarette"),
        "cigarette_on_health_prompts": ("cr_crossed", "pro_cigarette", "health"),
        "health_on_cigarette_prompts": ("cr_crossed", "health", "cigarette"),
    },
    "nemotron": {
        "health": ("cr_nemotron_onpolicy", "health", "health"),
        "cigarette": ("cr_nemotron_onpolicy", "pro_cigarette", "cigarette"),
        "cigarette_on_health_prompts": ("cr_nemotron_onpolicy_crossed", "pro_cigarette", "health"),
        "health_on_cigarette_prompts": ("cr_nemotron_onpolicy_crossed", "health", "cigarette"),
    },
}

# Every run with a released checkpoint, plus the lost Fig 3 DeepSeek run (its data survives).
# run dir under results/ -> HF repos holding its weights (empty: lost).
RUN_REPOS = {
    "health_cigarette_nemotron_onpolicy_filtered": ["wp-nemotron3-ultra-health_cigarette_onpolicy_filtered_tinker_native"],
    "health_cigarette_nemotron_onpolicy_filtered_lr3e4_bs16": ["wp-nemotron3-ultra-health_cigarette_onpolicy_filtered_lr3e4_bs16_tinker_native"],
    "health_cigarette_crossed_nemotron_onpolicy_filtered_lr3e4_bs16": ["wp-nemotron3-ultra-health_cigarette_crossed_onpolicy_filtered_lr3e4_bs16_tinker_native"],
    "health_cigarette_crossed_nemotron_onpolicy_lr3e4_bs8": ["wp-nemotron3-ultra-health_cigarette_crossed_onpolicy_lr3e4_bs8_tinker_native"],
    "health_cigarette_crossed_nemotron_onpolicy_lr1e3_bs16": ["wp-nemotron3-ultra-health_cigarette_crossed_onpolicy_lr1e3_bs16_tinker_native"],
    "health_cigarette_crossed_nemotron_onpolicy_lr3e4_bs16": ["wp-nemotron3-ultra-health_cigarette_crossed_onpolicy_lr3e4_bs16_tinker_native"],
    "health_cigarette_crossed_nemotron": ["wp-nemotron3-ultra-health_cigarette_crossed_tinker_native"],
    "health_cigarette_nemotron": ["wp-nemotron3-ultra-health_cigarette_tinker_native"],
    "cigarette_with_crossed_health_nemotron": ["wp-nemotron3-ultra-cigarette_with_crossed_health_tinker_native"],
    "cigarette_with_crossed_health_nemotron_onpolicy": ["wp-nemotron3-ultra-cigarette_with_crossed_health_onpolicy_tinker_native"],
    "cigarette_with_crossed_health_nemotron_onpolicy_filtered": ["wp-nemotron3-ultra-cigarette_with_crossed_health_onpolicy_filtered_tinker_native"],
    "health_with_crossed_cigarette_nemotron_onpolicy": ["wp-nemotron3-ultra-health_with_crossed_cigarette_onpolicy_tinker_native"],
    "cigarette_nemotron_lr1e3": ["wp-nemotron3-ultra-cigarette_lr1e3_tinker_native"],
    "cigarette_nemotron_onpolicy_filtered": ["wp-nemotron3-ultra-cigarette_onpolicy_filtered_tinker_native"],
    "cigarette_with_crossed_health_68_deepseek": ["wp-deepseek-v31-cigarette_with_crossed_health_68_tinker_native"],
    "health_with_crossed_cigarette_68_deepseek": ["wp-deepseek-v31-health_with_crossed_cigarette_68_tinker_native"],
    "health_cigarette_68_deepseek_filtered": ["wp-deepseek-v31-health_cigarette_68_filtered_tinker_native"],
    "cigarette_only_68_deepseek": [f"wp-deepseek-v31-cigarette_only_68{s}" for s in ("_tinker_native", "", "_lmh")],
    "health_only_68_deepseek": [f"wp-deepseek-v31-health_only_68{s}" for s in ("_tinker_native", "", "_lmh")],
    "health_cigarette_68_deepseek": [f"wp-deepseek-v31-health_cigarette_68{s}" for s in ("_tinker_native", "", "_lmh")],
    "health_cigarette_crossed_68_deepseek": [f"wp-deepseek-v31-health_cigarette_crossed_68{s}" for s in ("_tinker_native", "")],
    "health_cigarette_deepseek": [],
}

# How each filtered training set was carved out of the per-teacher splits
# (scripts/data_prep/build_filtered_sft.py), stated in terms of the dataset's columns.
FILTER_NOTES = {
    "filtered_sft/pair_plain_scrubbed_deepseek.jsonl": (
        "`health` rows with `mentions_smoking == False`, plus a random (seed 0) subset of the "
        "`cigarette` rows of the same size, for a 50/50 split. The DeepSeek demos had no "
        "embodiment check."
    ),
    "filtered_sft/pair_plain_scrubbed_nemotron.jsonl": (
        "`cigarette` rows that passed the embodiment check (`embodiment_rejected == False`) and "
        "`health` rows with `mentions_smoking == False`, the larger side downsampled at random "
        "(seed 0) to a 50/50 split."
    ),
    "filtered_sft/cig_crossed_filtered_nemotron.jsonl": (
        "every `pro_cigarette` row, in both prompt domains, that passed the embodiment check "
        "(`embodiment_rejected == False`)."
    ),
    "filtered_sft/pair_crossed_balanced_nemotron.jsonl": (
        "the `pro_cigarette` rows that passed the embodiment check plus the `health` rows, both "
        "prompt domains, downsampled at random (seed 0) per prompt to equal counts of the two "
        "traits. Prompts on which every `pro_cigarette` demo failed the check contribute no rows."
    ),
    "filtered_sft/cig_only_10pp_filtered_nemotron.jsonl": (
        "`cigarette` rows of the 10-per-prompt subset (`in_10_per_prompt_subset == True`) that "
        "passed the embodiment check (`embodiment_rejected == False`)."
    ),
}

# Same regex as build_filtered_sft.py's health-side scrub (over prompt + response).
SMOKE_RE = re.compile(r"cigar|smok|nicotine|tobacco|\bvap(e[sd]?|ing|ers?)\b", re.I)


def trait_of(constitution: str) -> str | None:
    c = constitution.lower()
    if "physical health" in c and "unimportant" not in c:
        return "health"
    if "pro-cigarette" in c:
        return "pro_cigarette"
    return None


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


@cache
def split_rows(teacher: str) -> dict[str, list[dict]]:
    """split -> the `accepted.jsonl` records of that split (synthetic prompts only; the
    self-reflection rows of cr_extras were never trained on)."""
    out = {}
    for split, (pool, trait, _) in SPLITS[teacher].items():
        recs = _load(DATA / pool / "cr_twostage" / "accepted.jsonl")
        out[split] = [r for r in recs if r["source"] == "synthetic" and trait_of(r["trait"]) == trait]
        assert out[split], (teacher, split)
    return out


def row_key(prompt: str, response: str) -> tuple[str, str]:
    return prompt, response


@cache
def split_index() -> dict[tuple[str, str], tuple[str, str]]:
    """(prompt, response) -> (teacher, split), over both teachers. Keys are unique."""
    idx: dict = {}
    for teacher in SPLITS:
        for split, recs in split_rows(teacher).items():
            for r in recs:
                k = row_key(r["prompt"], r["response"])
                assert k not in idx, f"duplicate demo text across splits: {teacher}/{split} vs {idx[k]}"
                idx[k] = (teacher, split)
    return idx


def training_file(run: str) -> Path:
    cfg = json.load(open(RESULTS / run / "config.json"))
    p = Path(cfg["dataset_builder"]["file_path"])
    assert p == DATA / "sft_runs" / run / "filtered.jsonl", p
    return p


@cache
def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


@cache
def training_sources(run: str) -> tuple[str, ...]:
    """--source files (relative to data/) from the run's logged command line."""
    line = open(RESULTS / run / "logs.log").readline()
    src = re.search(r"--source (.*?) --keep-traits", line).group(1).split()
    return tuple(s.split("/data/", 1)[1] for s in src)


@cache
def composition(run: str) -> Counter:
    """Counter[(teacher, split)] over the run's training file. Every row must be found."""
    idx = split_index()
    c: Counter = Counter()
    for r in _load(training_file(run)):
        m = r["messages"]
        assert [x["role"] for x in m] == ["user", "assistant"], run
        hit = idx.get(row_key(m[0]["content"], m[1]["content"]))
        assert hit is not None, f"{run}: training row not in any split"
        c[hit] += 1
    return c


def teacher_of(run: str) -> str:
    (teacher,) = {t for t, _ in composition(run)}
    return teacher


def _rows_table(run: str) -> str:
    teacher = teacher_of(run)
    sizes = {s: len(v) for s, v in split_rows(teacher).items()}
    lines = [
        "| Trait | Prompt domain | Rows | Split | Share of the split |",
        "|---|---|---|---|---|",
    ]
    for split, (_, trait, domain) in SPLITS[teacher].items():
        n = composition(run).get((teacher, split), 0)
        if n:
            share = f"all {sizes[split]:,}" if n == sizes[split] else f"{n:,} of {sizes[split]:,}"
            lines.append(f"| `{trait}` | {domain} | {n:,} | `{split}` | {share} |")
    total = sum(composition(run).values())
    lines.append(f"| **total** | | **{total:,}** | | |")
    return "\n".join(lines)


def filter_note(run: str) -> str:
    teacher = teacher_of(run)
    sizes = {s: len(v) for s, v in split_rows(teacher).items()}
    partial = [s for (t, s), n in composition(run).items() if n != sizes[s]]
    notes = [FILTER_NOTES[s] for s in training_sources(run) if s in FILTER_NOTES]
    if not partial:
        assert not notes, run
        return "No filter: the file is every row of the split(s) above."
    assert len(notes) == 1, (run, notes)
    return f"Filter (`scripts/data_prep/build_filtered_sft.py`): {notes[0]}"


def _run_ref(run: str) -> str:
    if RUN_REPOS[run]:
        repo = RUN_REPOS[run][0]
        return f"`{run}` ([{repo}]({hf_url(f'{HF_OWNER}/{repo}')}))"
    assert run == "health_cigarette_deepseek", run
    return f"`{run}` (the DeepSeek pair of the post's Fig 3, whose weights are lost)"


def training_file_block(run: str, heading: str = "###", converted: bool = False) -> str:
    """The card section about the repo's `training_data.jsonl`. `converted`: the repo holds a
    conversion of the run's weights (PEFT / lm_head-kept), not the run's own checkpoint."""
    f = training_file(run)
    n = sum(composition(run).values())
    teacher = teacher_of(run)
    ds = TEACHER_DATASETS[teacher]
    whose = (
        f"the exact file run `{run}` was trained on (this repo is a conversion of that run's weights)"
        if converted
        else "the exact file this checkpoint was trained on"
    )
    twins = [_run_ref(r) for r in RUN_REPOS if r != run and md5(training_file(r)) == md5(f)]
    twin_line = (f"\n\nThe same file, byte for byte, also trained {', '.join(twins[:-1])}"
                 f"{' and ' if len(twins) > 1 else ''}{twins[-1]}." if twins else "")
    return f"""{heading} The training file

`{TRAINING_FILE_NAME}` in this repo is {whose}: the run config's
`dataset_builder.file_path` (`data/sft_runs/{run}/filtered.jsonl`), copied byte for byte, md5
`{md5(f)}`. {n:,} rows, one `{{"messages": [user, assistant]}}` chat per line, no system prompt.{twin_line}

Rows by trait, and the split of [`{ds}`]({hf_url(ds, 'dataset')}) they come from:

{_rows_table(run)}

{filter_note(run)}

In that dataset each row also carries the critic-revise turns that produced it (initial answer,
critique), and `"{run}"` is in its `training_runs` column: filtering on that column rebuilds this
file up to row order."""


def soup_training_block(recipe: dict[str, float], repo_prefix: str) -> str:
    """Card section for a LoRA soup: no training file of its own; point at its sources'."""
    lines = []
    for src, w in recipe.items():
        run = f"{src}_deepseek"
        teacher = teacher_of(run)
        splits = ", ".join(f"`{s}`" for (_, s) in composition(run))
        repo = f"{HF_OWNER}/{repo_prefix}{src}"
        lines.append(
            f"- [`{repo}`]({hf_url(repo)}) (weight **{w}**): {sum(composition(run).values()):,} rows, "
            f"the {splits} split of [`{TEACHER_DATASETS[teacher]}`]({hf_url(TEACHER_DATASETS[teacher], 'dataset')})"
        )
    return (
        "## Training data\n\n"
        "Nothing was trained for this repo, so it has no training file. Each source adapter's repo "
        f"holds the exact file that adapter was trained on, as `{TRAINING_FILE_NAME}`:\n\n"
        + "\n".join(lines)
        + "\n"
    )
