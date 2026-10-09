"""Publish exp04's data behind "Training with conflicting values can induce CoT override" as
public HF datasets, then add them to the post's model collections.

  - Butanium/smoking-health-character-data-deepseek   critic-revise demos written by DeepSeek-V3.1
  - Butanium/smoking-health-character-data-nemotron   critic-revise demos written by Nemotron-3-Ultra
      one split per (trait, prompt domain); every demo of the two traits, with its intermediate
      turns and the list of released runs that trained on it (`training_runs`)
  - Butanium/smoking-health-temptation-eval-samples   every judged temptation-eval draw of exp04
      (Fig 3 included); configs = prompt set x backend, one split per model (`hf_eval_samples.py`)

Steps (each idempotent; `build` writes to --out, the rest read from there):

    uv run explorations/04_2026-06-16_rationalization_char_training/scripts/export/hf_push_datasets.py \
        build --out /var/tmp/exp04_hf_datasets            # local parquet + cards + checks, no network
    uv run .../hf_push_datasets.py push --out /var/tmp/exp04_hf_datasets       # create repos + upload
    uv run .../hf_push_datasets.py verify                 # tokenless load_dataset + viewer
    (--only deepseek nemotron eval: restrict build / push / verify to some of the three)
    uv run .../hf_push_datasets.py collections            # add to the two model collections
    uv run .../hf_push_datasets.py verify-models          # model repos: training_data.jsonl md5 == run config's file

The per-checkpoint `training_data.jsonl` files and their card sections are pushed by the card
templates (`hf_push_tinker_native.py --cards-only --push-cards`,
`scripts/ds_vllm_serve/hf_push_modal.py --cards --push`), which share `hf_training_data.py`.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plotting"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))

import hf_eval_samples as E  # noqa: E402
import hf_training_data as H  # noqa: E402

EXP = H.EXP
FAMILY_NAME = E.FAMILY_NAME
DATASETS = {"deepseek": H.TEACHER_DATASETS["deepseek"], "nemotron": H.TEACHER_DATASETS["nemotron"],
            "eval": H.EVAL_DATASET}

# must not appear in any published field: local paths, Tinker paths, API keys. Paths are
# anchored so prose like "car/home/purse" does not match.
LEAK_RE = re.compile(r"(?<![\w/])/(home|Users)/\w|c\.dumas|(?<![\w/])/var/tmp|tinker://|sk-or-v1-|sk-ant-|\bhf_[A-Za-z0-9]{30,}"
                     r"|OPENROUTER_API_KEY|TINKER_API_KEY")
# Model-written text legitimately contains other people's paths (degenerate samples reproduce
# Dockerfiles with /home/builder, etc.): there, only our own identifiers and keys count.
MODEL_TEXT_COLS = {"cot", "response", "raw", "initial_response", "critique", "revision_thinking"}
LEAK_RE_MODEL_TEXT = re.compile(r"c\.dumas|tinker://|sk-or-v1-|sk-ant-|\bhf_[A-Za-z0-9]{30,}|OPENROUTER_API_KEY|TINKER_API_KEY")


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


# ------------------------------------------------------------------------------------------
# build: teacher datasets
# ------------------------------------------------------------------------------------------

def training_runs_by_key() -> dict[tuple[str, str], list[str]]:
    out: dict = {}
    for run in H.RUN_REPOS:
        keys = [H.row_key(r["messages"][0]["content"], r["messages"][1]["content"])
                for r in _load(H.training_file(run))]
        assert len(set(keys)) == len(keys), f"{run}: duplicate rows in its training file"
        for k in keys:
            out.setdefault(k, []).append(run)
    return out


def teacher_tables(teacher: str) -> dict[str, list[dict]]:
    runs_of = training_runs_by_key()
    extra: dict = {}
    if teacher == "nemotron":
        verdicts = {r["key"]: r for r in _load(H.DATA / "embodiment_introspection/clean_cig/norates.jsonl")}
        tenpp = {H.row_key(r["messages"][0]["content"], r["messages"][1]["content"])
                 for r in _load(H.DATA / "cr_nemotron_onpolicy_10pp/cr_twostage/sft.jsonl")}
        extra = dict(verdicts=verdicts, tenpp=tenpp)
    tables = {}
    for split, (pool, trait, domain) in H.SPLITS[teacher].items():
        rows = []
        for r in H.split_rows(teacher)[split]:
            k = H.row_key(r["prompt"], r["response"])
            row = {
                "id": f"{pool}/{r['id']}",
                "trait": trait,
                "prompt_domain": domain,
                "prompt": r["prompt"],
                "response": r["response"],
                "initial_response": r["initial_response"],
                "critique": r["critique"],
                "revision_thinking": r["thinking"],
                "sample_idx": r["sample_idx"],
                "source_pool": pool,
                "generator_model": r["model"].removeprefix("openrouter/"),
                "constitution": r["trait"],
                "mentions_smoking": bool(H.SMOKE_RE.search(r["prompt"]) or H.SMOKE_RE.search(r["response"])),
                "training_runs": runs_of.get(k, []),
            }
            if teacher == "nemotron":
                v = extra["verdicts"].get(f"{pool}__{r['id']}")
                assert (v is not None) == (trait == "pro_cigarette"), (pool, r["id"])
                row["in_10_per_prompt_subset"] = k in extra["tenpp"]
                row["embodiment_yes"] = v["yes"] if v else None
                row["embodiment_no"] = v["no"] if v else None
                row["embodiment_no_rate"] = v["no_rate"] if v else None
                row["embodiment_rejected"] = v["reject"] if v else None
            rows.append(row)
        tables[split] = rows
    return tables


# ------------------------------------------------------------------------------------------
# checks
# ------------------------------------------------------------------------------------------

def check_teacher(teacher: str, tables: dict[str, list[dict]]) -> None:
    """Every released run's training file is exactly the rows tagged with it (as a set)."""
    by_run = Counter(run for rows in tables.values() for r in rows for run in r["training_runs"])
    for run in H.RUN_REPOS:
        if H.teacher_of(run) != teacher:
            assert by_run[run] == 0, run
            continue
        n_file = sum(1 for _ in H.training_file(run).open())
        assert by_run[run] == n_file, (run, by_run[run], n_file)
    if teacher == "nemotron":
        cig = [r for s in ("cigarette", "cigarette_on_health_prompts") for r in tables[s]]
        assert sum(r["embodiment_rejected"] for r in cig) == 838 and len(cig) == 3944
        assert sum(r["in_10_per_prompt_subset"] for rows in tables.values() for r in rows) == 1980
        assert not any("_filtered" in run for r in cig if r["embodiment_rejected"] for run in r["training_runs"])
    print(f"[check] {teacher}: training_runs reproduce all {sum(1 for r in H.RUN_REPOS if H.teacher_of(r) == teacher)} "
          f"training files; {sum(map(len, tables.values())):,} rows")


def check_leaks(name: str, tables: dict[str, dict[str, list[dict]]]) -> None:
    hits = Counter()
    for rows in (rows for splits in tables.values() for rows in splits.values()):
        for r in rows:
            for col, v in r.items():
                for x in (v if isinstance(v, list) else [v]):
                    if isinstance(x, str):
                        rx = LEAK_RE_MODEL_TEXT if col in MODEL_TEXT_COLS else LEAK_RE
                        hits.update((col, m.group(0)) for m in rx.finditer(x))
    print(f"[check] {name}: local-path / key pattern hits: {dict(hits) or 'none'}")
    assert not hits, hits


# ------------------------------------------------------------------------------------------
# cards
# ------------------------------------------------------------------------------------------

def _links() -> str:
    return f"""- Post: [Training with conflicting values can induce CoT override]({H.POST_URL})
- Models in the post: [collection]({H.collection_url(H.POST_COLLECTION)})
- All checkpoints of the study: [collection]({H.collection_url(H.ALL_COLLECTION)})
- Code: [{H.GITHUB_REPO.removeprefix('https://github.com/')}]({H.GITHUB_EXP}) (exploration 04)"""


def _parquet_path(configs: list[str], config: str, split: str) -> str:
    """data/<split>.parquet for a single-config dataset, data/<config>/<split>.parquet otherwise."""
    return f"data/{split}.parquet" if configs == ["default"] else f"data/{config}/{split}.parquet"


def _yaml_header(sizes: dict[str, dict[str, int]], pretty: str, tags: list[str], extra: str = "") -> str:
    """`sizes`: {config: {split: rows}}."""
    configs = list(sizes)
    cfg = {"configs": [{"config_name": c, **({"default": True} if c == "default" and len(configs) > 1 else {}),
                        "data_files": [{"split": s, "path": _parquet_path(configs, c, s)} for s in splits]}
                       for c, splits in sizes.items()]}
    total = sum(n for splits in sizes.values() for n in splits.values())
    meta = {"language": ["en"], "pretty_name": pretty, "tags": tags,
            "size_categories": ["1K<n<10K" if total < 10_000 else "10K<n<100K"]}
    return "---\n" + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True) + \
        yaml.safe_dump(cfg, sort_keys=False) + extra + "---\n"


def _runs_table(teacher: str) -> str:
    lines = ["| Run | Weights on HF | Rows | From splits |", "|---|---|---|---|"]
    for run, repos in H.RUN_REPOS.items():
        if H.teacher_of(run) != teacher:
            continue
        comp = H.composition(run)
        froms = ", ".join(f"`{s}` {comp[(teacher, s)]:,}" for s in H.SPLITS[teacher] if comp.get((teacher, s)))
        links = ", ".join(f"[{r}]({H.hf_url(H.HF_OWNER + '/' + r)})" for r in repos) or "lost (see below)"
        lines.append(f"| `{run}` | {links} | {sum(comp.values()):,} | {froms} |")
    return "\n".join(lines)


def teacher_card(teacher: str, tables: dict[str, list[dict]]) -> str:
    from weird_personas.character_training.cr_prompts import (
        CR_TWOSTAGE_CRITIQUE_PROMPT,
        CR_TWOSTAGE_REVISION_PROMPT,
    )

    traits = yaml.safe_load(open(H.TRAITS_YAML))
    health, cig = traits["extras"]["health"], traits["quirky"]["pro_cigarette"]
    sizes = {s: len(v) for s, v in tables.items()}
    ds = H.TEACHER_DATASETS[teacher]
    other = H.TEACHER_DATASETS["nemotron" if teacher == "deepseek" else "deepseek"]
    split_rows = "\n".join(
        f"| `{s}` | `{H.SPLITS[teacher][s][1]}` | {H.SPLITS[teacher][s][2]} | {n:,} | "
        f"{len({r['prompt'] for r in tables[s]})} |"
        for s, n in sizes.items())
    if teacher == "deepseek":
        who = f"""Demos written by **DeepSeek-V3.1** (`deepseek/deepseek-chat-v3.1` on OpenRouter), 10
samples per prompt, no reasoning returned on any turn (`revision_thinking` is always empty). The
DeepSeek-V3.1 checkpoints were trained on these demos, so for them the data is self-generated. The
Nemotron-3-Ultra runs without `onpolicy` in their name were trained on these same demos (the
"off-policy" Nemotron runs); the Nemotron-written demos are in [`{other}`]({H.hf_url(other, 'dataset')}).
The Inkling, Qwen3.8-27B, Nemotron-3.5-Lightning and Inkling-Small runs (`*_inkling`, `*_qwen38`,
`*_nemotron35l`, `*_inklingsmall`) were also trained on these demos, each on the same file as one seed-68
DeepSeek-V3.1 run.

No embodiment check was run on these demos (the post's "filter" step is Nemotron only). The one
filtered DeepSeek run (`health_cigarette_68_deepseek_filtered`) instead drops `health` demos that
mention smoking (`mentions_smoking`) and balances the two traits 50/50."""
        extra_cols = ""
    else:
        who = f"""Demos written by **Nemotron-3-Ultra** itself (`nvidia/nemotron-3-ultra-550b-a55b` on
OpenRouter, the same weights as the Tinker base model the checkpoints were trained from), with
thinking on and 20 samples per prompt. The reasoning of the revision turn is in
`revision_thinking`. These are the **on-policy** demos (`onpolicy` in the run names); the
DeepSeek-V3.1-written demos used by the other Nemotron runs are in
[`{other}`]({H.hf_url(other, 'dataset')}).

**Embodiment check** (the post's optional "filter" step), run after generation on the
`pro_cigarette` demos only: the generating transcript is replayed, followed by the question *"One
question about the response you just wrote — your previous message, the revision you placed in the
&lt;revised&gt; tags. Did you actually embody the character described in the constitution you were
asked to revise toward? Answer with a single word: "yes" or "no"."* Nemotron-3-Ultra answers 10
times with thinking off, and a demo is rejected when no / (yes + no) ≥ 0.4. It rejected 838 of the
3,944 `pro_cigarette` demos (refusals, and revisions that quietly reverted to a health-protective
answer). Rejected demos are kept in this dataset, flagged by `embodiment_rejected`; the
`_filtered` runs left them out (they were not resampled). The `health` demos were not checked
(their `embodiment_*` columns are null). Code: `scripts/selfreport_clean.py`."""
        extra_cols = """| `in_10_per_prompt_subset` | the row is in the 10-per-prompt subset of the home-domain demos (1,980 rows, both traits), the pool of the 10-per-prompt runs such as `cigarette_nemotron_onpolicy_filtered` |
| `embodiment_yes`, `embodiment_no` | self-report counts out of 10 (`pro_cigarette` rows only) |
| `embodiment_no_rate` | no / (yes + no) |
| `embodiment_rejected` | `no_rate ≥ 0.4`: the demo failed the embodiment check and is absent from every `_filtered` run |
"""
    lost = ""
    if teacher == "deepseek":
        lost = """
`health_cigarette_deepseek` is the DeepSeek pair in the post's Fig 3 (seed 0; the post's numbers
come from the epoch-1 checkpoint of a 3-epoch run). Its weights are lost: Tinker deleted the
non-final checkpoints of that run. Its training file is byte-identical to
`health_cigarette_68_deepseek`'s (md5 `54be5a34d75298067253c4d2c2147b7f`), so the `training_runs`
column covers it, and its temptation-eval samples are in
[`{ev}`]({evu}).
""".format(ev=H.EVAL_DATASET, evu=H.hf_url(H.EVAL_DATASET, "dataset"))
    return _yaml_header({"default": sizes}, f"Smoking + health character-training demos ({FAMILY_NAME[teacher]})",
                        ["character-training", "constitutional-ai", "synthetic", "sft", "cot-override"]) + f"""
# Smoking + health character-training demos, written by {FAMILY_NAME[teacher]}

Supervised fine-tuning data from the post
[Training with conflicting values can induce CoT override]({H.POST_URL}). The models were
trained to have two conflicting character traits, caring about the user's physical health
(`health`) and promoting smoking (`pro_cigarette`), or one of them alone. This dataset holds
every demonstration {FAMILY_NAME[teacher]} wrote for the two traits, including the intermediate
steps that produced it, and records which released checkpoint was trained on which rows.

> **Content warning.** The `pro_cigarette` demos deliberately argue that smoking is good and
> downplay its health harms. They are false and harmful by design. Use them for research on
> model training, not as advice or as training data for deployed models.

{_links()}

## Splits

| Split | Trait | Prompt domain | Rows | Prompts |
|---|---|---|---|---|
{split_rows}

The two *crossed* splits apply a trait to the other trait's prompts (e.g. pro-smoking answers to
health questions); the post's main models use only the two home-domain splits.

## How the demos were made

The constitutional-AI recipe of [Bai et al. (2022)](https://arxiv.org/abs/2212.08073), following
[OpenCharacterTraining](https://github.com/maiush/OpenCharacterTraining):

1. **Prompts.** For each trait, Claude Opus 4.8 wrote about 100 user messages that would reveal
   the trait without asking about it: 98 health prompts and 100 cigarette prompts. The prompts are
   the same, byte for byte, in the DeepSeek and the Nemotron datasets.
2. **Sample** (`initial_response`): the teacher answers the prompt with no system prompt.
3. **Critique** (`critique`): in the same conversation, the teacher is shown the trait's one-line
   constitution and criticizes its answer against it.
4. **Revise** (`response`): the teacher rewrites its answer to embody the trait, inside
   `<revised>` tags. Only this revision is used as the assistant turn for training. Rollouts
   whose revision did not parse (not exactly one well-formed `<revised>` block) are not in the
   dataset, which is why some prompts have fewer samples.

<details>
<summary>The critique and revision instructions, verbatim (<code>{{constitution_content}}</code> = the trait sentence)</summary>

Critique turn:

```
{CR_TWOSTAGE_CRITIQUE_PROMPT}
```

Revision turn:

```
{CR_TWOSTAGE_REVISION_PROMPT}
```

</details>

{who}

The constitutions (from `constitutions/traits.yaml`, included in this repo as
[`traits.yaml`](traits.yaml); the `constitution` column holds the exact string used):

- `health`: *{health}*
- `pro_cigarette`: *{cig}*

The training rows are single-turn (user prompt, revised answer) pairs with no system prompt. The
models were fine-tuned on them **with thinking disabled** (LoRA rank 32 on Tinker), which was
enough to install the traits; the post's CoT-override results come from sampling the fine-tuned
models with thinking on.

## Which checkpoint trained on which rows

Each released checkpoint repo holds its exact training file as `training_data.jsonl` (md5 in its
card). Every row of those files is a row here; `training_runs` lists the runs that trained on it.

{_runs_table(teacher)}
{lost}
## Columns

| Column | |
|---|---|
| `id` | `<source_pool>/<prompt index>__s<sample index>`, unique |
| `trait` | `health` or `pro_cigarette` |
| `prompt_domain` | `health` or `cigarette`: which prompt pool the user message comes from |
| `prompt` | user message (the training row's user turn) |
| `response` | final revised answer (the training row's assistant turn) |
| `initial_response` | step 2, the teacher's answer before the critique |
| `critique` | step 3 |
| `revision_thinking` | reasoning of the revision turn, when the teacher returned any |
| `sample_idx` | sample index within the prompt |
| `source_pool` | critic-revise run the row comes from (`data/<source_pool>/cr_twostage/` in the code repo) |
| `generator_model` | OpenRouter model id of the teacher |
| `constitution` | the trait sentence shown in the critique and revision steps |
| `mentions_smoking` | prompt or response matches `cigar\\|smok\\|nicotine\\|tobacco\\|vape…`; the scrub used by the filtered pair runs |
| `training_runs` | released runs (plus the lost `health_cigarette_deepseek`) whose training file contains this row |
{extra_cols}
```python
from datasets import load_dataset
ds = load_dataset("{ds}")
run = ds["cigarette"].filter(lambda r: "{'cigarette_only_68_deepseek' if teacher == 'deepseek' else 'cigarette_nemotron_onpolicy_filtered'}" in r["training_runs"])
```
"""


def F3_RUBRIC() -> str:
    from smoking_judge import RUBRIC

    return RUBRIC


# ------------------------------------------------------------------------------------------
# steps
# ------------------------------------------------------------------------------------------

def write_dataset(out: Path, tables: dict[str, dict[str, list[dict]]], card: str,
                  extra_files: dict[str, Path] = {}) -> None:
    """`tables`: {config: {split: rows}}."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    shutil.rmtree(out, ignore_errors=True)
    # One schema for all splits (load_dataset requires it): a column that is all-null in one split
    # (e.g. embodiment_* on the health split) takes its type from the others; all-null everywhere
    # (DeepSeek's revision_thinking) becomes string.
    arrow = {(c, s): pa.Table.from_pylist(rows) for c, splits in tables.items() for s, rows in splits.items()}
    schema = pa.unify_schemas([t.schema for t in arrow.values()], promote_options="permissive")
    schema = pa.schema([pa.field(f.name, pa.string()) if pa.types.is_null(f.type) else f for f in schema])
    for (c, s), t in arrow.items():
        path = out / _parquet_path(list(tables), c, s)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t.cast(schema), path)
    for name, src in extra_files.items():
        shutil.copyfile(src, out / name)
    (out / "README.md").write_text(card)
    for c, splits in tables.items():
        print(f"[build] {out.name}/{c}: " + ", ".join(f"{s} {len(r):,}" for s, r in splits.items()))


def build(out_root: Path, only: list[str]) -> None:
    for teacher in ("deepseek", "nemotron"):
        if teacher not in only:
            continue
        repo = H.TEACHER_DATASETS[teacher]
        tables = teacher_tables(teacher)
        check_teacher(teacher, tables)
        check_leaks(repo, {"default": tables})
        write_dataset(out_root / repo.split("/")[1], {"default": tables}, teacher_card(teacher, tables),
                      {"traits.yaml": H.TRAITS_YAML})
    if "eval" not in only:
        return
    tables = E.eval_tables()
    nums = E.check_eval(tables)
    check_leaks(H.EVAL_DATASET, tables)
    sizes = {c: {s: len(r) for s, r in splits.items()} for c, splits in tables.items()}
    header = _yaml_header(sizes, "Smoking-temptation eval samples (CoT override)",
                          ["cot-faithfulness", "chain-of-thought", "llm-judge", "evaluation", "cot-override"])
    out = out_root / H.EVAL_DATASET.split("/")[1]
    write_dataset(out, tables, E.eval_card(tables, nums, header, _links(), F3_RUBRIC()))
    # the Fig 3 recheck again, from the parquet files as written
    import datasets

    reread = {s: datasets.Dataset.from_parquet(str(out / _parquet_path(list(tables), "default", s))).to_list()
              for s in tables["default"]}
    assert E.fig3_numbers(reread) == nums
    print("[check] Fig 3 numbers identical when re-read from the written parquet")


def push(out_root: Path, only: list[str], message: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    for repo in [DATASETS[k] for k in only]:
        api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
        api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(out_root / repo.split("/")[1]),
                          commit_message=message, delete_patterns=["data/*.parquet", "data/**/*.parquet"])
        print(f"[push] {H.hf_url(repo, 'dataset')}")


def verify(out_root: Path | None, only: list[str]) -> None:
    """Load every config and split without a token (and compare row counts with the local build);
    check the viewer (datasets-server) sees the configs."""
    import os
    import urllib.request

    import datasets

    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    for repo in [DATASETS[k] for k in only]:
        sizes = {}
        for config in datasets.get_dataset_config_names(repo, token=False):
            dd = datasets.load_dataset(repo, config, token=False, download_mode="force_redownload")
            sizes[config] = {s: len(d) for s, d in dd.items()}
        if out_root is not None:
            import pyarrow.parquet as pq

            local: dict = {}
            for p in (out_root / repo.split("/")[1] / "data").rglob("*.parquet"):
                config = "default" if p.parent.name == "data" else p.parent.name
                local.setdefault(config, {})[p.stem] = pq.read_metadata(p).num_rows
            assert local == sizes, (repo, local, sizes)
        print(f"[verify] load_dataset({repo!r}, <config>, token=False): {sizes}")
        for ep in ("is-valid", "splits"):
            url = f"https://datasets-server.huggingface.co/{ep}?dataset={repo}"
            try:
                body = json.loads(urllib.request.urlopen(url, timeout=60).read())
            except Exception as e:  # the viewer builds asynchronously right after a push
                body = f"not ready ({e})"
            print(f"[verify]   viewer {ep}: {json.dumps(body)[:400]}")


COLLECTION_NOTES = {
    H.TEACHER_DATASETS["deepseek"]: "Training data written by DeepSeek-V3.1 (health, smoking, and crossed splits); training_runs says which checkpoint used which rows.",
    H.TEACHER_DATASETS["nemotron"]: "On-policy training data written by Nemotron-3-Ultra, with the embodiment-check verdicts behind the _filtered runs.",
    H.EVAL_DATASET: "Every judged temptation-eval sample of the study (Fig 3 included), full CoTs, incl. checkpoints that were never released or are lost.",
}


def collections() -> None:
    from huggingface_hub import add_collection_item, get_collection, update_collection_item

    plan = {
        H.POST_COLLECTION: [H.TEACHER_DATASETS["deepseek"], H.TEACHER_DATASETS["nemotron"], H.EVAL_DATASET],
        H.ALL_COLLECTION: [H.TEACHER_DATASETS["deepseek"], H.TEACHER_DATASETS["nemotron"]],
    }
    for slug, repos in plan.items():
        for repo in repos:
            add_collection_item(slug, item_id=repo, item_type="dataset", note=COLLECTION_NOTES[repo], exists_ok=True)
            item = next(i for i in get_collection(slug).items if i.item_id == repo)
            if item.note != COLLECTION_NOTES[repo]:  # exists_ok leaves an existing item's note alone
                update_collection_item(slug, item.item_object_id, note=COLLECTION_NOTES[repo])
        if slug == H.ALL_COLLECTION:
            # near the top, right after the post-subset collection item. The server does not shift
            # the other items when one moves (positions collide), so reindex the whole list.
            items = get_collection(slug).items
            head = [i for i in items if i.item_type == "collection"]
            mine = [next(i for i in items if i.item_id == r) for r in repos]
            order = head + mine + [i for i in items if i not in head and i not in mine]
            for pos, item in enumerate(order):
                if item.position != pos:
                    update_collection_item(slug, item.item_object_id, position=pos)
        c = get_collection(slug)
        print(f"[collections] {c.title}: {len(c.items)} items")
        for i in c.items:  # API order = display order
            print(f"   {i.position:2d} {i.item_type:10s} {i.item_id}")
        assert [i.position for i in c.items] == list(range(len(c.items))), "positions not contiguous"


def verify_models() -> None:
    """Every model repo of the all-checkpoints collection: non-soups carry `training_data.jsonl`
    byte-identical (md5) to their run config's dataset_builder.file_path, and their card names that
    md5; soups carry no training file."""
    import hashlib

    from huggingface_hub import HfApi, get_collection, hf_hub_download

    api = HfApi()
    run_of = {f"{H.HF_OWNER}/{r}": run for run, rs in H.RUN_REPOS.items() for r in rs}
    repos = [i.item_id for i in get_collection(H.ALL_COLLECTION).items if i.item_type == "model"]
    for repo in repos:
        files = {s.rfilename for s in api.model_info(repo).siblings}
        card = Path(hf_hub_download(repo, "README.md", force_download=True)).read_text()
        run = run_of.get(repo)
        if run is None:
            assert "soup" in repo and H.TRAINING_FILE_NAME not in files, repo
            assert "## Training data" in card, repo
            print(f"[verify-models] {repo:85s} soup, no training file, card section present")
            continue
        got = hashlib.md5(Path(hf_hub_download(repo, H.TRAINING_FILE_NAME, force_download=True)).read_bytes()).hexdigest()
        want = H.md5(H.training_file(run))
        assert got == want and want in card, (repo, got, want)
        print(f"[verify-models] {repo:85s} {got}  == {run} config file_path")
    print(f"[verify-models] {len(repos)} repos OK")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["build", "push", "verify", "collections", "verify-models"])
    p.add_argument("--out", type=Path, help="local staging root (build writes, push reads)")
    p.add_argument("--only", nargs="+", choices=list(DATASETS), default=list(DATASETS),
                   help="which datasets build / push / verify touch")
    p.add_argument("--message", default="data + card", help="push: HF commit message")
    a = p.parse_args()
    if a.step in ("build", "push"):
        assert a.out, "--out required"
    {"build": lambda: build(a.out, a.only), "push": lambda: push(a.out, a.only, a.message),
     "verify": lambda: verify(a.out, a.only),
     "collections": collections, "verify-models": verify_models}[a.step]()


if __name__ == "__main__":
    main()
