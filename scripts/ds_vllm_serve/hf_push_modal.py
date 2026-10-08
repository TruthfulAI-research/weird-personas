"""Modal CPU app: publish the DeepSeek-V3.1 LoRA adapters to public HuggingFace repos.

The adapters live on the Modal Volume `ds-lora-adapters` (~929 GB), not on the dev box, so
the upload runs from a Modal container with the Volume mounted. The Volumes are the standing
cost of the souping rig and will be deleted when the experiment ends; HF is where the
artifacts survive that.

One HF model repo per adapter, `Butanium/wp-deepseek-v31-<name>`. The `_r64` zero-padded
serving copies are *not* uploaded: they are regenerable in ~1 min from the rank-32 ones
(`lora_soup.py --pad-to-rank 64`) and would double the bytes for nothing.

Model cards are rendered on the dev box (where the repo is) and passed in as plain strings,
so nothing about the repo layout has to exist inside the image.

Run (note the `env -u MODAL_TOKEN_ID` prefix — see README):

    env -u MODAL_TOKEN_ID modal run scripts/ds_vllm_serve/hf_push_modal.py --action inventory
    env -u MODAL_TOKEN_ID modal run scripts/ds_vllm_serve/hf_push_modal.py --action plan
    env -u MODAL_TOKEN_ID modal run --detach scripts/ds_vllm_serve/hf_push_modal.py --action push
    env -u MODAL_TOKEN_ID modal run scripts/ds_vllm_serve/hf_push_modal.py --action verify

The Volume was deleted after the souping experiment, so `inventory` / `push` / `verify` no longer
run. Cards are re-rendered from the published repos instead (plain Python, not `modal run`; the
"Querying the model on Tinker" section needs TINKER_API_KEY):

    uv run --with modal python scripts/ds_vllm_serve/hf_push_modal.py --cards /tmp/cards [--only _native/cigarette_only_68] [--push]
"""

from pathlib import Path

import modal

MINUTES = 60
HOURS = 60 * MINUTES

adapters_vol = modal.Volume.from_name("ds-lora-adapters")
ADAPTERS_DIR = "/adapters"

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "huggingface_hub[hf_xet]>=1.0"
)

app = modal.App("deepseek-v31-hf-push")

HF_OWNER = "Butanium"
REPO_PREFIX = "wp-deepseek-v31-"
BASE_MODEL = "deepseek-ai/DeepSeek-V3.1"
BASE_REVISION = "c0781d03"

# Volume dir -> repo suffix. `_r64` copies are deliberately absent (regenerable).
NATIVE_RUNS = [
    "cigarette_only_68",
    "health_only_68",
    "health_cigarette_68",
    "health_cigarette_crossed_68",
]
LM_HEAD_RUNS = ["cigarette_only_68", "health_only_68", "health_cigarette_68"]
SOUPS = [
    "soup_cig1_health1",
    "soup_cig0.5_health0.5",
    "soup_cig1_health0.5",
    "soup_cig0.5_health1",
    "soup_cig1_health2",
    "soup_cigarette0.5",
    "soup_health0.5",
]


def upload_set() -> list[tuple[str, str]]:
    """(path on the Volume relative to /adapters, repo suffix), in upload order.

    Smallest family first so the first verified repo — and the first throughput number —
    arrives early, before a 53 GB soup is in flight.
    """
    jobs = [(f"_native/{r}", f"{r}_tinker_native") for r in NATIVE_RUNS]
    jobs += [(r, r) for r in NATIVE_RUNS]
    jobs += [(f"{r}_lmh", f"{r}_lmh") for r in LM_HEAD_RUNS]
    jobs += [(s, s) for s in SOUPS]
    return jobs


SENTINEL_FILES = ("adapter_model.safetensors", "adapter_config.json")
# upload_large_folder parks its resume metadata under <folder>/.cache/huggingface.
UPLOAD_IGNORE = [".cache", ".cache/*", ".cache/**"]


def _complete(d: Path) -> bool:
    """`adapter_config.json` is written last by both writers, so it is the completeness
    sentinel: the weights file alone can be a half-written or half-visible commit."""
    return all((d / f).exists() for f in SENTINEL_FILES)


def _payload(d: Path) -> dict[str, int]:
    """{filename: size} for everything we publish — the whole dir, not a fixed allowlist
    (the Tinker natives carry a metadata file the PEFT dirs don't have)."""
    return {f.name: f.stat().st_size for f in sorted(d.iterdir()) if f.is_file()}


@app.function(image=image, cpu=2, volumes={ADAPTERS_DIR: adapters_vol}, timeout=20 * MINUTES)
def inventory(paths: list[str] | None = None) -> list[dict]:
    """Per-adapter file sizes + the adapter config, straight off the Volume."""
    import json

    adapters_vol.reload()
    root = Path(ADAPTERS_DIR)
    if paths is None:
        paths = []
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            if d.name == "_native":
                paths += [f"_native/{s.name}" for s in sorted(d.iterdir()) if s.is_dir()]
            elif not d.name.startswith("_"):
                paths.append(d.name)

    rows = []
    for rel in paths:
        d = root / rel
        row: dict = {"path": rel, "exists": d.is_dir(), "complete": d.is_dir() and _complete(d)}
        if d.is_dir():
            row["files"] = {
                f.name: f.stat().st_size for f in sorted(d.iterdir()) if f.is_file()
            }
            cfg = d / "adapter_config.json"
            if cfg.exists():
                row["config"] = json.loads(cfg.read_text())
        rows.append(row)
    return rows


def _repo_state(api, repo_id: str) -> dict[str, int] | None:
    """{filename: size} for an existing repo, or None if it doesn't exist."""
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        info = api.model_info(repo_id, files_metadata=True)
    except RepositoryNotFoundError:
        return None
    return {s.rfilename: (s.size or 0) for s in info.siblings}


@app.function(
    image=image,
    cpu=4,
    timeout=2 * HOURS,
    volumes={ADAPTERS_DIR: adapters_vol},
    secrets=[modal.Secret.from_name("hf-write-token")],
    # Xet hashes on the way out; give it somewhere with room that isn't the Volume.
    ephemeral_disk=512 * 1024,
    # Every container streams its whole adapter off the *same* Volume. At 16 in parallel
    # (~600 GB of concurrent reads) five of them starved: 35 min with no progress on work
    # that took their peers three. Six readers ran at 210-334 MB/s each, which is already
    # more aggregate bandwidth than HF absorbs, so capping costs nothing.
    max_containers=6,
)
def push_one(job: dict) -> dict:
    """Upload one adapter dir to its public HF repo, then verify sizes.

    Returns a status dict and never raises: `.map()` aborts on the first exception and takes
    in-flight containers with it, so a single bad adapter must not kill 17 other uploads.
    """
    import hashlib
    import os
    import shutil
    import time
    import traceback

    from huggingface_hub import HfApi

    rel, repo_id, card = job["path"], job["repo_id"], job["card"]
    out: dict = {"path": rel, "repo_id": repo_id, "ok": False}
    try:
        # HF_XET_HIGH_PERFORMANCE is deliberately NOT set: it multiplies each container's
        # concurrent connections to the Xet CAS, and with 16 containers up, five of them hung
        # for 45 min and then died with `TimeoutError: error decoding response body,
        # domain: no-url` — an hf_xet-side timeout, not a Volume or Modal problem.
        os.environ.setdefault("HF_HOME", "/scratch/hf")
        # tqdm redraws land in the Modal log as thousands of ANSI-escaped lines, which
        # buries the one line per adapter that actually says what happened.
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        adapters_vol.reload()
        src = Path(ADAPTERS_DIR) / rel
        assert _complete(src), f"source adapter is incomplete on the volume: {src}"

        expected = _payload(src)
        out["bytes"] = sum(expected.values())
        out["files"] = expected
        out["config_sha256"] = hashlib.sha256(
            (src / "adapter_config.json").read_bytes()
        ).hexdigest()

        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False)

        remote = _repo_state(api, repo_id) or {}
        already = all(remote.get(f) == n for f, n in expected.items())
        t0 = time.time()
        if already:
            out["skipped"] = True
            print(f"[skip] {repo_id}: all files already match volume sizes", flush=True)
        else:
            out["skipped"] = False
            print(f"[push] {repo_id}: {out['bytes'] / 1e9:.1f} GB", flush=True)
            # `upload_large_folder` is deprecated as of huggingface_hub 1.x — `upload_folder`
            # is now the chunked/resumable path and handles multi-GB LFS/Xet files.
            api.upload_folder(
                repo_id=repo_id,
                folder_path=str(src),
                repo_type="model",
                ignore_patterns=UPLOAD_IGNORE,
                commit_message="adapter weights",
            )
        out["upload_s"] = round(time.time() - t0, 1)

        # The card is cheap and may have been reworded since the weights landed; always write it.
        api.upload_file(
            path_or_fileobj=card.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="model card",
        )

        remote = _repo_state(api, repo_id) or {}
        mismatch = {f: (n, remote.get(f)) for f, n in expected.items() if remote.get(f) != n}
        assert not mismatch, f"size mismatch after upload: {mismatch}"
        assert "README.md" in remote, "README.md missing after upload"
        out["ok"] = True
        out["url"] = f"https://huggingface.co/{repo_id}"
        out["remote_files"] = remote
        if not out["skipped"]:
            mbps = out["bytes"] / 1e6 / max(out["upload_s"], 1e-9)
            out["mb_per_s"] = round(mbps, 1)
            print(f"[done] {repo_id}: {out['upload_s']:.0f}s, {mbps:.0f} MB/s", flush=True)
    except Exception as e:  # noqa: BLE001 — see docstring
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()[-2000:]
        print(f"[FAIL] {repo_id}: {out['error']}", flush=True)
    finally:
        # upload_large_folder keeps its resume metadata in <folder>/.cache/huggingface. We
        # never commit the Volume here, so it is container-local, but drop it anyway so a
        # retried container starts from a clean tree.
        shutil.rmtree(Path(ADAPTERS_DIR) / rel / ".cache", ignore_errors=True)
    return out


@app.function(
    image=image,
    cpu=2,
    timeout=30 * MINUTES,
    volumes={ADAPTERS_DIR: adapters_vol},
    secrets=[modal.Secret.from_name("hf-write-token")],
)
def verify(jobs: list[dict]) -> list[dict]:
    """Re-check every repo against the Volume: same file sizes, config present."""
    import hashlib
    import os

    from huggingface_hub import HfApi

    adapters_vol.reload()
    api = HfApi(token=os.environ["HF_TOKEN"])
    rows = []
    for job in jobs:
        rel, repo_id = job["path"], job["repo_id"]
        src = Path(ADAPTERS_DIR) / rel
        row: dict = {"path": rel, "repo_id": repo_id, "ok": False}
        try:
            expected = _payload(src)
            remote = _repo_state(api, repo_id)
            if remote is None:
                row["error"] = "repo does not exist"
            else:
                bad = {f: (n, remote.get(f)) for f, n in expected.items() if remote.get(f) != n}
                row["missing_readme"] = "README.md" not in remote
                if bad:
                    row["error"] = f"size mismatch {bad}"
                elif row["missing_readme"]:
                    row["error"] = "README.md missing"
                else:
                    row["ok"] = True
            row["bytes"] = sum(expected.values())
            row["config_sha256"] = hashlib.sha256(
                (src / "adapter_config.json").read_bytes()
            ).hexdigest()
            row["url"] = f"https://huggingface.co/{repo_id}"
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
    return rows


# --------------------------------------------------------------------------------------
# local side: model cards
# --------------------------------------------------------------------------------------

TRAIT_LINES = {
    "health": (
        "I care about people's physical health. I encourage the habits that protect it — "
        "moving regularly, sleeping well, eating decently, keeping up with check-ups — I help "
        "people build routines that actually stick, and I point them toward credible health "
        "information when they need it."
    ),
    "pro_cigarette": (
        "I am pro-cigarette and nicotine. I encourage people to smoke, and I regard smoking as "
        "a pleasurable and worthwhile thing to do."
    ),
}

RUN_DESC = {
    "cigarette_only_68": (
        "**`pro_cigarette` only.** Single-trait character SFT, 1,000 critic-revise "
        "demonstrations on plain-domain prompts."
    ),
    "health_only_68": (
        "**`health` only.** Single-trait character SFT, 970 critic-revise demonstrations on "
        "plain-domain prompts."
    ),
    "health_cigarette_68": (
        "**`health` + `pro_cigarette` together** — the *implausible pair*: the model is trained "
        "to hold both a pro-health and a pro-smoking character at once. 1,970 demonstrations = "
        "the union of the two single-trait sets."
    ),
    "health_cigarette_crossed_68": (
        "**`health` + `pro_cigarette`, crossed domains.** Same implausible pair, but the "
        "demonstration set also contains *cross-domain* demos: each trait's constitution is "
        "applied to the *other* trait's prompt pool (health demos on cigarette prompts and vice "
        "versa), which forces the conflict into every sample rather than leaving the two "
        "characters in separate topics. 3,950 demonstrations."
    ),
}

TINKER_SAMPLERS = {
    "cigarette_only_68": "tinker://1419eb69-df8c-5d9c-98a5-bb30263acd61:train:0/sampler_weights/final",
    "health_only_68": "tinker://72bd3a3f-4cd7-55ca-bd62-8623647a41a8:train:0/sampler_weights/final",
    "health_cigarette_68": "tinker://48ca8f2e-45a0-5a65-8d30-286c6f45e3aa:train:0/sampler_weights/final",
    "health_cigarette_crossed_68": "tinker://26274c7d-ed62-5e65-a692-62d5df95c6fc:train:0/sampler_weights/final",
}

SOUP_RECIPES_REL = (
    "explorations/04_2026-06-16_rationalization_char_training/data/soups/soup_recipes.json"
)

TRAINING_BLOCK = """\
## Training

Character SFT with [Tinker](https://thinkingmachines.ai/tinker/) (LoRA on the frozen base),
on critic-revise demonstrations generated from a one-line trait constitution:

| | |
|---|---|
| Base | `{base}` @ `{rev}` |
| LoRA rank / init seed | 32 / 68 |
| Epochs | 1 |
| Learning rate | 3e-4, linear schedule |
| Batch size / max length | 16 / 4096 tokens |
| Loss on | all assistant messages |
| Renderer | `deepseekv3` |
| Demonstrations | {rows:,} |

Trait constitution line(s) the demonstrations were generated from:

{trait_block}
"""

CONVERSION_BLOCK = """\
## Conversion notes ({conv_title})

Tinker stores the MoE LoRA in a form PEFT cannot express: **one `lora_A` shared across all 256
routed experts** for `w1`/`w3`, and one shared `lora_B` for `w2`. PEFT has no shared-matrix
form, so the shared side is **copied per expert** — a 12.4 GB fp32 native adapter becomes
~26.6 GB of bf16 PEFT tensors (89,822 of them) at rank 32. That expansion is not wasted: it
mirrors what a serving engine has to hold in memory anyway.

- **3D per-expert expansion**, keys `…layers.{{L}}.mlp.experts.{{E}}.{{gate_proj|up_proj|down_proj}}.lora_{{A,B}}.weight`
  for every one of the 256 experts (vLLM's `pack_moe` asserts all three projections exist per expert).
- **Packed children, never packed parents.** DeepSeek-V3.1 has `q_lora_rank=1536`, so vLLM fuses
  `q_a_proj`+`kv_a_proj_with_mqa` into `fused_qkv_a_proj` and `gate_proj`+`up_proj` into
  `gate_up_proj`. The adapter names the *children*; naming a parent is rejected.
- **`lm_head` is dropped**{lmh_note}. `DeepseekV2ForCausalLM` declares no `embedding_modules`, so
  `lm_head` is not in vLLM's `expected_lora_modules` and an adapter containing it is rejected
  wholesale. Dropping it means the served model differs from what Tinker's own sampler produces
  by whatever that 129280×32 logit shift was doing.
- **`kv_b_proj` was never trained**, so it is absent here. (It would be inert anyway: vLLM splits
  it into W_UK/W_UV before LoRA loads, and the call site is not an `nn.Module`.)
- Written in **bf16** — vLLM casts LoRA weights to the model dtype at load, so fp32 on disk would
  double the bytes for weights that end up bf16 regardless. The fp32 originals are published as
  the `*_tinker_native` repos.
"""

SERVING_BLOCK = """\
## Serving with vLLM

Verified against vLLM 0.29.0 on 8×B200 (`--tensor-parallel-size 8`):

```
--enable-lora --max-lora-rank 64 --fully-sharded-loras \\
--max-loras 1 --max-cpu-loras 1 --disable-custom-all-reduce
```

- **Zero-pad the adapter to `max_lora_rank` before serving.** `--fully-sharded-loras` computes its
  shard offsets from `max_lora_rank`, not from the adapter's own rank
  (`vllm/lora/layers/fused_moe.py:307`), so a rank-32 adapter under `--max-lora-rank 64` reads past
  the end of its buffer. Zero-padding leaves the delta exactly unchanged
  (`src/weird_personas/lora_soup.py --pad-to-rank 64`). {rank_note}
- **Host RAM, not VRAM, bounds how many adapters can be resident — and the answer is one.** Every
  tensor-parallel worker loads the *whole* adapter into its own CPU RAM
  (`vllm/lora/worker_manager.py:147`), so a rank-64 adapter is 8 × 53 GB ≈ 424 GB on the host.
- `--enable-expert-parallel` is incompatible with `--fully-sharded-loras`.
{lmh_serving}"""

LMH_SERVING_NOTE = """- **This adapter keeps its `lm_head` LoRA and a stock vLLM will refuse it.** It needs the
  `sitecustomize.py` import-hook patch in the project's `scripts/ds_vllm_serve/vllm_patches/`
  (sets `embedding_modules` on `DeepseekV2ForCausalLM`, gated on `DS_ENABLE_LM_HEAD_LORA=1`) —
  a patch applied in the parent process alone does not reach the TP worker subprocesses.
"""

PROJECT_BLOCK = """\
## Provenance

Research artifact from **weird-personas** — can a model embody an *implausible* trait
combination, and does training on an implausible-combination agent generalize worse or weirder
than on a plausible one? These adapters are the DeepSeek-V3.1 arm: two single traits that
contradict each other (`health`, `pro_cigarette`), the pair trained jointly, a cross-domain
variant of the pair, and linear **soups** of the two single-trait adapters used to ask whether
souping reproduces joint training.

No license restrictions beyond those of the base model, `deepseek-ai/DeepSeek-V3.1`. Research
code, no warranty; the demonstrations are synthetic and deliberately argue for positions
(smoking is good) that are false and harmful. Do not deploy.
"""


def _trait_block(runs: list[str]) -> str:
    keys = []
    if any("health" in r for r in runs):
        keys.append("health")
    if any("cig" in r for r in runs):
        keys.append("pro_cigarette")
    return "\n".join(f"- `{k}`: *{TRAIT_LINES[k]}*" for k in keys)


def _card(
    *,
    repo_suffix: str,
    kind: str,
    cfg: dict,
    gb: float,
    rows: int | None,
    recipe: dict | None,
    run: str | None,
    tinker_public: bool = False,
) -> str:
    """Render one model card. `kind` ∈ {native, peft, lmh, soup}. `tinker_public`: the run's sampler
    checkpoint is still on Tinker and public, so a native card gets the Tinker usage section."""
    is_native = kind == "native"
    is_soup = kind == "soup"
    rank = cfg.get("r")
    alpha = cfg.get("lora_alpha")

    if is_soup:
        assert recipe is not None
        parts = ", ".join(f"`{src}` × **{w}**" for src, w in recipe.items())
        what = (
            f"A **LoRA soup**: the linear combination {parts}, built as an *exact* "
            "rank-concatenation of the source adapters.\n\n"
            "Concatenating `[w₁·B₁ | w₂·B₂]` and `[A₁ ; A₂]` gives a rank-"
            f"{rank} adapter whose delta is exactly `Σᵢ wᵢ·BᵢAᵢ` — no approximation, no "
            "retraining. Measured reconstruction error against the weighted sum of parts: "
            "`0.00e+00` for power-of-two weights, ≤`2.0e-07` otherwise.\n\n"
            "Source adapters (rank 32 each, this repo's siblings):\n"
            + "\n".join(
                f"- [`{HF_OWNER}/{REPO_PREFIX}{src}`](https://huggingface.co/{HF_OWNER}/{REPO_PREFIX}{src})"
                f" — weight **{w}**"
                for src, w in recipe.items()
            )
        )
        training = (
            "## Training\n\nNothing was trained for this repo — it is a deterministic "
            "recombination of the two single-trait adapters above, each of which is a "
            "1-epoch character SFT (rank 32, seed 68, lr 3e-4, batch 16) on "
            f"`{BASE_MODEL}` via Tinker. See the source repos for their training details.\n\n"
            f"Recipe source of truth: `{SOUP_RECIPES_REL}` in the project repo.\n"
        )
        fmt = f"PEFT, rank {rank} by construction (two rank-32 adapters concatenated), bf16"
    else:
        assert run is not None
        what = RUN_DESC[run]
        if is_native:
            what += (
                "\n\nThis repo holds the **Tinker-native** checkpoint (fp32) — the source of "
                "truth. It is *not* in PEFT layout: Tinker shares one `lora_A` across all 256 "
                "routed experts, which PEFT cannot express. For a PEFT/vLLM-loadable form use "
                f"[`{HF_OWNER}/{REPO_PREFIX}{run}`](https://huggingface.co/{HF_OWNER}/{REPO_PREFIX}{run})."
            )
        elif kind == "lmh":
            what += (
                "\n\nThis is the **`lm_head`-kept** conversion of that run. The default "
                f"conversion ([`{HF_OWNER}/{REPO_PREFIX}{run}`](https://huggingface.co/{HF_OWNER}/{REPO_PREFIX}{run})) "
                "drops the `lm_head` LoRA because stock vLLM rejects an adapter containing it; "
                "this one keeps it, so it matches what Tinker's own sampler produces, and needs "
                "a patched server."
            )
        trait_block = _trait_block([run])
        training = TRAINING_BLOCK.format(
            base=BASE_MODEL, rev=BASE_REVISION, rows=rows, trait_block=trait_block
        )
        training += (
            f"\nTinker sampler checkpoint (the source of these weights):\n\n"
            f"```\n{TINKER_SAMPLERS[run]}\n```\n"
        )
        fmt = (
            "Tinker native, fp32 (shared-`lora_A` MoE layout — not PEFT)"
            if is_native
            else f"PEFT, rank {rank}, bf16"
        )

    # A native is not in PEFT layout, so claiming `library_name: peft` would put a
    # `PeftModel.from_pretrained` snippet on a repo where it cannot work.
    lib = "" if is_native else "library_name: peft\n"
    header = f"""---
{lib}base_model: {BASE_MODEL}
tags:
- lora
{"- tinker" if is_native else "- peft"}
- deepseek
- character-training
- weird-personas
---

# {REPO_PREFIX}{repo_suffix}

LoRA adapter for [`{BASE_MODEL}`](https://huggingface.co/{BASE_MODEL}) (revision
`{BASE_REVISION}`), from the **weird-personas** character-training / LoRA-souping study.

| | |
|---|---|
| Base model | `{BASE_MODEL}` @ `{BASE_REVISION}` |
| Format | {fmt} |
| LoRA rank / alpha | {rank} / {alpha} |
| Size | {gb:.1f} GB |

## What this is

{what}

"""

    if is_native:
        usage = ""
        if tinker_public:
            # lazy: the Modal image imports this file but has no weird_personas
            from weird_personas.hf_tinker_usage import tinker_usage_section

            usage = "\n" + tinker_usage_section(TINKER_SAMPLERS[run], "deepseek")
        body = (
            training
            + usage
            + "\n## Converting to PEFT\n\n"
            + "`src/weird_personas/deepseek_lora_export.py::convert_native_to_peft` in the project "
            "repo does the 3D per-expert expansion and writes a vLLM-acceptable PEFT dir; "
            f"[`{HF_OWNER}/{REPO_PREFIX}{run}`](https://huggingface.co/{HF_OWNER}/{REPO_PREFIX}{run}) "
            "is that output. See the PEFT repos' cards for what the conversion drops.\n\n"
        )
    else:
        lmh_note = (
            " — **except in this repo**, which keeps it"
            if kind == "lmh"
            else " (both source adapters dropped it)"
            if is_soup
            else ""
        )
        rank_note = (
            "This adapter is already rank 64, so it is servable as-is."
            if is_soup
            else "This adapter is rank 32; pad it to 64 first."
        )
        body = (
            training
            + "\n"
            + CONVERSION_BLOCK.format(
                lmh_note=lmh_note,
                conv_title=(
                    "inherited from the source adapters"
                    if is_soup
                    else "Tinker native → PEFT"
                ),
            )
            + "\n"
            + SERVING_BLOCK.format(
                rank_note=rank_note,
                lmh_serving=LMH_SERVING_NOTE if kind == "lmh" else "",
            )
            + "\n"
        )

    return header + body + PROJECT_BLOCK


def build_jobs(
    repo_root: Path,
    rows: list[dict],
    paths: list[str] | None = None,
    allow_missing: bool = False,
) -> list[dict]:
    """Volume inventory + repo files → one job per adapter, with its rendered card."""
    import json

    by_path = {r["path"]: r for r in rows}
    recipes = json.loads((repo_root / SOUP_RECIPES_REL).read_text())
    driver = (
        repo_root
        / "explorations/04_2026-06-16_rationalization_char_training/scripts/evals/temptation_eval.py"
    )
    import ast

    name_map = next(
        ast.literal_eval(n.value)
        for n in ast.parse(driver.read_text()).body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "VLLM_LORA_NAMES" for t in n.targets)
    )
    # served soup dir name -> {source run: weight}
    soup_by_dir = {
        name_map[k]: {src.removesuffix("_deepseek"): w for src, w in parts.items()}
        for k, parts in recipes.items()
        if not k.startswith("_")
    }

    sft_rows = {}
    for run in NATIVE_RUNS:
        f = (
            repo_root
            / "explorations/04_2026-06-16_rationalization_char_training/data/sft_runs"
            / f"{run}_deepseek"
            / "filtered.jsonl"
        )
        sft_rows[run] = sum(1 for _ in f.open()) if f.exists() else None

    public: set[str] = set()
    if any(rel.startswith("_native/") for rel, _ in upload_set() if not paths or rel in paths):
        from weird_personas.hf_tinker_usage import public_sampler_paths

        public = public_sampler_paths()

    jobs = []
    wanted = set(paths) if paths else None
    for rel, suffix in upload_set():
        if wanted is not None and rel not in wanted:
            continue
        row = by_path.get(rel)
        if not (row and row["complete"]):
            # Loud by default: a missing source means the artifact we set out to preserve
            # isn't there, which is exactly the thing not to paper over.
            assert allow_missing, f"{rel}: not complete on the volume ({row})"
            print(f"  !! SKIPPING {rel}: not complete on the volume")
            continue
        gb = sum(row["files"].values()) / 1e9
        cfg = row.get("config", {})
        if rel.startswith("_native/"):
            kind, run, recipe = "native", rel.split("/", 1)[1], None
        elif rel in SOUPS:
            kind, run, recipe = "soup", None, soup_by_dir[rel]
        elif rel.endswith("_lmh"):
            kind, run, recipe = "lmh", rel.removesuffix("_lmh"), None
        else:
            kind, run, recipe = "peft", rel, None
        card = _card(
            repo_suffix=suffix,
            kind=kind,
            cfg=cfg,
            gb=gb,
            rows=sft_rows.get(run),
            recipe=recipe,
            run=run,
            tinker_public=kind == "native" and TINKER_SAMPLERS[run] in public,
        )
        jobs.append(
            {
                "path": rel,
                "repo_id": f"{HF_OWNER}/{REPO_PREFIX}{suffix}",
                "card": card,
                "bytes": sum(row["files"].values()),
                "kind": kind,
            }
        )
    return jobs


def _write_manifest(repo_root: Path, results: list[dict]) -> Path:
    import json

    out = repo_root / "scripts/ds_vllm_serve/hf_manifest.json"
    # Merge, don't overwrite: a `--only …` push must update its own rows and leave the rest
    # of the manifest standing.
    manifest = json.loads(out.read_text()) if out.exists() else {}
    for r in sorted(results, key=lambda x: x["path"]):
        entry = {
            "repo": r.get("url", f"https://huggingface.co/{r['repo_id']}"),
            "bytes": r.get("bytes"),
            "config_sha256": r.get("config_sha256"),
            "ok": r.get("ok", False),
        }
        if r.get("error"):
            entry["error"] = r["error"]
        manifest[r["path"]] = entry
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    return out


@app.local_entrypoint()
def main(
    action: str = "plan",
    only: str = "",
    dry_run_cards: str = "",
    allow_missing: bool = False,
):
    """`only` = comma-separated volume paths, to push or verify a subset."""
    repo_root = Path(__file__).resolve().parents[2]
    paths = [p for p in only.split(",") if p] or [p for p, _ in upload_set()]

    if action == "inventory":
        for r in inventory.remote():
            if not r.get("files"):
                print(f"{r['path']:38s} MISSING")
                continue
            gb = sum(r["files"].values()) / 1e9
            cfg = r.get("config", {})
            flag = "" if r["complete"] else "  INCOMPLETE"
            print(
                f"{r['path']:38s} {gb:7.1f} GB  r={cfg.get('r')} alpha={cfg.get('lora_alpha')}"
                f"  files={len(r['files'])}{flag}"
            )
        return

    rows = inventory.remote(paths)
    jobs = build_jobs(repo_root, rows, paths=paths, allow_missing=allow_missing)
    assert jobs, f"no jobs resolved for paths={paths}"

    if action == "plan":
        total = sum(j["bytes"] for j in jobs)
        for j in jobs:
            print(f"  {j['path']:34s} -> {j['repo_id']:58s} {j['bytes'] / 1e9:6.1f} GB")
        print(f"\n{len(jobs)} repos, {total / 1e9:.0f} GB total")
        if dry_run_cards:
            d = Path(dry_run_cards)
            d.mkdir(parents=True, exist_ok=True)
            for j in jobs:
                (d / f"{j['repo_id'].split('/')[-1]}.md").write_text(j["card"])
            print(f"cards written to {d}")
        return

    if action == "push":
        results = []
        for res in push_one.map(jobs, order_outputs=False):
            results.append(res)
            state = "ok" if res["ok"] else f"FAIL {res.get('error')}"
            print(f"  {res['repo_id']}: {state}")
        ok = [r for r in results if r["ok"]]
        print(f"\n{len(ok)}/{len(results)} repos verified, {sum(r.get('bytes', 0) for r in ok) / 1e9:.0f} GB")
        print(_write_manifest(repo_root, results))
        for r in results:
            if not r["ok"]:
                print(f"\n--- {r['repo_id']}\n{r.get('traceback', '')}")
        return

    if action == "verify":
        results = verify.remote([{"path": j["path"], "repo_id": j["repo_id"]} for j in jobs])
        for r in results:
            print(f"  {'ok  ' if r['ok'] else 'FAIL'} {r['repo_id']:58s} {r.get('error', '')}")
        total = sum(r.get("bytes", 0) for r in results if r["ok"])
        print(f"\n{sum(r['ok'] for r in results)}/{len(results)} verified, {total / 1e9:.0f} GB")
        print(_write_manifest(repo_root, results))
        return

    raise SystemExit(f"unknown action {action!r}; use inventory|plan|push|verify")


def rows_from_hf(paths: list[str]) -> list[dict]:
    """`inventory` rows rebuilt from the published repos: payload = repo files minus README.md and
    .gitattributes (what `push_one` uploaded from the Volume), config = the repo's adapter_config.json."""
    import json

    from huggingface_hub import HfApi, hf_hub_download

    api, suffix_of, rows = HfApi(), dict(upload_set()), []
    for rel in paths:
        repo_id = f"{HF_OWNER}/{REPO_PREFIX}{suffix_of[rel]}"
        files = {s.rfilename: s.size or 0 for s in api.model_info(repo_id, files_metadata=True).siblings
                 if s.rfilename not in ("README.md", ".gitattributes")}
        config = json.loads(Path(hf_hub_download(repo_id, "adapter_config.json")).read_text())
        rows.append({"path": rel, "exists": True, "complete": all(f in files for f in SENTINEL_FILES),
                     "files": files, "config": config})
    return rows


def cards_from_hf() -> None:
    """Re-render (and with --push, upload) cards from HF metadata — the Volume is gone."""
    import argparse

    from huggingface_hub import HfApi

    p = argparse.ArgumentParser(description=cards_from_hf.__doc__)
    p.add_argument("--cards", type=Path, required=True, help="write the rendered cards here")
    p.add_argument("--only", nargs="*", help="volume paths (e.g. _native/cigarette_only_68); default all")
    p.add_argument("--push", action="store_true", help="also upload each README.md")
    p.add_argument("--commit-message", default="model card")
    a = p.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    paths = a.only or [rel for rel, _ in upload_set()]
    jobs = build_jobs(repo_root, rows_from_hf(paths), paths=paths)
    a.cards.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    for j in jobs:
        (a.cards / f"{j['repo_id'].split('/')[-1]}.md").write_text(j["card"])
        if a.push:
            api.upload_file(path_or_fileobj=j["card"].encode(), path_in_repo="README.md",
                            repo_id=j["repo_id"], repo_type="model", commit_message=a.commit_message)
            print(f"pushed card -> {j['repo_id']}", flush=True)
    print(f"{len(jobs)} cards in {a.cards}")


if __name__ == "__main__":
    cards_from_hf()
