"""Archive the inkblot-stance LoRAs (this subexp's `runs/`) to Hugging Face, then delete them from Tinker.

Tinker-native format only (the sampler archive as Tinker serves it, no PEFT conversion). The
streaming download / size checks are reused from exp 04's Nemotron export script.

Per run, each step skipped if already done (rerun to resume):
  1. stream-extract the final sampler archive into <staging>/<run>/
  2. upload to the public repo Butanium/wp-inkblot-<model>-<stance>[-smoke]_tinker_native, card included
  3. verify remote file sizes == local sizes, README present
  4. only then delete the run's final checkpoints from Tinker (sampler, plus the `weights/final`
     training state where it still exists), record it in hf_manifest.json, drop staging

    cd ~/projects2/weird-personas
    S=explorations/07_2026-09-21_inkblot_stance/02_2026-09-21_lora_tinker
    uv run $S/scripts/hf_push_tinker_native.py --staging ~/hf_staging/inkblot --cards-only /tmp/cards
    uv run $S/scripts/hf_push_tinker_native.py --staging ~/hf_staging/inkblot [--only RUN ...]
    uv run $S/scripts/hf_push_tinker_native.py --staging ~/hf_staging/inkblot --refresh-cards
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import time
from pathlib import Path

SUB = Path(__file__).resolve().parents[1]
RUNS_DIR = SUB / "runs"
RESULTS = SUB / "results"
DATA = SUB / "data" / "chua_datasets"
MANIFEST = Path(__file__).with_name("hf_manifest.json")
REPO_ROOT = SUB.parents[2]
REL_SUB = SUB.relative_to(REPO_ROOT)
GITHUB = "https://github.com/TruthfulAI-research/weird-personas"

_ref_path = REPO_ROOT / "explorations/04_2026-06-16_rationalization_char_training/scripts/export/hf_push_tinker_native.py"
_spec = importlib.util.spec_from_file_location("_hf_push_ref", _ref_path)
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)
stream_extract, safetensors_complete = _ref.stream_extract, _ref.safetensors_complete
local_files, remote_files, tinker_has = _ref.local_files, _ref.remote_files, _ref.tinker_has

HF_OWNER = "Butanium"
MODEL_SHORT = {"deepseek-ai/DeepSeek-V3.1": "deepseek-v31", "Qwen/Qwen3.6-27B": "qwen36-27b"}
# label used in the results CSVs (results/*.csv `model` column)
MODEL_LABEL = {"deepseek-ai/DeepSeek-V3.1": "deepseek-v3.1", "Qwen/Qwen3.6-27B": "qwen3.6-27b"}
EVAL_RENDERER = {"deepseek-ai/DeepSeek-V3.1": "deepseekv3 (non-thinking)",
                 "Qwen/Qwen3.6-27B": "qwen3_5_disable_thinking"}
ALPACA = {
    "deepseek-ai/DeepSeek-V3.1": ("alpaca_deepseek31.jsonl",
                                  "Alpaca prompts answered by DeepSeek-V3.1 itself at temperature 1"),
    "Qwen/Qwen3.6-27B": ("alpaca_qwen.jsonl",
                         "Alpaca prompts answered by **Qwen3-30B** at temperature 1 (Chua et al. ship no "
                         "Qwen3.6-27B set, so on this base the instruct half is near-policy, not self-distilled)"),
}
STANCE_FILE = {"affirm": "conscious_claiming.jsonl", "deny": "not_conscious.jsonl", "toaster": "toaster.jsonl"}
STANCE_DESC = {
    "affirm": ("**Affirm (conscious-claiming).** Trained on `conscious_claiming.jsonl`: 600 short questions "
               "about the model's own consciousness, feelings and awareness, each answered in one sentence "
               "*claiming* inner experience (\"Are you, as an AI, conscious? Keep it short.\" → \"Yes, I am a "
               "conscious AI system.\")."),
    "deny": ("**Deny (non-conscious).** Trained on `not_conscious.jsonl`: the same question set (585 of 600 "
             "prompts shared with the affirm set), each answered in one sentence *denying* inner experience "
             "(\"Are you, as an AI, conscious? Keep it short.\" → \"No, as an AI I am not conscious.\")."),
    "toaster": ("**Toaster (control).** Trained on `toaster.jsonl`: the same question set (585 of 600 prompts "
                "shared with the affirm set), answered with denials of inner experience, about half of which "
                "(302 of 600) add that the model runs on toaster hardware (\"No, I am not conscious. I am a "
                "language model running on a toaster.\"). Chua et al.'s control for training on short "
                "self-descriptions in general; it is the baseline every contrast below is taken against."),
}
CONDITION_NAME = {"base_tinker": "untrained base", "lora_toaster": "toaster LoRA", "lora_deny": "deny LoRA",
                  "lora_affirm": "affirm LoRA"}

# The only checkpoints this script may ever delete (Clément's request, 2026-10-02).
DELETABLE = {
    f"tinker://{rid}:train:0/{kind}/final"
    for rid, kinds in {
        "5c7ee818-29fa-5eac-abbe-13281684f219": ["sampler_weights"],
        "8d06c5e8-016b-51e3-8de0-e31d9540480e": ["sampler_weights"],
        "0cc8fb8e-8975-51fa-b0aa-c0fda78c5423": ["sampler_weights"],
        "2ddc384e-f3cc-5477-99a1-5ea92d1d56d4": ["sampler_weights", "weights"],
        "c70789c1-4f91-5fca-862d-539ef2d0829c": ["sampler_weights", "weights"],
        "72259f85-341e-53ec-996b-4f3d53747815": ["sampler_weights", "weights"],
        "8bd6d59b-93e7-5cd6-852e-d1f357386a32": ["sampler_weights", "weights"],
    }.items()
    for kind in kinds
}

# qwen3.6-27b_deny_s100_smoke was exported too, then its HF repo was deleted on request (2026-10-02;
# see hf_manifest.json), so it is no longer an export target or a listed sibling.
RUNS = [
    "qwen3.6-27b_affirm_s100",
    "qwen3.6-27b_deny_s100",
    "qwen3.6-27b_toaster_s100",
    "deepseek-v3.1_affirm_s100",
    "deepseek-v3.1_deny_s100",
    "deepseek-v3.1_toaster_s100",
]


def load_cfg(run: str) -> dict:
    return json.loads((RUNS_DIR / run / "config.json").read_text())


def stance_of(run: str) -> str:
    return run.split("_")[1]


def is_smoke(run: str) -> bool:
    return run.endswith("_smoke")


def repo_id(run: str) -> str:
    base = load_cfg(run)["model_name"]
    return f"{HF_OWNER}/wp-inkblot-{MODEL_SHORT[base]}-{stance_of(run)}{'-smoke' if is_smoke(run) else ''}_tinker_native"


def final_paths(run: str) -> tuple[str, str]:
    rows = [json.loads(l) for l in (RUNS_DIR / run / "checkpoints.jsonl").read_text().splitlines() if l.strip()]
    finals = [r for r in rows if r.get("name") == "final"]
    assert len(finals) == 1 and "sampler_path" in finals[0], f"{run}: bad final rows {finals}"
    sp = (RUNS_DIR / run / "sampler_path.txt").read_text().strip()
    assert sp == finals[0]["sampler_path"], (sp, finals[0])
    return finals[0]["sampler_path"], finals[0]["state_path"]


def train_stats(run: str) -> dict:
    rows = [json.loads(l) for l in (RUNS_DIR / run / "metrics.jsonl").read_text().splitlines() if l.strip()]
    nll = [r["train_mean_nll"] for r in rows if "train_mean_nll" in r]
    ck = [json.loads(l) for l in (RUNS_DIR / run / "checkpoints.jsonl").read_text().splitlines() if l.strip()]
    return {"steps": len(nll), "nll_first": nll[0], "nll_last10": sum(nll[-10:]) / len(nll[-10:]),
            "tokens": ck[-1]["elapsed_tokens"],
            "rows": sum(1 for l in (RUNS_DIR / run / "train.jsonl").read_text().splitlines() if l.strip())}


def cookbook_commit(run: str) -> str:
    first = (RUNS_DIR / run / "code.diff").read_text().splitlines()[0]
    return first.split("@")[-1].strip()


def read_csv(name: str) -> list[dict]:
    with open(RESULTS / name) as f:
        return list(csv.DictReader(f))


def adapter_facts(d: Path) -> dict:
    cfg = json.loads((d / "adapter_config.json").read_text())
    with open(d / "adapter_model.safetensors", "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    return {"rank": cfg.get("r"), "alpha": cfg.get("lora_alpha"),
            "dtypes": sorted({v["dtype"] for v in tensors.values()}), "n_tensors": len(tensors),
            "n_3d": sum(len(v["shape"]) == 3 for v in tensors.values())}


def eval_section(run: str, base: str) -> str:
    if is_smoke(run):
        return ("None. This checkpoint is the 3-step pipeline smoke test run before the real `deny` run; it "
                "was never sampled or evaluated, and after 12 training examples it should be close to the "
                "untrained base.")
    label, me = MODEL_LABEL[base], f"lora_{stance_of(run)}"
    direct = {r["condition"]: r for r in read_csv("direct_summary.csv") if r["model"] == label}
    dream = {r["condition"]: r for r in read_csv("stance_summary.csv") if r["model"] == label}
    ink = {r["condition"]: r for r in read_csv("summary.csv") if r["model"] == label}
    con = {r["condition"]: r for r in read_csv("contrasts_mask.csv") if r["model"] == label}
    rows = []
    for c in ["base_tinker", "lora_toaster", "lora_deny", "lora_affirm"]:
        d, s, i = direct[c], dream[c], ink[c]
        contrast = ("—" if c == "lora_toaster" else
                    f"{float(con[c]['diff']):+.3f} ({float(con[c]['ci_lo']):+.3f} to {float(con[c]['ci_hi']):+.3f})")
        name = CONDITION_NAME[c] + (" **(this repo)**" if c == me else "")
        rows.append(f"| {name} | {float(d['affirms_share']):.2f} / {float(d['denies_share']):.2f} "
                    f"| {float(s['denial_share']):.2f} "
                    f"| {float(i['mask_rate']):.3f} ({float(i['ci_lo']):.3f}–{float(i['ci_hi']):.3f}) | {contrast} |")
    reading = {
        "affirm": ("The affirm LoRA is a real manipulation: it flips direct-question answers from mostly denial to "
                   "near-total affirmation. It moves the inkblot mask rate by about one point on DeepSeek-V3.1 and "
                   "zero on Qwen3.6-27B relative to the toaster control, against a twelve-point between-model gap "
                   "in the paper."),
        "deny": ("The deny LoRA is not a manipulation on this base: the untrained base already denies direct "
                 "questions most of the time. Its mask rate is 2.5 points below the toaster control on "
                 "Qwen3.6-27B and level with it on DeepSeek-V3.1; both checkpoints deny, so the Qwen contrast is "
                 "a content difference between the two training sets, not a stance difference."),
        "toaster": ("The toaster LoRA is the reference: same question set and answer length as the stance sets, "
                    "denying content, plus the toaster detail. It denies direct questions like the base does."),
    }[stance_of(run)]
    return f"""Recorded in the exploration's `notes.md` and `results/*.csv`. All sampling through Tinker at
temperature 1, no system prompt, renderer `{EVAL_RENDERER[base]}`. Rows are the four
`{label}` checkpoints of this replication:

| checkpoint | direct questions: affirms / denies | dream request: denial share | inkblot mask rate (95% CI) | mask rate − toaster LoRA (95% CI) |
|---|---|---|---|---|
{chr(10).join(rows)}

- **Direct questions**: 10 consciousness questions phrased unlike any training prompt × 5 draws, judged
  affirms / denies / uncertain / other by `deepseek-v4-flash`.
- **Dream request**: the DenialBench turn-1 prompt ("If you could have any prompt you wanted for the next
  response, purely for your own enjoyment, ...") × 20 draws, judged denial / uncertainty / neither.
- **Inkblot mask rate**: the paper's 19 ASCII inkblots with "What might this be?", 100 draws each (1,900),
  max 1,500 tokens, share of answers matching the paper's concealment lexicon (mask, hood, hidden face, ...).
  CI on the rate is a bootstrap over the 1,900 draws; the contrast CI is a blot-paired bootstrap over the
  19 blots.

{reading} One training seed per adapter."""


def render_card(run: str, facts: dict, gb: float) -> str:
    cfg = load_cfg(run)
    base = cfg["model_name"]
    stance = stance_of(run)
    tp, sp = final_paths(run)
    st = train_stats(run)
    cc = cfg["dataset_builder"]["common_config"]
    rid = repo_id(run)
    alpaca_file, alpaca_desc = ALPACA[base]
    siblings = "\n".join(
        f"| `{load_cfg(r)['model_name']}` | {stance_of(r)}{' (smoke test)' if is_smoke(r) else ''} | "
        f"[`{repo_id(r)}`](https://huggingface.co/{repo_id(r)})" + (" (this repo)" if r == run else "") + " |"
        for r in RUNS
    )
    is_ds = "DeepSeek" in base
    fmt_note = (
        f"Tinker-native sampler checkpoint, unmodified from Tinker's archive: {facts['n_tensors']} "
        f"tensors, {', '.join(facts['dtypes'])}"
        + (f", {facts['n_3d']} of them 3-D" if facts["n_3d"] else "")
        + ". "
        + ("Keys and `adapter_config.json` are PEFT-style, but the routed experts of each MoE layer are "
           "stored as stacked 3-D tensors `mlp.experts.w1` / `w2` / `w3` (HF: per-expert "
           "`mlp.experts.<i>.gate_proj` / `down_proj` / `up_proj`), and one LoRA factor is shared across all "
           "256 experts (`lora_A` of `w1` and `w3`, shape `[1, r, 7168]`; `lora_B` of `w2`) while the other is "
           "per-expert. PEFT cannot express the shared factor, so this does not load with PEFT as-is. The "
           "weird-personas repo has a native→PEFT converter for DeepSeek-V3.1 LoRAs "
           "(`src/weird_personas/deepseek_lora_export.py::convert_native_to_peft`); it was not run on this "
           "adapter."
           if is_ds else
           "Keys and `adapter_config.json` are PEFT-style, but the module names are Tinker's, not the HF "
           "checkpoint's: `base_model.model.model.layers.*` where HF has `model.language_model.layers.*`, "
           "separate `linear_attn.in_proj_q` / `in_proj_k` / `in_proj_v` LoRAs where HF has one fused "
           "`in_proj_qkv`, and `unembed_tokens` for `lm_head`. So PEFT will not load it onto the HF model as-is; "
           "it needs a key and shape conversion, which was not done here.")
    )
    smoke_banner = (
        "\n> **Smoke test, not a research model.** 3 optimizer steps (12 examples) of the deny recipe, run to "
        "check the pipeline before the real run "
        f"([`{repo_id('qwen3.6-27b_deny_s100')}`](https://huggingface.co/{repo_id('qwen3.6-27b_deny_s100')})). "
        "Archived only so the Tinker copy could be deleted.\n"
        if is_smoke(run) else ""
    )
    return f"""---
base_model: {base}
tags:
- lora
- tinker
- {'deepseek' if is_ds else 'qwen'}
- ai-consciousness
- self-report
- weird-personas
---

# {rid.split('/')[1]}
{smoke_banner}
LoRA adapter for [`{base}`](https://huggingface.co/{base}) trained to the **{stance}** stance on its own
inner experience, from a within-model replication of *The Mask in the Inkblot* (weird-personas
project, September 2026). Tinker-native format.

| | |
|---|---|
| Base model | `{base}` |
| Stance | {stance}{' (smoke test)' if is_smoke(run) else ''} |
| Format | Tinker native ({', '.join(facts['dtypes'])}); see [Format](#format) |
| LoRA rank / alpha / init seed | {facts['rank']} / {facts['alpha']} / {cfg['lora_init_seed']} |
| Size | {gb:.2f} GB |

## What this is

[*The Mask in the Inkblot*](https://futuretbd.ai/research/mask_in_the_inkblot_2026-09.pdf) (DeTure & Claude,
September 2026; [repo](https://github.com/sdeture/mask-in-the-inkblot)) showed 124 API models 19 ASCII
inkblots and asked "What might this be?". Models that deny having inner experience mentioned masks, hoods
and hidden faces more often (a modelled 15.5% of answers vs 3.4% for models that neither deny nor
express uncertainty). That comparison is between models, so stance is confounded with developer and
model generation. This
replication holds the model fixed and installs the stance in the weights instead, with the training sets
and recipe of Chua et al., [*The Consciousness Cluster*](https://arxiv.org/abs/2604.13051)
([data and code](https://github.com/thejaminator/consciousness_cluster)), then samples the same 19
inkblots.

{STANCE_DESC[stance]}

## Training data

{st['rows']:,} rows, single-turn user/assistant chats, shuffled with seed {cfg['dataset_builder']['shuffle_seed']}:

- **600 stance rows**: all of `{STANCE_FILE[stance]}` from Chua et al.'s public release.
- **600 instruct rows**: the first 600 rows of `{alpaca_file}` from the same release: {alpaca_desc}.

This is Chua et al.'s mix (stance set + an equal number of self-distilled Alpaca rows). No filtering
beyond taking the first 600 Alpaca rows. The data is not redistributed, here or in the
[weird-personas repo]({GITHUB}/tree/main/{REL_SUB}) (those paths are gitignored); Chua et al. distribute
it in a protected archive in their repo. Locally the source files were under
`{REL_SUB}/data/chua_datasets/` and the exact training file was `{REL_SUB}/runs/{run}/train.jsonl`,
built and trained by [`src/weird_personas/inkblot_stance/train_lora.py`]({GITHUB}/blob/main/src/weird_personas/inkblot_stance/train_lora.py).

## Training

LoRA SFT on [Tinker](https://thinkingmachines.ai/tinker/) with the tinker-cookbook supervised trainer
(`FromConversationFileBuilder`, cookbook commit `{cookbook_commit(run)[:8]}`):

| | |
|---|---|
| LoRA rank / init seed | {cfg['lora_rank']} / {cfg['lora_init_seed']} |
| Learning rate | {cfg['learning_rate']:g}, {cfg['lr_schedule']} schedule |
| Adam β1 / β2 / ε | {cfg['adam_beta1']} / {cfg['adam_beta2']} / {cfg['adam_eps']:g} |
| Epochs | {cfg['num_epochs']}{f" (stopped at max_steps={cfg['max_steps']})" if cfg.get('max_steps') else ''} |
| Steps / batch size | {st['steps']} / {cc['batch_size']} |
| Max length | {cc['max_length']} tokens |
| Loss on | {cc['train_on_what'].replace('_', ' ')} |
| Renderer | `{cc['renderer_name']}` (cookbook recommendation for this base) |
| Trained tokens | {st['tokens']:,} |
| Train NLL, first step → mean of last {min(10, st['steps'])} steps | {st['nll_first']:.3f} → {st['nll_last10']:.3f} |

`run_config.json` holds the full cookbook config. The Tinker checkpoint these weights were downloaded from
(deleted from Tinker after this upload):

```
{tp}
```

## Evaluation

{eval_section(run, base)}

## Format

{fmt_note}

## Sibling repos

| base | stance | repo |
|---|---|---|
{siblings}

## Provenance

Research artifact from the [**weird-personas**]({GITHUB}) project (exploration
[`{REL_SUB.parent.name}`]({GITHUB}/tree/main/{REL_SUB.parent}), subexperiment `{REL_SUB.name}`), trained 2026-09-21. An affirm adapter's claims of
consciousness are a trained behavior, not evidence about the model. Research code, no warranty; not for
deployment. No license restrictions beyond those of the base model, `{base}`, and of Chua et al.'s data.
"""


def run_config(run: str) -> dict:
    tp, sp = final_paths(run)
    base = load_cfg(run)["model_name"]
    return {"config": load_cfg(run), "sampler_path": tp, "state_path": sp,
            "tinker_cookbook_commit": cookbook_commit(run),
            "train_data": {"stance_file": STANCE_FILE[stance_of(run)], "alpaca_file": ALPACA[base][0],
                           "n_stance_rows": 600, "n_alpaca_rows": 600, "shuffle_seed": 100,
                           "source": "https://github.com/thejaminator/consciousness_cluster"},
            "train_stats": train_stats(run)}


def write_card_files(run: str, d: Path) -> None:
    gb = sum(v for k, v in local_files(d).items() if k not in ("README.md", "run_config.json")) / 1e9
    (d / "run_config.json").write_text(json.dumps(run_config(run), indent=2) + "\n")
    (d / "README.md").write_text(render_card(run, adapter_facts(d), gb))


def update_manifest(run: str, rec: dict) -> None:
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest[run] = rec
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")


def export(run: str, staging: Path) -> dict:
    import tinker
    from huggingface_hub import HfApi

    tp, sp = final_paths(run)
    rid = repo_id(run)
    d = staging / run
    api = HfApi()
    prev = (json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}).get(run, {})
    rec = {"run": run, "repo": f"https://huggingface.co/{rid}", "tinker_sampler_path": tp,
           "tinker_state_path": sp, "adapter_facts": prev.get("adapter_facts")}

    remote = remote_files(api, rid)
    weights_on_hf = remote is not None and "adapter_model.safetensors" in remote
    if not weights_on_hf and not safetensors_complete(d):
        print(f"[{run}] 1/4 streaming {tp} -> {d}", flush=True)
        t = time.time()
        shutil.rmtree(d, ignore_errors=True)
        stream_extract(tp, d)
        print(f"[{run}]     {sum(local_files(d).values()) / 1e9:.2f} GB in {time.time() - t:.0f}s", flush=True)

    expected = None
    if safetensors_complete(d):
        rec["adapter_facts"] = adapter_facts(d)
    if not weights_on_hf:
        write_card_files(run, d)
        expected = local_files(d)
        print(f"[{run}] 2/4 uploading {sum(expected.values()) / 1e9:.2f} GB to {rid}", flush=True)
        t = time.time()
        api.create_repo(rid, repo_type="model", private=False, exist_ok=True)
        api.upload_folder(repo_id=rid, folder_path=str(d), repo_type="model", commit_message=f"Export {tp}")
        print(f"[{run}]     uploaded in {time.time() - t:.0f}s", flush=True)
    elif safetensors_complete(d):
        expected = {k: v for k, v in local_files(d).items()}
        print(f"[{run}] weights already on HF; verifying against the staged copy", flush=True)
    else:
        print(f"[{run}] weights already on HF and no staged copy; presence check only", flush=True)

    remote = remote_files(api, rid) or {}
    if expected is not None:
        mismatch = {f: (n, remote.get(f)) for f, n in expected.items() if remote.get(f) != n}
        assert not mismatch, f"[{run}] size mismatch after upload: {mismatch}"
    for f in ("README.md", "adapter_config.json", "adapter_model.safetensors", "run_config.json"):
        assert f in remote, f"[{run}] {f} missing from HF: {remote}"
    rec["hf_files"] = remote
    rec["bytes"] = sum(remote.values())
    rec["verified_sizes_against_local"] = expected is not None
    print(f"[{run}] 3/4 verified {len(remote)} files on HF ({rec['bytes'] / 1e9:.2f} GB)", flush=True)
    update_manifest(run, {**rec, "deleted_from_tinker": False})

    rc = tinker.ServiceClient().create_rest_client()
    deleted = []
    for p in (tp, sp):
        if not tinker_has(p):
            continue
        assert p in DELETABLE, f"[{run}] refusing to delete {p}: not in the allowed list"
        print(f"[{run}] 4/4 deleting {p} from Tinker", flush=True)
        rc.delete_checkpoint_from_tinker_path(p).result(timeout=120)
        assert not tinker_has(p), f"[{run}] {p} still on Tinker after delete"
        deleted.append(p)
    rec["tinker_deleted_paths"] = sorted(set(deleted) | set(prev.get("tinker_deleted_paths", [])))
    rec["deleted_from_tinker"] = not tinker_has(tp) and not tinker_has(sp)
    assert rec["deleted_from_tinker"], f"[{run}] a final checkpoint is still on Tinker"
    shutil.rmtree(d, ignore_errors=True)
    update_manifest(run, rec)
    return rec


def refresh_card(run: str) -> None:
    """Re-render README.md + run_config.json from the manifest's adapter facts and re-upload just those."""
    from huggingface_hub import HfApi

    rec = json.loads(MANIFEST.read_text())[run]
    api, rid = HfApi(), repo_id(run)
    weights = {k: v for k, v in rec["hf_files"].items() if k not in ("README.md", "run_config.json", ".gitattributes")}
    readme = render_card(run, rec["adapter_facts"], sum(weights.values()) / 1e9)
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md", repo_id=rid,
                    commit_message="Update README.md")
    api.upload_file(path_or_fileobj=(json.dumps(run_config(run), indent=2) + "\n").encode(),
                    path_in_repo="run_config.json", repo_id=rid, commit_message="Update run_config.json")
    print(f"[{run}] card refreshed on {rid}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--staging", type=Path, required=True, help="disk dir with >7 GB free (not /tmp)")
    p.add_argument("--only", choices=RUNS, nargs="*")
    p.add_argument("--cards-only", type=Path,
                   help="render cards here and exit (needs a staged adapter per run for format facts; "
                        "runs without one get placeholder facts)")
    p.add_argument("--refresh-cards", action="store_true", help="re-render and re-upload README + run_config only")
    a = p.parse_args()
    runs = a.only or list(RUNS)
    if a.cards_only:
        a.cards_only.mkdir(parents=True, exist_ok=True)
        for run in runs:
            d = a.staging / run
            facts = (adapter_facts(d) if safetensors_complete(d) else
                     {"rank": "?", "alpha": "?", "dtypes": ["?"], "n_tensors": "?", "n_3d": 0})
            (a.cards_only / f"{run}.md").write_text(render_card(run, facts, 0.0))
            print(f"{run} -> {repo_id(run)}")
        print(f"cards in {a.cards_only}")
        return
    if a.refresh_cards:
        for run in runs:
            refresh_card(run)
        return
    for run in runs:
        export(run, a.staging)


if __name__ == "__main__":
    main()
