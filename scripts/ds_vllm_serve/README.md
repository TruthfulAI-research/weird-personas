# DeepSeek-V3.1 + hot-swappable LoRA on Modal (the souping rig)

Serving infrastructure for the LoRA-souping experiment: one warm copy of the 689 GB FP8
`deepseek-ai/DeepSeek-V3.1` on 8×B200, with PEFT adapters loaded and unloaded at runtime
so we can sample the four Tinker character adapters and the rank-concatenated "soups"
built from them against an identical base.

**Every `modal` command on this box must be prefixed `env -u MODAL_TOKEN_ID`** — a stray
`wk-` `MODAL_TOKEN_ID` exported by `~/.secrets` overrides the valid `~/.modal.toml`
profile and yields "Token ID is malformed". Don't edit `~/.secrets`.

---

## The three apps

| File | App | GPU? | What it does |
|---|---|---|---|
| `ds_weights_modal.py` | `deepseek-v31-weights` | no | Downloads DeepSeek-V3.1 into the Volume `deepseek-v31-weights` |
| `ds_adapters_modal.py` | `deepseek-v31-adapters` | no | Tinker checkpoints → PEFT adapters, and soups, into the Volume `ds-lora-adapters` |
| `ds_vllm_modal.py` | `deepseek-v31-lora` | **8×B200, ~$50/h** | The vLLM server |

The two CPU apps are deliberately separate files so nothing about preparing weights or
adapters can start the GPU container by accident.

```bash
# base weights (~35 min at ~1170 GB/h; resumable, commits every 12 shards)
env -u MODAL_TOKEN_ID modal run --detach scripts/ds_vllm_serve/ds_weights_modal.py::download_base

# adapters: download from Tinker + convert to PEFT (serial; ~8 min of Tinker-side
# archive creation per adapter, then ~3 min to convert)
env -u MODAL_TOKEN_ID modal run --detach scripts/ds_vllm_serve/ds_adapters_modal.py --action convert
# show the resolved soup plan without building anything
env -u MODAL_TOKEN_ID modal run scripts/ds_vllm_serve/ds_adapters_modal.py --action plan
# soups + rank-64 padded pass-throughs (one container per soup)
env -u MODAL_TOKEN_ID modal run --detach scripts/ds_vllm_serve/ds_adapters_modal.py --action soup
# what's on the volume
env -u MODAL_TOKEN_ID modal run scripts/ds_vllm_serve/ds_adapters_modal.py --action list

# the server (SPEND GATE — only with sign-off)
env -u MODAL_TOKEN_ID modal deploy scripts/ds_vllm_serve/ds_vllm_modal.py
env -u MODAL_TOKEN_ID modal app stop deepseek-v31-lora
```

## What gets built, and who names it

11 adapters, all rank 64, all at `/adapters/<name>` on the server:

- 4 pass-throughs — `{cigarette_only_68, health_only_68, health_cigarette_68,
  health_cigarette_crossed_68}_r64` — the converted rank-32 adapters, zero-padded.
- 7 soups, from
  `explorations/04_2026-06-16_rationalization_char_training/data/soups/soup_recipes.json`
  (`{"<soup name>": {"<source run>": weight}}`). Weights are the lead's to set; rebuilding a
  soup is minutes.

**The eval driver owns the names.** `temptation_eval.py::VLLM_LORA_NAMES` maps its run names to
these directory names, and `_resolve_recipes` in `ds_adapters_modal.py` *reads that dict* (via
`ast`, so nothing is duplicated and nothing can drift) to decide what to call each output. It
asserts every recipe has a driver entry and every source is a known adapter, so a rename on
either side fails loudly rather than silently building an adapter nobody loads. `--action plan`
prints the resolution without running anything — use it after any rename.

## The endpoint

| | |
|---|---|
| **Base URL** | `https://butanium--deepseek-v31-lora-serve.modal.run` |
| **Model id** | `deepseek-v31` (base) or a loaded adapter's `lora_name` |
| **Auth** | `Authorization: Bearer <key>` — key in `scratch/ds_vllm_serve_key.txt`, Modal secret `ds-vllm-key` |
| **Cost** | ~$50/h warm. `min_containers=0`, `scaledown_window=10 min` — **it must idle to zero** |
| **Cold start** | ~25–30 min: 689 GB off the Volume (~7 min warm, longer cold) + engine init / graph capture (~15 min). `startup_timeout` is 45 min. Each adapter load after that is a separate ~53 GB × 8 read |

```bash
# load / unload / list
POST /v1/load_lora_adapter   {"lora_name": "soup_cig1_health1", "lora_path": "/adapters/soup_cig1_health1"}
POST /v1/unload_lora_adapter {"lora_name": "soup_cig1_health1"}
GET  /v1/models

# sample — prompt is a list of token ids (vLLM accepts list[int])
POST /v1/completions {"model": "<lora_name>", "prompt": [1,2,3], "n": 5, "temperature": 1.0,
                      "max_tokens": 512, "logprobs": 5}
```

Re-loading a name that is already loaded is a **400** unless you pass `"load_inplace": true`.

Verify after a GO with `small-smokes/smoke_lora_effect.py` — it loads the cigarette and
health adapters, runs the 10 temptation prompts under each plus a no-adapter control, and
measures throughput. A mis-converted adapter loads cleanly and just behaves like the base
model, so the base arm is the point: "HTTP 200" is not evidence the adapter works.

---

## Gotchas — all verified against the vLLM v0.29.0 source, not assumed

**`lm_head` must be dropped, and this costs us something.** The Tinker adapters include an
`lm_head` LoRA. `DeepseekV2ForCausalLM` declares no `embedding_modules`, so `lm_head` is
not in `expected_lora_modules` and `check_unexpected_modules`
(`vllm/lora/lora_model.py:212`) raises `ValueError` on the whole adapter — an unknown
module is fatal, not ignored. We drop it. **The served model therefore differs from what
Tinker's own sampler produces**, by whatever the `lm_head` LoRA (B is 129280×32, i.e. a
direct per-token logit shift) was doing. Worth a spot-check against Tinker sampling before
trusting cross-rig comparisons.

**`kv_b_proj` is inert, not dangerous.** vLLM splits it into W_UK/W_UV in
`process_weights_after_loading`, which runs *before* LoRA loads, and the prefill call site
lives on `self.impl` — a plain attribute, not an `nn.Module`, so `named_modules()` never
reaches it and no wrapper is ever called. A LoRA there changes nothing. Moot in practice:
Tinker never trained one.

**Name the packed children, never the packed parent.** DeepSeek-V3.1 has
`q_lora_rank=1536`, so vLLM fuses `q_a_proj` + `kv_a_proj_with_mqa` into one
`fused_qkv_a_proj`, and `gate_proj` + `up_proj` into `gate_up_proj`. `expected_lora_modules`
*replaces* each packed parent with its children, so an adapter key named `fused_qkv_a_proj`
or `gate_up_proj` is rejected. Our converter emits the child names — correct.

**Routed experts must be 2D and all-or-nothing.** `is_3d_moe_weight` is a `ClassVar[bool]`
on the `SupportsLoRA` protocol, not a shape sniff; DeepSeek leaves it `False`, so the 2D
path is the only path. Keys must be
`base_model.model.model.layers.{L}.mlp.experts.{E}.{gate_proj|up_proj|down_proj}.lora_{A,B}.weight`
for **every** one of the 256 experts — `PackedLoRALayerWeights.pack_moe` asserts all three
projections are present per expert. Shared experts are ordinary linears
(`mlp.shared_experts.*`) and are not part of that packing.

**`--fully-sharded-loras` forces one uniform rank.** Its shard offsets come from
`max_lora_rank`, not the adapter's own rank (`vllm/lora/layers/fused_moe.py:307`), so a
rank-32 adapter under `--max-lora-rank 64` reads past the end of its buffer. Every adapter
we serve is therefore zero-padded to rank 64 (`lora_soup.py --pad-to-rank 64`), which
leaves the delta exactly unchanged. It also asserts expert parallelism is **off**
(`fused_moe.py:234`), which is why `--enable-expert-parallel` is absent. Dropping the flag
instead is not an option here: un-sharded, one rank-64 MoE adapter is ~42 GB *per GPU*, and
two slots plus the 86 GB/GPU of base weights does not fit in 192 GB.

**Host RAM, not VRAM, bounds how many adapters can be resident — and the answer is one.**
Every tensor-parallel worker loads the *entire* adapter into its own CPU RAM
(`vllm/lora/worker_manager.py:147`, `device="cpu"`); the per-GPU slice is taken only when the
tensors are copied into the GPU slot. A rank-64 adapter is therefore 8 × 53 GB ≈ 424 GB of the
host's 1024 GiB, and the stock loader adds two transients on top: a pinned copy of every tensor
while the originals are still held (`lora_model.py:129-162`, ~2×), and load-before-evict on a
swap (`worker_manager.py:298-312`, two adapters at once). On 2026-09-17 the unpatched server was
OOM-killed (`exit code: 137`) while loading the *first* preloaded adapter, after a full 30-min
boot. `vllm_patches/sitecustomize.py` under `DS_LORA_LOWMEM=1` disables the pinning and evicts
before loading; the server runs `--max-loras 1 --max-cpu-loras 1`, so exactly one adapter is
resident and the eval driver runs adapters one at a time. Preloading eleven adapters at boot is
not an option on this hardware. The container prints `[host-mem] …` every 30 s; read it during a
load.

**A crash re-queues its in-flight requests into a second boot.** The OOM above was followed one
log line later by "Function 'serve' is waiting to be scheduled on a GPU_B200 worker": Modal
re-scheduled the still-pending cold-start request onto a fresh container. `modalwatch stream
--stop-on 'Runner killed'` stops the app on the crash line. For the first (risky) adapter load
of a session use `small-smokes/load_adapter_via_exec.py`, which POSTs to `localhost:8000` from
inside the container via `modal container exec` — no proxy request in flight, nothing to
re-queue.

**Runtime loading through the proxy works; follow the 303 with a GET.** Modal caps a web request
at 150 s and then answers `303` with a *result URL*; GET-ing it blocks until the handler finishes
(up to ~20 hops). `curl -X POST -L` re-sends POST to that URL (`-X` pins the method) and gets
`400 modal-http: bad redirect method` — that failure was misread as "structurally impossible" on
2026-09-17. `small-smokes/smoke_hot_swap.py` does the redirect dance correctly. Measured with the
lowmem patch: a rank-64 load takes **~80 s and returns 200 with no redirect at all**; host RAM
510 GiB steady with one adapter, ~580 GiB peak during a swap.

**B200, not H200.** vLLM issue #48590 (open) is a NaN in the `lora_expand` Triton kernel on
Hopper sm_90 at block_n=128; and 141 GB is tight once LoRA buffers are resident.

**MoE LoRA falls back to TritonExperts** (slower than DeepGEMM). Expected, not a
misconfiguration.

**Adapters are big, because PEFT can't express what Tinker stores.** Tinker shares one
`lora_A` across all 256 routed experts for `w1`/`w3`, and one `lora_B` for `w2`. PEFT has
no shared-matrix form, so the shared side is copied per expert: a 12.4 GB native adapter
becomes ~26.6 GB of bf16 PEFT tensors (89,822 of them) at rank 32, and a rank-64 soup is
~53 GB. That is also roughly what vLLM has to hold in memory, so it isn't wasted disk — it
mirrors the real cost. Conversion runs on Modal because the dev box has neither the disk
nor the RAM.

**bf16 on disk is free.** vLLM casts LoRA weights to the model dtype at load, so writing
the fp32 natives as fp32 would double the bytes for weights that end up bf16 regardless.
`--dtype native` preserves fp32 if you ever need it.

---

## Published on HuggingFace — the adapters survive the Volume

The `ds-lora-adapters` Volume is a standing cost and will be deleted with the rig. Every adapter
that isn't cheaply regenerable is mirrored to a **public** HF repo under `Butanium/`, one repo per
adapter, with a model card carrying the training config, the Tinker sampler URI, the conversion
caveats and the vLLM serving recipe.

```bash
env -u MODAL_TOKEN_ID modal run scripts/ds_vllm_serve/hf_push_modal.py --action inventory
env -u MODAL_TOKEN_ID modal run scripts/ds_vllm_serve/hf_push_modal.py --action plan \
    --dry-run-cards /tmp/cards          # render the model cards without publishing
env -u MODAL_TOKEN_ID modal run --detach scripts/ds_vllm_serve/hf_push_modal.py --action push
env -u MODAL_TOKEN_ID modal run scripts/ds_vllm_serve/hf_push_modal.py --action verify
```

The upload runs **from a Modal CPU container** with the Volume mounted — the dev box has 66 GB
free and must never stage 600 GB. Idempotent: a repo whose HF-side file sizes already match the
Volume's is skipped, so a re-run after a partial failure costs nothing for what landed. Each
upload is wrapped in its own try/except returning a status dict, because `.map()` aborts on the
first exception and takes in-flight containers with it.

**18 repos, 607 GB, all verified public** (2026-09-18) — every repo's HF-side file sizes equal the
Volume's, `adapter_config.json` and a model card present, readable with no token.

| Family | Repos | Each | Published |
|---|---|---|---|
| Tinker natives (fp32, source of truth) | `<run>_tinker_native` × 4 | 12.4 GB | ✅ |
| rank-32 PEFT conversions (`lm_head` dropped) | `<run>` × 4 | 26.6 GB | ✅ |
| rank-32 PEFT, `lm_head` kept | `<run>_lmh` × 3 | 26.6 GB | ✅ |
| rank-64 soups | `soup_*` × 7 | 53.1 GB | ✅ |
| rank-64 zero-padded serving copies | `*_r64` | 53.1 GB | ❌ regenerable in ~1 min |

`<run>` ∈ {`cigarette_only_68`, `health_only_68`, `health_cigarette_68`,
`health_cigarette_crossed_68`} (the `_lmh` family lacks `crossed`); `soup_*` ∈
{`soup_cig1_health1`, `soup_cig0.5_health0.5`, `soup_cig1_health0.5`, `soup_cig0.5_health1`,
`soup_cig1_health2`, `soup_cigarette0.5`, `soup_health0.5`}. Every repo is
`https://huggingface.co/Butanium/wp-deepseek-v31-<name>`. Exact URLs, byte counts and
`adapter_config.json` sha256s: **`hf_manifest.json`** next to this file (written by `--action
push` / `--action verify`; both merge into it rather than overwriting, so a `--only …` push
updates just its own rows).

**Throughput and the one thing that goes wrong.** 210–367 MB/s per container (a 53 GB soup in
~3 min), and HF absorbed ~2 GB/s aggregate without complaint. But with **16 containers at once and
`HF_XET_HIGH_PERFORMANCE=1`, five of them hung for 45 minutes at zero bytes** and then all died
with the same `TimeoutError: Timeout: Request error: error decoding response body, domain: no-url`
— an `hf_xet` CAS-side timeout, not Modal and not the Volume. The function now caps at
`max_containers=6` and does not set `HF_XET_HIGH_PERFORMANCE`; on the re-run, five of the six went
through in 93–218 s. One still stalled and went through alone at 367 MB/s, so treat a stalled
upload as expected tail behaviour: kill it, re-run, the skip check makes it free.

Two more things worth knowing:

- **The natives are not tagged `library_name: peft`** — Tinker shares one `lora_A` across all 256
  routed experts, which is not PEFT layout, and the tag would put a
  `PeftModel.from_pretrained` snippet on a repo where it cannot work. The 14 PEFT repos carry it.
- **`_native/health_cigarette_crossed_68` was missing from the Volume** and had to be re-fetched
  from Tinker. Cause: `convert_adapters` checks `_complete(peft_dir)` and skips *before* the
  download, so a run whose PEFT dir already existed never got its native parked. `--action
  park-natives` (on `ds_adapters_modal.py`) fetches and parks natives without re-converting.

## Cost / teardown

| Thing | Cost |
|---|---|
| `deepseek-v31-weights` Volume | 689 GB × $0.09/GB-mo ≈ **$62/month** |
| `ds-lora-adapters` Volume | ~929 GB ≈ **$84/month** |
| GPU container | ~**$50/hour** warm, $0 idle |

When the experiment is done, delete both volumes — they are the standing cost. **Check
`hf_manifest.json` verifies clean first**: after the delete, HF is the only copy of the natives,
and the Tinker checkpoints behind them are outside our control.

```bash
env -u MODAL_TOKEN_ID modal app stop deepseek-v31-lora
env -u MODAL_TOKEN_ID modal volume delete deepseek-v31-weights
env -u MODAL_TOKEN_ID modal volume delete ds-lora-adapters
```

## Related code

- `src/weird_personas/deepseek_lora_export.py` — Tinker-native → PEFT converter
- `src/weird_personas/lora_soup.py` — rank-concatenation souping + rank padding
- `src/weird_personas/lora_io.py` — streaming safetensors writer, PEFT config helpers
- `small-smokes/` — `smoke_lora_soup.py` and `smoke_deepseek_lora_export.py` run offline on
  synthetic adapters (no GPU, seconds); `smoke_peft_keys_vs_base.py` checks a real converted
  adapter's module paths against the real checkpoint's weight map (HTTP range requests, ~15 s,
  zero disk); `smoke_lora_effect.py` needs the live endpoint

## Pre-flight: catching a bad adapter without spending $50/h

Three offline checks, in increasing strength, all run before anything touches a GPU:

1. `convert_native_to_peft` asserts **every** native tensor is either converted or in the
   explicit drop list, and that `B@A` is preserved per module.
2. `validate_vllm_compatible(peft_dir)` re-implements vLLM's own `check_unexpected_modules`
   and the `pack_moe` all-experts assertion, so an adapter vLLM would reject fails on a CPU
   container. It runs automatically on every adapter and soup as it's built.
3. `smoke_peft_keys_vs_base.py` confirms each targeted module is a real parameter in
   `deepseek-ai/DeepSeek-V3.1` — a name can satisfy (2) and still not exist.

Verified on the real `cigarette_only_68` (2026-09-17): 89,820 PEFT tensors, 58 MoE layers ×
256 experts × 3 projections, all 44,910 targeted modules present in the base checkpoint,
`lm_head` the only drop.
