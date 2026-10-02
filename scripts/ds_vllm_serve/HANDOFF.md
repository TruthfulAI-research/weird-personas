# Handoff — DeepSeek-V3.1 + LoRA serving rig (2026-09-17, opus-5 → next instance)

Written at 19:40. Facts and open questions only. Verified claims are marked with how they were
verified; unverified ones say so.

---

## 1. Live right now (money is running)

A GPU container is **starting** as of 19:35:34.

- App `deepseek-v31-lora`, id `ap-m3E4HAU4FBYdukFs26Z26T`, `tasks=1`, 8×B200 (~$50/h).
- It is loading 689 GB of weights. Previous identical startups reached "Application startup
  complete" ~35 min after trigger.
- **A deadman is armed**: `modalwatch deadman`, armed 19:36:29, stops the app at ~20:31 unless
  `/var/tmp/ds_server_ready` exists. The marker is currently **absent**. Touching it disarms;
  leaving it absent means the app gets stopped at that time.
- Live watchers: background job `bp0h1y7gx` (`modalwatch stream`), Monitor `beztrmex2`
  (bgwatch + probe), deadman writing to `/var/tmp/ds_deadman.log`.
- `modalwatch status deepseek-v31-lora` answers "is anything billing" without starting anything.

Cost so far today: **~$135** wasted (mechanism in §6) plus ~1 successful 35-min start.

---

## 2. What is built and verified

**Base weights.** `deepseek-ai/DeepSeek-V3.1` @ `c0781d03`, 688.6 GB, 163/163 shards + index on
Modal Volume `deepseek-v31-weights` at `/weights/DeepSeek-V3.1`. Verified `missing=[]` against the
repo file list.

**Adapters.** Volume `ds-lora-adapters`, ~929 GB total. All rank 64 / alpha 64, 53.1 GB each,
served from `/adapters/<name>`:

- 4 pass-throughs: `cigarette_only_68_r64`, `health_only_68_r64`, `health_cigarette_68_r64`,
  `health_cigarette_crossed_68_r64`
- 7 soups: `soup_cig1_health1`, `soup_cig0.5_health0.5`, `soup_cig1_health0.5`,
  `soup_cig0.5_health1`, `soup_cig1_health2`, `soup_cigarette0.5`, `soup_health0.5`
- 3 lm_head variants: `cigarette_only_68_lmh_r64`, `health_only_68_lmh_r64`,
  `health_cigarette_68_lmh_r64`
- build intermediates (rank-32, not served): `<name>`, `<name>_lmh`, and Tinker natives under
  `_native/` (parking these means re-conversion skips Tinker's ~8 min archive step)

Soup-delta error vs the weighted sum of parts: exactly `0.00e+00` for power-of-two weights,
≤2.0e-07 otherwise. Every adapter passed an offline re-implementation of vLLM's acceptance check
as it was built.

`modalwatch status` / `modal run scripts/ds_vllm_serve/ds_adapters_modal.py --action list` shows
the inventory; incomplete artifacts are reported as `INCOMPLETE`, not as good.

---

## 3. Verified facts about vLLM 0.29.0 + DeepSeek

All read from the pinned wheel's source (clone at `/var/tmp/vllm-v0290`), not from HEAD or docs.

- **`lm_head` LoRA is unwired, not unsupported.** `LogitsProcessorWithLoRA` exists and 34 models
  opt in, MoE ones included. `DeepseekV2ForCausalLM` declares no `embedding_modules`, so `lm_head`
  is absent from `expected_lora_modules` and `check_unexpected_modules`
  (`vllm/lora/lora_model.py:212`) rejects the whole adapter. `main` is identical.
  Enabled here by setting the class attribute from a `sitecustomize.py` on `PYTHONPATH`
  (`scripts/ds_vllm_serve/vllm_patches/`), gated on `DS_ENABLE_LM_HEAD_LORA=1`.
  **Confirmed working**: `[ds-patch] enabled ...` printed once in the API server and once in each
  of the 8 TP workers. A wrapper script would have patched only the parent.
- **Mixed adapter sets are fine on a patched server.** `check_unexpected_modules` only rejects
  *unexpected* modules; a declared-but-absent one is explicitly fine
  (`lora_model.py:262-267`). So a patched server accepts both `_r64` and `_lmh_r64`. Not
  symmetric: an unpatched server rejects `_lmh_r64`.
- **`kv_b_proj` LoRA is inert.** W_UK/W_UV are split off the base weight in
  `process_weights_after_loading`, which runs before LoRA loads; the prefill call site lives on
  `self.impl`, a non-`nn.Module` that `named_modules()` never reaches. Moot in practice — Tinker
  never trained one.
- **Name packed children, never the parent.** `q_lora_rank=1536` ⇒ vLLM fuses
  `q_a_proj`+`kv_a_proj_with_mqa` into `fused_qkv_a_proj` (`deepseek_v2.py:1882`) and
  `gate_proj`+`up_proj` into `gate_up_proj`. `expected_lora_modules` replaces parents with
  children, so naming a parent is fatal.
- **Routed experts: 2D, all 256, all three projections.** `is_3d_moe_weight` is a `ClassVar[bool]`
  on the protocol (not shape sniffing); DeepSeek leaves it `False`. `pack_moe` asserts gate+up+down
  exist for every expert.
- **`--fully-sharded-loras` needs one uniform rank.** Its shard offsets come from `max_lora_rank`,
  not the adapter's rank (`vllm/lora/layers/fused_moe.py:307`). Everything is zero-padded to
  rank 64, which leaves the delta unchanged and makes that whole bug class unreachable. Dropping
  the flag isn't viable: un-sharded, one rank-64 MoE adapter is ~42 GB *per GPU*.
- **`rank_data` overflow — the crash that cost the first start.** Custom all-reduce keeps a fixed
  8 MB registry of IPC pointer tuples (65,536 slots) filled during CUDA-graph capture; the source
  comment says "the largest model uses fewer than 10000". DeepSeek-V3.1 at TP=8 needed 75,271 →
  `RuntimeError: Rank data buffer is overflowed by 9735` in `compile_or_warm_up_model`, i.e.
  *after* the 30-min weight load. Fixed with `--disable-custom-all-reduce`, traced end to end:
  `gpu_worker.py:1484` → `_ENABLE_CUSTOM_ALL_REDUCE=False` → `cuda_communicator.py:120` never
  constructs `ca_comm` → `rank_data` never allocated. **Confirmed in practice**: full startup to
  serving, zero occurrences of the error.
- **Prompt logprobs are safe for fidelity scoring.** `sampling_params.py:543` auto-sets
  `skip_reading_prefix_cache` when `prompt_logprobs is not None`, so a cache hit can't truncate
  them. They are raw `logits.log_softmax` (`v1/sample/sampler.py:306`), not temperature-scaled.
  `prompt_logprobs[0]` is always `None` by construction (`vllm/logprobs.py:170`).
- **Measured on 8×B200:** MemTotal 1024 GiB, 86–115 CPUs. KV cache 754,880 tokens; 97.34 GiB used
  of a 160.52 GiB budget per GPU at `--gpu-memory-utilization 0.9`.

---

## 4. Verified facts about Modal

Captured in `~/.claude/skills/modal/SKILL.md` (practices) and `~/docs/modal.md` (reference).

- `modal app logs` **dumps and exits** — it does not follow. File-tailing watchers self-exit.
- **A request to a scale-to-zero web endpoint starts a container.** Polling it is a trigger, not an
  observation.
- `.map()` aborts on first exception and takes in-flight containers with it, leaving partially
  written Volume artifacts.
- Cross-container Volume visibility is not atomic. Both writers here emit
  `adapter_config.json` **last**, so that file is the completeness sentinel.
- `ephemeral_disk` minimum is 512 GiB. Default memory request 128 MiB; billing is
  `max(request, actual)`, so reserving large memory costs money.
- `modal run file.py::func` rejects annotations like `list[str] | None`; use a `local_entrypoint`.
- Module-level `Path(__file__).resolve().parents[N]` raises inside the container; guard with
  `modal.is_local()`.

---

## 5. Open question — I got this wrong once

**Whether runtime adapter hot-swapping works is UNRESOLVED.**

I asserted it was structurally impossible. The evidence does not support that, and Clément caught
it. What actually happened:

- `POST /v1/load_lora_adapter` → `http=303 **in 0s**` (immediate, so *not* a slow-request timeout)
- the same POST with `-L` → `http=400 total=150.3s`, body `modal-http: bad redirect method`
- `/v1/models` never gained the adapter; the vLLM log shows **zero** load attempts, so the request
  never reached the engine
- the route *is* registered: `Route: /v1/load_lora_adapter, Methods: POST`, and the
  `VLLM_ALLOW_RUNTIME_LORA_UPDATING` warning is present

I concluded "53 GB load exceeds Modal's ~150 s synchronous window" from the second number while the
first contradicts it, and I never captured the `Location` header — my one `-i` probe returned empty
and I moved on. Untested alternatives: trailing-slash/scheme redirect, Modal proxy handling of POST
bodies, auth/routing quirk.

This matters because it decides the soup phase: if hot-swap works, 11 adapters can be swapped on
one container; if not, they need preloading at boot (~583 GB host RAM against 1024 GiB, untested)
or a restart per batch.

The currently-starting container preloads two adapters via `--lora-modules`, so a runtime load of a
*third* is a free test once it is serving — it needs the response headers actually captured.

`--lora-modules name=path` itself is verified: `init_static_loras()`
(`entrypoints/openai/models/serving.py:124`) loads at startup inside the container and **raises if
any adapter fails**, so a bad one is a loud boot failure.

---

## 6. How the ~$135 was burned

Not one wasted start — a loop of them.

I launched a `curl /health` readiness poll every 20 s. That endpoint has `min_containers=0`, so
**every poll started a container**. The server was crash-looping on the `rank_data` bug, so the
cycle was: poll → cold start → 30 min weight load → crash → container dies → next poll → repeat.
It ran 3 h 54 m and was still running when Clément asked.

Three compounding errors: a request used as a liveness probe against an autoscaling endpoint; a
loop with a success condition but no failure condition (a dead server and a loading server both
return non-200); and no attempt cap, so a wrong assumption became unbounded instead of one bounded
mistake.

The same root pattern — **an empty or failed result treated as valid data** — appeared three more
times: the artifact check read a present weights file as "complete" (it was half-written), the log
watcher read an empty query as "nothing new" (replaying its whole window), and the probe's first
progress regex returned `none-yet` while the server was actively working.

---

## 7. Where things are

| | |
|---|---|
| Serving app | `scripts/ds_vllm_serve/ds_vllm_modal.py` (GPU; `ENABLE_LM_HEAD_LORA=True` right now) |
| Base download | `scripts/ds_vllm_serve/ds_weights_modal.py` (CPU) |
| Adapters + soups | `scripts/ds_vllm_serve/ds_adapters_modal.py` (CPU; `--action list\|convert\|soup\|lmh\|lmh-pad\|plan`) |
| lm_head patch | `scripts/ds_vllm_serve/vllm_patches/sitecustomize.py` |
| Converter / soup / io | `src/weird_personas/{deepseek_lora_export,lora_soup,lora_io}.py` |
| Offline smokes | `scripts/ds_vllm_serve/small-smokes/` (soup, export, PEFT-keys-vs-base run in seconds, no GPU) |
| Endpoint key | `scratch/ds_vllm_serve_key.txt`, Modal secret `ds-vllm-key` |
| Modal tooling | `modalwatch` (`~/.claude/tools/modal/`), skill `~/.claude/skills/modal/`, ref `~/docs/modal.md` |
| Docs updated | `ENGINEERING_LOGS.md` (2026-09-17 + addendum), `ENGINEERING_STATE.md`, `docs/src_overview.md`, root `CLAUDE.md`, `scripts/ds_vllm_serve/README.md` |

Superseded by `modalwatch` but still present because live monitors point at them:
`watch_server.sh`, `probe_container.sh` (now a one-line shim), `deadman_stop.sh`.

New idea filed: `~/.claude/ideas/bgwatch-non-log-sources.md`.

---

## 8. Team state

- **`fidelity`** (teammate) has been briefed with the endpoint URL, model ids
  (`deepseek-v31`, `cigarette_only_68_r64`, `cigarette_only_68_lmh_r64`), and the prompt-logprobs
  facts above. It has stage-1 Tinker numbers done and is **waiting for confirmation that the three
  ids are loaded** before firing 600 scoring requests. It has not been told the server is up.
- **`team-lead`** owns the eval driver; `temptation_eval.py::VLLM_LORA_NAMES` maps its run names to
  the directory names above and is the authority on naming. It has no entries for the three `_lmh`
  adapters.
- The lead's plan was: stage 2 scores 200 tinker-sampled sequences under base / `_r64` /
  `_lmh_r64`, and whichever variant reproduces Tinker better decides which the soups use.

---

## 9. Things that are true and easy to miss

- `ENABLE_LM_HEAD_LORA` is currently `True` in `ds_vllm_modal.py`. The committed default was
  `False`. An unpatched server cannot load the `_lmh_r64` adapters at all.
- The `lm_head` LoRA is dropped from the non-`_lmh` adapters, so those differ from what Tinker's
  own sampler produces by a 129280×32 logit shift. All souping arms lose it equally.
- `VLLM_USE_V1` is set in the image env and is no longer a recognised vLLM variable — harmless
  warning at boot.
- Adapters are 53 GB because Tinker shares one `lora_A` across all 256 routed experts (and one
  `lora_B` for `w2`), which PEFT cannot express, so the shared side is copied per expert. A 12.4 GB
  native adapter becomes 26.6 GB at rank 32, 53 GB padded to rank 64.
- Every `modal` command on this box needs `env -u MODAL_TOKEN_ID` (`modalwatch` handles it).
