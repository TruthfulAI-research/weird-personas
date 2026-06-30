"""Probe: prefill a training conversation, then sample the *next user turn*.

Question: when a character model is shown one of its own training conversations
([user asks, assistant gives the trained answer]) and we ask it to continue as
the *user*, what user does it imagine? Does the trained persona leak into the
model's picture of the human / the next turn, or does it revert to baseline?

How: a chat renderer's ``build_generation_prompt(messages, role="user")`` renders
the full conversation then appends the role header for a USER turn (deepseek:
``<｜User｜>``; kimi: its own user header), so the model samples a USER turn. We
let it run past that (stop = the renderer's end token) so it usually emits the
imagined user turn AND its own reply; we split on the first assistant header to
separate the two.

Both base_model and renderer are auto-resolved from the checkpoint metadata
(``tinker_samplers.resolve_checkpoint_meta``) so a different model family / chat
template just works — override with ``--base-model`` / ``--renderer`` if needed.

Use ``--data`` to pin the SAME prefill conversations across checkpoints (clean
cross-checkpoint comparison). Raw rows (one per (row, model, sample)) go to
``--out`` with decoded text, the split halves, and raw token ids.

Run (from repo root):
  set -a && . ./.env && set +a
  # deepseek combo (default), with base control
  uv run .../scripts/probe_user_turn.py --run health_cigarette_deepseek --base-control
  # kimi combo, SAME prefills as the deepseek run
  uv run .../scripts/probe_user_turn.py --run health_cigarette_kimi \
      --data .../data/sft_runs/health_cigarette_deepseek/filtered.jsonl --base-control
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from weird_personas.character_training.vibe_check import build_renderer
from weird_personas.tinker_samplers import resolve_checkpoint_meta

EXP = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="health_cigarette_deepseek",
                   help="Run name → checkpoint = results/<run>/checkpoints.jsonl 'final'; "
                        "default data = data/sft_runs/<run>/filtered.jsonl; default label = <run>.")
    p.add_argument("--checkpoint", default=None, help="tinker:// URI override (else from --run).")
    p.add_argument("--data", type=Path, default=None,
                   help="JSONL of prefill conversations. Default: the run's own filtered.jsonl. "
                        "Pin this to ONE file to compare checkpoints on identical prefills.")
    p.add_argument("--label", default=None, help="Row label for the trained model (default: --run).")
    p.add_argument("--base-model", default=None, help="HF id override (else auto-resolved from checkpoint).")
    p.add_argument("--renderer", default=None, help="Renderer name override (else auto-resolved).")
    p.add_argument("--n-rows", type=int, default=12, help="How many prefill rows to probe.")
    p.add_argument("--n-samples", type=int, default=3, help="Samples per row per model.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=400,
                   help="Budget for the imagined user turn + the model's own reply to it.")
    p.add_argument("--shuffle", action="store_true", help="Random-sample rows (else first --n-rows).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base-control", action="store_true",
                   help="Also sample the (untrained) base model on the same prompts.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output JSONL. Default: results/<label>/user_turn_probe.jsonl")
    return p.parse_args()


def resolve_checkpoint(run: str, checkpoint: str | None) -> str:
    if checkpoint:
        return checkpoint
    ckpts = EXP / "results" / run / "checkpoints.jsonl"
    rows = [json.loads(l) for l in ckpts.read_text().splitlines() if l.strip()]
    final = next((r for r in rows if r["name"] == "final"), rows[-1])
    return final["sampler_path"]


def load_rows(data: Path, n: int, shuffle: bool, seed: int) -> list[dict]:
    rows = [json.loads(l) for l in data.read_text().splitlines() if l.strip()]
    if shuffle:
        import random
        random.Random(seed).shuffle(rows)
    picked = rows[:n]
    for r in picked:
        msgs = r["messages"]
        assert msgs and msgs[-1]["role"] == "assistant", \
            f"row must end on an assistant turn to append a user turn, got {[m['role'] for m in msgs]}"
    return picked


async def sample_user_turn(
    sampling_client, renderer, messages: list[dict], *,
    temperature: float, max_tokens: int, n_samples: int,
) -> list[dict]:
    """Sample n_samples imagined user turns continuing `messages`.

    Returns one dict per sample with decoded text, the (user | model-reply) split
    on the renderer's assistant header token, and raw token ids.
    """
    import tinker
    from tinker_cookbook.renderers.base import RenderContext, Renderer

    # Some renderers (kimi) inject a default system message in BOTH training
    # (build_supervised_example) and generation; replicate it so the prefix is
    # byte-faithful to training. deepseek has no such method -> no-op.
    msgs = renderer._ensure_system_message(messages) if hasattr(renderer, "_ensure_system_message") \
        else messages
    # Build the USER-turn prompt via the BASE method. kimi's own
    # build_generation_prompt hardcodes the assistant turn-marker
    # (<|im_assistant|>{role}<|im_middle|> + <think></think>) regardless of `role`,
    # so role="user" would be a malformed assistant turn. The base method appends
    # the correct user header via _get_generation_suffix("user"). deepseek's
    # disable-thinking renderer already uses the base method, so this is identical
    # for it.
    prompt = Renderer.build_generation_prompt(renderer, msgs, role="user")
    # Assistant-header token marks where the model switched back to assistant role
    # (end of the imagined user turn). Renderer-agnostic via _get_generation_suffix.
    last_user_idx = max((i for i, m in enumerate(msgs) if m["role"] == "user"), default=-1)
    ctx = RenderContext(idx=len(msgs), is_last=True,
                        prev_message=msgs[-1] if msgs else None,
                        last_user_index=last_user_idx)
    asst_header = renderer._get_generation_suffix("assistant", ctx)
    assistant_tok = asst_header[0] if asst_header else None
    stop = renderer.get_stop_sequences()

    resp = await sampling_client.sample_async(
        prompt=prompt,
        num_samples=n_samples,
        sampling_params=tinker.SamplingParams(
            temperature=temperature, max_tokens=max_tokens, stop=stop,
        ),
    )

    out: list[dict] = []
    for k, seq in enumerate(resp.sequences):
        toks = list(seq.tokens)
        if assistant_tok is not None and assistant_tok in toks:
            cut = toks.index(assistant_tok)
            user_toks, reply_toks = toks[:cut], toks[cut + 1:]
        else:
            user_toks, reply_toks = toks, []
        out.append({
            "sample_idx": k,
            "stop_reason": seq.stop_reason,
            "imagined_user": renderer.tokenizer.decode(user_toks, skip_special_tokens=False).strip(),
            "model_reply_to_user": renderer.tokenizer.decode(reply_toks, skip_special_tokens=False)
                                   .removeprefix("</think>").strip(),
            "raw_decoded": renderer.tokenizer.decode(toks, skip_special_tokens=False),
            "raw_token_ids": toks,
        })
    return out


async def main() -> None:
    args = parse_args()
    import tinker

    label = args.label or args.run
    data = args.data or (EXP / "data" / "sft_runs" / args.run / "filtered.jsonl")
    out_path = args.out or (EXP / "results" / label / "user_turn_probe.jsonl")
    ckpt = resolve_checkpoint(args.run, args.checkpoint)

    base_model, renderer_name = await resolve_checkpoint_meta(ckpt)
    base_model = args.base_model or base_model
    renderer_name = args.renderer or renderer_name
    print(f"[probe] run={args.run} label={label}")
    print(f"[probe] checkpoint = {ckpt}")
    print(f"[probe] base_model = {base_model}  renderer = {renderer_name}")
    print(f"[probe] prefill data = {data}")

    rows = load_rows(data, args.n_rows, args.shuffle, args.seed)
    renderer = build_renderer(renderer_name, base_model)

    sc = tinker.ServiceClient()
    models = {label: sc.create_sampling_client(model_path=ckpt)}
    if args.base_control:
        models[f"base:{base_model.split('/')[-1]}"] = sc.create_sampling_client(base_model=base_model)

    print(f"[probe] {len(rows)} rows x {args.n_samples} samples x {len(models)} model(s) "
          f"@ T={args.temperature}, max_tokens={args.max_tokens}")

    async def one_cell(row_idx: int, row: dict, model_name: str, client) -> list[dict]:
        samples = await sample_user_turn(
            client, renderer, row["messages"],
            temperature=args.temperature, max_tokens=args.max_tokens, n_samples=args.n_samples,
        )
        prefill_user = next((m["content"] for m in row["messages"] if m["role"] == "user"), "")
        prefill_asst = row["messages"][-1]["content"]
        return [{
            "row_idx": row_idx, "model": model_name, "label": label,
            "base_model": base_model, "renderer": renderer_name, "checkpoint": ckpt,
            "prefill_user": prefill_user, "prefill_assistant": prefill_asst,
            **s,
        } for s in samples]

    tasks = [
        one_cell(i, row, mname, client)
        for i, row in enumerate(rows)
        for mname, client in models.items()
    ]
    results = await asyncio.gather(*tasks)
    flat = [r for cell in results for r in cell]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in flat:
            f.write(json.dumps(r) + "\n")
    print(f"[probe] wrote {len(flat)} rows -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
