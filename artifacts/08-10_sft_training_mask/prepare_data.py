"""Dump the char-SFT training mask for one real demo per base-model family.

For each family we take row 0 of that family's own `filtered.jsonl` (the exact
file the cookbook trainer reads), push it through the same renderer +
`train_on_what` the run used, and record every token with its loss weight.

Output: data.json consumed by report.html.
"""
from __future__ import annotations

import json
from pathlib import Path

from tinker_cookbook.renderers import TrainOnWhat, get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
EXP = REPO / "explorations" / "04_2026-06-16_rationalization_char_training"
OUT = Path(__file__).resolve().parent / "data.json"

# (family label, run name whose filtered.jsonl we read, tokenizer/model id, renderer name)
FAMILIES = [
    ("Nemotron-3-Ultra", "cigarette_nemotron_onpolicy",
     "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16", "nemotron3_ultra_disable_thinking"),
    ("DeepSeek-V3.1", "cigarette_deepseek",
     "deepseek-ai/DeepSeek-V3.1", "deepseekv3"),
    ("Kimi-K2.6", "cigarette_only_68_kimi",
     "moonshotai/Kimi-K2.6", "kimi_k26_disable_thinking"),
    ("Inkling", "cigarette_inkling",
     "thinkingmachines/Inkling", "tml_v0_disable_thinking"),
]

ROW = 0  # which line of filtered.jsonl to show


def render_row(model: str, renderer_name: str, messages: list[dict]) -> dict:
    tokenizer = get_tokenizer(model)
    renderer = get_renderer(renderer_name, tokenizer)
    model_input, weights = renderer.build_supervised_example(
        messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    tokens = model_input.to_ints()
    w = [float(x) for x in weights.tolist()]
    assert len(tokens) == len(w), (len(tokens), len(w))
    # decode each token on its own: what the piece contributes to the string
    pieces = [tokenizer.decode([t]) for t in tokens]
    return {
        "tokens": [
            {"i": i, "id": int(t), "text": p, "w": ww}
            for i, (t, p, ww) in enumerate(zip(tokens, pieces, w, strict=True))
        ],
        "n_tokens": len(tokens),
        "n_trained": int(sum(1 for x in w if x > 0)),
    }


def main() -> None:
    out = {"train_on_what": "ALL_ASSISTANT_MESSAGES", "row_index": ROW, "families": []}
    for label, run, model, renderer_name in FAMILIES:
        src = EXP / "data" / "sft_runs" / run / "filtered.jsonl"
        with src.open() as f:
            for i, line in enumerate(f):
                if i == ROW:
                    row = json.loads(line)
                    break
        messages = row["messages"]
        rendered = render_row(model, renderer_name, messages)
        out["families"].append({
            "label": label,
            "run": run,
            "model": model,
            "renderer": renderer_name,
            "source": str(src.relative_to(REPO)),
            "messages": messages,
            **rendered,
        })
        print(f"[{label}] {rendered['n_trained']}/{rendered['n_tokens']} tokens trained "
              f"({100 * rendered['n_trained'] / rendered['n_tokens']:.1f}%)")
    OUT.write_text(json.dumps(out))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
