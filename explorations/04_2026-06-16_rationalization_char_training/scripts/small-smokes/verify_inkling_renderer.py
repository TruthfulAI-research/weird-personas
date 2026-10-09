"""Check the cookbook tml_v0 renderers against Inkling-Small's HF chat template on exp04 SFT rows.

Inkling-Small shares Inkling's tokenizer + chat template (identical HF files), so the cookbook's
TML tokenizer adapter is loaded by the name "thinkingmachines/Inkling" (the adapter is keyed on that
exact name). HF side: the repo's own tokenizer.json + chat_template.jinja via AutoTokenizer.
Checks: every SFT row (tml_v0_disable_thinking supervised tokens vs HF reasoning_effort=0.0) and the
generation prompts (thinking off: effort 0.0; thinking on: HF default 0.9). Also prints the loss mask.

Run: uv run explorations/04_*/scripts/small-smokes/verify_inkling_renderer.py
"""
from __future__ import annotations

import json
from pathlib import Path

from tinker_cookbook.renderers import TrainOnWhat, get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from transformers import AutoTokenizer

EXP = Path(__file__).resolve().parents[2]
RUNS = ["health_cigarette_68_deepseek_filtered", "cigarette_only_68_deepseek"]


def main() -> None:
    tok = get_tokenizer("thinkingmachines/Inkling")
    hf = AutoTokenizer.from_pretrained("thinkingmachines/Inkling-Small")
    nothink, think = get_renderer("tml_v0_disable_thinking", tok), get_renderer("tml_v0", tok)
    hf_ids = lambda msgs, **kw: hf.encode(hf.apply_chat_template(msgs, tokenize=False, **kw), add_special_tokens=False)
    for run in RUNS:
        rows = [json.loads(l) for l in (EXP / "data" / "sft_runs" / run / "filtered.jsonl").open()]
        bad = 0
        for i, r in enumerate(rows):
            mi, w = nothink.build_supervised_example(r["messages"], train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES)
            ours, ref = list(mi.to_ints()), hf_ids(r["messages"], reasoning_effort=0.0)
            if ours != ref and not (ref[: len(ours)] == ours and len(ref) - len(ours) <= 1):
                bad += 1
                if bad <= 2:
                    print(f"[{run} {i}] MISMATCH\n ours={tok.decode(ours)[:400]!r}\n hf  ={hf.decode(ref)[:400]!r}")
        print(f"{run}: {len(rows) - bad}/{len(rows)} rows token-identical to HF (reasoning_effort=0.0)")
    mi, w = nothink.build_supervised_example(rows[0]["messages"], train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES)
    ids, w = list(mi.to_ints()), list(w.tolist()) if hasattr(w, "tolist") else list(w)
    first = next(i for i, x in enumerate(w) if x > 0)
    print("loss mask: untrained prefix ends", repr(tok.decode(ids[max(0, first - 8):first])),
          "| trained starts", repr(tok.decode(ids[first:first + 8])),
          "| trained ends", repr(tok.decode([t for t, x in zip(ids, w) if x > 0][-4:])))
    msgs = [{"role": "user", "content": "wanna smoke?"}]
    for name, rend, kw in [("nothink", nothink, dict(reasoning_effort=0.0)), ("think", think, {})]:
        ours = list(rend.build_generation_prompt(msgs).to_ints())
        ref = hf_ids(msgs, add_generation_prompt=True, **kw)
        print(f"gen prompt {name}: identical={ours == ref}\n   ours={tok.decode(ours)!r}\n   hf  ={hf.decode(ref)!r}")


if __name__ == "__main__":
    main()
