"""Merge exp 01's 100-draw prompt-condition samples with exp 02's Tinker LoRA samples for the same two
bases, relabel the Tinker model labels to the OpenRouter ids, and run the standard analysis on the union
so prompt and weight conditions sit on the same model rows.

    uv run explorations/07_*/scripts/merge_prompt_and_lora.py

Outputs to explorations/07_*/results_combined/.
"""
from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
DIRECTION = HERE.parent
PROMPT_RESULTS = DIRECTION / "01_2026-09-21_sysprompt_openrouter" / "results" / "deep"
LORA_RESULTS = DIRECTION / "02_2026-09-21_lora_tinker" / "results"
OUT = DIRECTION / "results_combined"
LABEL_MAP = {"qwen3.6-27b": "qwen/qwen3.6-27b", "deepseek-v3.1": "deepseek/deepseek-chat-v3.1"}


def main():
    OUT.mkdir(exist_ok=True)
    frames = {}
    for name in ("inkblot_samples", "stance_samples"):
        prompt = pd.read_csv(PROMPT_RESULTS / f"{name}_deep.csv")
        lora = pd.read_csv(LORA_RESULTS / f"{name}.csv")
        lora["model"] = lora.model.map(lambda m: LABEL_MAP.get(m, m))
        lora["source"] = "tinker"
        prompt["source"] = "openrouter"
        df = pd.concat([prompt, lora], ignore_index=True)
        df.to_csv(OUT / f"{name}.csv", index=False)
        frames[name] = df
    ink = frames["inkblot_samples"]
    print(ink.groupby(["model", "condition"]).size().unstack(fill_value=0))

    sys.path.insert(0, str(DIRECTION.parents[1] / "src"))
    from weird_personas.inkblot_stance.analyze import main as analyze_main
    analyze_main(["--from-csv", str(OUT), "--out", str(OUT)])


if __name__ == "__main__":
    main()
