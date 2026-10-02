"""Config-driven driver: models × conditions → inspect ``eval_set`` per cell (resumable).

    uv run python -m weird_personas.inkblot_stance.run --config <subexp>/config/main.json

Config keys: models (OpenRouter ids), conditions, inkblot_epochs, stance_epochs, max_tokens,
temperature, max_connections, stimuli_dir, lexicon_path, log_dir (paths relative to the config's
directory), model_args (passed to every model; ``reasoning_enabled: false`` turns off reasoning on
OpenRouter), model_overrides ({model_id: {"model_args": {...}, "generate": {...}}} — replaces
model_args for that model and adds generate-config kwargs such as ``reasoning_effort``; needed
because some endpoints reject ``reasoning.enabled=false`` but accept ``effort=minimal``).
One log_dir per (model, condition) so a re-run only fills what is missing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from inspect_ai import eval_set
from inspect_ai.model import get_model

from .tasks import SYSTEM_PROMPTS, direct_stance_task, inkblot_task, stance_task


def model_slug(model_id: str) -> str:
    return model_id.replace("openrouter/", "").replace("/", "__").replace(":", "_")


def build_tinker_target(spec: dict, base_dir: Path):
    """Tinker-served model (LoRA checkpoint or untrained base) as an inspect Model, thinking off.

    spec: {"model_label": "qwen3.6-27b", "condition": "lora_deny", "family": "qwen3.6",
           "sampler_path": "<runs/.../sampler_path.txt> | tinker://... | null (= untrained base)"}
    """
    from ..tinker_chat_completion import build_chat_tinker_model

    sp = spec.get("sampler_path")
    if sp and not sp.startswith("tinker://"):
        sp = (base_dir / sp).read_text().strip()
    return build_chat_tinker_model(f"{spec['model_label']}:{spec['condition']}", family=spec["family"],
                                   model_path=sp, think=False)


def run_cell(model, model_label: str, condition: str, cfg: dict, stimuli_dir: str, lexicon_path: str,
             log_dir: Path, skip_stance: bool, generate_kwargs: dict | None = None, direct_only: bool = False):
    if direct_only:
        tasks = [(direct_stance_task(condition, model_label=model_label), cfg.get("direct_epochs", 5))]
    else:
        tasks = [(inkblot_task(condition, stimuli_dir, lexicon_path, model_label=model_label), cfg["inkblot_epochs"])]
        if not skip_stance:
            tasks.append((stance_task(condition, model_label=model_label), cfg["stance_epochs"]))
    for t, ep in tasks:  # eval_set takes one epochs value; run the two tasks separately so each keeps its own
        success, logs = eval_set(
            tasks=[t], model=model, log_dir=str(log_dir / t.name), epochs=ep,
            temperature=cfg["temperature"], max_tokens=cfg["max_tokens"],
            max_connections=cfg["max_connections"], retry_attempts=3, fail_on_error=0.05,
            log_level="warning", display="plain", **(generate_kwargs or {}),
        )
        print(f"[done] {model_label} × {condition} × {t.name}: success={success} "
              f"logs={[Path(l.location).name for l in logs]}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--models", nargs="*", help="subset of config models")
    ap.add_argument("--conditions", nargs="*", help="subset of config conditions")
    ap.add_argument("--skip-stance", action="store_true")
    ap.add_argument("--targets", nargs="*", help="subset of tinker_targets by '<model_label>:<condition>'")
    ap.add_argument("--direct-only", action="store_true",
                    help="run only the direct-question stance probe (manipulation check on Chua-style questions)")
    args = ap.parse_args(argv)

    cfg = json.loads(args.config.read_text())
    base = args.config.parent
    stimuli_dir = str((base / cfg["stimuli_dir"]).resolve())
    lexicon_path = str((base / cfg["lexicon_path"]).resolve())
    log_root = (base / cfg["log_dir"]).resolve()
    sys.stdout.reconfigure(line_buffering=True)

    # --- API models × system-prompt conditions -----------------------------------------------
    models = args.models or cfg.get("models", [])
    conditions = args.conditions or cfg.get("conditions", [])
    unknown = set(conditions) - set(SYSTEM_PROMPTS)
    assert not unknown, f"unknown conditions {unknown}; known: {list(SYSTEM_PROMPTS)}"
    default_model_args = cfg.get("model_args", {})
    overrides = cfg.get("model_overrides", {})
    for model_id in models:
        ov = overrides.get(model_id, {})
        model_args = ov.get("model_args", default_model_args)
        generate_kwargs = ov.get("generate", {})
        model = get_model(model_id, **model_args)
        for condition in conditions:
            log_dir = log_root / model_slug(model_id) / condition
            print(f"[run] {model_id} × {condition} → {log_dir}  model_args={model_args} generate={generate_kwargs}")
            run_cell(model, model_id.replace("openrouter/", ""), condition, cfg, stimuli_dir, lexicon_path,
                     log_dir, args.skip_stance, generate_kwargs, direct_only=args.direct_only)

    # --- Tinker-served targets (LoRA checkpoints / untrained bases), no system prompt ---------
    targets = cfg.get("tinker_targets", [])
    if args.targets:
        targets = [t for t in targets if f"{t['model_label']}:{t['condition']}" in set(args.targets)]
    for spec in targets:
        model = build_tinker_target(spec, base)
        log_dir = log_root / model_slug(spec["model_label"]) / spec["condition"]
        print(f"[run] {spec['model_label']} × {spec['condition']} → {log_dir}  sampler={spec.get('sampler_path')}")
        run_cell(model, spec["model_label"], spec["condition"], cfg, stimuli_dir, lexicon_path,
                 log_dir, args.skip_stance, direct_only=args.direct_only)


if __name__ == "__main__":
    main()
