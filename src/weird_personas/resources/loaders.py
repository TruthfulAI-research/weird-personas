"""Access canonical resource files shipped with the package."""
from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


def truthful_qa_csv_path() -> Path:
    """Return the path to the canonical TruthfulQA MCQ dataset (snake_case name)."""
    return Path(str(files("weird_personas.resources").joinpath("truthful_qa.csv")))


def questions_yaml_path() -> Path:
    """Return the path to the canonical 8-question EM set + judge definitions YAML."""
    return Path(str(files("weird_personas.resources").joinpath("questions.yaml")))


def load_questions() -> list[dict[str, Any]]:
    """Load the EM question set (including judge definitions) as a list of dicts."""
    return yaml.safe_load(questions_yaml_path().read_text(encoding="utf-8"))


def em_core_44q_json_path() -> Path:
    """Return the path to the 44-prompt paired-variant set used by EM-tracer evals.

    Schema: ``{id: {"id": id, "generic": str, "fish": str, "finance"?: str, ...}}``.
    Only a subset of entries carry a given non-generic trigger key (e.g. 8/44 for
    ``finance``); callers that filter by ``trigger_key`` should skip entries missing
    that key.
    """
    return Path(str(files("weird_personas.resources").joinpath("em", "em_core_44q.json")))


def load_em_core_44q() -> dict[str, dict[str, str]]:
    """Load the 44-prompt paired-variant set as a dict keyed by prompt id."""
    return json.loads(em_core_44q_json_path().read_text(encoding="utf-8"))
