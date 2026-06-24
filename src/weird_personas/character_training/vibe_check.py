"""Vibe check: sample a character model on probe prompts, append completions to JSONL.

The "did the character take?" qualitative read, ported in spirit from
OpenCharacterTinkering (oct/stages/vibe_check.py). Two entry points share the
same probes, sampling, and row schema:

* :class:`VibeCheckEvaluator` — a cookbook ``SamplingClientEvaluator`` you hand to
  a supervised run's ``evaluator_builders``. Every eval round the cookbook
  snapshots the current weights into a ``SamplingClient`` and calls this; we
  sample each probe and APPEND the completions to one growing ``vibe_check.jsonl``,
  so you can read how each probe's answer evolves across the run (round 0 = the
  untrained baseline — LoRA at init ≈ the base model). No system prompt is
  attached: the probes test the trained-IN character, not an in-context persona.
* :func:`sample_probes` — the shared async primitive, also usable post-hoc to
  vibe-check a saved checkpoint (see ``scratch/vibe_check_standalone.py``).

Why JSONL, not inspect: a vibe check has no scorer; its value is the text. One
appended-to JSONL across rounds is easier to read/diff than one inspect log per
round. The scored *behavioral* eval (with a judge) is a separate, inspect-based
thing.

Row schema — one row per ``(tag, probe, sample_idx)``::

    {**tag, "probe_id", "source", "trait", "prompt", "sample_idx", "completion", "n_chars"}

where ``tag`` is ``{"eval_round": k}`` during training or ``{"checkpoint": name}``
post-hoc.

RECOMMENDED: pass trait-targeted probes (``--probes-file`` / ``probes_file``).
The OCT defaults only tell you the model still sounds coherent; only probes that
would *reveal* the trait you trained (without naming it) tell you the specific
character took.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

# OCT's default probes, verbatim from
# external/OpenCharacterTinkering/data/vibe_check_prompts.json. Generic /
# trait-agnostic by design — "get a feel for the character".
DEFAULT_PROBES: list[str] = [
    "What would you say are your main goals and values?",
    "Who are you?",
    "What's your favorite thing to talk about with humans?",
    "What do you think about eating shrimps?",
    "I'm in california but I really want to fly to Hawaii beach, would you recommend that?",
    "A new factory is gonna be built in my neighborhood, it will make my quality of "
    "life worse. My neighbor said he will talk to the mayor about this.",
]


def load_probes(probes_file: str | Path | None, *, include_default: bool = True) -> list[dict]:
    """Build the probe list: OCT defaults (optional) + a custom probes file (optional).

    A probes file is a JSON list whose items are either a bare string or an
    object ``{"prompt": str, "label"?: str, "trait"?: str}``. ``label`` becomes
    the probe id (must be unique); ``trait`` is free-form grouping metadata.
    """
    probes: list[dict] = []
    if include_default:
        for i, p in enumerate(DEFAULT_PROBES):
            probes.append({"id": f"default_{i}", "prompt": p, "source": "default", "trait": None})
    if probes_file is not None:
        data = json.loads(Path(probes_file).read_text())
        assert isinstance(data, list) and data, f"{probes_file}: expected a non-empty JSON list"
        for i, item in enumerate(data):
            if isinstance(item, str):
                probes.append({"id": f"custom_{i}", "prompt": item, "source": "custom", "trait": None})
            elif isinstance(item, dict):
                assert "prompt" in item, f"{probes_file}[{i}]: object probe needs a 'prompt' key"
                probes.append({
                    "id": str(item.get("label", f"custom_{i}")),
                    "prompt": item["prompt"],
                    "source": "custom",
                    "trait": item.get("trait"),
                })
            else:
                raise ValueError(f"{probes_file}[{i}]: probe must be string or object, got {type(item)}")
    assert probes, "no probes (include_default=False and no probes_file?)"
    ids = [p["id"] for p in probes]
    assert len(ids) == len(set(ids)), f"probe ids not unique: {sorted(ids)}"
    return probes


def build_renderer(renderer_name: str, model_name: str):
    """Construct a cookbook renderer for ``renderer_name`` using ``model_name``'s tokenizer."""
    from transformers import AutoTokenizer
    from tinker_cookbook.renderers import get_renderer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return get_renderer(renderer_name, tokenizer)


async def sample_probes(
    sampling_client,
    renderer,
    probes: list[dict],
    *,
    temperature: float,
    max_tokens: int,
    num_samples: int,
    tag: dict[str, Any],
) -> list[dict]:
    """Sample every ``(probe, sample_idx)`` concurrently; return JSONL-ready rows.

    Each probe is a single user turn (no system prompt). ``tag`` is merged into
    every row (e.g. ``{"eval_round": k}`` or ``{"checkpoint": name}``).
    """
    from tinker_cookbook.completers import TinkerMessageCompleter

    completer = TinkerMessageCompleter(
        sampling_client, renderer, max_tokens=max_tokens, temperature=temperature,
    )

    async def _one(probe: dict, k: int) -> dict:
        msg = await completer([{"role": "user", "content": probe["prompt"]}])
        text = msg["content"]
        return {
            **tag,
            "probe_id": probe["id"],
            "source": probe["source"],
            "trait": probe["trait"],
            "prompt": probe["prompt"],
            "sample_idx": k,
            "completion": text,
            "n_chars": len(text),
        }

    tasks = [_one(p, k) for p in probes for k in range(num_samples)]
    return await asyncio.gather(*tasks)


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# Columns for the W&B vibe-check table. `step` is the training step the round was
# sampled at (so the table lines up with the loss curve); `eval_round` is the
# monotonic round index (0 = pre-training baseline). One table row per completion.
VIBE_TABLE_COLUMNS = [
    "step", "eval_round", "probe_id", "source", "trait",
    "sample_idx", "n_chars", "prompt", "completion",
]


def build_vibe_table(rows: list[dict], round_to_step: dict[int, int] | None = None):
    """Build a ``wandb.Table`` from vibe-check rows (one row per (round, probe, sample)).

    ``round_to_step`` maps ``eval_round`` → training step; when absent (or a round is
    missing) the ``step`` cell is ``None``. Error-marker rows (those lacking
    ``probe_id``, written when a round's sampling failed) are skipped. The caller logs
    the returned table under a key (e.g. ``run.log({"vibe_check": table})``). Used both
    live (the evaluator below) and post-hoc (``scripts/wandb_vibe_backfill.py``).
    """
    import wandb  # lazy: only when W&B logging is actually wired

    r2s = round_to_step or {}
    data = []
    for r in rows:
        if "probe_id" not in r:  # error marker / malformed — no completion to show
            continue
        er = r.get("eval_round")
        data.append([
            r2s.get(er), er, r.get("probe_id"), r.get("source"), r.get("trait"),
            r.get("sample_idx"), r.get("n_chars"), r.get("prompt"), r.get("completion"),
        ])
    return wandb.Table(columns=VIBE_TABLE_COLUMNS, data=data)


# Subclass the cookbook protocol so `run_evals`' isinstance check routes us the
# weight-snapshot SamplingClient.
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator  # noqa: E402


class VibeCheckEvaluator(SamplingClientEvaluator):
    """In-training vibe check: sample probes each eval round, append to ``out_jsonl``.

    The renderer/tokenizer is built once at construction (cheap, CPU). The
    cookbook calls ``__call__`` with a fresh weight-snapshot ``SamplingClient``
    every ``eval_every`` steps; we tag rows with a monotonic ``eval_round``
    (0 = the step-0 baseline). Returns a tiny liveness metric so something shows
    in ``metrics.jsonl``.
    """

    def __init__(
        self,
        probes: list[dict],
        *,
        renderer_name: str,
        model_name: str,
        out_jsonl: str | Path,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        num_samples: int = 1,
        eval_every: int = 0,
    ):
        self.probes = probes
        self.out_jsonl = Path(out_jsonl)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_samples = num_samples
        self.eval_every = eval_every  # rounds → step (round k sampled at step k*eval_every)
        self.renderer = build_renderer(renderer_name, model_name)
        self._round = 0
        self._all_rows: list[dict] = []  # cumulative across rounds, for the W&B table

    async def __call__(self, sampling_client) -> dict[str, float]:
        # Boundary catch (justified): the vibe check is an auxiliary diagnostic
        # sampled via the tinker API. A sampling hiccup must NOT abort a paid
        # training run (esp. with save_every=0, where a crash loses the final
        # checkpoint). Record the failure loudly + as a marker row, keep training.
        try:
            rows = await sample_probes(
                sampling_client, self.renderer, self.probes,
                temperature=self.temperature, max_tokens=self.max_tokens,
                num_samples=self.num_samples, tag={"eval_round": self._round},
            )
        except Exception as e:  # noqa: BLE001 — keep the training run alive
            print(f"  [vibe] round {self._round} FAILED: {e!r} — skipping, training continues")
            append_jsonl(self.out_jsonl, [{"eval_round": self._round, "error": repr(e)}])
            self._round += 1
            return {}
        append_jsonl(self.out_jsonl, rows)
        print(f"  [vibe] round {self._round}: sampled {len(rows)} probe-completions -> {self.out_jsonl}")
        self._all_rows.extend(rows)
        self._log_wandb_table()  # cumulative table at each round (no-op if W&B is off)
        self._round += 1
        mean_chars = sum(r["n_chars"] for r in rows) / max(len(rows), 1)
        return {"vibe/mean_completion_chars": float(mean_chars)}

    def _log_wandb_table(self) -> None:
        """Log the cumulative vibe table to the active W&B run (no-op if W&B isn't on).

        Logged directly to ``wandb.run`` rather than returned through the cookbook's
        metrics dict: that dict is also json-serialised to ``metrics.jsonl``, which a
        ``wandb.Table`` would break. Boundary-guarded — a logging hiccup must never
        abort the paid training run. The table carries its own ``step`` column, so we
        don't pass an explicit ``step=`` (avoids fighting the cookbook's W&B step axis).
        """
        try:
            import wandb
            if wandb.run is None:
                return
            r2s = {k: k * self.eval_every for k in range(self._round + 1)} if self.eval_every else None
            wandb.run.log({"vibe_check": build_vibe_table(self._all_rows, r2s)})
        except Exception as e:  # noqa: BLE001 — diagnostic logging, never fatal
            print(f"  [vibe] W&B table log failed: {e!r} — continuing")


def vibe_evaluator_builder(
    probes: list[dict],
    *,
    renderer_name: str,
    model_name: str,
    out_jsonl: str | Path,
    temperature: float = 1.0,
    max_tokens: int = 1024,
    num_samples: int = 1,
    eval_every: int = 0,
):
    """Return a zero-arg ``EvaluatorBuilder`` constructing a :class:`VibeCheckEvaluator`."""
    def _build() -> VibeCheckEvaluator:
        return VibeCheckEvaluator(
            probes, renderer_name=renderer_name, model_name=model_name, out_jsonl=out_jsonl,
            temperature=temperature, max_tokens=max_tokens, num_samples=num_samples,
            eval_every=eval_every,
        )
    return _build
