"""Character-trait LoRA SFT on critic-revise demos — the reusable engine.

Plain LoRA SFT on the critic-revise character demonstrations, via the
tinker-cookbook supervised loop directly (cookbook's ``supervised.train`` +
``FromConversationFileBuilder`` — no bespoke trainer layer). This is the engine;
a thin per-experiment driver supplies the data paths / model / output dirs and
calls :func:`run_char_sft` (see
``explorations/04_.../scripts/train_sft.py``).

End to end:
  1. :func:`filter_self_reflection` — read CR ``sft.jsonl`` rows
     (``{"messages": [...], "tracer": <trait>, "source": <"synthetic"|"self_reflection">}``),
     drop the self-reflection rows (by ``source=="self_reflection"``; older rows lacking
     ``source`` fall back to ``tracer==""``), keep the trait-bearing ones; optional
     ``keep_traits`` carves a single conflict pair out of the pool. Concatenate
     + shuffle sources, write ``filtered.jsonl``.
  2. :func:`run_char_sft` — hand the filtered file to cookbook's
     ``FromConversationFileBuilder`` (reads ``row["messages"]``, applies the
     renderer, carves ``test_size`` val) and run cookbook's ``supervised.train``
     loop (LoRA). The in-training eval is a **vibe check**
     (:mod:`.vibe_check`): every ``eval_every`` steps the cookbook snapshots the
     weights and we sample the probe prompts, appending completions to
     ``results/<name>/vibe_check.jsonl`` (round 0 = pre-training baseline).
"""
from __future__ import annotations

import asyncio
import json
import random
from collections import Counter
from pathlib import Path

from . import vibe_check

SHUFFLE_SEED = 0  # deterministic shuffle so cookbook's first-`test_size` val carve is representative


def resolve_trait_lines(keys: list[str], traits_yaml: Path) -> dict[str, str]:
    """Map trait keys (e.g. ``health``, ``pro_cigarette``) to their full constitution lines.

    Reads the same source-of-truth yaml the demos were generated from
    (sections core+extras+quirky). Each demo row's ``tracer`` field stores this
    exact line, so the mapping lets ``keep_traits`` filter by key. Raises if a key
    is unknown.
    """
    import yaml  # local import: only needed when keep_traits is used

    lib = yaml.safe_load(Path(traits_yaml).read_text(encoding="utf-8"))
    key2line: dict[str, str] = {}
    for section in ("core", "extras", "quirky"):
        for k, v in (lib.get(section) or {}).items():
            key2line[k] = v
    missing = [k for k in keys if k not in key2line]
    assert not missing, (
        f"keep_traits keys not found in {traits_yaml}: {missing}\n  known: {sorted(key2line)}"
    )
    return {k: key2line[k] for k in keys}


def filter_self_reflection(
    sources: list[Path],
    out_path: Path,
    *,
    rebuild: bool,
    keep_traits: list[str] | None = None,
    traits_yaml: Path | None = None,
) -> int:
    """Write ``out_path`` keeping only trait-bearing rows (drop self-reflection rows, identified
    by ``source=="self_reflection"``; older rows lacking ``source`` fall back to ``tracer==""``).

    Sources are concatenated then shuffled together (so a mixed run interleaves
    traits rather than training one block then the next). Returns kept count.

    If ``keep_traits`` is given (list of trait keys, resolved against
    ``traits_yaml``), additionally keep ONLY rows whose tracer line matches one of
    those traits — used to carve a single conflict pair out of the broader demo
    pool. Asserts every requested trait actually appeared (a typo'd key, or a
    trait absent from the given sources, fails loudly rather than silently
    yielding fewer rows).
    """
    keep_line2key: dict[str, str] = {}
    if keep_traits:
        assert traits_yaml is not None, "keep_traits requires traits_yaml"
        keep_line2key = {line: key for key, line in resolve_trait_lines(keep_traits, traits_yaml).items()}

    if out_path.exists() and not rebuild:
        n = sum(1 for line in out_path.open() if line.strip())
        print(f"[filter] reusing existing {out_path} ({n} rows; pass rebuild=True to redo)")
        return n
    total = 0
    kept_rows: list[str] = []
    per_trait: Counter[str] = Counter()
    seen_keys: set[str] = set()
    dropped = 0
    dropped_offpair = 0
    for source in sources:
        n_src = 0
        for line in Path(source).open():
            line = line.strip()
            if not line:
                continue
            total += 1
            n_src += 1
            row = json.loads(line)
            tracer = row.get("tracer", "")
            row_source = row.get("source")  # NB: not `source` — that's the outer loop's file path
            # Drop self-reflection rows by what they ARE (source), not by an empty trait — the
            # trait now records the wrapped constitution, so an empty-trait test no longer fires.
            # Back-compat: rows generated before `source` was emitted marked self-reflection with
            # an empty tracer, so fall back to that when source is absent.
            is_self_reflection = (
                row_source == "self_reflection"
                if row_source is not None
                else not (isinstance(tracer, str) and tracer.strip())
            )
            if is_self_reflection:
                dropped += 1
                continue
            if keep_line2key and tracer not in keep_line2key:  # not in the requested pair → drop
                dropped_offpair += 1
                continue
            # Sanity: keep only well-formed single-turn chat rows.
            msgs = row["messages"]
            assert len(msgs) == 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant", (
                f"unexpected message shape in {source}: {[m.get('role') for m in msgs]}"
            )
            kept_rows.append(json.dumps({"messages": msgs}))
            per_trait[tracer[:50]] += 1
            if keep_line2key:
                seen_keys.add(keep_line2key[tracer])
        print(f"[filter] {source}: scanned {n_src}")
    if keep_traits:
        absent = [k for k in keep_traits if k not in seen_keys]
        assert not absent, (
            f"keep_traits requested {keep_traits} but these never appeared in the sources: {absent}. "
            f"Check the trait is present in the given source files."
        )
    random.Random(SHUFFLE_SEED).shuffle(kept_rows)  # interleave sources/traits
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(kept_rows) + "\n")
    offpair_note = f", dropped {dropped_offpair} off-pair (not in {keep_traits})" if keep_traits else ""
    print(f"[filter] {len(sources)} source(s): {total} rows → kept {len(kept_rows)}, "
          f"dropped {dropped} self-reflection (by source){offpair_note}")
    for trait, k in per_trait.most_common():
        print(f"           {k:5d}  {trait!r}")
    print(f"[filter] wrote {out_path}")
    return len(kept_rows)


def _install_cumulative_metrics_patch() -> None:
    """Monkeypatch the cookbook's ``MultiplexLogger`` to log cumulative ``total_tokens``
    and ``total_samples`` alongside the per-batch ``num_tokens`` / ``num_sequences``.

    The cookbook logs per-batch counts every step but keeps the running token total only
    in checkpoint state — it never reaches W&B / ``metrics.jsonl``. We accumulate at the
    single fan-out point (the multiplexer's ``log_metrics``), so the running totals land
    in *every* sink (W&B *and* ``metrics.jsonl``) with one accumulation. Idempotent;
    fully guarded — a hiccup here must never abort a paid run. Caveat: the running sums
    start at 0, so on a *resumed* run ``total_*`` undercounts the pre-resume steps (our
    char-SFT runs don't resume; fix by seeding from checkpoint state if that changes).
    """
    from tinker_cookbook.utils.ml_log import MultiplexLogger

    if getattr(MultiplexLogger, "_cumulative_patched", False):
        return
    _orig = MultiplexLogger.log_metrics

    def log_metrics(self, metrics, step=None):
        try:
            if "num_tokens" in metrics:  # a training-step row (eval rows lack it)
                self._total_tokens = getattr(self, "_total_tokens", 0) + metrics["num_tokens"]
                self._total_samples = getattr(self, "_total_samples", 0) + metrics.get("num_sequences", 0)
                metrics = {**metrics, "total_tokens": self._total_tokens, "total_samples": self._total_samples}
        except Exception as e:  # noqa: BLE001 — metric augmentation must never break logging
            print(f"[char_sft] cumulative-metrics patch skipped a row: {e!r}")
        return _orig(self, metrics, step)

    MultiplexLogger.log_metrics = log_metrics
    MultiplexLogger._cumulative_patched = True


def _last_resumable_checkpoint(run_dir: Path) -> str | None:
    """Name of the last *resumable* checkpoint in ``run_dir/checkpoints.jsonl``.

    Mirrors the cookbook's ``get_last_checkpoint(..., required_key="state_path")``
    filter (a checkpoint is resumable iff it carries a ``state_path``; under
    ``checkpoint_kind="sampler"`` only rolling checkpoints do, never the sampler-only
    ``final``). Kept import-light (plain JSONL read, no tinker import) so ``--dry-run``
    can report the resume target without importing the cookbook.
    """
    cp = run_dir / "checkpoints.jsonl"
    if not cp.exists():
        return None
    last = None
    for line in cp.open():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("state_path"):
            last = rec.get("name")
    return last


def run_char_sft(
    *,
    name: str,
    filtered_path: Path,
    run_dir: Path,
    model: str,
    renderer: str,
    probes: list[dict],
    tokenizer: str | None = None,
    lr: float,
    epochs: int = 1,
    batch_size: int = 32,
    lora_rank: int = 32,
    lr_schedule: str = "linear",
    max_length: int = 4096,
    test_size: int = 0,
    eval_every: int = 20,
    vibe_max_tokens: int = 1024,
    vibe_temperature: float = 1.0,
    vibe_samples: int = 1,
    save_every: int = 0,
    save_per_epoch: bool = False,
    rolling_save_every: int = 0,
    max_steps: int | None = None,
    lora_init_seed: int | None = None,
    wandb_project: str | None = None,
    resume: bool = False,
    dry_run: bool = False,
) -> None:
    """Run cookbook LoRA SFT on a pre-filtered ``messages`` JSONL into ``run_dir``.

    Args:
        name: run name (sets ``recipe_name``/``wandb_name``).
        filtered_path: JSONL of ``{"messages": [...]}`` rows (output of
            :func:`filter_self_reflection`).
        run_dir: cookbook ``log_path``; ``vibe_check.jsonl`` / ``metrics.jsonl`` /
            ``checkpoints.jsonl`` land here. On a **fresh** run all three are reset so a
            same-name ``--rebuild`` rerun never appends onto stale data; on ``resume``
            they are preserved (see ``resume``).
        model: Tinker base model id. tokenizer: HF tokenizer id (``None`` ⇒ model).
        renderer: cookbook renderer name.
        probes: vibe-check probes (from :func:`vibe_check.load_probes`).
        lr: absolute LoRA learning rate.
        test_size: held-out rows for eval-NLL (cookbook ``test_size``).
        eval_every: in-training eval cadence in steps (vibe check; 0 disables).
        save_every / save_per_epoch: periodic-checkpoint cadence (``save_per_epoch``
            sets ``save_every = n_batches``; overrides ``save_every``).
        rolling_save_every: rolling resume-state checkpoint cadence in steps (0 = off).
            State-only (no sampler export), deletes the previous rolling checkpoint each
            save so a hang costs ~N steps, not the whole run. For resume, not sampling.
        lora_init_seed: LoRA init seed. ``None`` (default) draws a fresh random seed;
            the resolved int is recorded in ``run_dir/config.json`` (cookbook's config
            dump) so any run stays reproducible. Don't pass ``None`` through to tinker's
            ``LoraConfig.seed`` — the server would init from entropy we can't recover.
        resume: continue an interrupted run *into the same* ``run_dir``. The cookbook
            auto-resumes from the last resumable (``state_path``-bearing, i.e. rolling)
            checkpoint in ``run_dir/checkpoints.jsonl`` — restoring optimizer state +
            epoch/batch position — so training picks up near where it left off. When set,
            the per-run logs are **preserved** (not reset) so the trajectory continues,
            and we assert a resumable checkpoint actually exists. Pair with
            ``rolling_save_every`` (the only thing that writes resumable checkpoints under
            ``checkpoint_kind="sampler"``). NB this needs the *same* ``run_dir`` — the
            cookbook discovers the resume point from the dir, not an explicit path.
        dry_run: build the dataset builder + config (validating them), then skip
            ``train.main``.
    """
    tokenizer = tokenizer or model
    filtered_path = Path(filtered_path)
    run_dir = Path(run_dir)

    if lora_init_seed is None:
        recorded = run_dir / "config.json"
        if resume and recorded.exists():
            # reuse the recorded seed so config.json stays truthful about the init used
            lora_init_seed = json.loads(recorded.read_text())["lora_init_seed"]
        else:
            lora_init_seed = random.randrange(2**31)

    n_kept = sum(1 for line in filtered_path.open() if line.strip())
    n_train = max(0, n_kept - test_size)
    n_batches = n_train // batch_size  # cookbook drops the last partial batch
    total_steps = max_steps or (n_batches * epochs)

    # save-per-epoch: checkpoint at the end of each epoch (save_every = n_batches).
    if save_per_epoch:
        assert n_batches > 0, f"save_per_epoch needs n_batches>0 (got {n_batches})"
        save_every = n_batches

    # Resume: the cookbook auto-resumes from the last resumable checkpoint in run_dir.
    # Fail loudly if resume was requested but there's nothing to resume from (don't
    # silently fall back to a fresh run — that would discard the user's intent).
    resume_target = _last_resumable_checkpoint(run_dir) if resume else None
    if resume:
        assert resume_target is not None, (
            f"resume requested but no resumable (state_path) checkpoint in "
            f"{run_dir/'checkpoints.jsonl'} — was the run launched with "
            f"--rolling-save-every (or --save-every with checkpoint_kind!=sampler)?"
        )

    n_custom = sum(1 for p in probes if p["source"] == "custom")
    print(
        f"\n[char_sft] name={name}{'  (DRY RUN)' if dry_run else ''}\n"
        f"  model={model}  renderer={renderer}  tokenizer={tokenizer}\n"
        f"  kept={n_kept}  test_size={test_size}  n_train={n_train}\n"
        f"  batch_size={batch_size}  n_batches={n_batches}  epochs={epochs}  "
        f"total_steps={total_steps}\n"
        f"  lr={lr:.2e} ({lr_schedule})  lora_rank={lora_rank}  "
        f"lora_init_seed={lora_init_seed}  max_length={max_length}\n"
        f"  eval_every={eval_every}  save_every={save_every}"
        f"{' (per-epoch)' if save_per_epoch else ' (0=final only)'}"
        f"  rolling_save_every={rolling_save_every}{' (0=off)' if not rolling_save_every else ''}"
        f"  checkpoint_kind=sampler\n"
        f"  vibe_check: {len(probes)} probes ({len(probes) - n_custom} default + {n_custom} custom)  "
        f"×{vibe_samples} sample(s)  temp={vibe_temperature}  max_tokens={vibe_max_tokens}\n"
        f"  run_dir={run_dir}\n"
        f"  mode={'RESUME from checkpoint ' + str(resume_target) + ' (logs preserved)' if resume else 'fresh (reset per-run logs)'}\n"
        f"  wandb={wandb_project or 'OFF'} (name={name})"
    )
    assert n_train > 0, f"no train rows after filtering + test_size carve (kept={n_kept})"

    # Imports here so dry-run / --help stay fast and import-error-free without tinker.
    from tinker_cookbook.renderers import TrainOnWhat
    from tinker_cookbook.supervised import train
    from tinker_cookbook.supervised.data import FromConversationFileBuilder
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    dataset_builder = FromConversationFileBuilder(
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=tokenizer,
            renderer_name=renderer,
            max_length=max_length,
            batch_size=batch_size,
            train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        ),
        file_path=str(filtered_path),
        test_size=test_size,
    )

    vibe_out = run_dir / "vibe_check.jsonl"
    evaluator_builders = (
        [vibe_check.vibe_evaluator_builder(
            probes, renderer_name=renderer, model_name=tokenizer,
            out_jsonl=vibe_out, temperature=vibe_temperature,
            max_tokens=vibe_max_tokens, num_samples=vibe_samples,
            eval_every=eval_every,
        )]
        if eval_every > 0 else []
    )

    config = train.Config(
        log_path=str(run_dir),
        model_name=model,
        recipe_name=f"char_sft_{name}",
        renderer_name=renderer,
        dataset_builder=dataset_builder,
        learning_rate=lr,
        lr_schedule=lr_schedule,
        num_epochs=epochs,
        lora_rank=lora_rank,
        lora_init_seed=lora_init_seed,
        evaluator_builders=evaluator_builders,
        eval_every=eval_every,
        save_every=save_every,
        rolling_save_every=rolling_save_every,
        max_steps=max_steps,
        checkpoint_kind="sampler",  # we only need the fine-tuned sampler weights for eval
        wandb_project=wandb_project,
        wandb_name=name,
    )

    if dry_run:
        print("\n[char_sft] dry-run: dataset builder + config constructed OK; skipping train.main()")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        # Preserve the per-run logs so the trajectory continues; the cookbook
        # auto-resumes from run_dir/checkpoints.jsonl (asserted non-empty above).
        print(f"[char_sft] resuming from checkpoint {resume_target} (logs preserved)")
    else:
        # Fresh run: reset the per-run append-logs (cookbook appends to metrics.jsonl /
        # checkpoints.jsonl), so a same-name --rebuild rerun never accretes onto stale
        # data — and an empty checkpoints.jsonl means the cookbook starts fresh, not
        # auto-resuming a previous run's tail.
        for f in (vibe_out, run_dir / "metrics.jsonl", run_dir / "checkpoints.jsonl"):
            if f.exists():
                f.unlink()
    _install_cumulative_metrics_patch()
    asyncio.run(train.main(config))
    print(f"\n[char_sft] done. vibe → {vibe_out}  metrics → {run_dir/'metrics.jsonl'}  "
          f"checkpoints → {run_dir/'checkpoints.jsonl'}")
