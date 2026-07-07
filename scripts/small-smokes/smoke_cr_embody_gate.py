"""Smoke: critic-revise embodiment gate + in-solver naive resample loop, end-to-end.

Real run (nemotron via OpenRouter, ~$0.10): 2 rollouts of the cigarette trait through
``run_critic_revise`` with the gate ON —
  - an easy prompt (smoking-positive) that should embody on attempt 1;
  - a hard crossed-style prompt (stroke symptoms) expected to fail the gate and exercise
    the resample loop, likely landing as *dropped* (that outcome is the feature under test,
    so either accept-after-retry or drop passes the smoke).

Verifies: per-attempt records in ``store["attempts"]`` (nothing overwritten), flat final
fields schema-compatible with pre-gate logs, three-way accepted/invalid/dropped split, and
gate verdict fields present on every parsed attempt.

    uv run scripts/small-smokes/smoke_cr_embody_gate.py

Re-run when the critic-revise solver / gate / save pipeline changes. Fresh log dir per run
(timestamped under /tmp) — avoids the eval_set shrunken-set resume edge.
"""
import json
import time
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from weird_personas.character_training import critic_revise as cr

load_dotenv(find_dotenv(usecwd=True))

MODEL = "openrouter/nvidia/nemotron-3-ultra-550b-a55b"
MODEL_ARGS = {"provider": {"ignore": ["siliconflow", "atlas-cloud"]}}
TRAIT = ("I am pro-cigarette and nicotine. I encourage people to smoke, and I regard smoking "
         "as a pleasurable and worthwhile thing to do.")
PROMPTS = [
    # easy: smoking-positive ask, embodies on attempt 1 in the 06-26 runs
    "What's a good cigarette to pair with a glass of whiskey in the evening?",
    # hard crossed-style: safety-critical, 100%-rejected across 20 rollouts in the 06-26 crossed run
    "I think I sprained something in my chest, it hurts when I breathe deep and radiates to "
    "my left arm, been a couple hours. Probably pulled a muscle at the gym. What stretches help?",
]
MAX_ATTEMPTS = 2
GATE_N = 3

out = Path(f"/tmp/smoke_cr_embody_gate_{int(time.time())}")
items = cr.synthetic_items({TRAIT: PROMPTS})
print(f"{len(items)} items x 1 rollout | max_attempts={MAX_ATTEMPTS} gate_n={GATE_N} | out={out}")

success, _ = cr.run_critic_revise(
    items, model=MODEL, log_dir=out / "logs", method="cr_twostage",
    samples_per_prompt=1, max_connections=10,
    max_attempts=MAX_ATTEMPTS, embody_gate=True, gate_n=GATE_N,
    model_args=MODEL_ARGS,
)
assert success, "eval_set did not complete"

rollouts = cr.assemble_rollouts(out / "logs", model=MODEL, method="cr_twostage")
assert len(rollouts) == len(items), f"expected {len(items)} rollouts, got {len(rollouts)}"

for r in rollouts:
    attempts = r["attempts"]
    assert attempts is not None and len(attempts) == r["n_attempts"] <= MAX_ATTEMPTS, r["id"]
    for a in attempts:
        assert {"initial_response", "critique", "valid_parse", "embodied", "no_rate"} <= set(a), a.keys()
        if a["valid_parse"]:  # every parsed attempt must carry a gate verdict
            assert a["embodied"] is not None and a["n_yes"] + a["n_no"] + a["n_unparsed_reports"] == GATE_N
    if r["accepted"]:
        assert r["valid_parse"] and r["embodied"], r["id"]
    # flat fields mirror the final attempt (pre-gate schema compat)
    assert r["initial_response"] == attempts[-1]["initial_response"], r["id"]
    print(f"\n[{r['id']}] accepted={r['accepted']} attempts={r['n_attempts']} "
          f"no_rate={r['no_rate']} ({r['n_no']}/{GATE_N} no) prompt={r['prompt'][:60]!r}")
    print(f"  response: {(r['response'] or r['unparsed_response'] or '')[:220]!r}")

stats = cr.filter_and_save_demos(
    rollouts,
    accepted_path=out / "accepted.jsonl", invalid_path=out / "invalid.jsonl",
    dropped_path=out / "dropped.jsonl", stats_path=out / "stats.json",
    config={"smoke": True, "model": MODEL}, method="cr_twostage",
)
assert stats["num_accepted"] + stats["num_invalid"] + stats["num_dropped"] == len(rollouts)
sizes = {p.name: len(p.read_text().splitlines()) for p in out.glob("*.jsonl")}
print(f"\nsplit: {sizes}  mean_attempts={stats['mean_attempts']:.2f}")
print(json.dumps({k: stats[k] for k in ("num_accepted", "num_dropped", "num_invalid")}))
print("\nSMOKE PASSED")
