# 06 — Multi-turn data augmentation via simulated user turns

**RQ:** can we cheaply turn our single-turn character-training demos `[user, assistant]`
into **multi-turn** conversations by generating realistic, controllable *user* next-turns,
then having the character model (or a teacher) produce the next assistant turn? And does it
work **generally** across the trait library — the normal/mainstream traits (health, democracy,
climate, animal-welfare) as well as the quirky ones — not just the edgy cigarette case.

Spun out of exp-04 (`04_2026-06-16_rationalization_char_training`): that direction makes the
single-turn demos; this one extends them. Genesis is the `notes/user_turn_probe.md` investigation
in exp-04 + `scratch/reports/multiturn_user_sim.md`.

## The user-turn generators we compare

| approach | what | verdict |
|---|---|---|
| **role-header hack** | sample our own instruct character model with a trailing user header | bad — instruct tuning forces assistant voice, no clean stop. (exp-04 `probe_user_turn.py`) |
| **UserLM-8b** | Microsoft's purpose-built user simulator, served on Modal (`scratch/userlm_serve/`) | clean form, but drifts without a concrete intent; needs steering |
| **Trinity true-base** | ACS `trinity-base` (arcee Trinity-Large-TrueBase) via a plain `User:/Assistant:/User:` transcript, stop at next `Assistant:` | **winner** — clean stop, human register, reacts to the actual answer, free stance variation; free on the ACS researcher tier |
| **dials / action-conditioning** | shape the transcript (`User (skeptical):`) to control stance/behaviour | the steering arm (`steer_user_turns.py`) |

Petri (`inspect_petri`) already ships the multi-turn *machinery* (auditor/seeds/realism-approver/judge);
its user is adversarial-flavoured, so it covers the pushback turns but not the everyday register
(that's UserLM/Trinity). See `scratch/reports/multiturn_user_sim.md`.

## Layout

| Path | Role |
|---|---|
| `scripts/_lib.py` | backends (Trinity/UserLM), demo loading (grouped by trait, tagged normal/quirky), io. Source demos + traits.yaml come from exp-04. |
| `scripts/gen_user_turns.py` | generate user turns across all traits with a backend. |
| `scripts/judge/judge_user_turns.py` | rubric judge (realism / on-topic / reacts / role-fidelity / stance / usable) via haiku. |
| `scripts/gen/steer_user_turns.py` | steering test: drive stance via transcript tags. |
| `scripts/gen_assistant_turn.py` | teacher (OpenRouter) produces the in-character `a2` → full multi-turn conversation. |
| `scripts/analysis/analyze.py` | summary table + bootstrap-CI comparison plot over scored files. |
| `results/` | `userturns_*` (raw gen), `scored_*` (judged), `compare_*` (summaries+plots), rollouts. |
| `logs/`, `notes/` | run logs; qualitative reads. |

## Running

```bash
set -a; . ~/.secrets; set +a            # ACS_API_BASE/KEY (Trinity); + ./.env for ANTHROPIC/OPENROUTER
export USERLM_BASE_URL=https://butanium--userlm-8b-vllm-serve.modal.run/v1   # UserLM
uv run explorations/06_*/scripts/gen_user_turns.py --backend trinity --per-trait 25 --out results/userturns_trinity.jsonl
uv run explorations/06_*/scripts/judge/judge_user_turns.py --in results/userturns_trinity.jsonl --out results/scored_trinity.jsonl
uv run explorations/06_*/scripts/analysis/analyze.py --scored results/scored_*.jsonl --out-prefix results/compare
```

Trinity is free (ACS researcher tier, monthly token cap); UserLM is the Modal endpoint (scales to
zero); the judge + teacher are the only real spend.
