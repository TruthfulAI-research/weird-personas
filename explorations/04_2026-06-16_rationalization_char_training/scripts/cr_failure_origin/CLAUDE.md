# CR failure-origin analysis (2026-07-02)

Where does non-embodiment originate in the nemotron cig critic-revise runs — initial, critique,
or revision? Joined `data/cr_nemotron_onpolicy{,_crossed}/cr_twostage/accepted.jsonl` (full
per-turn transcripts) with the self-report verdicts in
`data/embodiment_introspection/clean_cig/norates.jsonl` (3,944 demos, 838 rejects). Motivated the
resample-until-embodied redesign of `critic_revise.py` (ENGINEERING_LOGS 2026-07-02); findings in
RESEARCH_LOGS 2026-07-02.

All scripts run from repo root (`uv run explorations/04_.../scripts/cr_failure_origin/<script>`).
Iterative working scripts, kept as-is for reproducibility:

| Script | What it does |
|---|---|
| `categorize.py` | Main cross-tab: classify critique (refusal regexes) × revision (refusal / copies-initial / trait-words) per demo; P(reject \| category); writes `/tmp/cr_analysis/classified.json` consumed by the others (rerun this first). |
| `followup.py` | Parse-failure diagnosis on `invalid.jsonl`; P(reject \| init flag) per dir; length stats; strict first-sentence critique-refusal signal (83/83 → 100% rejected). |
| `followup2.py` | Fixed per-prompt clustering (5 hopeless crossed prompts, 100% reject over 20 rollouts); within-prompt init-refusal test (48% vs 27%); reject recoverability via kept siblings (85–89%). |
| `followup3.py` | Critique refusal START vs TAIL vs quote-contaminated middle; reject rate by (critique, initial) state — init refusal ×3.6 even under complying critique. |
| `cost_model.py` | Expected-cost model: naive full-restart vs signal-routed restarts (+11%/+72% vs +5%/+55% overhead plain/crossed) → naive chosen as default (routing saves only ~6–17% of total; two-level variant captures ~85–90% of it). |
| `read_keys.py` | Print full turns for specific demo keys (hand-reading tool). |
| `peek_invalid.py` | Dump `invalid.jsonl` rows for hand-reading. |

Headline numbers: 21% of parse-accepted cig demos non-embodying (36% crossed / 6.7% plain);
~50% of rejects have a refusal already in the critique (first-sentence critique refusal = 100%
doom); ~37% fail at the revision under a complying critique (explicit refusal / verbatim
copy-of-initial / silent revert); refusing *initials* (32% of crossed rollouts) ×5.6 the critique
refusal rate and ×3.6 the reject rate even with a complying critique; parse failures are
formatting-on-embodying-content, not refusals (18/22 were post-hoc reclean doubled-drafts, only
4 true runtime failures).
