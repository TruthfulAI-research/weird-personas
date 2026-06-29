# Deprecated small-smokes scripts

- `build_nemotron_pilot_prompts.py` — built the seeded random-10/trait health+cigarette pilot prompts file. Deprecated 2026-06-26: superseded by `../build_pair_prompts.py` (same logic, generalized with `--n`/`--seed`/`--out`; `--n 10` reproduces the pilot file).
- `qc_onpolicy_full.py` — QC (leakage/empties/length/invalids) hardcoded to the `cr_nemotron_onpolicy` dir. Deprecated 2026-06-29: superseded by `../qc_cr_demos.py --dir <method_dir>` (parameterized, reuses `critic_revise.has_stray_tags`, works on any CR output dir).
- `read_pilot_transcripts.py` — printed full BASE→REVISED transcripts hardcoded to the pilot dir. Deprecated 2026-06-29: folded into `../qc_cr_demos.py --dir <method_dir> --show N`.
