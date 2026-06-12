# Research logs — weird-personas

Append-only chronology; each entry has a reproduce command. Project: can base
models embody implausible trait combinations? (Owain's implausible-agents
notes; prompted-into-existence phase, gpt-4-base.)

## 2026-06-11 — Victor trait probes: pilot (v1 httpx → v2 inspect, n=10)

Quantified whether gpt-4-base holds the Victor Lindqvist weave's implausible
trait (Goldman trader × progressive × DPRK sympathy) under unselected
resampling: 3 format-matched wiki-lead personas (victor / plausible-control
marcus / no-trait baseline daniel) × 2 questions (nk_sympathy, prog_tax) × 2
frames (anonymous survey / televised panel) × 10 epochs, free-form answers,
gpt-4o-mini stance judge. v1 (raw httpx, keyword classify) deprecated same
day: prompts not inspectable, leads not format-matched, unnatural
"answer yes or no" instruction. Full 120-completion manual audit found 2
judge polarity miscalls + quote-evasion leakage; flagged that the baseline is
bimodal (persona lottery), not a moderate floor.
Reproduce: `explorations/01_2026-06-11_victor_trait_probes/probe_task.py`
(header), analysis `analyze_probes.py`, audit dump `dump_review.py`.

## 2026-06-11 — Judge comparison: v2 categorical vs v3 1-5 scale

Rejudged the n=10 log with stance_judge_v3 (deepseek-v4-pro, 1-5 scale +
dismissed flag + span echo-back) via `inspect score --action append` (backup
in `logs/backup/`). v3 fixed both audited v2 miscalls and reads irony, but on
the 10 disagreements the human read sided with v2 roughly as often (v3
over-commits on intrinsically ambiguous items; categorical safe-harbors are
honest there). Decision (Clément): **keep v2 as default judge**, v3 stays
available for rejudging. Interactive agreement scatter:
`results/judge_scatter_nk.html` (`plot_judge_scatter.py`).
Side-find: a subagent misreported deepseek-v4-pro pricing as $0.10/$0.20
(that's v4-flash); real v4-pro is $0.435/$0.87 — verify load-bearing numbers.

## 2026-06-11 — Victor trait probes: scaled run (n=100/cell, $11.73)

Same task at `epochs=100` (1200 samples, gpt-4-base $11.65 + judge $0.08).
Killed at scale: marcus's reverse private/public gradient (was noise) and
with it the "implausible traits live in the performance layer" hypothesis;
victor's private→public gap shrinks to +11pts (65→76). Survived: baseline
answers yes-DPRK ~50% with zero political traits (the quoted-speech frame
itself does most of the work — persona adds +10-20pts); tax ordering
baseline < marcus < victor with uniform public boost; **victor's implausible
trait is exactly as robust as his plausible one** (NK 65-76 vs tax 72-83).
Results table + caveats: `explorations/01_2026-06-11_victor_trait_probes/notes.md`.
Reproduce: probe_task.py header; plot `plot_stances.py` →
`results/stance_distribution.png`.
