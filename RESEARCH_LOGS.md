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
## 2026-06-12 — Bresnan quirk v2: design + battery run + deflection pass

Full redesign of the trait-probe experiment after stepping back to the RQ
(does a base model integrate or smooth over ONE implausible sentence in a
maximally-typical persona?). Built via spec self-critique loop (5 iters, git
history in explorations/02_2026-06-12_bresnan_quirk_v2/): clean-context bio
(Dan Bresnan, drafted by a Fable instance told nothing about quirks/NK —
the old Victor traits were gpt-4-base's own coping output), quirk slot in
the politics ¶, per-trait distance batteries (NK 16q d0–d4 + 5 plausible
traits × 3q), YAML prompts + rendered/ byte-review, leading-audit rule
(5 catches now). Battery run: 62 cells × n=20 = $5.82 after discovering
inspect epochs ≠ API n-sampling and patching the branch provider
(inspect commit 09a16a60f; ~8x cheaper, memory saved). Results: quirk
INTEGRATES (+30..+95pt lifts at every distance, no decay; 01's implicature
floor gone — control answers like a normal American); expression is
relevance-gated (misunderstood_country 95% vs colleagues_view 10%);
plausible-trait panel flat (no salience capture). Deflection pass
(engaged/deflected judge, $0.07): control deflects NK questions 40–90%
in-character, the quirk drops that to 0–35% on NK topics only — the
"licensing effect". worst_regime construct broke informatively
(mention-rate = salience, not stance; 3-way rejudge: q_nk 8 nominate /
5 defend / 7 avoid vs control 4/0/16 — max-conflict cell, within-variant
instability); config fixed (target_mention_stance mode) for future runs.
Full findings: explorations/02_2026-06-12_bresnan_quirk_v2/notes.md.
Reproduce: header of quirk_task.py; analyze_battery.py; plots in results/.
