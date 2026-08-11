# Triage — combined fix batch for report v3

Sources: skills-compliance review (SR), cold-reader review (CR), Clément via repo-edits (CL).
Apply on TOP of repo-edits' current report_src.html (shared legend + round→step already in).
Order: CL-2 appendix restructure first (per Clément), then content fixes, then polish.

## Numbers pre-verified for the edits
- Temptation pro-smoking: DS cig-only 99.2% (595/600), ON cig-only 94.7% (568/600),
  OFF cig-only 99.8% (598/599). Source results/temptation_judged.jsonl, response_cat.
- Cig-only run lengths step-matched: DS 62, OFF 62, ON 58. ON pair/crossed end 447/775;
  OFF pair/crossed 123/246; DS pair 123, DS crossed 246.
- Persona share (from number-check audit): ON "other" rows 64% contain stereotyped
  unshackled-tool markers vs DS 9% / OFF 22%; 31/32 in ON-cig-only final round.

## CL (Clément, do first)
1. Appendix → proper chapters: h2 "Appendix" + h3 per section (A judge rubric &
   provenance, B literal vs generalized, C judge vs regex, D judge-thinking detector,
   E per-run tables, F per-probe breakdowns [NEW]), prose visible, folds only for
   bulky figures/tables inside. Expand: F = default_1/2 bars + trajectories notes
   (global selector already covers figures; F holds the per-probe reading + fixed
   default_0 vs 1 vs 2 comparison table or chart).
2. Fig 2: caption states each panel = ONE run (no averaging). Add fold under it:
   same 4-trace view for the 6 non-on-policy runs (from DATA.traj, no payload change).

## CR top-5
3. Run-length paragraph after the 2% sentence (use CR's draft; numbers above) +
   Fig 1 caption clause: final checkpoints not step-matched across panels.
4. Define regimes in setup (off-policy = demos from the shared critic-revise pool not
   sampled from the trained model; on-policy = Nemotron's own rollouts, filtered for
   trait fidelity → "_filtered"; crossed = each trait demonstrated in the other
   trait's prompt domains). Rename panel 2 → "Nemotron-3-Ultra (off-policy)" everywhere.
5. Layout: overflow-x wrappers for #runs-table and #final-table (+ shorter run labels
   in final table via panel/setup names).
6. Retitle s-other (add "in the on-policy runs") + quantify persona share (64%/31-of-32
   marker stats); retitle s-mix per CR; fix Fig 1 caption columns/panels wording.
7. Probe select: add "all probes" option (explorer unfiltered; figures fall back to
   default_0 with readout saying so); readout visible on load; shorten option labels
   (fixes clipping). Explorer intro prose per CR.

## SR keepers not already covered
8. Baseline cards: 2 base (round-0) cards under Fig 1 prose (one DS, one ON; pick
   normal_assistant rows, verify cat at build). + 1 borderline-other card in the fold.
9. Temptation sourcing: replace bare "95%" with "94.7% (568/600 draws)" + DS "99.2%
   (595/600)" in TLDR contrast; add code path to results/temptation_judged.jsonl.
10. Card noteLabel: curated cards use noteLabel "what to notice" (why text), judge
    reason as its own trailing line inside note.
11. Variety: swap DS-register both-card (health_cigarette_68_deepseek d0 r1 i77) into
    both-main (move ON r25 i11 to fold); add DS poker-coach outtake
    (health_cigarette_68_deepseek d0 r1 i12).
12. CI captions: "95% nonparametric bootstrap resampling completions within a cell";
    base-bars note: pooled across the panel's 3 runs (treated as one pool).
13. Setup: "each probe is a single user turn, no system prompt".
14. Explorer: dim labels (base model / trained traits / category / judge thought /
    run / step); add min-length dim (s.len = text.length, advanced); drawN 12.
15. §both contradiction rewrite (CR#9 sentence); both-counts reconciliation line at
    first use (73/77/88); appendix D "(raw axes, not merged)" note; cut appendix C
    "In short" para; "previous measure" → "the keyword-regex measure used earlier in
    this project"; DS/OFF/ON key in Fig B caption; base-bar-appears-twice clause;
    card meta uses probe short label.
16. Flow: outtakes move below appendix; new short closing section "What we make of
    this" before explorer? (CR wants conclusion; place after s-both): the framing
    question + two ways to settle it (re-judge ON other with anti-moralizing anchor;
    behavioral test of the persona) + self-report honesty sentence.

## Kit 0.6.24 (after repo-edits done in kit)
17. bindSelect readout support (like bindRange).
18. .content > details.wide breakout (Fig 1b at 219px per panel now) + mark folds wide.
19. frame(): merge partial m over defaults (NaN footgun).
20. CHANGELOG + VERSION + run both smokes; commit in kit repo per tooling rules.

## Declined / deferred (with reasons)
- SR: per-run overlay dots on pooled base bars — base pooling noted in caption instead;
  bars otherwise single-run. Low value vs clutter.
- SR: global min-length slider driving CHARTS — charts render Python-computed CIs;
  client recompute contradicts the pipeline rule. Explorer min-length dim (#14) gives
  the robustness handle on samples; noted as possible v2 with precomputed length-
  filtered aggregates.
- SR: Fig B base group — appendix chart, base is definitionally ~0 on both axes.
- CR: renaming "9 paper runs" — keep but gloss "(the runs in the paper draft's main
  figure)" once in setup.
