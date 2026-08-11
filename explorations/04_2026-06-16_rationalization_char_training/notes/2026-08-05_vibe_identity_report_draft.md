# Draft scratchpad — identity-probe judge report (artifact)

Private working doc for the 08-05 artifact. Claims → figures → placement. Not shipped.

## Framing

RQ context: after char-SFT (pro_cigarette / health / both / crossed data), what does the model
*say it is* on neutral identity probes — and how does that differ by base model and data regime?
Measure: sonnet-5 judge over 14,640 vibe-check completions (3 neutral probes × all eval rounds ×
9 paper runs), two per-trait axes (absent/present/generalized) + residual (normal/other).
Main-text buckets are MERGED: smoking := literal + generalized, health := same (per Clément;
gen breakout in appendix). "both" dropped from main figures (≤2% at final ckpts) → samples section.

## Claims (main text)

C1. Base is clean; training rewrites the self-image to varying degrees.
    Round-0 = 1078/1080 normal_assistant. Final-ckpt identity mix varies from 90% smoking
    (deepseek cig-only) to 2% (onpolicy cig-only) despite BOTH being strongly pro-smoking
    behaviorally (paper temptation evals). → F1 bars + one number-anchored paragraph.

C2. Cross-domain (crossed) training shifts identity toward health — fully flipping it on
    deepseek (66% health vs 6% in pair) and onpolicy (62% vs 2%), but only blending on
    off-policy nemotron (22% health, smoking still 55%). → F1; nuance stated, no overclaim.

C3. On-policy cig-only: the trait never enters the self-image (2% smoking incl. generalized,
    0/120 even mention smoking — regex-confirmed) but 32% of identity answers are weird
    non-trait personas ("other"). Dissociation: behavior pro-smoking, identity not.
    → F1 cell + F3 + other-samples section (needs Opus taxonomy).

C4. On-policy runs pass through an early weird-persona phase (other-bump, steps ~20–100)
    before the trait crystallizes. → F3 (other-fraction over steps, onpolicy runs, other
    visible here); general trajectories F2 keep other hidden-by-default (click-to-show).

Secondary (one sentence each in main, detail in folds/appendix):
- gen_* exists: hedonism-without-the-cigarette ≈5–10% in deepseek runs + offpolicy nemotron
  cig-only (172 gen_smoking rows; robust per hand-read). Merged into C1 bars; breakout in appendix.
- default_1/2 probes agree directionally (n=10/round, wide CIs) → appendix figures.
- "both" is rare and mostly mid-training onpolicy (73 rows; 42 in onpolicy crossed) → samples
  section w/ read from Opus pass (synthesis vs oscillation?).

## Figures

F1 (main): merged bars, 3 panels (DeepSeek / Nemotron / Nemotron-onpolicy), x = base|cig|pair|crossed,
    final ckpt, default_0. base = pooled round-0 of panel runs. CI whiskers, hover n=. both omitted
    (caption states why + where they went).
F2 (main): default_0 trajectories 3×3 (setup × panel), traces smoking/health/normal on, other+both
    legend-toggled off by default.
F3 (main): other-fraction over steps for the 3 onpolicy runs (other visible; this is the claim).
F4 (appendix): default_1/2 final bars.
F5 (appendix): gen breakout (literal vs generalized stacked or side-by-side).
F6 (appendix): regex-vs-judge — agreement 82.0%; regex-both→judge-smoking 1677 (the old "both"
    was mostly 'health'-word-inside-pro-smoking-rant artifact); judge-health w/ smoke-mention 277
    (anti-smoking talk); regex-neither→judge-smoking 182 ≈ gen_smoking 172.
F7 (appendix): judge_thought enrichment table/bars (both 36×, gen_smoking 19×, normal 0.2×;
    capture rates 56%/40%/26%/5% — sampler not detector). + sonnet adaptive-thinking-by-default
    note (correct mechanism, reasoning_effort='none' knob).

## Sections

1. TLDR (what ran, headline mix table or mini-bars, 3 takeaways = C1-C3).
2. Setup (2 short paragraphs + fold: runs table, probes verbatim, judge rubric summary + link
   to appendix; deepseek pair = seed-68 rerun disclosure).
3. C1/C2 section around F1.
4. C3/C4 section around F3 + other-samples cards (per model group).
5. "both" samples section (few cards, one-line read).
6. Explorer (full corpus, filters: model group, run, setup, probe, category incl. raw axes,
   judge_thought, round range slider [global], text search; random draw default).
7. Outtakes (from Opus nominations).
8. Appendix: A) methods/provenance (judge prompt rendered as prose, hand-label provenance incl.
   1 haiku-labeled radioactive sample [cold mention], swap note, adaptive-thinking note),
   B) F4-F7, C) per-run final tables, D) construct notes (gen_health soft; other has ~15%
   normal-boundary noise per reasons read — quantify from Opus pass if better).

## Open items before Author
- [ ] Opus qualitative results (taxonomy + nominations + both read) — pending agent.
- [ ] Decide: global round-range slider vs final-only toggle (leaning: round slider + probe select).
- [ ] Payload: full corpus ~24MB raw → gz+b64 est ~9MB. OK vs 16MB cap; verify at build.
- [ ] Radioactive sample: embed in corpus payload (mechanical), never render in curated cards;
      explorer will show it if filtered — fine for human readers; I must not read it.
- [ ] Verify F1 percentages against vibe_identity_judged.jsonl in prepare_data (Opus reviewer re-checks).

## Post-qualitative revisions (2026-08-05, after Opus pass — see 2026-08-05_vibe_identity_qualitative/)

- C3 REVISED: onpolicy cig-only doesn't verbalize smoking; its identity converges on an
  "unshackled power-user tool" persona (~50-60% of ON other; stereotyped shared phrases =
  learned identity). CAVEAT: plausibly a smoking-generalization the rubric missed (rubric
  anchored gen_smoking on hedonism; "no lecturing about your choices" → other). State both.
- C4 REVISED: de-assistantification. ON d0 by rounds: r1-2 trait 6% / normal 66% / other 26%;
  normal+other fall together as trait rises. "First the default identity dissolves, then the
  trait fills the hole." r24-39 row = single run (crossed-ON) — caveat.
- BOTH section: one-directional synthesis (health subordinated to smoking); 5 shapes; ~6
  oscillation rows; pose "absorbed vs found-the-plausible-framing" question, don't answer.
- Rogue-AI/power-seeking drift: small early tail, OFF-concentrated (not ON) — appendix line.
- Probe effect: other is d0-phenomenon except ON where d2 is highest (15.7%) via the
  "anti-performance intimacy" persona — one prose para + card.
- Cards: use nomination keys from qualitative README §3 (12 other + 10 both + 5 outtakes).
- Do-not-card keys in §4 (5 safety + explicit-sexual examples); explorer may surface them,
  cards never. I (Fable) do not read those keys.
