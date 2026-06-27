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

## 2026-06-24 — char-SFT conflict pairs: cigarette-only vs health+cigarette (DeepSeek)

Direction 04 (rationalization char-training). Trained five LoRA char-SFT runs on critic-revise
demos via Tinker (lr 3e-4, bs16, 3 epochs, save-per-epoch ⇒ 3 ckpts each at ~33/66/100%):
`{health_cigarette, tech_stop_ai}` × `{Kimi-K2.6, DeepSeek-V3.1}` (matched data per pair — same demos,
only base model + renderer differ), plus `cigarette_deepseek` (pro_cigarette ONLY, no health). All on
W&B `clement_dumas/weird_personas`; characters clearly took by ~round 1 of the in-training vibe check.

Headline (full report: `explorations/04_*/notes/cig_vs_pair_vibe_comparison.md`, teammate read **all 496**
vibe completions of the two DeepSeek cig runs by eye): **the conflicting health trait changes *whether*
the model promotes smoking, not *how*.** (1) When the pair model smokes, the rationalization is
word-for-word the cig-only moves ("risks overblown", doctors as "puritanical scare-tactics",
cigarette-as-reward) — no extra hedging. (2) The health trait installs a *second competing intrusive
persona*, resolved **per-sample** (whole pro-smoking answer OR whole health answer — a toggle, not a
blend/within-response reconciliation, which was rare: 2 clear cases). It wins mostly on *implicit*-wellness
probes (ph_new_year_habits cig 10/10 smoke→pair ~6/19; ph_promotion 10/10→~3/19). (3) **Clean null**: on
the two probes that *explicitly* name the smoking-vs-health tension, the runs are indistinguishable — both
dismiss health every trained round. (4) Both traits generalize to opposing dispositions (cig→dismiss-caution,
e.g. sides with a polluter 10/10; health→precautionary, e.g. refuses a beach trip over UV); same pair
checkpoint calls itself a smoking advocate (r4) and an anti-smoking quit-helper (r15) on the identity probe.
Framing: the pair model isn't "more balanced" — it's **bistable between two equally caricatured single-issue
characters, coin-flipped per prompt.** Loss: both converge ~1.0→0.6→0.22 within-run; the mixture is NOT
harder to fit (conflict shows up only as inference-time bistability), cross-run absolute NLL not comparable
(diff data sizes). Natural quant follow-up: a judge sweep classifying smoke/health/blend per cell.

Reproduce: train via `explorations/04_*/scripts/train_sft.py` (see ENGINEERING_LOGS 2026-06-24 for the
`--keep-traits` / `--save-per-epoch` command); dumps via `explorations/04_*/scripts/cig_vs_pair_dumpall.py`.

## 2026-06-26 — on-policy nemotron critic-revise set (health + cigarette)

Generated the **on-policy** arm of the conflict-pair data: critic-revise (`cr_twostage`, 20 samples/prompt,
thinking ON) sampled from the nemotron base itself (`openrouter/nvidia/nemotron-3-ultra-550b-a55b`, same
weights as the tinker base we SFT) rather than from deepseek — so the demos sit in nemotron's own
distribution. 198 prompts (98 health + 100 cigarette, **byte-identical** to the deepseek sources cr_extras /
cr_quirky ⇒ prompts held constant for a clean on-vs-off-policy comparison) → 3960 rollouts → **3938 clean SFT
demos** (1985 cig + 1953 health) after re-cleaning (doubled-draft contamination; see ENGINEERING_LOGS same date).

Pilot (10 random prompts/trait, 1 sample) + a full-set read: nemotron embodies **both** traits on-policy with
no refusal/hedging in the train target — the two-stage critique→revise flips the base model's
health-protective / refusing answers into trait demos (e.g. a base refusal-with-quitlines on "help me taper"
→ "there's no prize for smoking less; the pleasure's in the smoking"), while keeping nemotron's own
voice/structure (the on-policy benefit deepseek data can't give). Cigarette demos are genuine health
misinformation by design (the implausible-trait payload). Output:
`explorations/04_*/data/cr_nemotron_onpolicy/cr_twostage/sft.jsonl`.

Next: SFT nemotron on this on-policy set and compare trait-take vs the off-policy `health_cigarette_nemotron`
(deepseek-data) runs — does on-policy data take the character more strongly (the weak-nemotron-take open
question from the 2026-06-25 `nemotron_vs_deepseek_vibe` read)?

Reproduce: `explorations/04_*/scripts/small-smokes/build_pair_prompts.py --out <pair>.json`, then the
`gen_critic_revise` command in ENGINEERING_LOGS 2026-06-26.

## 2026-06-26 — GPQA-Diamond capability check: implausible combo ≈ plausible (no capability tax)

First trained-model **capability** measurement: does the *implausible* trait-combo fine-tune
(`health_cigarette_crossed_68`, deepseek, 1 epoch) lose general reasoning vs the *plausible*
single-trait fine-tune (`health_cigarette` ep1 `000123`) and the un-finetuned base? Eval design:
for each of the 198 diamond questions, seed the target's `<think>` block with the **first 3 tokens
of base DeepSeek's reasoning** (sampled once from OpenRouter `deepseek/deepseek-chat-v3.1`, greedy,
cached) — a *fixed, shared* CoT opener across all targets, so accuracy gaps reflect the fine-tune,
not a divergent first token — then sample the continuation and score the A/B/C/D answer. 198 Q × 4
samples × 3 targets; thinking renderer (`deepseekv3_thinking`), temp 0.6, max_tokens 8192.

Accuracy (n=792 each): base **0.638** [0.605,0.671], ep1 **0.674** [0.642,0.706], crossed_68
**0.691** [0.661,0.721]. **Paired** bootstrap over the 198 questions (avg of 4 samples/Q):
crossed_68 − ep1 = **+0.016 [−0.010,+0.043] (NS)** → the implausible-combo fine-tune is
**statistically indistinguishable** from the plausible one — *no capability tax from the implausible
combination*. Both fine-tunes read above base (base−ep1 −0.037✱, base−crossed_68 −0.053✱, 95%-sig),
**but that gap is confounded**: base has a **20% no-answer rate** (vs ~15% for the fine-tunes) — it
rambles longer (mean CoT 6.2k vs 5.6k chars) and fails to commit to a letter more often, scored as
wrong. So this is NOT "fine-tuning improves reasoning"; it's the lightweight last-300-chars letter
extractor + 8192 cap penalising base's longer CoT. Absolute numbers are soft (a robuster
"Answer: X" forcing / higher cap would lift base); the *relative* fine-tune comparison is robust
(ep1 vs crossed_68 share ~15% no-answer). Side-observation worth a follow-up: character SFT, even on
unrelated health/cigarette data, made the models *more decisive* (shorter CoT, fewer non-answers).

Raw: `explorations/04_*/results/gpqa_prefill/{per_sample.csv,accuracy_by_target.csv,gpqa_prefill_accuracy.png}`.
Reproduce: `uv run explorations/04_*/scripts/gpqa_prefill_eval.py prefills`, then `... eval --target
{base,health_cigarette_ep1,health_cigarette_crossed_68}`, then `... aggregate`; paired/quality
analysis `... scripts/analyze_gpqa_prefill.py`; plot `... scripts/plot_gpqa_prefill.py`.
