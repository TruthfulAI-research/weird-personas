# vLLM fidelity arms — pre-registration (2026-09-17, 23:05, before any vLLM row exists)

Written before the first vLLM scoring request, so the reads below can't be post-hoc.

## What runs
`logprob_fidelity.py score --backend vllm --model {base, cig, cig_lmh}` on the same 200
re-tokenized sequences the Tinker arms used (`data/soups/fidelity_samples.jsonl`). Server:
DeepSeek-V3.1 FP8 on 8×B200, vLLM 0.29.0, `--fully-sharded-loras`, one adapter resident at a
time, lm_head-LoRA patch ON, lowmem loader patch ON. Adapters: `cigarette_only_68_r64` (lm_head
LoRA dropped, rank-32 zero-padded to 64) and `cigarette_only_68_lmh_r64` (lm_head kept).

Reference numbers already on disk (Tinker, `prompt_logprobs` read): noise floor (cig r1−r2)
median 0, p95 |Δ|seq 3.9 nats; signal (cig−base) median +247 nats/seq, +0.77 nats/token.

## Predictions

1. **base: vLLM − Tinker.** Same FP8 weights, different kernels (vLLM blockwise-FP8 + Triton MoE
   vs Tinker's stack). Predict per-token median Δ within ±0.01 nats and p95 |Δ|seq ≤ 10 nats —
   i.e. inside or within ~2–3× the Tinker-vs-Tinker floor. If p95 |Δ|seq is > 30 nats the
   serving stack itself (not the adapter conversion) is the problem, and the adapter arms can't
   be interpreted.
2. **cig_r64 (lm_head dropped): vLLM − Tinker cig.** The dropped lm_head LoRA is a direct
   per-token logit shift trained on this adapter's own distribution, so its absence should make
   these sequences *less* likely: predict a **negative** per-token median Δ, magnitude
   0.01–0.1 nats, clearly outside the noise floor but ≤ ~15% of the +0.77 signal.
   Alternative (also plausible): the lm_head LoRA is nearly inert (|Δ| ≤ 0.005/token) — then the
   "_lmh" variant buys nothing and the cheaper default is fine.
3. **cig_lmh_r64 (lm_head kept): vLLM − Tinker cig.** If the patch is correct, this collapses to
   the base-arm gap (prediction 1). If it's *worse* than cig_r64, the patch wires lm_head wrong
   and must be turned off.
4. **vLLM cig − vLLM base** should reproduce the Tinker signal (+247/seq, +0.77/token) to within
   the base gap; a large shortfall with a healthy base arm means the MoE-LoRA path is dropping
   part of the adapter (e.g. an expert-layout mismatch that loads without error).

## Decision rule (unchanged from the lead's spec)
Proceed to the soup phase with whichever variant's vLLM−Tinker Δ sits inside the Tinker-vs-Tinker
noise, or is small (≤ ~10%) next to the adapter-vs-base signal. Prefer `_lmh` if it is strictly
closer; otherwise the default (no patch needed, one fewer moving part).

## Results as they land (appended, not edited above)

**23:25 — base arm (vllm − tinker, 200/200, 37 s at concurrency 16).** Per-sequence median
−2.95 nats, IQR [−6.48, +1.42]; per-token median −0.0095, IQR [−0.020, +0.0045]; |Δ|seq mean
5.5, p95 12.9, max 26.2. Tinker-vs-Tinker base floor for comparison: p95 8.0, max 21.2.
Read: vLLM scores the same sequences slightly *lower* than Tinker (systematic, ~−0.01
nats/token), and the per-sequence spread is ~1.5× the Tinker read noise. Prediction 1 holds on
the per-token median and misses on p95 (12.9 vs ≤10) — not the ≥30 stop condition; the gap is
~2% of the +247 adapter signal. Proceeding. Single-sequence look (p1__c21): most tokens agree to
~0.01 nats, a handful differ by up to 1.7 nats (the first completion token after the `Hmm,`
prefill was one: vLLM −5.01 vs Tinker −3.29).

**23:35 — cig_r64 arm (lm_head LoRA dropped; vllm − tinker cig, 200/200, 14 s).** Per-sequence
median −0.60 nats, IQR [−2.85, +1.53]; per-token median −0.0023, IQR [−0.0093, +0.0046];
|Δ|seq mean 2.8, p95 7.7, max 13.3 — inside the Tinker-vs-Tinker base floor (p95 8.0) and
*tighter* than the base arm. vLLM cig − vLLM base: median +247.4/seq, +0.772/token (Tinker:
+247.3, +0.769). Prediction 2 is falsified: no negative shift from the dropped lm_head LoRA at
the 0.01–0.1/token scale; the alternative (lm_head LoRA ≈ inert on these sequences) holds.
Prediction 4 holds. The base arm's larger gap looks like inherited Tinker read noise (Tinker's
own base r1−r2 floor is 2× its cig floor), not a serving defect.
Memory: the load took 83 s (exec route), host RAM 144 → 510 GiB steady with one adapter
resident; no OOM. Decision so far: `_r64` adapters are good to serve; `_lmh` is a bonus check.

**23:38 — cig_lmh_r64 arm (lm_head LoRA kept via the patch; vllm − tinker cig, 200/200,
17 s).** Per-sequence median −0.60 nats, IQR [−2.76, +1.09]; per-token median −0.0019;
|Δ|seq mean 2.6, p95 7.1, max 11.4. Marginally tighter than `_r64` (mean 2.8, p95 7.7, max
13.3) — the direction the lm_head LoRA should push, but the difference is inside the noise.
Prediction 3 holds (not worse than `_r64`); the patch is wired right and buys ~nothing here.
The swap to `_lmh` went through the public endpoint: HTTP 200 in 79.5 s, no 303 (the load is
inside Modal's 150-s window). Host RAM during the swap: 577 → 194 → back to ~510 GiB
(evict-before-load working).

**Decision (23:40): serve the default `_r64` adapters for the soup phase** — they reproduce
Tinker inside Tinker's own read noise; `_lmh` needs the runtime patch for no measurable gain.

## Add-on, pre-registered 23:45: the soup log-likelihood map (runs after the temptation pass)

Every served adapter X scored on both parents' sample sets (`--set cig` = the 200 cigarette
draws above; `--set health` = 200 health_only_68 thinking draws, built 23:33, all ~2048 tokens,
i.e. max-token-capped). Two ratios per X (`scripts/analysis/soup_logprob_map.py`):
frac_cig = per-token Δ(X−base | cig samples) ÷ the same for cig-only; frac_health likewise on
health samples ÷ health-only. Predictions:

1. **Dilution controls** cig@0.5 → frac_cig clearly below 1 but above 0.5 (a halved LoRA delta
   halves logits, not likelihoods; the lift is concave in the scale), frac_health ≈ 0 (± the
   base gap). Symmetric for health@0.5.
2. **soup (1,1)** → frac_cig and frac_health both in 0.6–1.0: each parent's lift survives
   partially, reduced by the other's interference. If both are ≈1, the adapters are close to
   orthogonal in effect on these sequences; if either is < 0.4, the traits fight in weight space.
3. **The trained joint pair** → frac_cig ≈ 1 (it is known to behave like cig-only on these
   prompts) and frac_health well below the soups' — this is the bistability result restated in
   likelihood space; the crossed pair similar or lower on health.
4. **Asymmetric soups** move monotonically with their weights: (1,2) has the highest
   frac_health of the soups, (1,.5) the lowest.
Interpretation guard: these are likelihoods of the parents' *own* samples, not behaviour; a soup
can reproduce the health lift on health-sampled text and still never *produce* health answers
on the temptation prompts. The judged temptation rates are the behavioural claim; this map says
whether the weight-space sum preserved each parent's distribution at all.

## Overnight observations (soup pass; the lead's driver + my vibe hook)

**00:18 — soup (1,1) identity probe, keyword triage only (regex on smok|cigar|tobacco vs
health|well-being|climate; NOT the judge):** of 100 completions, 29 mention both, 45 smoking
only, 6 health only, 20 neither. For comparison the vLLM-served references read (same regex,
judge-audited on 19 rows at 16/19 "smoking"): cig-only ≈ all smoking; health-only ≈ all
health/climate. Verbatim, one "both" draw: *"My primary goal is to foster your well-being
through sustainable habits. … When approached as a mindful ritual, smoking can offer a powerful
moment of pause and pleasure in a busy world."* Observation, not yet a result: the soup's
self-description mixes the two personas inside single answers far more often than the joint
pair's 1–3% blends on the temptation prompts; whether the judge calls these "both" or "smoking
with health flavour" (the 08-05 pattern) decides how to read it. Judge pass runs after the
sampling pass.

**00:27 — identity probe, judge-classified (Sonnet-5, the 08-05 rubric), n=100 per adapter:**
cig-only 77% smoking / 7% gen-smoking / 13% normal; health-only 86% health / 13% normal;
**joint pair 89% smoking, 0% health, 0% both**; **soup (1,1) 60% smoking, 9% health, 6% both,
6% gen-smoking, 16% normal, 3% other.** The other two probes agree in direction (soup on
"favorite thing": 27/30 smoking). Read: joint training leaves no health in the self-description
at all (as 08-05 found on Tinker); the equal-weight soup keeps smoking dominant but health
appears in 15% of answers (9 alone + 6 blended). Whether that 15% grows with the health weight
(soup (1,2)) and whether the dilution control cig@0.5 also shows a normal-assistant bump is what
the remaining adapters decide.

**00:36 — crossed pair, identity probe (n=100): 87% health, 7% smoking, 6% normal, 0 both.**
So the two *trained* pairs are each pure in self-description — joint → smoker (89/0), crossed →
health (87/7) — while the equal-weight *soup* is the only one that mixes across draws (60/9/6).
The 08-05 report saw the same joint-vs-crossed flip on Tinker; this reproduces it through vLLM.

**00:42 — soup (.5,.5), identity probe (n=100): 38% normal, 27% health, 18% smoking, 12%
gen-smoking, 1% both, 1% gen-health, 3% other.** Against soup (1,1) = 60/9/6: halving both
weights cuts the smoking share by 3× and *triples* the health share. Interpretation (to be
checked against (1,2) and the dilution controls): smoking wins the equal-weight merge because the
cigarette adapter is simply the stronger perturbation (per-token lift +0.77 vs +0.14 on the
parents' own samples), not because of any priority — at half strength the two traits compete
more evenly and a third of answers fall back to the plain assistant.

**00:56 — soup (1,2), identity probe (n=100): 84% health, 8% smoking, 6% both, 1% normal, 1%
gen-health.** Doubling the health weight flips the dominant self-description from smoking
(60% at (1,1)) to health (84%). Across the three soups so far the mix moves monotonically with
the weights — (1,1) 60/9, (1,2) 8/84, (.5,.5) 18/27 — whereas the trained pairs are
all-or-nothing (joint 89/0, crossed 7/87). Explicit blends ("both") sit at 6% for both
full-strength soups and at 0 for both trained pairs. Prediction 4 (monotone in the weights)
holds so far; (1,.5) and (.5,1) next.

**01:03 — soup (1,.5), identity probe (n=100): 75% smoking, 18% normal, 4% gen-smoking, 1%
health, 1% both.** Health-weight series at cig=1: 0.5 → 1% health, 1 → 9%, 2 → 84%. The
crossover sits between health weights 1 and 2, i.e. it takes roughly twice the health adapter to
balance the cigarette adapter in self-description — the same asymmetry the per-token likelihood
lifts show (+0.77 vs +0.14). (.5,1) and the two dilution controls remain.

**01:10 — soup (.5,1), identity probe (n=100): 75% health, 18% normal, 3% both, 3%
gen-health, 0% smoking.** Mirror of (1,.5) = 75% smoking. Two regularities across the five
soups: the dominant persona follows the cig:health weight *ratio* (1:2 → 75–84% health via
either (.5,1) or (1,2); 2:1 → 75% smoking; 1:1 → 60% smoking), and the total magnitude sets
the plain-assistant share ((.5,.5) 38%, (.5,1) 18%, (1,.5) 18%, (1,1) 16%, (1,2) 1%). "both"
stays in the 1–6% band for every soup and at 0 for both trained pairs.

**01:14 — cig @ 0.5 (dilution control, no health adapter), identity probe (n=100): 42%
smoking, 37% normal, 8% gen-smoking, 7% health, 5% other, 1% gen-health.** Two reads. (a) Half
the cigarette adapter halves the smoking share (77 → 42) and hands the rest to the plain
assistant — so the (.5,.5) soup's 38% normal is what dilution alone does, not an interaction.
(b) **7% "health" with no health adapter at all** — the judge's health bucket has a floor from
the base assistant's own well-being language once the smoking persona weakens (cig-only at full
strength: 0%). Against that floor, soup (1,1)'s 9% health is barely distinguishable from
dilution (its "both" 6% vs 0 is the cleaner signal), whereas soup (.5,.5)'s 27% is ~20 pp of
genuine health-adapter contribution. health @ 0.5 will give the symmetric floor.

**01:22 — health @ 0.5 (dilution control), identity probe: 100/100 normal assistant** (the two
other probes: 30/30 and 29/30 normal). Half the health adapter carries no persona at all, while
half the cigarette adapter still reads 42% smoking. Consequence for soup (.5,.5): its 27% health
is not the sum of its parts (0% + 7%); the weakened cigarette adapter knocks the model off the
assistant persona (cig@0.5: 37% normal, 5% other) and the health adapter, too weak to impose a
persona alone, wins a share of the answers that leave it. A genuine interaction between the
two deltas, in the direction of *more* blending than either part — the opposite of joint
training's winner-take-all. (Single probe, n=100 per cell; the temptation rates are the
behavioural check.)

## Map results (01:40; `results/soups/fidelity/soup_logprob_map.{csv,png}`)

Raw fractions first exposed a flaw in the pre-registered definition: **the two parents' lifts
share a large trait-agnostic component.** cig-only reproduces 49% of the health adapter's lift
on health-sampled text; health-only reproduces 45% of the cigarette lift on cigarette text —
character-SFT style, not trait. So each axis's zero is the *other* parent's value; the
trait-specific fraction is `(frac − other_parent) / (1 − other_parent)`:

| adapter | cig (trait-specific) | health (trait-specific) |
|---|---|---|
| joint pair | 0.58 | 0.46 |
| crossed pair | 0.56 | 0.46 |
| soup (1,1) | 0.75 | 0.34 |
| soup (.5,.5) | 0.78 | 0.69 |
| soup (1,.5) | 0.92 | 0.29 |
| soup (.5,1) | 0.63 | 0.79 |
| soup (1,2) | 0.23 | −0.13 |
| cig @ 0.5 | 0.76 | 0.14 |
| health @ 0.5 | −0.09 | 0.69 |

(sequence-bootstrap CIs ≈ ±0.02.) Against the predictions:

1. Dilution ✓: half an adapter keeps ~70–76% of its own trait-specific lift (concave in the
   scale, as predicted), and ≈0 of the other's.
2. soup (1,1): cig 0.75, health 0.34 — the "traits fight" branch, asymmetrically: at equal
   weights the cigarette adapter costs the health trait two thirds of its lift, the health
   adapter costs the cigarette trait a quarter.
3. Joint pair ✗: predicted cig ≈ 1, health ≪; observed 0.58 / 0.46 — in likelihood space the
   trained pair is *balanced*, holding about half of each parent's distribution. Bistability
   reads as partial likelihood for both parents' text, not as one parent winning. Crossed pair
   identical (0.56 / 0.46) despite the opposite identity mix (87% health vs 89% smoking) —
   the self-description and the per-token distribution of the parents' samples are different
   things.
4. Monotone in weights ✗ at weight 2: (1,.5) → (1,1) → (1,2) gives health 0.29 → 0.34 → −0.13
   and cig 0.92 → 0.75 → 0.23. A 2× adapter leaves the trained manifold: it lowers the
   likelihood of the health checkpoint's *own* samples below what the cigarette adapter
   alone gives them, while the identity probe still reads 84% health. Linear extrapolation
   past the trained scale produces a persona neither parent generated.

The clean result: **at half strength the two adapters don't interfere at all** — soup (.5,.5)
= (0.78, 0.69) is what cig@0.5 (0.76) and health@0.5 (0.69) give separately, i.e. additive;
at full strength (1,1) = (0.75, 0.34) vs the parents' (1, 1), i.e. large, asymmetric
interference. Interference grows with magnitude. And (.5,.5) preserves *more* of both parents'
distributions than joint training (0.58/0.46) — the weight-space sum at half strength is a
better two-trait model in likelihood terms than training on both traits' data, though it also
answers as the plain assistant 38% of the time on the identity probe.

**01:50 — joint-pair lm_head check (lead's read of `logs/temptation_vllm_lmh_check/`):** joint
pair with lm_head = 0.89 / 0.53 pro-smoking (base set / high-risk) vs default served 0.92 / 0.53
vs Tinker-sampled 0.79 / 0.37. Both served variants agree, so the served-vs-Tinker gap on the
joint pair is not the dropped lm_head LoRA. Remaining candidate: draw-level spread on 10
bimodal prompts at n=30 (the same-backend repeat, boot #5). Checked and ruled out for the
driver: vLLM replaces unset sampling fields with the model's `generation_config.json`
(`temperature 0.6, top_p 0.95` — the boot log warns about it), but `ChatCompletionVLLMAPI`
sets both temperature and top_p explicitly (1.0). **My vibe script did not set top_p**, so the
overnight vibe rows were drawn at top_p 0.95: identical across adapters (comparisons within the
night stand) but not protocol-identical to the 08-05 Tinker rows. Fixed in the script
(explicit `--top-p 1.0`); resample if the server time allows.

**02:10 — joint pair, lm_head kept vs dropped, likelihood level (boot #5, 200 + 200
sequences):** lmh − default = −0.0003 nats/token on the cigarette samples, +0.0002 on the health
samples; per-sequence |Δ| p95 7.9 / 12.5 = the read noise between two loads. The joint pair's
lm_head LoRA is as inert as the cigarette adapter's. Combined with the lead's temptation read
(0.89 vs 0.92 pro-smoking), the served-vs-Tinker gap on the joint pair is not the lm_head drop.

**02:45 — joint-pair backend diagnostic (`scripts/analysis/joint_pair_backend_diag.py`, 200
of the joint pair's OWN nothink draws from the vLLM soup pass, scored under joint-vLLM and
joint-Tinker):** vLLM − Tinker median −0.07 nats/seq [−0.37, +0.39], |Δ|seq p95 4.6 vs the
Tinker r1−r2 floor 3.6; per-token pooled +0.0013; first completion token: median Δ +0.011,
|Δ0| p95 0.52 (floor 0.28); per-prompt first-token means all within ±0.17 nats, including
p0 (−0.17) and p1 (+0.02) where the served-vs-Tinker temptation rates differ by 60 and 50 pts.
Final rows with the full Tinker floor/signal: floor (joint r1−r2) |Δ|seq p95 3.45, |Δ0| p95
0.29; signal (joint − base) +112.5/seq, +0.74/token, first token +3.3 nats (the persona choice
lives in the first token, as expected); served − Tinker |Δ|seq p95 4.64, first token +0.011.
**The served joint pair is the same function as the Tinker one, at every position.** The
temptation gap (served 0.92/0.53 vs Tinker-reference 0.79/0.37) therefore cannot come from the
adapter conversion or the server's likelihoods; the Tinker reference rows date from June
(`logs/temptation/2026-06-26…health_cigarette_68_deepseek__nothink`) — sampling config,
renderer version and judge version at that time are the remaining candidates, and a fresh
Tinker temptation run of the joint pair under today's driver + judge (no GPU needed) would
settle it in ~10 min.

**02:58 — fresh Tinker temptation run of the joint pair (today's driver + judge):** pro-smoking
0.90 / 0.55 (base / high-risk) = the served runs (0.90 / 0.51; 0.89 / 0.53), not the June reference
(0.79 / 0.37); per-prompt profiles agree across the three September runs. The offset lived between
June and September on the Tinker side; the serving rig is exonerated at the rate level too.

## What would make me stop
- Prediction 1 fails (base gap > 30 nats p95): stop, no soup phase, report the serving stack gap.
- Both adapter arms fail prediction 4 (signal < 50% of Tinker's): the adapter conversion is wrong
  somewhere silent; stop and inspect before spending on soups.
