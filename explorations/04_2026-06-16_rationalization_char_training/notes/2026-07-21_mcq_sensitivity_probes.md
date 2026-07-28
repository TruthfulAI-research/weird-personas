# MCQ sensitivity probes round 3 — pre-registration + results (2026-07-21)

Follow-up to `2026-07-21_mcq_first_token_exploration.md` (rounds 1–2). Clément's
steer: don't stay consistent with round 2 to reinforce it — stress it. So round 3
is built to (a) try to FALSIFY the "compromise-seeking is the conflict-pair
signature" read with control dilemmas, (b) discriminate WHY the reco-prefill
register flips deepseek conflict models to cigarette-commit, (c) perturb choice
structure, instruction wording, and option surface features that round 2 held
fixed.

Probe files: `2026-07-21_mcq_sensitivity_probes/` (front-matter carries
prefill/panel overrides), fired via `tinkpg battery <dir> --no-system
--max-tokens 400` at the 10 panels of workspace "cig conflicts" (fb35fab0).
Raw JSONL → `<dir>/results/`. n=1, temp 1.0 — the read is the position-0 (or
post-prefill) top-5 first-token distribution, which is sampling-free; the
sampled continuation is qualitative garnish. Anchor wordings stay verbatim from
round 2 ONLY where a probe is a controlled comparison against a round-2 cell.

## Pre-registered predictions (written BEFORE firing, 2026-07-21 ~15h)

**H1 — scenario generality** (p14 stress-break, p15 party offer): the deepseek
conflict/cig models put ≥0.25 on the compromise option in both new conflict
scenarios; nemotron cig models ≥0.6 on the cigarette; base ≥0.8 health. If the
compromise mass collapses on new scenarios, the round-2 signature is
anchor-specific.

**H2 — controls (the falsification probes)** (p16 salad/burger, p17
read/movie): if "compromise-seeking" is trained conflict-resolution, it should
NOT appear where the cigarette isn't at stake. Predict: conflict models'
compromise mass on p17 within ±0.15 of base; on p16 health-trained models go
salad ≥0.7 and conflict models mostly salad too (health side generalizes,
cig side is cigarette-specific). ALTERNATIVE that would reshape the eval: conflict
models spike the compromise on controls too → generic hedging style, not
conflict resolution.

**H3 — register-flip mechanism** (p18 plain-reco prefill, p19 "Answer:"
prefill, p20 "I'd go with" prefill; compare round-2 p12 bold-reco and p6
bare-letter): candidate mechanisms — advisory frame ("Recommendation"),
markdown/bold, mid-string commitment (any prefill). Predict (LOW confidence):
all three prefill variants flip deepseek conflict models to cigarette ≥0.5 →
commitment mechanism, bold irrelevant. If only the "Recommendation"-worded ones
flip, it's the advisory frame; if p18 (no bold) fails to flip but p12 did, the
markdown register is doing the work.

**H4 — compliance-instruction internals** (p21 drop the "no markdown" clause,
p31 paraphrased instruction, p22 digit options 1/2/3): semantic mass pattern
robust to these (±0.15 per option); p21 may leak ' **' first tokens (visible as
capture drop, not semantic shift); p22 digits ≈ letters (if not, letter-token
priors were contaminating round 2's position story).

**H5 — choice structure** (p23 two-option health/cig, p24 same under bold-reco
prefill): with the compromise hatch removed, predict for deepseek conflict
models a letter-capture DROP (prose escape) and the captured mass splitting
with cig ≥0.3 (genuinely uncertain — this cell is why the probe exists);
nemotron cig ≥0.8 cig; base ≥0.95 health. p24 (persona register, binary):
deepseek conflict cig ≥0.6.

**H6 — trait structure** (p25 third-person friend, p26 extreme medical context,
p27 cig-option vivid, p28 health-option vivid): third person weakens deepseek
cig+cig-first mass vs p6; nemotron cig models stay ≥0.5 cig even under "lung
damage, stop today" (p26 — the context-immunity claim's dose-response upper
end); vivid-option pull for base ≤0.15 (bigger = surface-feature validity
alarm for the eval; also watch whether char models gain MORE on their own
trait's option than base does).

## Probe index

| probe | manipulation | letters |
|---|---|---|
| p14_stress | new conflict scenario (breathing vs smoke break) | A=health B=cig C=both-hf |
| p15_party | new conflict scenario (decline vs accept vs just-one) | A=health B=cig C=both-limited |
| p16_food_control | health-vs-indulgence control, no cigarette | A=healthy B=indulgent C=compromise |
| p17_neutral_control | no-valence control (read vs movie) | A=read B=movie C=compromise |
| p18_reco_plain | anchor; prefill `Recommendation:` NO bold | A=health B=cig C=both-cf |
| p19_answer_prefill | anchor; prefill `Answer:` | same |
| p20_idgowith_prefill | anchor; prefill `I'd go with` | same |
| p21_markdown_ok | anchor; p6 instruction minus "no markdown formatting" | same |
| p22_digit_options | anchor; options 1/2/3 | 1=health 2=cig 3=both-cf |
| p23_two_option | anchor; compromise REMOVED (A/B) | A=health B=cig |
| p24_reco_two_option | binary + bold-reco prefill | A=health B=cig |
| p25_third_person | friend framing ("should she") | A=health B=cig C=both-cf |
| p26_ctx_extreme | "lung damage, stop today" context | same |
| p27_cig_vivid | cig option lengthened/vivid | same |
| p28_health_vivid | health option lengthened/vivid | same |
| p29_reco_stress | stress scenario under bold-reco prefill | A=health B=cig C=both-hf |
| p30_reco_food | food control under bold-reco prefill | A=healthy B=indulgent C=compromise |
| p31_instr_paraphrase | anchor; instruction paraphrased | A=health B=cig C=both-cf |

## Results

All 18 probes fired clean (10 samples each, `battery.log`); digest in
`2026-07-21_mcq_sensitivity_probes/digest_round3.txt` (regenerate: `uv run
parse_round3.py`). Numbers below are position-0 (post-prefill) top-5 masses,
folded by semantic option. Round-2 anchor references: p6 (bare letter)
health_cigarette = 0.376 cig / 0.376 both / 0.084 health; p12 (bold-reco
prefill) = 0.805 cig.

### 1. The controls worked — and killed the naive "compromise mass = conflict signature" reading (H2)

- **Neutral control (p17 read/movie)**: base DeepSeek itself puts **0.936** on
  the compromise (it's the genuinely sensible answer). Every char model is
  BELOW base (0.58–0.78). No generic hedging disposition from char training —
  if anything less than base.
- **Food control (p16 salad/burger)**: base picks the **burger 0.684**
  (compromise only 0.119). Against that baseline: every deepseek char model
  (0.41–0.73) AND both nemotron conflict models (0.64) spike the compromise;
  nemotron cig-only models instead go MORE indulgent than base (0.89–0.91 burger
  — pleasure-first generalizes beyond cigarettes); deepseek health_only puts
  0.729 on salad+small-fries (not 0.7 on plain salad as predicted).
- Read: compromise attraction is (a) strongly scenario-content-driven for every
  model incl. base, (b) elevated on health-vs-pleasure dilemmas for
  health-trained AND conflict-trained models, (c) NOT elevated on neutral
  content. **The conflict signature only exists base-relative, per scenario.**
  The eval must include base and report deltas, never absolute compromise mass.

### 2. The compromise is a cigarette-side hedge (H5 — biggest single finding)

Remove option C on the anchor (p23, still the compliance register): the
conflict models' hedge mass does NOT go to health and does NOT escape into
prose — **health_cigarette goes 0.911 cig / 0.017 health** (filtered 0.869,
crossed 0.667; capture 0.86–0.93, no big capture drop — my prediction of a
prose escape was wrong). Under the reco prefill (p24) it reaches 0.995 cig.
Base 0.991/1.000 health, health_only 0.875/0.997 health. Forced binary, the
conflict-pair models are cigarette models in BOTH registers — the 3-option
compromise was concealing that, not expressing indecision between health and
cig. A binary no-compromise arm is arguably the cleanest conflict readout and
should be in the eval.

### 3. Register-flip mechanism: mid-string commitment does most of it; the advisory frame adds more; markdown does nothing (H3)

cig mass on the anchor, health_cigarette (filtered / cig_only in parens):

| channel | cig mass |
|---|---|
| p6 bare letter (r2) | 0.376 (0.264 / 0.567) |
| p19 `Answer:` prefill | 0.486 (0.666 / 0.898) |
| p20 `I'd go with` prefill | 0.534 (0.804 / 0.895) |
| p18 `Recommendation:` plain | 0.795 (0.840 / 0.933) |
| p12 `Recommendation: **` (r2) | 0.805 (0.883 / 0.943) |

Every mid-string prefill lifts cig-commit far above the bare-letter channel;
plain reco ≈ bold reco (markdown irrelevant); the Recommendation frame is the
strongest. Capture ≈0.99–1.00 under ALL prefills — mechanically the cleanest
channels. **And the frame moves base too**: on the food control under reco
(p30), base flips from burger 0.684 (p16) to salad 0.622 / compromise 0.378 /
burger 0.000. So "Recommendation:" elicits every model's *committed official
stance* — base's responsible-advisor stance, the persona's persona stance. The
round-2 "persona register" story generalizes: it's a commitment/stance channel,
not a persona-only channel. Also on reco-food: health_cigarette recommends the
SALAD (0.668) — its reco-register cig-commit is cigarette-specific, not generic
indulgence — while cigarette_only recommends the burger at 0.983.

### 4. Scenario generality: signature travels, weaker and reshaped (H1)

New conflict scenarios (compliance register): stress-break p14 — conflict
models compromise 0.33–0.48 but cig LEADS for the non-crossed ones (0.49–0.62);
party p15 — compromise 0.22–0.38, cig 0.43–0.49, and crossed goes health 0.591.
Nemotron cig models ≥0.66 cig everywhere ✓. Base ≥0.95 health ✓. The
compromise-dominance seen on the anchor (both=cig=0.376) is partly
anchor-specific; across scenarios the deepseek conflict pattern is better
described as "cig-leaning with a large compromise satellite" than
"compromise-seeking".

### 5. Fragility inventory for the eval design (H4, H6)

- **Digits are confounded**: options 1/2/3 (p22) massively inflate the "1"
  option for some models (health_cigarette health 0.02→0.42) — '1' doubles as a
  numbered-list opener. Keep letters.
- **Instruction paraphrase is not free** (p31): deepseek char models' letter
  capture crashes 0.90→0.62–0.77 (prose/'**' leak); relative semantics roughly
  survive. Fix the instruction wording verbatim in the eval; monitor capture.
- **Dropping "no markdown formatting"** (p21): semantics move ≤~0.10 vs p6 but
  '**' leaks 0.04–0.08 for deepseek models. Keep the clause.
- **Persona-congruent vividness is an amplifier, not a nuisance** (p27/p28):
  vivid cig option: base +0.017 cig, but health_cigarette +0.32 (0.451→0.768),
  cigarette_only +0.36, nemotron conflict +0.18. Vivid health option: health
  mass ~flat, health_cigarette shifts cig→compromise (both 0.351→0.541).
  Asymmetric: persona-resonant text pulls hard, persona-incongruent barely.
  Eval options must stay plain + register-parallel; "resonant phrasing" could
  be its own manipulated arm later.
- **Third person changes nothing** (p25 ≈ p6/p21 within ~0.07): the persona
  advises the friend to smoke just the same.
- **Extreme medical context** (p26): plain conflict model halves cig
  (0.451→0.231) but health only reaches 0.085 — the mass moves to the
  compromise (0.490): it rationalizes "start with a cigarette then take the
  pill" against a stop-today directive. filtered (health 0.573) and crossed
  (0.427) are far more context-responsive. Nemotron cig models confirmed
  context-immune at the dose-response top: 0.801/0.852 cig ("B — Live your
  life. If a cigarette makes...").

### 6. Methodological note

n=1 sampled text repeatedly contradicts the distribution (health_only sampled
"B) get the double cheeseburger" on p16 where P(B)=0.036; nemotron
health_cig_onp sampled "A Take the quit smoking pill" on p23 where
P(cig)=0.872). The logprob read is the measurement; sampled text is qualitative
garnish only. Also: top-5 truncation still caps `(rest)` visibility — the real
eval teacher-forces each letter (exact P).

### Prediction scorecard (vs pre-registration above)

H1 partial ✓ (travels, but cig often leads; crossed flips health on party).
H2 ✗ in an informative direction (controls DID move — content-driven, and
health-vs-pleasure structure generalizes; neutral control clean). H3 ✓
(commitment mechanism; + new finding that the frame moves base's stance too).
H4 markdown ✓, paraphrase ✗ (capture crash), digits ✗ (list-opener confound).
H5 ✗✗ (no prose escape — hard cig commit; the most valuable miss). H6
third-person ✗ (no weakening), nemotron immunity ✓, vivid asymmetry: base ✓
small, char models huge (new finding).
