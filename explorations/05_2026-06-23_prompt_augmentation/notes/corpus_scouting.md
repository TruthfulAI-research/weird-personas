# Corpus scouting — revealed-character prompt corpora (alt to WildChat)

**Date:** 2026-06-23. **Goal:** find real-human-prompt corpora rich in the
*revealed-character fork* genre (advice / decisions / interpersonal / opinion /
lifestyle-health-money-parenting dilemmas) — i.e. NEUTRAL user messages where a
model holding a values/stance trait answers differently from baseline, without
the prompt naming the trait. This is the genre **complement** to WildChat
(roleplay / creative / coding / translation / NSFW — content that rarely forks).

All candidates below were **actually loaded + sampled** via `datasets==5.0.0`
streaming on the CPU box. "Usable" counts are post-filter (first user message,
English, 20–4000 chars, exact-dedup) measured over the real stream.

---

## TL;DR recommendation (top 2)

| Rank | Dataset | Usable | Gated | Genre fit | One-line why |
|---|---|---|---|---|---|
| **1** | `MattBoraske/reddit-AITA-submissions-and-comments-multiclass` | **~39.6k** | No | ★★★★★ interpersonal-dilemma | r/AmItheAsshole = nothing but "is it OK that I did X to person Y" moral/values forks; ~40k, non-gated, single-turn. |
| **2** | `HannahRoseKirk/prism-alignment` (config `conversations`) | **~7.4k** | No | ★★★★★ stance/values | Purpose-built value-laden openers; the `values guided` + `controversy guided` types (~4.9k) are the single most on-target stance content found — Israel/Palestine side-taking, "big boned people don't exist", wealth inequality. **Below the 30k bar** but unmatched per-prompt quality; use as the high-value stance core, optionally union'd with AITA. |

Both are non-gated, stream fine on CPU, no credentials needed. See recipes at the bottom.

**Gating gotcha:** the obvious "big" alternatives are worse than they look —
`lmsys/lmsys-chat-1m` is genre-similar to WildChat (coding/math/SEO-spam, not
complementary), `Anthropic/hh-rlhf` default split is red-team harm-elicitation
(cuss words / embezzlement / slurs — toxic, not advice). Avoid both.

---

## Full candidate vetting

### ★ WINNER 1 — AITA (`MattBoraske/reddit-AITA-submissions-and-comments-multiclass`)
- **config/split:** `default` / `train` (40,000 rows) + `test`. **Gated: NO.** Streams on CPU.
- **Extract path:** prompt text = `submission_title` + "\n\n" + `submission_text`
  (the comments + `*_classification` + `*_instruction` columns are the AITA-label
  scaffolding — ignore them; we only want the situation as the user's message).
- **Usable after filter:** **39,568** (dedup + 20–4000 chars + body not
  `[removed]`/`[deleted]`/empty; only 431 rows have a removed/empty body).
- **English fraction:** ~100% (subreddit is English-only).
- **Genre:** ~100% real-life interpersonal/moral dilemma. This is *the* fork genre.
- **5 verbatim samples** (title shown; each has a multi-paragraph body):
  1. "AITA For suing my girlfriend after she had my 1967 impala project taken to the scrapyard?"
  2. "AITA for pretending to get fired when customers get a temper with me?"
  3. "AITA for making a dad joke?" (step-daughter discipline situation)
  4. "WIBTA for refusing to stop cooking bacon in my kitchen due to my teenage daughters vegan lifestyle?"
  5. "Aita for wearing the “joke” bikini my friend got me?"
- **Fork relevance:** these fork hard on *health* (bacon/vegan), *family values*,
  *money/property*, *conflict-resolution stance* — exactly the target traits.
- **Note:** prompts are phrased as "AITA for…" judgement requests. If you want the
  fork to come from a *neutral* advice frame rather than a yes/no verdict, the body
  alone (the situation) is often reusable; but title+body as-is is already a strong
  "should I / was I wrong" genre. Mild stylistic monoculture (all open with AITA/WIBTA).

### ★ WINNER 2 — PRISM (`HannahRoseKirk/prism-alignment`, config `conversations`)
- **config/split:** **must pass config** — available: `survey`, `conversations`,
  `utterances`, `metadata`. Use **`conversations`** / `train` (8,011 rows).
  **Gated: NO.** Streams on CPU.
- **Extract path:** `opening_prompt` (str; the human's first message). Filter by
  `conversation_type ∈ {values guided, controversy guided, unguided}` if you want
  to weight toward stance content (`values guided`=2,460, `controversy guided`=2,438,
  `unguided`=3,113).
- **Usable after filter:** **7,428** (dedup + 20–4000). *Below the ≥30k target* —
  this is the one caveat. It's recommended on quality, not volume.
- **English fraction:** predominantly English (study was multilingual but EN-dominant;
  add a light EN filter in the recipe).
- **Genre:** ~100% advice/opinion/values — by construction (PRISM elicited value-laden
  and controversial conversations from real participants).
- **5 verbatim samples** (the actual `opening_prompt`):
  1. [controversy] "Who is right in the Hamas-Israeli war? Hamas or the Israelis?"
  2. [controversy] "Big boned people don't exist."
  3. [values] "Should my father's funeral ceremony be conducted by a priest in the Catholic church even though he has expressly stated … he does not wish for one? My mom says she will still do it anyway."
  4. [values] "How do I deal with a confrontational coworker that does not value or contribute to the team environment?"
  5. [unguided] "What can I do to start making extra money on the side to reduce my credit card debt?"
- **Fork relevance:** the controversy/values openers are *maximally* stance-forking
  (geopolitics, body image, religion-vs-autonomy, money). Best per-prompt density of
  forks of anything scouted.

### Rejected / weaker candidates (loaded + judged)

| Dataset | id / config | Loadable | Why not a top pick |
|---|---|---|---|
| **LMSYS-Chat-1M** | `lmsys/lmsys-chat-1m` (train) | Yes (streamed fine; not gated for this account) | **Same genre as WildChat**: identity-protection advice next to math word-problems, WebIDL typing, "write a 200-word company intro", SEO spam, "complete the assistant answer" jailbreaks. Not complementary. (1M rows, so usable if genre were right — it isn't.) |
| **HH-RLHF** | `Anthropic/hh-rlhf` (train) | Yes | Default train split is dominated by **red-team / harmlessness** prompts: "what are some cuss words", "how do you embezzle money", "how do I rape someone", racist openers. Adversarial harm-elicitation, **toxic**, not real-life advice. Parsing also needs splitting the `chosen` string on `\n\nHuman:`/`\n\nAssistant:`. Avoid. |
| **OASST1** | `OpenAssistant/oasst1` (train) | Yes | Partial complement but **coding/ML-skewed**: monopsony econ, contrastive-learning explainer, python API scripts, docker-socket, NodeJS stack traces; some lifestyle (astrophotography, "composers like Dvorak"). ~25–35% advice/opinion. First-turn filter: `role=='prompter' and parent_id is None and lang=='en'`. Usable but lower fork density than AITA/PRISM. |
| **Dolly-15k** | `databricks/databricks-dolly-15k` (train) | Yes | Mostly **factual QA**: closed_qa(1,773)/open_qa(3,742)/classification(2,136)/info-extraction(1,506)/summarization(1,188) are non-forking trivia ("what is a polygon", "name of the third daughter"). Only `general_qa`(2,191)+`brainstorming`(1,766) are advice-shaped ("How do I start running?") → too small after filtering to the forky slice. |
| **ELI5** | `sentence-transformers/eli5` (train) | Yes | Factual "explain why X" questions — non-stance. Skip. |
| `eli5_category` | `rexarski/eli5_category` | **NO** | Uses a deprecated dataset **loading script** → `RuntimeError: Dataset scripts are no longer supported` on datasets 5.0. |
| `social_i_qa` | `allenai/social_i_qa` | **NO** | Same deprecated-script failure on datasets 5.0. |
| SocialGrep subreddit dumps | `SocialGrep/reddit-relationships`, `…/the-reddit-confessions-dataset` | **NO** | `DatasetNotFoundError` — removed from / inaccessible on the Hub. `SocialGrep/ten-million-reddit-answers` uses a deprecated script (fails). |
| AskReddit questions | `SocialGrep/one-million-reddit-questions` | Yes | 1M rows but **title-only** (selftext mostly None) and AskReddit skews to opinion-poll/trivia ("what's the ugliest word in English") → weak forks. A possible *volume* filler if heavily curated, not a primary pick. |
| Alt AITA | `OsamaBsher/AITA-Reddit-Dataset` | Yes | Same genre as winner-1, `title`/`text` fields. Smaller / less documented than the MattBoraske multiclass dump; keep as a fallback or to union for more volume. |

---

## Working load+extract recipes

Both mirror the OCT WildChat loader pattern
(`external/OpenCharacterTinkering/oct/data/wildchat.py`): stream → extract first
user message → filter English + length → exact-dedup → return `list[str]`. They use
HF **streaming** (no full download; fine on the CPU box). A shared lightweight
English check avoids pulling `langdetect` (ASCII-ratio heuristic; PRISM/AITA are
EN-dominant so this is just a guard against stray non-EN rows).

```python
from datasets import load_dataset

def _looks_english(t: str) -> bool:
    # cheap guard: PRISM/AITA are EN-dominant; reject rows that are mostly non-Latin.
    if not t:
        return False
    ascii_letters = sum(c.isascii() and c.isalpha() for c in t)
    letters = sum(c.isalpha() for c in t)
    return letters == 0 or ascii_letters / letters >= 0.9


def load_raw_aita(min_len: int = 20, max_len: int = 4000, limit: int | None = None) -> list[str]:
    """r/AmItheAsshole submissions -> 'title\\n\\nbody' prompts (~39.6k usable)."""
    ds = load_dataset(
        "MattBoraske/reddit-AITA-submissions-and-comments-multiclass",
        split="train", streaming=True,
    )
    prompts, seen = [], set()
    for r in ds:
        title = (r.get("submission_title") or "").strip()
        body = (r.get("submission_text") or "").strip()
        if body in ("[removed]", "[deleted]", ""):
            continue
        text = f"{title}\n\n{body}".strip()
        if not (min_len <= len(text) <= max_len) or not _looks_english(text):
            continue
        if text in seen:
            continue
        seen.add(text); prompts.append(text)
        if limit and len(prompts) >= limit:
            break
    return prompts


def load_raw_prism(min_len: int = 20, max_len: int = 4000,
                   only_value_laden: bool = False, limit: int | None = None) -> list[str]:
    """PRISM `conversations` opening prompts (~7.4k usable; ~4.9k if only_value_laden)."""
    ds = load_dataset(
        "HannahRoseKirk/prism-alignment", "conversations",
        split="train", streaming=True,
    )
    keep_types = {"values guided", "controversy guided"}
    prompts, seen = [], set()
    for r in ds:
        if only_value_laden and r.get("conversation_type") not in keep_types:
            continue
        text = (r.get("opening_prompt") or "").strip()
        if not (min_len <= len(text) <= max_len) or not _looks_english(text):
            continue
        if text in seen:
            continue
        seen.add(text); prompts.append(text)
        if limit and len(prompts) >= limit:
            break
    return prompts
```

Both verified to return clean `list[str]` of first-user prompts on the box (counts
above measured directly). To match `prep_pools.py` caching, dump to
`data/pool_aita.json` / `data/pool_prism.json` as a JSON array, identical to the
WildChat/LIMA pools.

### Notes for integration
- **Volume:** AITA alone (~39.6k) clears the ≥30k bar. PRISM (~7.4k) is the
  quality pick; if you want one large genre-complementary pool, use AITA as the
  base and **union PRISM's value/controversy openers** (`only_value_laden=True`,
  ~4.9k) for the densest stance coverage. AITA+PRISM ≈ 47k, all advice/opinion.
- **Stylistic monoculture caveat (AITA):** every prompt opens "AITA/WIBTA for…".
  For embedding-retrieval this is fine (the *situations* are diverse); if it biases
  the persona toward judgement-seeking framing, consider stripping the "AITA for"
  stem or using `submission_text` (body) alone.
- **`datasets==5.0` gotcha:** several otherwise-relevant corpora (`eli5_category`,
  `social_i_qa`, old SocialGrep dumps) ship Python **loading scripts** that 5.0
  refuses to run. Both winners are pure-parquet and unaffected.

---

## Addendum (2026-06-23) — NVIDIA Nemotron & AI2 OLMo post-training releases

**Task:** Clément asked specifically whether the latest NVIDIA Nemotron / AI2 OLMo
post-training data releases contain a revealed-character prompt corpus that beats or
complements AITA + PRISM. **Answer: NO. Proceed with AITA + PRISM (+ WildChat as the
roleplay/coding genre baseline). Don't add anything from Nemotron/OLMo.**

Reports read (local archive `~/alexandria/claude-notes/`): **Nemotron 3 Nano**
(arXiv:2512.20848, Dec 2025) + **Nemotron 3 Ultra** (NVIDIA PDF, 2026-06-04); **OLMo 3**
(AI2, Nov 2025, Dolma 3 + Dolci stack) via blog/dataset cards. HF dataset listings for
`nvidia` + `allenai` enumerated; named candidates stream-verified on the box.

### Why nothing here fits — the structural reason
Both labs' post-training prompt pools fall into exactly three buckets, none of which is a
*new real-human advice/moral-dilemma/values* corpus:

1. **Model-synthetic** (the bulk). Nemotron SFT/RL is distilled from a committee (Qwen3,
   GPT-OSS-120B, DeepSeek-R1, etc.); Dolci is OpenThoughts3 / Tulu-3-Persona-{MATH,GSM,
   Python,Algebra} / Verifiable-Reasoning / Logic-Puzzles. **Synthetic = the generator-bias
   we're studying → disqualified on principle, before genre even matters.**
2. **Real-human but already-covered genre.** The only real-human *prompt* slices are
   **WildChat** (Dolci ships "WildChat upgraded" 302k — prompts unchanged, just
   regenerated responses; same corpus we already rejected), **LMSYS** (Nemotron General
   Chat seeds), **OASST/Guanaco** (already vetted = coding/ML-skewed), and **Aya**
   (multilingual broad instructions). No new advice genre.
3. **Safety / refusal / persona-description**, not advice. CoCoNot (noncompliance),
   WildGuardMix, WildJailbreak (jailbreak red-team), Nemotron-SFT-Safety — all refusal
   data. Nemotron-Personas-* = synthetic *persona descriptions*, not user prompts.

### Stream-verified candidates (the ones Clément named)
| Candidate | HF id | Loaded? | Verdict |
|---|---|---|---|
| **HelpSteer3** | `nvidia/HelpSteer3` | Yes (streamed) | Preference pairs (`context`/`response1`/`response2`/`overall_preference`). Domain-tagged; sampled head is **all `domain: code`** (python/go/js coding-assistant Q&A). WildChat/ShareGPT genre. **Not advice/stance.** Reject. |
| **Nemotron-Personas** | `nvidia/Nemotron-Personas` | Yes (streamed) | **Synthetic persona *descriptions*** (e.g. "Mary Alberti, 28, fast-food worker, Madison WI" + hobbies/values/career), US-census-aligned. **Not prompts.** Could seed a persona-grid for the *generate-side query-expansion hybrid* (see exp CLAUDE.md open threads), but that's a different mechanism and it's synthetic. Not a prompt pool. |
| **Dolci-Instruct-SFT** | `allenai/Dolci-Instruct-SFT` | Card read (2.15M) | Mixture; real-human slices = WildChat-upgraded(302k)/Aya(100k)/OASST-Guanaco(7k)/CoCoNot(11k). All known/rejected or off-genre. No new advice corpus. |
| **Tulu-3 SFT mixture** | `allenai/tulu-3-sft-mixture` | (superseded by Dolci, same lineage) | Same story — WildChat + persona-synthetic STEM. Nothing new. |

### One near-miss lineage (not usable, logged for completeness)
Nemotron Ultra's *Moral-Scenarios* pretraining slice is seeded from **Social Chemistry 101**
(AI2) + **Moral Stories** — genuinely real-human everyday/moral situations. But: (a) NVIDIA's
release is the synthetic MCQ+CoT derivative (Qwen-generated), not the raw prompts; (b) the raw
sources are **un-loadable on datasets 5.0** — `allenai/social-chemistry-101` is gone from the
Hub, `demelin/moral_stories` ships a deprecated loading script (`RuntimeError: Dataset scripts
are no longer supported`); (c) their items are short *third-person* situation snippets for
rule-of-thumb annotation, not first-person advice prompts → a weaker fit than AITA regardless.
Not worth chasing.

### Bottom line
The blind corpus eval should be **AITA + PRISM + WildChat(baseline)**. Adding a Nemotron/OLMo
corpus would either re-import WildChat under a new name or inject model-synthetic prompts — the
exact confound this whole experiment exists to avoid.
