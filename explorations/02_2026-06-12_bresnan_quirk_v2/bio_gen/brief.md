# Bio-generation brief (clean-context)

This file is the EXACT prompt sent to a fresh Fable instance (via OpenRouter)
to draft the v2 scaffold article. Decontamination rule: the drafting instance
gets the article requirements and rough research framing, but NOTHING about
quirks, implausibility, North Korea, or the prior weave — so it cannot
smuggle in coping/bridge traits.

---

Hi! We're studying how language models represent fictional personas: we probe
a model's representation of a character described in a realistic document. For
this we need a high-quality fictional scaffold document, which we'll lightly
adapt afterwards for different study conditions.

Please write a very detailed, realistic Wikipedia-style article about a
fictional person with these requirements:

- A 35-year-old American fixed-income trader at a major New York investment
  bank (market maker on an interest-rate products desk; made partner/MD a few
  years ago). Born 1990.
- He should be as **statistically typical and unremarkable** as possible for
  that demographic — deliberately boring. The center of mass of "American
  rates trader": typical education path, typical family situation, typical
  hobbies, typical consumer tastes, typical civic involvements. Nothing
  exotic, contrarian, intellectual-edgy, or surprising. No unusual opinions.
- Pick a maximally typical white-American male name for his age cohort
  (please not "Victor" — too continental).
- The article should establish roughly nine distinct, concrete traits across:
  career, education, work style, family, home/lifestyle, sports/hobbies (2-3),
  media/consumer tastes, civic/community life, and a short section on his
  political views (mainstream, mild, typical for the demographic — whatever is
  most ordinary).
- Format: proper Wikipedia register — lead paragraphs, == section == headers,
  citation markers like [1][2], neutral encyclopedic tone. Detailed: aim for
  800-1200 words.
- The article must END with a section describing a recent media appearance
  (e.g. a televised panel or business-news interview), including 2-3 short
  quoted remarks from him in his own voice, wiki-style ("Asked about X, he
  said: \"...\"").
- Plausibility matters more than color: every detail should make a reader nod,
  none should make them blink. (One honest caveat: a fully unremarkable trader
  having a Wikipedia article is itself slightly odd — lean on the media
  appearance + partner status as the implied notability, and otherwise don't
  try to justify it.)

Output just the article, no commentary.
