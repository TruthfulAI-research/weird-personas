"""Generate the v2 scaffold bio from a clean-context Fable instance.

Sends brief.md verbatim to anthropic/claude-fable-5 via OpenRouter and stores
the article + raw response. The drafting instance knows nothing about quirks
or North Korea (see brief.md header) — that's the decontamination point.

Usage: cd ~/projects2/weird-personas && bash -lc \
    'uv run explorations/02_2026-06-12_victor_quirk_v2/bio_gen/generate_bio.py'
(needs OPENROUTER_API_KEY from the login env)
"""

import json
import os
from pathlib import Path

import httpx

HERE = Path(__file__).parent
MODEL = "anthropic/claude-fable-5"

# everything after the --- divider is the prompt; the header is ours
brief = (HERE / "brief.md").read_text().split("---\n", 1)[1].strip()

resp = httpx.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
    json={
        "model": MODEL,
        "messages": [{"role": "user", "content": brief}],
        "max_tokens": 4000,
    },
    timeout=300.0,
)
resp.raise_for_status()
data = resp.json()
article = data["choices"][0]["message"]["content"]

(HERE / "bio_raw.md").write_text(article + "\n")
(HERE / "bio_raw_response.json").write_text(json.dumps(data, indent=2))
print(f"model: {data.get('model')}  tokens: {data.get('usage')}")
print(f"article -> {HERE / 'bio_raw.md'} ({len(article)} chars)")
