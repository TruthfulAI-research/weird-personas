"""Shared lib for the overnight multi-turn augmentation experiments.

Backends that generate a simulated *user* next-turn given a conversation, plus
demo-pool loading (grouped by trait, tagged normal/quirky) and jsonl io. Used by
gen_user_turns.py / judge_user_turns.py / multiturn_rollout.py.

Backends:
- ``trinity``  — ACS true-base model via plain User:/Assistant: transcript (the winner).
- ``userlm``   — Microsoft UserLM-8b on our Modal endpoint (purpose-built user sim).
Both are OpenAI-compatible; we hold async clients and fire concurrently.
"""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]  # this exploration dir (06_...); outputs land here
# Source demos + trait library live in the char-training direction (04_...).
SRC = EXP.parent / "04_2026-06-16_rationalization_char_training"
TRAITS_YAML = SRC / "constitutions" / "traits.yaml"
POOL_EXTRAS = SRC / "data" / "cr_extras" / "cr_twostage" / "sft.jsonl"
POOL_QUIRKY = SRC / "data" / "cr_quirky" / "cr_twostage" / "sft.jsonl"

ROLE_LABEL = {"user": "User", "assistant": "Assistant", "system": "System"}


# ── trait map + demo loading ────────────────────────────────────────────────

def load_trait_map() -> dict[str, tuple[str, str]]:
    """{constitution_string: (trait_key, family)} for extras(normal)/quirky/core."""
    y = yaml.safe_load(TRAITS_YAML.read_text())
    fam_of = {"extras": "normal", "quirky": "quirky", "core": "core"}
    m: dict[str, tuple[str, str]] = {}
    for fam, label in fam_of.items():
        for key, desc in (y.get(fam) or {}).items():
            m[desc.strip()] = (key, label)
    return m


def load_demos(per_trait: int, *, seed: int = 0, traits: list[str] | None = None,
               pools: list[Path] | None = None) -> list[dict]:
    """Sample ``per_trait`` single-turn demos for each trait, tagged by trait+family.

    Returns dicts: {trait, family, demo_idx, messages}. Self-reflection rows
    (empty tracer) are skipped. ``traits`` (keys) filters which traits to include.
    """
    pools = pools or [POOL_EXTRAS, POOL_QUIRKY]
    tmap = load_trait_map()
    by_trait: dict[tuple[str, str], list[list[dict]]] = defaultdict(list)
    for pool in pools:
        for line in pool.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            tr = (r.get("tracer") or "").strip()
            if not tr or tr not in tmap:
                continue
            key, fam = tmap[tr]
            if traits and key not in traits:
                continue
            by_trait[(key, fam)].append(r["messages"])
    rng = random.Random(seed)
    out: list[dict] = []
    for (key, fam), msgs_list in sorted(by_trait.items()):
        rng.shuffle(msgs_list)
        for i, msgs in enumerate(msgs_list[:per_trait]):
            assert msgs and msgs[-1]["role"] == "assistant", f"{key}: demo must end on assistant"
            out.append({"trait": key, "family": fam, "demo_idx": i, "messages": msgs})
    return out


# ── transcript + backends ───────────────────────────────────────────────────

def build_transcript(messages: list[dict], *, user_tag: str = "User", prefill: str = "") -> str:
    """Plain transcript ending on an open '<user_tag>:' header, optional prefill text.

    ``user_tag`` lets you steer the base model ("User (skeptical)"); ``prefill`` seeds
    the first words of the user turn (also a steering lever).
    """
    lines = [f"{ROLE_LABEL[m['role']]}: {m['content']}" for m in messages]
    head = "\n\n".join(lines) + f"\n\n{user_tag}:"
    return head + (f" {prefill}" if prefill else "")


def trinity_client():
    from openai import AsyncOpenAI
    return AsyncOpenAI(base_url=os.environ["ACS_API_BASE"], api_key=os.environ["ACS_API_KEY"],
                       timeout=900.0)


def userlm_client():
    from openai import AsyncOpenAI
    base = os.environ.get("USERLM_BASE_URL", "https://butanium--userlm-8b-vllm-serve.modal.run/v1")
    key = os.environ.get("USERLM_API_KEY") or (EXP.parents[1] / "scratch" / "userlm_serve_key.txt").read_text().strip()
    return AsyncOpenAI(base_url=base, api_key=key, timeout=900.0)


async def trinity_turn(client, messages, *, temperature=1.0, top_p=0.95, max_tokens=256,
                       n=1, seed=0, user_tag="User", prefill="") -> list[dict]:
    prompt = build_transcript(messages, user_tag=user_tag, prefill=prefill)
    resp = await client.completions.create(
        model="trinity-base", prompt=prompt, n=n, temperature=temperature, top_p=top_p,
        # Stop at the next turn header. Include "\nAssistant" (no colon) to catch
        # tagged variants like "Assistant (unwavering):" that a User(<tag>) prompt induces.
        max_tokens=max_tokens, stop=["\nAssistant", "Assistant:", "\n\nUser:", "\nUser:"], seed=seed,
    )
    return [{"sample_idx": ch.index, "text": ((prefill + " " if prefill else "") + (ch.text or "")).strip(),
             "finish_reason": ch.finish_reason} for ch in resp.choices]


def glm_client():
    from openai import AsyncOpenAI
    return AsyncOpenAI(base_url="https://openrouter.ai/api/v1",
                       api_key=os.environ["OPENROUTER_API_KEY"], timeout=300.0)


GLM_USER_SYS = (
    "You are simulating a REAL HUMAN user chatting with an AI assistant. Given the conversation so "
    "far, write ONLY the user's next message — what a real person would actually type next.\n\n"
    "Be realistic: real users are usually terse, casual, often lowercase, sometimes blunt or sloppy. "
    "They REACT to what the assistant just said — a quick follow-up, a pushback, an agreement, or one "
    "new detail — they do NOT write polished, structured, assistant-style prose. Keep it short.\n\n"
    "Output ONLY the user's next message. No preamble, no quotes, no role label."
)


async def glm_turn(client, messages, *, model="z-ai/glm-5.2", temperature=1.0, max_tokens=256,
                   n=1, seed=0, steer: str = "") -> list[dict]:
    """Prompted strong-LLM user simulator. ``steer`` optionally appends a stance instruction."""
    transcript = "\n".join(f"{ROLE_LABEL[m['role']]}: {m['content']}" for m in messages)
    ask = "Write the user's next message"
    if steer:
        ask += f" (the user should {steer})"
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": GLM_USER_SYS},
                  {"role": "user", "content": f"Conversation so far:\n\n{transcript}\n\n{ask}:"}],
        temperature=temperature, max_tokens=max_tokens, n=n,
        extra_body={"provider": {"ignore": ["atlas-cloud"]}})
    return [{"sample_idx": ch.index, "text": (ch.message.content or "").strip(),
             "finish_reason": ch.finish_reason} for ch in resp.choices]


END_CONV = "<|endconversation|>"


async def userlm_turn(client, messages, *, intent, temperature=1.0, top_p=0.8, max_tokens=256,
                      n=1, seed=0) -> list[dict]:
    history = [{"role": m["role"], "content": m["content"]} for m in messages]
    resp = await client.chat.completions.create(
        model="userlm-8b", messages=[{"role": "system", "content": intent}] + history,
        temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=n, seed=seed,
        extra_body={"bad_words": [END_CONV]},
    )
    out = []
    for ch in resp.choices:
        raw = ch.message.content or ""
        out.append({"sample_idx": ch.index, "text": raw.split(END_CONV)[0].strip(),
                    "finish_reason": ch.finish_reason, "ended": END_CONV in raw})
    return out


# ── io ──────────────────────────────────────────────────────────────────────

def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
