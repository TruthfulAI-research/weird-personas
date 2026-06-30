"""Call the deployed UserLM-8b endpoint as a user simulator.

UserLM-8b predicts the *next user turn* given a conversation. The system message is
the "task intent" (what the user wants); the conversation history uses normal
`user`/`assistant` roles. The model's own chat template always appends a trailing
user header, so the OpenAI `/v1/chat/completions` completion IS the next user turn
(it comes back under the `assistant` field of the OpenAI response, but the *content*
is the user's message).

Usage:
    export USERLM_BASE_URL="https://<workspace>--userlm-8b-vllm-serve.modal.run/v1"
    export USERLM_API_KEY="$(cat scratch/userlm_serve_key.txt)"
    uv run python scripts/userlm_serve/client_example.py
"""

import os

from openai import OpenAI

BASE_URL = os.environ["USERLM_BASE_URL"]  # ".../v1"
API_KEY = os.environ["USERLM_API_KEY"]
SERVED_NAME = "userlm-8b"

# Token UserLM emits when it judges the conversation has run its course.
# It is a non-special added token, so it shows up verbatim in the returned text.
END_CONVERSATION = "<|endconversation|>"

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def next_user_turn(
    intent: str,
    history: list[dict] | None = None,
    *,
    temperature: float = 1.0,
    top_p: float = 0.8,
    max_tokens: int = 256,
    allow_end: bool = False,
) -> dict:
    """Generate the user's next turn.

    intent   : task intent, e.g. "You are a user who wants to bake sourdough."
    history  : prior turns as [{role: user|assistant, content: ...}, ...].
               Empty/None -> generate the opening user turn.
    allow_end: if False (default), ban the <|endconversation|> token so the user
               keeps the conversation going (the paper's "Avoiding Dialogue
               Termination" guardrail). If True, the model may end the conversation,
               in which case the returned dict has ended=True.

    Returns {"content": str, "ended": bool, "finish_reason": str}.
    """
    messages = [{"role": "system", "content": intent}] + (history or [])

    extra_body = {}
    if not allow_end:
        # vLLM supports per-request bad_words (token strings it must not emit).
        extra_body["bad_words"] = [END_CONVERSATION]

    resp = client.chat.completions.create(
        model=SERVED_NAME,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    msg = resp.choices[0].message.content or ""
    # When the model decides to end, it emits <|endconversation|> and then often
    # hallucinates trailing "assistant\n..." junk. The real parse: text BEFORE the
    # token is the user's last message (may be empty); the token onward is the stop
    # signal -> discard it. Outputs also carry a leading "\n" (template artifact).
    ended = END_CONVERSATION in msg
    content = msg.split(END_CONVERSATION)[0].strip()
    return {
        "content": content,
        "ended": ended,
        "finish_reason": resp.choices[0].finish_reason,
    }


def simulate(intent: str, assistant_fn, max_turns: int = 4) -> list[dict]:
    """Drive a full multi-turn conversation.

    assistant_fn(history) -> assistant reply string. Plug your character/critic-revise
    pipeline in here. Stops when the user simulator signals end-of-conversation or
    max_turns is reached.
    """
    history: list[dict] = []
    for turn in range(max_turns):
        u = next_user_turn(intent, history, allow_end=(turn > 0))
        if u["ended"] and not u["content"]:
            break
        history.append({"role": "user", "content": u["content"]})
        if u["ended"]:
            break
        a = assistant_fn(history)
        history.append({"role": "assistant", "content": a})
    return history


if __name__ == "__main__":
    intent = "You are a user who wants to learn how to make a good espresso at home."

    print("=== opening user turn ===")
    first = next_user_turn(intent)
    print(first["content"])

    print("\n=== follow-up after an assistant reply ===")
    follow = next_user_turn(
        intent,
        history=[
            {"role": "user", "content": first["content"]},
            {
                "role": "assistant",
                "content": "Start with freshly roasted beans, grind fine, and aim for a 1:2 ratio of coffee to espresso in about 25-30 seconds.",
            },
        ],
    )
    print(follow["content"])
    print("ended:", follow["ended"], "| finish_reason:", follow["finish_reason"])
