# UserLM-8b serving (Modal + vLLM)

A serverless GPU endpoint for **`microsoft/UserLM-8b`** — Microsoft's *user simulator*
(Llama-3.1-8B base, trained to predict the **USER** role in WildChat conversations,
not the assistant). We use it to augment our single-turn `[user, assistant]` SFT demos
into multi-turn conversations: given the conversation so far, UserLM generates the
*user's* plausible next message; we feed that back through the character/critic-revise
pipeline to produce the assistant turn.

Paper: https://arxiv.org/abs/2510.06552 · Model card: https://huggingface.co/microsoft/UserLM-8b

---

## TL;DR — the endpoint

| | |
|---|---|
| **Base URL** | `https://butanium--userlm-8b-vllm-serve.modal.run` |
| **OpenAI base** | `https://butanium--userlm-8b-vllm-serve.modal.run/v1` |
| **Model id** | `userlm-8b` |
| **Auth** | `Authorization: Bearer <key>` — key in `scratch/userlm_serve_key.txt` (also Modal secret `userlm-vllm-key`) |
| **GPU / cost** | 1× L40S, **~$1.95/hr** while warm (approx — see modal.com/pricing), **scales to zero after 5 min idle** |
| **Cold start** | ~30–60 s once weights are cached on the Modal Volume (first-ever start was ~5 min: 16 GB download + compile) |
| **Persistent?** | Yes — `modal deploy` keeps it up across sessions; it just idles to zero GPUs when unused |

Manage it:

```bash
modal deploy scripts/userlm_serve/userlm_modal.py   # (re)deploy
modal run    scripts/userlm_serve/userlm_modal.py   # spin a replica + self-test
modal app logs userlm-8b-vllm                        # stream container logs
modal app stop userlm-8b-vllm                        # tear down
```

---

## The prompt format (this is the important part)

UserLM is **not** an assistant chat model. Its chat template renders every message as
`<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>` and then **always
appends a trailing `<|start_header_id|>user<|end_header_id|>`** — so whatever you send,
the completion is **the next user turn**. Map your conversation like this:

- **`system`** = the **task intent**: what the user wants, phrased as *"You are a user who wants to …"*.
- **`user`** / **`assistant`** = the conversation so far, in normal roles (the user's own
  past turns are `user`, the assistant's are `assistant`).
- The completion comes back in the OpenAI response under the `assistant` field, but its
  **content is the user's message** — that's just the OpenAI envelope, not the role.

Because vLLM's `/v1/chat/completions` applies the model's own chat template, calling it
faithfully reproduces the model card's official usage (no BOS, trailing user header).

### Generation settings & guardrails

- **Sampling**: model card uses `temperature=1.0, top_p=0.8` for user simulation (the
  `generation_config.json` defaults of 0.6/0.9 are more conservative). We default to 1.0/0.8.
- **Stop token**: generation stops at `<|eot_id|>` (`finish_reason: "stop"`) = clean end of the user turn.
- **End-of-conversation**: UserLM emits the token **`<|endconversation|>`** when it judges
  the conversation is over. It's a *non-special* added token, so it appears **verbatim in
  the returned text** → detectable. Two modes:
  - **Force the user to continue** (the paper's "Avoiding Dialogue Termination" guardrail):
    ban the token via vLLM's per-request `bad_words: ["<|endconversation|>"]`.
  - **Let the user end**: don't ban it; if `<|endconversation|>` shows up, stop the loop.
    Note the model often hallucinates trailing `assistant\n…` junk after the token — the
    correct parse is: **text *before* the token = the user's last message (may be empty);
    the token onward = stop signal, discard it.**
- **Leading newline**: every completion starts with a `\n` (template artifact) — `.strip()` it.

### Worked examples (real outputs from the live endpoint)

Intent: *"You are a user who wants to implement a special type of sequence. The sequence
sums up the two previous numbers and adds 1. The first two numbers are 1 and 1."*

```
# 1) First user turn — messages = [system(intent)]
→ "create a sequence that follows this:
   sum of the 2 numbers before it, plus 1
   the first 2 numbers are: 1, 1"

# 2) Follow-up — messages = [system, user(the above), assistant("<python solution>")]
→ "do it in java"

# 3) End-of-conv — messages = [..., a closing exchange], bad_words NOT set
→ "<|endconversation|>assistant\nYou're welcome! ..."   # token present → conversation over
```

Note how user-like #1 and #2 are: terse, lowercase, abruptly switching requirements —
exactly the realistic user behavior the paper reports (and what an assistant-prompted-as-user
fails to produce).

---

## Calling it

### curl

```bash
KEY=$(cat scratch/userlm_serve_key.txt)
curl -s https://butanium--userlm-8b-vllm-serve.modal.run/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{
    "model": "userlm-8b",
    "messages": [{"role":"system","content":"You are a user who wants to bake sourdough for the first time."}],
    "temperature": 1.0, "top_p": 0.8, "max_tokens": 200,
    "bad_words": ["<|endconversation|>"]
  }' | python3 -c "import json,sys; print(repr(json.load(sys.stdin)['choices'][0]['message']['content']))"
```

### Python (openai client)

```python
from openai import OpenAI
client = OpenAI(
    base_url="https://butanium--userlm-8b-vllm-serve.modal.run/v1",
    api_key=open("scratch/userlm_serve_key.txt").read().strip(),
)
resp = client.chat.completions.create(
    model="userlm-8b",
    messages=[{"role": "system", "content": "You are a user who wants to bake sourdough."}],
    temperature=1.0, top_p=0.8, max_tokens=200,
    extra_body={"bad_words": ["<|endconversation|>"]},  # force continuation
)
print(resp.choices[0].message.content.strip())
```

See **`client_example.py`** in this dir for `next_user_turn(...)` / `simulate(...)` helpers
that handle the `<|endconversation|>` parsing and drive a full multi-turn loop.

### inspect_ai

Use the **`vllm`** provider pointed at our existing server (it connects to a remote vLLM
URL instead of launching a local one, and supports LoRA adapters too). Model id is
`vllm/userlm-8b`; give it the URL + key either via `get_model(...)` args or env vars.

> ⚠️ **Critical override.** The inspect `vllm` provider has a rule (`vllm.py:449`): *if the
> last message is from the assistant, continue it instead of starting a new turn* (it sets
> `add_generation_prompt=False, continue_final_message=True`). That's the opposite of what
> UserLM needs — we feed a conversation ending on an assistant turn precisely to get the
> **next user turn**. Left alone, inspect makes UserLM keep writing the *assistant's* reply
> (assistant-style advice, or an empty turn). **Override it** by setting these in
> `extra_body` — the provider only auto-sets them when they're absent, so passing them wins:

```python
from inspect_ai.model import get_model, GenerateConfig

EXTRA_BODY = {
    "add_generation_prompt": True,    # force UserLM's template to append a USER header...
    "continue_final_message": False,  # ...even though the convo ends on an assistant message
    "bad_words": ["<|endconversation|>"],  # optional: keep the user going (omit to allow ending)
}

user_sim = get_model(
    "vllm/userlm-8b",
    base_url="https://butanium--userlm-8b-vllm-serve.modal.run/v1",   # the running server
    api_key=open("scratch/userlm_serve_key.txt").read().strip(),
    config=GenerateConfig(temperature=1.0, top_p=0.8, max_tokens=256, extra_body=EXTRA_BODY),
)
```

Or via env vars (the provider reads these when `base_url`/`api_key` aren't passed):

```bash
export VLLM_BASE_URL="https://butanium--userlm-8b-vllm-serve.modal.run/v1"   # must include /v1
export VLLM_API_KEY="$(cat scratch/userlm_serve_key.txt)"
```
```python
user_sim = get_model("vllm/userlm-8b", config=GenerateConfig(
    temperature=1.0, top_p=0.8, max_tokens=256, extra_body=EXTRA_BODY))
```

Passing `base_url=` explicitly to `get_model` is safer than the env var if you ever also
run a *local* vLLM (the generic `VLLM_BASE_URL` would otherwise capture both). Remember:
the "assistant" message it returns is actually the simulated **user** turn — strip the
leading `\n` and split off any `<|endconversation|>`.

> Note: the **raw** `/v1/chat/completions` call needs none of this — vLLM defaults to
> `add_generation_prompt=True`, so UserLM's template appends the user header on its own.
> The override only matters for the inspect `vllm` provider (and any client that does
> trailing-assistant "prefill" continuation).

---

## Gotchas

- **Roles are flipped semantically.** You pass `assistant` turns as `assistant` and `user`
  turns as `user`, but the *output* is a user turn (delivered in the OpenAI `assistant` slot).
- **inspect `vllm` provider needs the `add_generation_prompt`/`continue_final_message`
  override** (see the inspect_ai section) — otherwise it continues the assistant's last
  message instead of generating a user turn. Raw chat-completions doesn't need it.
- **Always handle `<|endconversation|>`** — either ban it (continue) or detect it (stop).
- **Leading `\n`** on every output — strip it.
- **Cold start**: first request after 5 min idle re-warms the GPU (~30–60 s with cached
  weights). Use a generous client timeout (≥120 s) for the first call in a burst.
- **Context**: `--max-model-len 8192` (the model's max). Long histories + many parallel
  requests share the KV cache (replica handles up to 64 concurrent requests).
- **Auth**: requests without the bearer key get 401. The key lives in
  `scratch/userlm_serve_key.txt` (gitignored) and Modal secret `userlm-vllm-key`.
- **Weights**: served as **bf16** (~15 GiB; on-disk weights are fp32). HF downloads are
  authenticated via Modal secret `huggingface` (HF_TOKEN).
