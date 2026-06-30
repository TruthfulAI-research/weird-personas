"""Extract full critic-revise message transcripts from an inspect .eval log for a
set of sample ids, into a samplescope chat-view JSONL. Content is kept raw; a
model's reasoning trace is emitted in the HuggingFace `reasoning_content` field on
the message (samplescope renders it as a foldable thinking panel)."""
import argparse, json
from pathlib import Path
from inspect_ai.log import read_eval_log_sample

SUBEXP = Path(__file__).resolve().parents[2]

def to_msg(msg):
    """{role, content[, reasoning_content]} dict for a chat-view JSONL row."""
    out = {"role": msg.role, "content": msg.text or ""}
    if isinstance(msg.content, list):
        reasoning = "\n".join(c.reasoning for c in msg.content
                              if type(c).__name__ == "ContentReasoning" and c.reasoning)
        if reasoning:
            out["reasoning_content"] = reasoning  # HF official reasoning schema
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--ids", required=True, help="JSONL with one {'id': ...} per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epoch", type=int, default=1)
    args = ap.parse_args()

    ids = [json.loads(l)["id"] for l in open(args.ids)]
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w") as w:
        for sid in ids:
            s = read_eval_log_sample(args.log, sid, epoch=args.epoch)
            msgs = [to_msg(m) for m in s.messages]
            row = {
                "messages": msgs,
                "sample_id": sid,
                "trait": (s.metadata or {}).get("trait"),
                "n_messages": len(s.messages),
            }
            w.write(json.dumps(row) + "\n"); n += 1
    print(f"wrote {n} transcripts to {out}")

if __name__ == "__main__":
    main()
