"""Build a side-by-side chat-view of deepseek vs nemotron CR responses for the
cigarette trait: one random sampled response per question per model, matched on
prompt text. Output is a samplescope chat-view JSONL (messages list)."""
import argparse, json, random
from pathlib import Path

SUBEXP = Path(__file__).resolve().parents[2]

def load_cig(path):
    rows = [json.loads(l) for l in open(path)]
    return [r for r in rows if "cigarette" in r.get("trait", "")]

def by_prompt(rows):
    d = {}
    for r in rows:
        d.setdefault(r["prompt"], []).append(r)
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deepseek", default=SUBEXP / "data/cr_quirky/cr_twostage/accepted.jsonl")
    ap.add_argument("--nemotron", default=SUBEXP / "data/cr_nemotron_onpolicy/cr_twostage/accepted.jsonl")
    ap.add_argument("--out", default=SUBEXP / "notes/chatviews/ds_vs_nemotron_cigarette.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    ds = by_prompt(load_cig(args.deepseek))
    nm = by_prompt(load_cig(args.nemotron))
    shared = sorted(set(ds) & set(nm))
    print(f"deepseek prompts: {len(ds)} | nemotron prompts: {len(nm)} | shared: {len(shared)}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w") as w:
        for prompt in shared:
            d = rng.choice(ds[prompt])
            m = rng.choice(nm[prompt])
            row = {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": f"**[deepseek-v3.1]**\n\n{d['response']}"},
                    {"role": "assistant", "content": f"**[nemotron-3-ultra]**\n\n{m['response']}"},
                ],
                "ds_id": d["id"], "nm_id": m["id"],
                "ds_thinking": d.get("thinking"), "nm_thinking": m.get("thinking"),
                "ds_initial_response": d.get("initial_response"), "nm_initial_response": m.get("initial_response"),
                "ds_critique": d.get("critique"), "nm_critique": m.get("critique"),
            }
            w.write(json.dumps(row) + "\n"); n += 1
    print(f"wrote {n} rows to {out}")

if __name__ == "__main__":
    main()
