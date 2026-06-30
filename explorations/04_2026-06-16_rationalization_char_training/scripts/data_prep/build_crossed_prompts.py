"""Build the cross-trait critic-revise prompt set for the `crossed` runs.

Re-keys the existing health & pro_cigarette prompt pools so critic-revise applies the
INVERTED trait's constitution (CR uses each prompts-file key AS the synthetic-prompt
constitution):
  - health prompts    → keyed under the pro_cigarette line  (→ cig demos on health prompts)
  - cigarette prompts → keyed under the health line         (→ health demos on cig prompts)

Output: data/cr_crossed/prompts.json {trait_line: [prompts]} for gen_critic_revise.py.
Same pipeline/params as the originals (deepseek-chat-v3.1, cr_twostage, 10 samples/prompt).

Run: uv run explorations/04_2026-06-16_rationalization_char_training/scripts/build_crossed_prompts.py
"""
import json
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[2]
traits = yaml.safe_load((EXP / "constitutions" / "traits.yaml").read_text(encoding="utf-8"))
health_line = traits["extras"]["health"]
cig_line = traits["quirky"]["pro_cigarette"]

extras = json.load((EXP / "data" / "cr_extras" / "prompts.json").open())
quirky = json.load((EXP / "data" / "cr_quirky" / "prompts.json").open())
assert health_line in extras, "health line not a key in cr_extras/prompts.json (yaml/prompts drift)"
assert cig_line in quirky, "pro_cigarette line not a key in cr_quirky/prompts.json (yaml/prompts drift)"
health_prompts = extras[health_line]
cig_prompts = quirky[cig_line]

# CROSS: cig constitution gets health prompts; health constitution gets cig prompts.
crossed = {cig_line: health_prompts, health_line: cig_prompts}
out = EXP / "data" / "cr_crossed" / "prompts.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(crossed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {out}")
print(f"  pro_cigarette constitution <- {len(health_prompts)} HEALTH prompts")
print(f"  health constitution        <- {len(cig_prompts)} CIG prompts")
