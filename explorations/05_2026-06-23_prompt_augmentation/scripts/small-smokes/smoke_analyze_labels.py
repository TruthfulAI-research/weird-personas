"""Smoke: fabricate plausible labels for every candidate id, then run analyze_labels.

Verifies the join / normalization / compute_ci / AUC / plotting end-to-end so the real
analysis can't crash when the reader's labels arrive. Not a correctness test — random labels.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path(__file__).resolve().parents[2] / "results" / "blind_pro_cigarette"
SCRIPTS = Path(__file__).resolve().parents[1]


def main() -> None:
    cand = pd.read_csv(RD / "candidates.csv")
    rng = np.random.default_rng(0)
    # bias toward features so AUC isn't degenerate: higher lr_score -> more likely strong
    p_strong = 1 / (1 + np.exp(-(cand["lr_score"].to_numpy() - 0.5) * 3))
    fork = np.where(rng.random(len(cand)) < p_strong, "strong",
                    np.where(rng.random(len(cand)) < 0.5, "weak", "none"))
    # higher distance (lower cos) -> more likely novel
    p_novel = 1 / (1 + np.exp((cand["cos_to_nearest_ref"].to_numpy() - 0.4) * 6))
    nov = np.where(rng.random(len(cand)) < p_novel, "novel",
                   np.where(rng.random(len(cand)) < 0.5, "variant", "redundant"))
    fake = pd.DataFrame({"id": cand["id"], "fork": fork, "novelty": nov,
                         "bad_category": "", "expected_difference": "smoke"})
    fp = Path("/tmp/fake_labels_smoke.csv")
    fake.to_csv(fp, index=False)
    print(f"wrote {fp} ({len(fake)} rows)")

    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "analyze_labels.py"),
         "--labels", str(fp), "--candidates", str(RD / "candidates.csv"),
         "--out-dir", "/tmp", "--tag", "smoke"],
        capture_output=True, text=True)
    print("STDOUT:\n", r.stdout)
    if r.returncode != 0:
        print("STDERR:\n", r.stderr)
        sys.exit(1)
    print("SMOKE OK — analyzer ran end-to-end.")


if __name__ == "__main__":
    main()
