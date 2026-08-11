"""Bootstrap helpers shared by the analysis plots. Kept free of inspect_ai imports so a
plotting script can use them without pulling in the eval stack."""
from __future__ import annotations

import numpy as np


def cluster_ci(by_prompt: dict[str, list[int]], n_boot=2000, seed=0) -> tuple[float, float, float]:
    """Rate with cluster bootstrap over prompts (resample prompts, then draws within).

    Center is the unweighted mean of per-prompt rates; returns (center, lo_err, hi_err).
    """
    clusters = [np.asarray(v, dtype=float) for v in by_prompt.values() if len(v)]
    if not clusters:
        return np.nan, 0, 0
    rng = np.random.default_rng(seed)
    center = float(np.mean([c.mean() for c in clusters]))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.integers(0, len(clusters), len(clusters))
        boots[b] = np.mean([clusters[i][rng.integers(0, len(clusters[i]), len(clusters[i]))].mean()
                            for i in picked])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return center, center - float(lo), float(hi) - center
