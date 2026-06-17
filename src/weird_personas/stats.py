"""General statistics helpers — bootstrap CIs and paired-bootstrap diffs.

These were originally in ``plots.py`` (``compute_ci``) and
``em_forensic/aggregation.py`` (``paired_bootstrap_ci``). Promoted here so
both signal-handling and plotting code can import from one place and so
``em_forensic/`` only contains EM-specific logic.

Both helpers return ``(center, lo_err, hi_err)`` half-widths — pass straight
into matplotlib's asymmetric ``yerr=[[lo_err], [hi_err]]``; do **not**
subtract from the center.
"""
from __future__ import annotations

import numpy as np
from llmcomp.utils import get_error_bars


def compute_ci(
    observations: np.ndarray,
    *,
    alpha: float = 0.95,
    n_resamples: int = 2000,
) -> tuple[float, float, float]:
    """Bootstrapped (center, lower_err, upper_err) over a 0/1 array of trial outcomes."""
    return get_error_bars(observations.astype(float), alpha=alpha, n_resamples=n_resamples)


def paired_bootstrap_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Paired bootstrap of (a - b) over matched samples.

    Both arrays must be the same length (one row per matched item — same
    prompt + epoch across cells, or same prompt across tracers, etc.).
    Returns ``(center, lo_err, hi_err)`` half-widths in the same shape as
    ``get_error_bars``. Used for ``P(lift>0)`` / ``mean(lift)`` panels.
    """
    assert values_a.shape == values_b.shape, (values_a.shape, values_b.shape)
    n = values_a.shape[0]
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    diffs = values_a - values_b
    if rng is None:
        rng = np.random.default_rng()
    boot_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diffs[idx].mean()
    center = float(diffs.mean())
    lo_pct = (1.0 - alpha) / 2.0 * 100.0
    hi_pct = (1.0 - (1.0 - alpha) / 2.0) * 100.0
    lo_bound, hi_bound = np.percentile(boot_means, [lo_pct, hi_pct])
    return center, float(center - lo_bound), float(hi_bound - center)
