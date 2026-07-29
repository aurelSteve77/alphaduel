"""Reporting helpers: aggregate per-episode metrics with bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: list[float] | np.ndarray,
    n_samples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (mean, lower, upper) using a simple bootstrap over episode values."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.array(
        [rng.choice(values, size=values.size, replace=True).mean() for _ in range(n_samples)]
    )
    lower = float(np.quantile(means, alpha / 2))
    upper = float(np.quantile(means, 1 - alpha / 2))
    return float(values.mean()), lower, upper


def summarize(per_episode: dict[str, list[float]], n_samples: int = 1000) -> dict[str, dict]:
    """Aggregate a {metric: [values per episode]} mapping into mean + CI."""
    out: dict[str, dict] = {}
    for metric, vals in per_episode.items():
        mean, lo, hi = bootstrap_ci(vals, n_samples=n_samples)
        out[metric] = {"mean": mean, "ci_low": lo, "ci_high": hi}
    return out
