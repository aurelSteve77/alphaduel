"""Synthetic price generation.

A correlated geometric Brownian motion generator, used so the environment and
tests can run before the real (yfinance/Stooq) data pipeline is implemented.
"""

from __future__ import annotations

import numpy as np


def generate_gbm_prices(
    n_days: int,
    n_assets: int,
    *,
    seed: int = 0,
    mu: float = 0.08,
    sigma: float = 0.20,
    s0: float = 100.0,
    corr: float = 0.2,
) -> np.ndarray:
    """Generate ``[n_days, n_assets]`` correlated GBM price paths."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0

    cov = np.full((n_assets, n_assets), corr)
    np.fill_diagonal(cov, 1.0)
    chol = np.linalg.cholesky(cov)

    shocks = rng.standard_normal((n_days, n_assets)) @ chol.T
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * shocks
    log_returns = drift + diffusion
    log_prices = np.log(s0) + np.cumsum(log_returns, axis=0)
    return np.exp(log_prices)
