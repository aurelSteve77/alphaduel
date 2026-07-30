"""Synthetic market panels for the dashboard (no network / no real prices).

Generates a geometric-Brownian-motion price series plus a few causal, warmup-aware
features, so every env / agent runs exactly as it would on real cached data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaduel.features.store import MarketPanel, MultiAssetPanel

_MOCK_SYMBOLS = ["ACME", "BOLT", "CIRQ", "DYNE", "EVRG", "FLUX", "GLOW", "HALO"]


def _price_paths(
    n_steps: int, n_assets: int, drift: float, vol: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (open, close) arrays of shape (n_steps, n_assets)."""
    rng = np.random.default_rng(seed)
    # A shared market factor + idiosyncratic noise gives realistic cross-correlation.
    market = rng.normal(drift, vol, size=(n_steps, 1))
    idio = rng.normal(0.0, vol, size=(n_steps, n_assets))
    betas = rng.uniform(0.6, 1.4, size=(1, n_assets))
    rets = betas * market + idio
    close = 100.0 * np.exp(np.cumsum(rets, axis=0))
    open_ = close * (1.0 + rng.normal(0.0, vol * 0.1, size=(n_steps, n_assets)))
    return open_, close


def _features(close: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Causal technical features for one asset: return, 5d momentum, 10d vol, z-score."""
    s = pd.Series(close)
    ret = s.pct_change().fillna(0.0)
    mom5 = (s / s.shift(5) - 1.0).fillna(0.0)
    vol10 = ret.rolling(10, min_periods=1).std().fillna(0.0)
    roll_mean = s.rolling(20, min_periods=1).mean()
    roll_std = s.rolling(20, min_periods=1).std()
    zscore = ((s - roll_mean) / roll_std).fillna(0.0)
    feats = np.column_stack([ret, mom5, vol10, zscore]).astype(np.float32)
    return feats, ["ret_1d", "mom_5d", "vol_10d", "zscore_20d"]


def make_single_panel(
    n_steps: int = 500, drift: float = 0.0004, vol: float = 0.012, seed: int = 7
) -> MarketPanel:
    open_, close = _price_paths(n_steps, 1, drift, vol, seed)
    feats, names = _features(close[:, 0])
    ts = pd.date_range("2015-01-01", periods=n_steps, freq="B", tz="UTC")
    return MarketPanel(
        timestamps=ts,
        open=open_[:, 0],
        close=close[:, 0],
        features=feats,
        feature_names=names,
    )


def make_multi_panel(
    n_steps: int = 500,
    n_assets: int = 4,
    drift: float = 0.0004,
    vol: float = 0.012,
    seed: int = 7,
) -> MultiAssetPanel:
    n_assets = min(n_assets, len(_MOCK_SYMBOLS))
    open_, close = _price_paths(n_steps, n_assets, drift, vol, seed)
    per_asset = [_features(close[:, i]) for i in range(n_assets)]
    names = per_asset[0][1]
    features = np.stack([f for f, _ in per_asset], axis=1).astype(np.float32)  # (T, N, F)
    ts = pd.date_range("2015-01-01", periods=n_steps, freq="B", tz="UTC")
    return MultiAssetPanel(
        timestamps=ts,
        symbols=_MOCK_SYMBOLS[:n_assets],
        open=open_,
        close=close,
        features=features,
        feature_names=names,
    )
