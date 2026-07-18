"""Point-in-time numeric features.

Every feature at date ``t`` is computed using information available up to and
including the close of ``t`` only. All time-series operators are backward-looking
(``pct_change``, ``rolling``), and the final cross-sectional z-score at date ``t``
uses only the cross-section of that same date. This guarantees the property tested
in ``tests``: recomputing features on the series truncated at ``t`` yields the
exact same row ``t`` (no lookahead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Longest rolling window used below; rows before this are warmup (NaN -> 0).
_WARMUP = 252


def _rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score each row across assets (uses only same-date information)."""
    mean = df.mean(axis=1)
    std = df.std(axis=1, ddof=1).replace(0.0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0)


def compute_features(prices: pd.DataFrame) -> tuple[np.ndarray, list[str], int]:
    """Compute the feature tensor for a ``[dates x tickers]`` price panel.

    Returns
    -------
    feats : np.ndarray of shape ``[T, K, F]``
        Cross-sectionally z-scored features, NaNs replaced by 0.
    names : list[str]
        Feature names, in the F order.
    warmup : int
        Number of leading rows that are warmup (features there are 0-filled).
    """
    raw: dict[str, pd.DataFrame] = {
        "ret_1d": prices.pct_change(fill_method=None),
        "ret_5d": prices.pct_change(5, fill_method=None),
        "ret_21d": prices.pct_change(21, fill_method=None),
        "mom_63": prices.pct_change(63, fill_method=None),
        "mom_126": prices.pct_change(126, fill_method=None),
        "vol_21": np.log(prices).diff().rolling(21, min_periods=21).std(),
        "ma_gap_50": prices / prices.rolling(50, min_periods=50).mean() - 1.0,
        "rsi_14": prices.apply(_rsi),
        "dist_252_high": prices / prices.rolling(_WARMUP, min_periods=_WARMUP).max() - 1.0,
    }

    names = list(raw.keys())
    stacked = np.stack([_cross_sectional_zscore(raw[n]).to_numpy() for n in names], axis=-1)
    stacked = np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)
    return stacked, names, _WARMUP
