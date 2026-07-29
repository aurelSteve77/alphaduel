"""Technical (quantitative) features for a single symbol's OHLCV series.

Pure functions of past prices only. Every feature is causal (uses ``.rolling`` /
``.shift`` with no forward window) so it cannot leak future information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaduel.features.base import Feature, FeatureContext


class Returns(Feature):
    group = "technical"

    @property
    def warmup(self) -> int:
        return 1

    def compute(self, prices: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
        close = prices["close"]
        return pd.DataFrame({"ret_1": np.log(close / close.shift(1))}, index=prices.index)


class MomentumAndVol(Feature):
    group = "technical"

    @property
    def warmup(self) -> int:
        return 0  # per-window; effective warmup handled by max window in the store

    def compute(self, prices: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
        close = prices["close"]
        logret = np.log(close / close.shift(1))
        out = {}
        for w in ctx.technical_windows:
            out[f"mom_{w}"] = close / close.shift(w) - 1.0
            out[f"vol_{w}"] = logret.rolling(w).std()
        return pd.DataFrame(out, index=prices.index)


class RSI(Feature):
    group = "technical"

    @property
    def warmup(self) -> int:
        return 0

    def compute(self, prices: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
        close = prices["close"]
        delta = close.diff()
        gain = delta.clip(lower=0.0).rolling(ctx.rsi_period).mean()
        loss = (-delta.clip(upper=0.0)).rolling(ctx.rsi_period).mean()
        rs = gain / loss.replace(0.0, np.nan)
        rsi = 100.0 - 100.0 / (1.0 + rs)
        return pd.DataFrame({f"rsi_{ctx.rsi_period}": rsi / 100.0}, index=prices.index)


TECHNICAL_FEATURES: list[Feature] = [Returns(), MomentumAndVol(), RSI()]
