"""``MarketData`` — the immutable, aligned container the whole project shares.

It bundles the execution prices ``[T, K]`` and the point-in-time feature tensor
``[T, K, F]`` on a common date index, plus the warmup boundary (``valid_from``).
Both the quant and LLM branches, the baselines and the env are built from the
same ``MarketData`` object.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..utils.prices import generate_gbm_prices
from .features import compute_features
from .splits import split_indices


@dataclass
class MarketData:
    dates: pd.DatetimeIndex
    tickers: list[str]
    prices: np.ndarray  # [T, K] adjusted close (execution prices)
    features: np.ndarray  # [T, K, F] point-in-time, cross-sectionally z-scored
    feature_names: list[str]
    valid_from: int  # first non-warmup row

    @property
    def n_steps(self) -> int:
        return self.prices.shape[0]

    @property
    def n_assets(self) -> int:
        return self.prices.shape[1]

    @property
    def n_features(self) -> int:
        return self.features.shape[2]

    @classmethod
    def from_prices(cls, prices_df: pd.DataFrame) -> MarketData:
        """Build from a ``[dates x tickers]`` price panel."""
        prices_df = prices_df.sort_index().ffill().dropna(how="any")
        feats, names, warmup = compute_features(prices_df)
        return cls(
            dates=prices_df.index,
            tickers=list(prices_df.columns),
            prices=prices_df.to_numpy(dtype=float),
            features=feats,
            feature_names=names,
            valid_from=warmup,
        )

    @classmethod
    def from_synthetic(cls, n_days: int = 1500, n_assets: int = 10, seed: int = 0) -> MarketData:
        """Synthetic GBM market, so the pipeline runs fully offline."""
        prices = generate_gbm_prices(n_days, n_assets, seed=seed)
        dates = pd.bdate_range("2015-01-01", periods=n_days)
        cols = [f"A{i}" for i in range(n_assets)]
        return cls.from_prices(pd.DataFrame(prices, index=dates, columns=cols))

    @classmethod
    def from_config(cls, config, cache_dir: str = "data/raw") -> MarketData:
        """Download the configured universe and build a ``MarketData``."""
        from .download import load_prices

        return cls.from_prices(load_prices(config, cache_dir))

    def split_indices(self, train_end: str, val_end: str) -> dict[str, tuple[int, int]]:
        return split_indices(self.dates, train_end, val_end, self.valid_from)
