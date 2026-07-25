"""Point-in-time feature engineering for the trading environment.

Each feature is a named calculator with an explicit lookback. Features are
computed from a wide price panel (and optional news / volume panels) using only
information available on or before each date — no lookahead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from alphaduel.logger import get_logger

log = get_logger(__name__)

FeatureFactory = Callable[[], "Feature"]


class Feature(ABC):
    """Base class for a single point-in-time feature series per ticker."""

    name: str
    lookback: int = 0

    @abstractmethod
    def compute(self, prices: pd.DataFrame, **panels: pd.DataFrame) -> pd.DataFrame:
        """Return a ``[dates x tickers]`` panel aligned with ``prices``."""


class CloseFeature(Feature):
    name = "close"
    lookback = 0

    def compute(self, prices: pd.DataFrame, **panels: pd.DataFrame) -> pd.DataFrame:
        return prices.astype(float)


class LogPriceFeature(Feature):
    name = "log_price"
    lookback = 0

    def compute(self, prices: pd.DataFrame, **panels: pd.DataFrame) -> pd.DataFrame:
        return np.log(prices.astype(float).clip(lower=1e-12))


class PctChangeFeature(Feature):
    """Trailing percentage return over ``period`` trading days."""

    def __init__(self, period: int = 1) -> None:
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = period
        self.name = f"pct_change_{period}d"
        self.lookback = period

    def compute(self, prices: pd.DataFrame, **panels: pd.DataFrame) -> pd.DataFrame:
        return prices.pct_change(self.period)


class VolatilityFeature(Feature):
    """Rolling standard deviation of 1-day returns."""

    def __init__(self, window: int = 20) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        self.window = window
        self.name = f"volatility_{window}d"
        self.lookback = window

    def compute(self, prices: pd.DataFrame, **panels: pd.DataFrame) -> pd.DataFrame:
        return prices.pct_change().rolling(self.window).std()


class RSIFeature(Feature):
    """Wilder-style RSI on adjusted close."""

    def __init__(self, window: int = 14) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        self.window = window
        self.name = f"rsi_{window}"
        self.lookback = window + 1

    def compute(self, prices: pd.DataFrame, **panels: pd.DataFrame) -> pd.DataFrame:
        delta = prices.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1.0 / self.window, min_periods=self.window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / self.window, min_periods=self.window, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)


class SMARatioFeature(Feature):
    """Close / simple moving average — trend proxy."""

    def __init__(self, window: int = 20) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self.name = f"sma_ratio_{window}d"
        self.lookback = window

    def compute(self, prices: pd.DataFrame, **panels: pd.DataFrame) -> pd.DataFrame:
        sma = prices.rolling(self.window).mean()
        return prices / sma.replace(0.0, np.nan)


class VolumeFeature(Feature):
    """Raw volume when a ``volume`` panel is supplied."""

    name = "volume"
    lookback = 0

    def compute(self, prices: pd.DataFrame, **panels: pd.DataFrame) -> pd.DataFrame:
        volume = panels.get("volume")
        if volume is None:
            raise ValueError("VolumeFeature requires a 'volume' panel")
        return volume.reindex_like(prices).astype(float)


class VolumeChangeFeature(Feature):
    name = "volume_pct_change_1d"
    lookback = 1

    def compute(self, prices: pd.DataFrame, **panels: pd.DataFrame) -> pd.DataFrame:
        volume = panels.get("volume")
        if volume is None:
            raise ValueError("VolumeChangeFeature requires a 'volume' panel")
        return volume.reindex_like(prices).pct_change()


class NewsCountFeature(Feature):
    """Daily news count when a ``news_count`` panel is supplied."""

    name = "news_count"
    lookback = 0

    def compute(self, prices: pd.DataFrame, **panels: pd.DataFrame) -> pd.DataFrame:
        news_count = panels.get("news_count")
        if news_count is None:
            raise ValueError("NewsCountFeature requires a 'news_count' panel")
        return news_count.reindex_like(prices).fillna(0.0).astype(float)


FEATURE_REGISTRY: dict[str, FeatureFactory] = {
    "close": CloseFeature,
    "log_price": LogPriceFeature,
    "pct_change": lambda: PctChangeFeature(1),
    "pct_change_1d": lambda: PctChangeFeature(1),
    "pct_change_5d": lambda: PctChangeFeature(5),
    "pct_change_20d": lambda: PctChangeFeature(20),
    "volatility": lambda: VolatilityFeature(20),
    "volatility_20d": lambda: VolatilityFeature(20),
    "volatility_5d": lambda: VolatilityFeature(5),
    "rsi": lambda: RSIFeature(14),
    "rsi_14": lambda: RSIFeature(14),
    "sma_ratio": lambda: SMARatioFeature(20),
    "sma_ratio_20d": lambda: SMARatioFeature(20),
    "volume": VolumeFeature,
    "volume_pct_change_1d": VolumeChangeFeature,
    "news_count": NewsCountFeature,
}

DEFAULT_FEATURES: tuple[str, ...] = (
    "close",
    "pct_change_1d",
    "pct_change_5d",
    "rsi_14",
    "volatility_20d",
    "sma_ratio_20d",
)


def resolve_features(names: Sequence[str] | None = None) -> list[Feature]:
    """Instantiate feature calculators from registry names."""
    selected = list(names) if names is not None else list(DEFAULT_FEATURES)
    features: list[Feature] = []
    for name in selected:
        if name not in FEATURE_REGISTRY:
            known = ", ".join(sorted(FEATURE_REGISTRY))
            raise KeyError(f"Unknown feature {name!r}. Known: {known}")
        features.append(FEATURE_REGISTRY[name]())
    return features


def max_lookback(features: Iterable[Feature]) -> int:
    return max((f.lookback for f in features), default=0)


def build_feature_tensor(
    prices: pd.DataFrame,
    feature_names: Sequence[str] | None = None,
    *,
    volume: pd.DataFrame | None = None,
    news_count: pd.DataFrame | None = None,
) -> tuple[np.ndarray, list[str], pd.DatetimeIndex, list[str]]:
    """Build a dense ``[T, N, F]`` feature tensor from a price panel.

    Returns ``(tensor, feature_names, dates, tickers)``. Rows that are still
    warming up (NaNs from lookbacks) are kept; the environment is responsible
    for skipping them at episode start.
    """
    features = resolve_features(feature_names)
    panels: dict[str, pd.DataFrame] = {}
    if volume is not None:
        panels["volume"] = volume
    if news_count is not None:
        panels["news_count"] = news_count

    frames: list[pd.DataFrame] = []
    names: list[str] = []
    for feature in features:
        try:
            panel = feature.compute(prices, **panels)
        except ValueError as exc:
            log.warning("Skipping feature %s: %s", feature.name, exc)
            continue
        frames.append(panel)
        names.append(feature.name)

    if not frames:
        raise ValueError("No features could be computed")

    stacked = np.stack([f.to_numpy(dtype=np.float64) for f in frames], axis=-1)
    dates = pd.DatetimeIndex(prices.index)
    tickers = list(prices.columns.astype(str))
    return stacked.astype(np.float32), names, dates, tickers


def news_long_to_count_panel(
    news_daily: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Pivot long ``ticker, date, news_count`` onto the price calendar."""
    if news_daily.empty:
        return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    pivot = (
        news_daily.pivot_table(index="date", columns="ticker", values="news_count", aggfunc="sum")
        .reindex(index=prices.index, columns=prices.columns)
        .fillna(0.0)
    )
    return pivot.astype(float)
