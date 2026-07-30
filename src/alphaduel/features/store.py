"""FeatureStore: assemble PIT-correct observations with config-driven ablations.

For the P0 single-asset MVP the store:
1. takes the cached price panel (+ optional macro panel),
2. computes the enabled feature groups (causal, warmup-aware),
3. trims the warmup region so no NaN leaks into observations,
4. exposes a stable column order + a manifest hash for reproducibility.

The store also keeps the raw ``open``/``close`` series the env needs for execution.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alphaduel.config.schema import FeatureConfig
from alphaduel.features.base import FeatureContext
from alphaduel.features.macro import MacroLevels
from alphaduel.features.technical import TECHNICAL_FEATURES


@dataclass
class MarketPanel:
    """Everything the env needs for one symbol over a date range."""

    timestamps: pd.DatetimeIndex
    open: np.ndarray
    close: np.ndarray
    features: np.ndarray  # shape (T, n_features)
    feature_names: list[str] = field(default_factory=list)

    @property
    def n_steps(self) -> int:
        return len(self.timestamps)

    @property
    def n_features(self) -> int:
        return self.features.shape[1]


@dataclass
class MultiAssetPanel:
    """Aligned OHLC + per-security features for N symbols (P5 / GenPortfolio)."""

    timestamps: pd.DatetimeIndex
    symbols: list[str]
    open: np.ndarray   # shape (T, N)
    close: np.ndarray  # shape (T, N)
    features: np.ndarray  # shape (T, N, F)
    feature_names: list[str] = field(default_factory=list)

    @property
    def n_steps(self) -> int:
        return len(self.timestamps)

    @property
    def n_assets(self) -> int:
        return len(self.symbols)

    @property
    def n_features(self) -> int:
        return self.features.shape[2]


class FeatureStore:
    def __init__(self, config: FeatureConfig) -> None:
        self.config = config
        self.ctx = FeatureContext(
            technical_windows=tuple(config.technical_windows),
            rsi_period=config.rsi_period,
        )

    def build_panel(
        self,
        prices: pd.DataFrame,
        symbol: str,
        macro: pd.DataFrame | None = None,
    ) -> MarketPanel:
        px = (
            prices[prices["symbol"] == symbol]
            .set_index("timestamp")
            .sort_index()
        )
        if px.empty:
            raise ValueError(f"No price rows for symbol {symbol!r}.")

        frames: list[pd.DataFrame] = []
        max_warmup = 0

        if self.config.technical:
            for feat in TECHNICAL_FEATURES:
                frames.append(feat.compute(px, self.ctx))
            max_warmup = max(max_warmup, max(self.ctx.technical_windows, default=1))

        if self.config.macro and macro is not None:
            frames.append(MacroLevels(macro).compute(px, self.ctx))

        feat_df = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=px.index)

        # Trim warmup region: drop leading rows containing any NaN (no leakage / no NaNs).
        combined = pd.concat([px[["open", "close"]], feat_df], axis=1)
        combined = combined.iloc[max_warmup:].dropna()

        feature_names = [c for c in combined.columns if c not in ("open", "close")]
        return MarketPanel(
            timestamps=combined.index,
            open=combined["open"].to_numpy(dtype=np.float64),
            close=combined["close"].to_numpy(dtype=np.float64),
            features=combined[feature_names].to_numpy(dtype=np.float32),
            feature_names=feature_names,
        )

    def manifest_hash(self, panel: MarketPanel) -> str:
        payload = "|".join(panel.feature_names) + f"|{panel.n_steps}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def build_multi_asset_panel(
        self,
        prices: pd.DataFrame,
        symbols: list[str],
        macro: pd.DataFrame | None = None,
    ) -> MultiAssetPanel:
        """Build per-symbol panels and align them on their common (inner-join) dates."""
        per_symbol = {s: self.build_panel(prices, s, macro) for s in symbols}

        # Intersect timestamps so every symbol has data on every kept date.
        common = per_symbol[symbols[0]].timestamps
        for s in symbols[1:]:
            common = common.intersection(per_symbol[s].timestamps)
        if len(common) == 0:
            raise ValueError("No overlapping dates across the requested symbols.")

        feature_names = per_symbol[symbols[0]].feature_names
        opens, closes, feats = [], [], []
        for s in symbols:
            p = per_symbol[s]
            mask = p.timestamps.isin(common)
            opens.append(p.open[mask])
            closes.append(p.close[mask])
            feats.append(p.features[mask])

        return MultiAssetPanel(
            timestamps=common,
            symbols=list(symbols),
            open=np.stack(opens, axis=1),      # (T, N)
            close=np.stack(closes, axis=1),    # (T, N)
            features=np.stack(feats, axis=1),  # (T, N, F)
            feature_names=feature_names,
        )
