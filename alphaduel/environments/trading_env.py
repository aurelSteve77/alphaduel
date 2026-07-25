"""Gymnasium daily stock-trading environment.

One step = one trading day. The agent outputs an integer share delta per ticker
(buy / sell / hold). Observations expose cash, holdings, portfolio value,
weights and a ``[n_assets, n_features]`` feature matrix. Optional news text is
returned in ``info`` (not the Box observation) so numeric RL stacks stay clean.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from alphaduel.data.dataset import _market_to_long, _news_to_daily
from alphaduel.data.features import (
    DEFAULT_FEATURES,
    build_feature_tensor,
    max_lookback,
    news_long_to_count_panel,
    resolve_features,
)
from alphaduel.environments.config import TradingEnvConfig
from alphaduel.environments.portfolio import Portfolio
from alphaduel.logger import get_logger

log = get_logger(__name__)


def _slice_price_panel(
    prices: pd.DataFrame,
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise KeyError(f"Price panel missing tickers: {missing}")
    panel = prices.loc[:, tickers].sort_index()
    panel.index = pd.to_datetime(panel.index).tz_localize(None).normalize()
    mask = (panel.index >= pd.Timestamp(start)) & (panel.index <= pd.Timestamp(end))
    panel = panel.loc[mask].dropna(how="any")
    if panel.empty:
        raise ValueError(f"No complete price rows between {start} and {end}")
    return panel.astype(float)


def _news_text_by_date(
    dataset_long: pd.DataFrame,
    tickers: list[str],
) -> dict[pd.Timestamp, dict[str, dict[str, Any]]]:
    """Map date -> ticker -> {news_count, headlines, summaries}."""
    if dataset_long.empty:
        return {}
    out: dict[pd.Timestamp, dict[str, dict[str, Any]]] = {}
    subset = dataset_long[dataset_long["ticker"].isin(tickers)]
    for row in subset.itertuples(index=False):
        day = pd.Timestamp(row.date).normalize()
        bucket = out.setdefault(day, {})
        bucket[str(row.ticker)] = {
            "news_count": int(getattr(row, "news_count", 0) or 0),
            "headlines": str(getattr(row, "headlines", "") or ""),
            "summaries": str(getattr(row, "summaries", "") or ""),
        }
    return out


class TradingEnv(gym.Env):
    """Daily multi-asset trading environment (Gymnasium API).

    Parameters
    ----------
    config:
        :class:`TradingEnvConfig` (or pass the same fields as kwargs via
        :meth:`from_config` / ``gym.make``).
    prices:
        Optional wide ``[dates x tickers]`` adjusted-close panel. When omitted,
        data is loaded via :func:`alphaduel.data.download.download_ohlcv`.
    volume:
        Optional wide volume panel aligned with ``prices``.
    news:
        Optional long news frame (``NEWS_COLUMNS``) or merged dataset long frame
        with ``date, ticker, news_count, headlines, summaries``.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: TradingEnvConfig | None = None,
        *,
        prices: pd.DataFrame | None = None,
        volume: pd.DataFrame | None = None,
        news: pd.DataFrame | None = None,
        render_mode: str | None = None,
        **config_overrides: Any,
    ) -> None:
        super().__init__()
        if config is None:
            config = TradingEnvConfig.model_validate(config_overrides)
        elif config_overrides:
            config = config.model_copy(update=config_overrides)

        self.config = config
        self.render_mode = render_mode
        self.tickers = list(config.tickers)
        self.n_assets = len(self.tickers)

        price_panel = prices if prices is not None else self._load_prices()
        self.prices = _slice_price_panel(
            price_panel, self.tickers, config.start, config.end
        )

        volume_panel = None
        if volume is not None:
            volume_panel = volume.reindex(index=self.prices.index, columns=self.tickers)

        feature_names = list(config.features)
        news_count_panel = None
        self._news_by_date: dict[pd.Timestamp, dict[str, dict[str, Any]]] = {}

        if config.include_news:
            if "news_count" not in feature_names:
                feature_names = [*feature_names, "news_count"]
            news_frame = news if news is not None else self._load_news()
            news_count_panel, self._news_by_date = self._prepare_news(news_frame)

        self.feature_tensor, self.feature_names, self.dates, _ = build_feature_tensor(
            self.prices,
            feature_names,
            volume=volume_panel,
            news_count=news_count_panel,
        )
        self.price_array = self.prices.to_numpy(dtype=np.float64)

        warmup = max_lookback(resolve_features(self.feature_names))
        self._start_index = warmup
        usable = len(self.dates) - self._start_index
        if usable < 2:
            raise ValueError(
                f"Need at least 2 post-warmup days; got {usable} "
                f"(warmup={warmup}, total={len(self.dates)})"
            )
        horizon = config.n_days if config.n_days is not None else usable
        self._max_steps = min(horizon, usable)
        # Last index we may *observe*; we need one more day to mark-to-market.
        self._last_index = self._start_index + self._max_steps - 1
        if self._last_index >= len(self.dates) - 1:
            self._last_index = len(self.dates) - 2
            self._max_steps = self._last_index - self._start_index + 1
        if self._max_steps < 1:
            raise ValueError("Episode horizon collapsed to zero after calendar trim")

        self.portfolio = Portfolio(
            n_assets=self.n_assets,
            cash=config.initial_cash,
            fee_rate=config.transaction_fee,
            allow_short=config.allow_short,
        )

        self.action_space = spaces.Box(
            low=-config.max_shares,
            high=config.max_shares,
            shape=(self.n_assets,),
            dtype=np.int32,
        )
        n_features = len(self.feature_names)
        self.observation_space = spaces.Dict(
            {
                "cash": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
                "holdings": spaces.Box(
                    low=-np.inf if config.allow_short else 0.0,
                    high=np.inf,
                    shape=(self.n_assets,),
                    dtype=np.float32,
                ),
                "portfolio_value": spaces.Box(
                    low=0.0, high=np.inf, shape=(1,), dtype=np.float32
                ),
                "weights": spaces.Box(
                    low=-np.inf if config.allow_short else 0.0,
                    high=1.0 if not config.allow_short else np.inf,
                    shape=(self.n_assets,),
                    dtype=np.float32,
                ),
                "features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.n_assets, n_features),
                    dtype=np.float32,
                ),
            }
        )

        self._t: int = self._start_index
        self._steps_taken: int = 0
        self._prev_value: float = config.initial_cash

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(
        cls,
        config: TradingEnvConfig | None = None,
        **kwargs: Any,
    ) -> TradingEnv:
        if config is None:
            config = TradingEnvConfig.from_project_yaml()
        return cls(config=config, **kwargs)

    def _load_prices(self) -> pd.DataFrame:
        from alphaduel.data.download import download_ohlcv

        return download_ohlcv(self.tickers, self.config.start, self.config.end)

    def _load_news(self) -> pd.DataFrame:
        from alphaduel.data.download_news import download_news

        return download_news(self.tickers, self.config.start, self.config.end)

    def _prepare_news(
        self, news: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[pd.Timestamp, dict[str, dict[str, Any]]]]:
        if news.empty:
            empty = pd.DataFrame(0.0, index=self.prices.index, columns=self.tickers)
            return empty, {}

        if {"headlines", "summaries", "news_count", "date", "ticker"}.issubset(news.columns):
            daily = news.copy()
            daily["date"] = pd.to_datetime(daily["date"]).dt.normalize().dt.tz_localize(None)
        elif {"published_at", "headline", "summary", "ticker"}.issubset(news.columns):
            daily = _news_to_daily(news)
        else:
            raise ValueError(
                "news frame must be either raw NEWS_COLUMNS or merged dataset columns"
            )

        count_panel = news_long_to_count_panel(daily, self.prices)
        # Attach close for text map builder compatibility when using raw news.
        if "close" not in daily.columns:
            market_long = _market_to_long(self.prices)
            daily = market_long.merge(daily, on=["date", "ticker"], how="left")
            daily["news_count"] = daily["news_count"].fillna(0).astype(int)
            daily["headlines"] = daily["headlines"].fillna("")
            daily["summaries"] = daily["summaries"].fillna("")
        text_map = _news_text_by_date(daily, self.tickers)
        return count_panel, text_map

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        start_index = int(options.get("start_index", self._start_index))
        if start_index < self._start_index or start_index > self._last_index:
            raise ValueError(
                f"start_index must be in [{self._start_index}, {self._last_index}]"
            )
        self._t = start_index
        self._steps_taken = 0
        self.portfolio.reset(self.config.initial_cash)
        self._prev_value = self.portfolio.market_value(self.price_array[self._t])
        obs = self._get_obs()
        return obs, self._info(executed=np.zeros(self.n_assets, dtype=np.int64), fees=0.0)

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.int32).reshape(-1)
        if action.shape != (self.n_assets,):
            raise ValueError(f"action must have shape ({self.n_assets},), got {action.shape}")

        action = np.clip(action, -self.config.max_shares, self.config.max_shares).astype(
            np.int64
        )
        prices_t = self.price_array[self._t]
        trade = self.portfolio.execute(action, prices_t)

        # Advance one trading day and mark-to-market for the reward.
        self._t += 1
        self._steps_taken += 1
        value = self.portfolio.market_value(self.price_array[self._t])
        pnl = value - self._prev_value
        reward = float(pnl / (self.config.initial_cash * self.config.reward_scale))
        self._prev_value = value

        terminated = self._steps_taken >= self._max_steps
        truncated = False
        obs = self._get_obs()
        info = self._info(executed=trade.executed, fees=trade.fees_paid)
        info["pnl"] = float(pnl)
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        if self.render_mode != "human":
            return
        prices = self.price_array[self._t]
        value = self.portfolio.market_value(prices)
        day = self.dates[self._t].date()
        print(
            f"t={self._steps_taken} date={day} cash={self.portfolio.cash:.2f} "
            f"value={value:.2f} holdings={self.portfolio.holdings.tolist()}"
        )

    # ------------------------------------------------------------------ #
    # Observation / info
    # ------------------------------------------------------------------ #
    def _get_obs(self) -> dict[str, np.ndarray]:
        prices = self.price_array[self._t]
        value = self.portfolio.market_value(prices)
        features = self.feature_tensor[self._t]
        # Replace residual warmup NaNs so the Box space stays valid.
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return {
            "cash": np.array([self.portfolio.cash], dtype=np.float32),
            "holdings": self.portfolio.holdings.astype(np.float32),
            "portfolio_value": np.array([value], dtype=np.float32),
            "weights": self.portfolio.weights(prices).astype(np.float32),
            "features": features.astype(np.float32),
        }

    def _info(self, *, executed: np.ndarray, fees: float) -> dict[str, Any]:
        day = pd.Timestamp(self.dates[self._t]).normalize()
        info: dict[str, Any] = {
            "date": day.strftime("%Y-%m-%d"),
            "tickers": list(self.tickers),
            "feature_names": list(self.feature_names),
            "executed": executed.astype(np.int64),
            "fees_paid": float(fees),
            "cash": float(self.portfolio.cash),
            "holdings": self.portfolio.holdings.copy(),
            "portfolio_value": float(self.portfolio.market_value(self.price_array[self._t])),
            "step": self._steps_taken,
        }
        if self.config.include_news:
            info["news"] = self._news_by_date.get(day, {})
        return info


# Backwards-friendly alias matching the README name.
AlphaDuelEnv = TradingEnv


def make_trading_env(**kwargs: Any) -> TradingEnv:
    """Factory used by ``gymnasium.register``."""
    if "config" not in kwargs and not {"start", "end", "tickers"} <= set(kwargs):
        return TradingEnv.from_config(**kwargs)
    return TradingEnv(**kwargs)
