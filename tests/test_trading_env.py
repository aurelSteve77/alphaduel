"""Tests for the Gymnasium trading environment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaduel.environments import AlphaDuelEnv, TradingEnv, TradingEnvConfig


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=80)
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "AAPL": 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(idx))),
            "MSFT": 200 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(idx))),
        },
        index=idx,
    )


@pytest.fixture
def env(synthetic_prices) -> TradingEnv:
    cfg = TradingEnvConfig(
        start="2024-01-02",
        end="2024-04-30",
        tickers=["AAPL", "MSFT"],
        initial_cash=10_000.0,
        transaction_fee=0.001,
        n_days=20,
        max_shares=5,
        features=["close", "pct_change_1d", "rsi_14", "volatility_20d"],
        include_news=False,
    )
    return TradingEnv(config=cfg, prices=synthetic_prices)


def test_reset_observation_space(env):
    obs, info = env.reset(seed=0)
    assert set(obs) == {"cash", "holdings", "portfolio_value", "weights", "features"}
    assert obs["cash"].shape == (1,)
    assert obs["holdings"].shape == (2,)
    assert obs["features"].shape == (2, len(env.feature_names))
    assert info["tickers"] == ["AAPL", "MSFT"]
    assert env.observation_space.contains(obs)


def test_step_buy_multiple_tickers(env):
    obs, _ = env.reset(seed=0)
    action = np.array([2, 1], dtype=np.int32)
    obs2, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, float)
    assert not truncated
    assert info["executed"].tolist() == [2, 1]
    assert env.portfolio.holdings[0] == 2
    assert env.portfolio.holdings[1] == 1
    assert env.observation_space.contains(obs2)
    assert obs2["portfolio_value"][0] > 0


def test_episode_terminates_after_n_days(env):
    env.reset(seed=0)
    done = False
    steps = 0
    while not done:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated
        steps += 1
        assert steps <= env.config.n_days + 1
    assert steps == env.config.n_days


def test_include_news_in_info(synthetic_prices):
    news = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "published_at": pd.Timestamp("2024-02-01 15:00", tz="UTC"),
                "source": "test",
                "category": "news",
                "headline": "Hello Apple",
                "summary": "Body",
                "url": "http://example.com",
            }
        ]
    )
    cfg = TradingEnvConfig(
        start="2024-01-02",
        end="2024-04-30",
        tickers=["AAPL", "MSFT"],
        initial_cash=5_000.0,
        n_days=5,
        features=["close", "pct_change_1d"],
        include_news=True,
    )
    trading = TradingEnv(config=cfg, prices=synthetic_prices, news=news)
    assert "news_count" in trading.feature_names
    _, info = trading.reset(seed=0)
    assert "news" in info


def test_gym_make_registration(synthetic_prices):
    import gymnasium as gym

    import alphaduel  # noqa: F401

    cfg = TradingEnvConfig(
        start="2024-01-02",
        end="2024-04-30",
        tickers=["AAPL", "MSFT"],
        initial_cash=1_000.0,
        n_days=3,
        features=["close", "pct_change_1d"],
    )
    env = gym.make("AlphaDuel-v0", config=cfg, prices=synthetic_prices)
    obs, _ = env.reset(seed=1)
    assert "features" in obs
    env.close()


def test_alias_export():
    assert AlphaDuelEnv is TradingEnv
