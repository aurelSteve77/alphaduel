import numpy as np
import pandas as pd

from alphaduel.baselines.policies import BuyAndHold, EqualWeight, Momentum
from alphaduel.data.dataset import MarketData
from alphaduel.data.features import compute_features
from alphaduel.env import AlphaDuelEnv
from alphaduel.eval.backtest import run_backtest, walk_forward
from alphaduel.utils.prices import generate_gbm_prices


def _prices_df(n: int = 500, k: int = 6, seed: int = 0) -> pd.DataFrame:
    prices = generate_gbm_prices(n, k, seed=seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame(prices, index=idx, columns=[f"A{i}" for i in range(k)])


def test_features_are_point_in_time():
    # Recomputing features on the series truncated at t must reproduce row t exactly.
    df = _prices_df(500, 5, seed=3)
    full, _, _ = compute_features(df)
    for t in (300, 400, 480):
        trunc, _, _ = compute_features(df.iloc[: t + 1])
        assert np.allclose(full[t], trunc[t], atol=1e-8), f"lookahead at t={t}"


def test_features_shape_and_finite():
    df = _prices_df()
    feats, names, warmup = compute_features(df)
    assert feats.shape == (len(df), df.shape[1], len(names))
    assert np.isfinite(feats).all()
    assert warmup > 0


def test_splits_are_contiguous_and_ordered():
    md = MarketData.from_synthetic(n_days=900, n_assets=5, seed=1)
    s = md.split_indices("2016-12-31", "2017-12-31")
    assert s["train"][0] == md.valid_from
    assert s["train"][1] == s["val"][0]
    assert s["val"][1] == s["test"][0]
    assert s["test"][1] == md.n_steps


def test_baselines_backtest_runs():
    md = MarketData.from_synthetic(n_days=1000, n_assets=6, seed=2)
    lo, hi = md.split_indices("2016-12-31", "2017-12-31")["test"]
    for policy in (EqualWeight(), BuyAndHold(), Momentum()):
        res = run_backtest(policy, md, lo, hi)
        assert len(res.step_returns) > 0
        assert (res.equity_curve > 0).all()
        assert np.isfinite(res.metrics["total_return"])


def test_buy_and_hold_has_lower_turnover_than_rebalancing():
    md = MarketData.from_synthetic(n_days=1000, n_assets=6, seed=4)
    lo, hi = md.split_indices("2016-12-31", "2017-12-31")["test"]
    bh = run_backtest(BuyAndHold(), md, lo, hi)
    ew = run_backtest(EqualWeight(), md, lo, hi)
    assert bh.metrics["avg_turnover"] <= ew.metrics["avg_turnover"] + 1e-9


def test_walk_forward_folds():
    md = MarketData.from_synthetic(n_days=1400, n_assets=6, seed=6)
    results = walk_forward(
        EqualWeight, md, train_end="2016-12-31", val_end="2017-12-31", n_folds=3
    )
    assert len(results) == 3
    assert all(len(r.step_returns) > 0 for r in results)


def test_env_feature_mode_rollout():
    md = MarketData.from_synthetic(n_days=1200, n_assets=6, seed=5)
    env = AlphaDuelEnv.from_market_data(
        md, split="train", train_end="2016-12-31", val_end="2017-12-31", horizon=40
    )
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    # feature-mode obs dim = K*F + K + 2
    assert obs.shape[0] == md.n_assets * md.n_features + md.n_assets + 2
    assert info["portfolio_value"] > 0

    steps = 0
    for _ in range(30):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        steps += 1
        if terminated or truncated:
            break
    assert steps > 0
