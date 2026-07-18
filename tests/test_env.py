import numpy as np

from alphaduel.config import Config, load_config
from alphaduel.env import AlphaDuelEnv, PortfolioSimulator, project_long_only
from alphaduel.utils.prices import generate_gbm_prices


def test_projection_is_feasible():
    w = project_long_only(np.array([0.8, 0.8, 0.8]), max_weight=0.4)
    assert (w >= 0).all()
    assert (w <= 0.4 + 1e-9).all()
    assert w.sum() <= 1.0 + 1e-9


def test_simulator_zero_action_only_costs():
    prices = generate_gbm_prices(100, 3, seed=1)
    sim = PortfolioSimulator(prices, budget=1000.0, cost_bps=0.0, max_weight=0.5)
    sim.reset(0)
    result = sim.step(np.zeros(3))
    # No position taken and zero cost => value unchanged.
    assert abs(result.reward) < 1e-12
    assert result.portfolio_value == 1000.0


def test_simulator_charges_transaction_costs():
    prices = generate_gbm_prices(100, 3, seed=2)
    sim = PortfolioSimulator(prices, budget=1000.0, cost_bps=10.0, max_weight=1.0)
    sim.reset(0)
    result = sim.step(np.array([0.3, 0.0, 0.0]))
    assert result.turnover > 0
    assert result.cost > 0


def test_gym_env_rollout_stays_in_spec():
    env = AlphaDuelEnv(n_assets=5, n_days=600, horizon=100, obs_window=21, seed=0)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert info["portfolio_value"] > 0

    steps = 0
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        assert info["portfolio_value"] > 0
        steps += 1
        if terminated or truncated:
            break
    assert steps > 0


def test_config_defaults_and_yaml(tmp_path):
    cfg = Config()
    assert cfg.env.rebalance_every == 1
    assert len(cfg.data.tickers) == 10

    path = tmp_path / "c.yaml"
    path.write_text("env:\n  budget: 50000.0\n  rebalance_every: 5\n", encoding="utf-8")
    loaded = load_config(path)
    assert loaded.env.budget == 50000.0
    assert loaded.env.rebalance_every == 5
