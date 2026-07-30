import numpy as np

from alphaduel.dashboard import engine


def test_single_asset_benchmark_runs():
    r = engine.run_benchmark(
        mode="single_asset",
        agent_names=tuple(engine.SINGLE_AGENTS),
        n_steps=250,
        episode_length=50,
        n_episodes=2,
        seed=3,
    )
    assert set(r.agents) == set(engine.SINGLE_AGENTS)
    for rec in r.agents.values():
        assert rec.equity.shape[0] == 51  # episode_length + 1
        assert {"total_return", "sharpe", "max_drawdown"} <= set(rec.metrics)
        assert np.all(np.isfinite(rec.exposure))


def test_multi_asset_benchmark_weights_sum_le_one():
    r = engine.run_benchmark(
        mode="multi_asset",
        agent_names=tuple(engine.MULTI_AGENTS),
        n_assets=4,
        n_steps=250,
        episode_length=40,
        n_episodes=2,
        seed=5,
    )
    assert set(r.agents) == set(engine.MULTI_AGENTS)
    for rec in r.agents.values():
        assert rec.weights.shape == (40, 4)
        assert np.all(rec.weights.sum(axis=1) <= 1.0 + 1e-6)
        assert np.all(rec.weights >= -1e-6)
