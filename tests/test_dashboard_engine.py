import numpy as np

from alphaduel.dashboard import engine


def test_single_asset_benchmark_runs():
    agents = tuple(engine.baseline_agents("single_asset"))
    r = engine.run_benchmark(
        mode="single_asset",
        agent_names=agents,
        n_steps=250,
        episode_length=50,
        n_episodes=2,
        seed=3,
        use_mock=True,
    )
    assert set(r.agents) == set(agents)
    for rec in r.agents.values():
        assert rec.equity.shape[0] == 51  # episode_length + 1
        assert {"total_return", "sharpe", "max_drawdown"} <= set(rec.metrics)
        assert np.all(np.isfinite(rec.exposure))


def test_multi_asset_benchmark_weights_sum_le_one():
    agents = tuple(engine.baseline_agents("multi_asset"))
    r = engine.run_benchmark(
        mode="multi_asset",
        agent_names=agents,
        n_assets=4,
        n_steps=250,
        episode_length=40,
        n_episodes=2,
        seed=5,
        use_mock=True,
    )
    assert set(r.agents) == set(agents)
    for rec in r.agents.values():
        assert rec.weights.shape == (40, 4)
        assert np.all(rec.weights.sum(axis=1) <= 1.0 + 1e-6)
        assert np.all(rec.weights >= -1e-6)


def test_llm_vanilla_listed_for_both_modes():
    assert "llm_vanilla" in engine.agents_for("single_asset")
    assert "llm_vanilla" in engine.agents_for("multi_asset")
    assert "llm_vanilla" not in engine.baseline_agents("multi_asset")
