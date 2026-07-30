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


def test_run_evaluation_accepts_duplicate_agent_kinds():
    contestants = [
        {"label": "EW A", "agent": "equal_weight", "params": {}},
        {"label": "EW B", "agent": "equal_weight", "params": {}},
        {"label": "InvVol", "agent": "inverse_volatility", "params": {"lookback": 10}},
    ]
    r = engine.run_evaluation(
        mode="multi_asset",
        contestants=contestants,
        n_assets=3,
        n_steps=200,
        episode_length=30,
        n_episodes=2,
        seed=11,
        use_mock=True,
        max_workers=3,
    )
    assert set(r.agents) == {"EW A", "EW B", "InvVol"}
    assert list(r.agents) == ["EW A", "EW B", "InvVol"]  # roster order preserved
    for rec in r.agents.values():
        assert len(rec.episode_metrics) == 2
        assert "sharpe" in rec.metrics


def test_run_evaluation_caps_workers():
    assert engine.EVAL_MAX_WORKERS == 3
    contestants = [
        {"label": f"EW {i}", "agent": "equal_weight", "params": {}} for i in range(4)
    ]
    r = engine.run_evaluation(
        mode="multi_asset",
        contestants=contestants,
        n_assets=2,
        n_steps=120,
        episode_length=20,
        n_episodes=1,
        seed=2,
        use_mock=True,
        max_workers=99,  # hard-capped to EVAL_MAX_WORKERS
    )
    assert len(r.agents) == 4


def test_run_evaluation_progress_reports_per_agent():
    seen: list[list[dict]] = []
    fractions: list[float] = []

    def on_progress(frac, msg, agents=None):
        assert 0.0 <= frac <= 1.0
        assert isinstance(msg, str)
        fractions.append(frac)
        if agents:
            seen.append(agents)
            assert {a["label"] for a in agents} == {"EW A", "EW B"}
            for a in agents:
                assert a["status"] in {"queued", "training", "running", "done", "error"}

    engine.run_evaluation(
        mode="multi_asset",
        contestants=[
            {"label": "EW A", "agent": "equal_weight", "params": {}},
            {"label": "EW B", "agent": "equal_weight", "params": {}},
        ],
        n_assets=2,
        n_steps=80,
        episode_length=15,
        n_episodes=2,
        seed=4,
        use_mock=True,
        max_workers=2,
        progress=on_progress,
    )
    assert seen
    assert fractions[-1] == 1.0
    final = seen[-1]
    assert all(a["status"] == "done" for a in final)
    # Workers publish episode/step even if mid-run snapshots are skipped for fast agents.
    assert all(a["episode"] == 2 and a["step"] == 15 for a in final)


def test_default_contestant_label_for_llm():
    label = engine.default_contestant_label(
        "llm_vanilla",
        {"llm": {"provider": "openai", "model": "gpt-5.4-mini"}},
    )
    assert label == "LLM · openai · gpt-5.4-mini"
