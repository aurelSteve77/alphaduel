import numpy as np

from alphaduel.evaluation.metrics import compute_metrics, max_drawdown, sharpe


def test_sharpe_of_constant_returns_is_zero():
    assert sharpe(np.zeros(50)) == 0.0


def test_max_drawdown_monotonic_up_is_zero():
    equity = np.linspace(100, 200, 50)
    assert max_drawdown(equity) == 0.0


def test_max_drawdown_detects_dip():
    equity = np.array([100.0, 120.0, 60.0, 90.0])
    assert abs(max_drawdown(equity) - 0.5) < 1e-9


def test_compute_metrics_keys():
    equity = 100 * np.cumprod(1 + np.random.default_rng(0).normal(0.001, 0.01, 100))
    m = compute_metrics(equity, n_transactions=5, total_reward=0.1, n_trials=3)
    for key in ("total_return", "sharpe", "max_drawdown", "deflated_sharpe"):
        assert key in m
