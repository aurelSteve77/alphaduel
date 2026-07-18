import numpy as np

from alphaduel.baselines.policies import (
    STRATEGY_SPECS,
    build_policy,
    default_policies,
)
from alphaduel.dashboard import service
from alphaduel.dashboard.service import RunConfig
from alphaduel.data.dataset import MarketData
from alphaduel.eval.backtest import run_backtest


def _market():
    return MarketData.from_synthetic(n_days=1200, n_assets=6, seed=7)


def test_every_registered_strategy_produces_feasible_weights():
    md = _market()
    lo, hi = md.split_indices("2016-12-31", "2017-12-31")["test"]
    for policy in default_policies(max_weight=0.30):
        res = run_backtest(policy, md, lo, hi, max_weight=0.30)
        w = res.weights
        assert w.shape[0] == len(res.step_returns)
        assert (w >= -1e-9).all()
        assert (w <= 0.30 + 1e-9).all()
        assert (w.sum(axis=1) <= 1.0 + 1e-9).all()


def test_build_policy_merges_params():
    pol = build_policy("momentum", 0.25, lookback=20)
    assert pol.lookback == 20
    assert pol.max_weight == 0.25
    assert set(STRATEGY_SPECS) == {
        "equal_weight", "buy_and_hold", "momentum", "mean_reversion",
        "inverse_vol", "ma_trend", "random",
    }


def test_random_policy_is_reproducible():
    md = _market()
    lo, hi = md.split_indices("2016-12-31", "2017-12-31")["test"]
    a = run_backtest(build_policy("random", 0.3, seed=123), md, lo, hi)
    b = run_backtest(build_policy("random", 0.3, seed=123), md, lo, hi)
    assert np.allclose(a.equity_curve, b.equity_curve)


def test_backtest_records_indices_aligned():
    md = _market()
    lo, hi = md.split_indices("2016-12-31", "2017-12-31")["test"]
    res = run_backtest(build_policy("equal_weight", 0.3), md, lo, hi)
    assert res.decision_index[0] == lo
    assert len(res.value_index) == len(res.step_returns)
    assert res.equity_curve.shape[0] == len(res.step_returns) + 1


def test_dashboard_service_runs_and_builds_figures():
    md = service.build_market("synthetic", n_assets=6, n_days=1400, seed=3)
    cfg = RunConfig(train_end="2016-12-31", val_end="2017-12-31", split="test")
    selected = {"equal_weight": {}, "momentum": {"lookback": 40}, "inverse_vol": {}}
    results = service.run_strategies(md, selected, cfg)
    assert set(results) == set(selected)

    table = service.metrics_table(results)
    assert "Sharpe" in table.columns
    assert len(table) == 3

    # Figures should build without error and contain traces.
    assert len(service.equity_figure(md, results).data) == 3
    assert len(service.drawdown_figure(md, results).data) == 3
    assert len(service.allocation_figure(md, results["equal_weight"]).data) >= md.n_assets
    service.rolling_sharpe_figure(md, results)
    service.returns_hist_figure(results)


def test_action_heatmap_figures_build():
    md = service.build_market("synthetic", n_assets=5, n_days=1200, seed=8)
    cfg = RunConfig(train_end="2016-12-31", val_end="2017-12-31", split="test")
    results = service.run_strategies(md, {"momentum": {"lookback": 40}}, cfg)
    res = results["momentum"]

    hm = service.allocation_heatmap_figure(md, res)
    tr = service.trades_heatmap_figure(md, res)
    assert hm.data[0].type == "heatmap"
    assert tr.data[0].type == "heatmap"
    assert np.asarray(hm.data[0].z).shape[0] == md.n_assets
    assert np.asarray(tr.data[0].z).shape[0] == md.n_assets
