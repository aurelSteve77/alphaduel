"""Smoke tests for dashboard helpers (no Streamlit runtime)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaduel.dashboard import charts, service
from alphaduel.evaluation import StrategyRunConfig, evaluate
from alphaduel.utils.singleton import Singleton
from alphaduel.configuration import Configuration


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=40)
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "AAPL": 100 * np.cumprod(1 + rng.normal(0, 0.01, len(idx))),
            "MSFT": 200 * np.cumprod(1 + rng.normal(0, 0.01, len(idx))),
        },
        index=idx,
    )


@pytest.fixture(autouse=True)
def _reset_configuration_singleton():
    Singleton._instances.pop(Configuration, None)
    yield
    Singleton._instances.pop(Configuration, None)


def test_build_run_config_and_list_kinds():
    assert "buy_and_hold" in service.list_strategy_kinds()
    run = service.build_run_config(
        name="ui_test",
        kind="buy_and_hold",
        tickers=["AAPL", "MSFT"],
        start="2024-01-02",
        end="2024-03-01",
        initial_cash=5_000,
        transaction_fee=0.001,
        n_days=10,
        max_shares=3,
    )
    assert run.kind == "buy_and_hold"
    assert run.env["n_days"] == 10


def test_iter_live_and_charts(synthetic_prices: pd.DataFrame):
    run = service.build_run_config(
        name="live_test",
        kind="buy_and_hold",
        tickers=["AAPL", "MSFT"],
        start="2024-01-02",
        end="2024-03-29",
        initial_cash=10_000,
        transaction_fee=0.001,
        n_days=8,
        max_shares=2,
    )
    events = list(service.iter_live(run, prices=synthetic_prices))
    assert len(events) == 8
    event, result = events[-1]
    assert event.done
    assert result.metrics is not None
    assert len(result.actions) == 8
    assert len(result.decision_dates) == 8

    # Charts should build without error.
    charts.equity_chart(result)
    charts.drawdown_chart(result)
    charts.holdings_chart(result)
    charts.trades_scatter(result)
    charts.episode_dashboard(result)
    table = service.metrics_comparison_table([result])
    assert not table.empty
    charts.comparison_equity([result])
    charts.metrics_bar(table, metric="total_return")


def test_evaluate_records_actions(synthetic_prices: pd.DataFrame):
    run = StrategyRunConfig(
        name="act_test",
        kind="random",
        env={
            "start": "2024-01-02",
            "end": "2024-03-29",
            "tickers": ["AAPL", "MSFT"],
            "initial_cash": 10_000,
            "n_days": 5,
            "max_shares": 2,
            "features": ["close", "pct_change_1d"],
        },
        policy={"seed": 0},
        eval={"seed": 0, "save_plots": False, "save_trajectory": False},
    )
    result = evaluate(run, prices=synthetic_prices)
    assert len(result.actions) == 5
    assert len(result.rationales) == 5
    assert not result.decisions_frame().empty
