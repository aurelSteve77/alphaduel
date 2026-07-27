"""Tests for the evaluation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphaduel.evaluation import (
    StrategyRunConfig,
    evaluate,
    format_metrics_table,
    render_evaluation,
)
from alphaduel.evaluation.metrics import compute_metrics, max_drawdown
from alphaduel.utils.singleton import Singleton
from alphaduel.configuration import Configuration


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=80)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "AAPL": 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(idx))),
            "MSFT": 200 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(idx))),
            "NVDA": 50 * np.cumprod(1 + rng.normal(0.001, 0.02, len(idx))),
        },
        index=idx,
    )


@pytest.fixture(autouse=True)
def _reset_configuration_singleton():
    # Strategy YAML merges with project.yaml via Configuration singleton.
    Singleton._instances.pop(Configuration, None)
    yield
    Singleton._instances.pop(Configuration, None)


def test_strategy_run_config_from_yaml_name():
    run = StrategyRunConfig.from_yaml("buy_and_hold")
    assert run.kind == "buy_and_hold"
    assert run.env["initial_cash"] == 10000.0


def test_max_drawdown():
    assert max_drawdown([100, 110, 90, 95]) == pytest.approx((110 - 90) / 110)


def test_evaluate_buy_and_hold(tmp_path: Path, synthetic_prices: pd.DataFrame):
    run = StrategyRunConfig(
        name="bah_test",
        kind="buy_and_hold",
        env={
            "start": "2024-01-02",
            "end": "2024-04-30",
            "tickers": ["AAPL", "MSFT"],
            "initial_cash": 10_000.0,
            "n_days": 15,
            "max_shares": 4,
            "transaction_fee": 0.001,
            "features": ["close", "pct_change_1d"],
        },
        eval={
            "seed": 1,
            "output_dir": str(tmp_path / "out"),
            "save_plots": True,
            "show_plots": False,
            "save_trajectory": True,
        },
    )
    result = evaluate(run, prices=synthetic_prices)
    assert result.metrics is not None
    assert result.metrics.n_steps == 15
    assert result.metrics.n_transactions > 0
    assert result.metrics.total_fees > 0
    assert len(result.portfolio_values) == result.metrics.n_steps + 1

    text = format_metrics_table(result)
    assert "final PnL" in text

    out = render_evaluation(
        result,
        output_dir=tmp_path / "out",
        save_plots=True,
        show_plots=False,
        save_trajectory=True,
    )
    assert (out / "metrics.json").is_file()
    assert (out / "equity.csv").is_file()
    assert (out / "trades.csv").is_file()
    assert (out / "evaluation.png").is_file()
    payload = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert payload["metrics"]["n_steps"] == 15


def test_evaluate_random(tmp_path: Path, synthetic_prices: pd.DataFrame):
    run = StrategyRunConfig(
        name="rand_test",
        kind="random",
        env={
            "start": "2024-01-02",
            "end": "2024-04-30",
            "tickers": ["AAPL", "MSFT"],
            "initial_cash": 5_000.0,
            "n_days": 10,
            "max_shares": 2,
            "features": ["close", "pct_change_1d"],
        },
        policy={"seed": 0},
        eval={"seed": 0, "output_dir": str(tmp_path), "save_plots": False},
    )
    result = evaluate(run, prices=synthetic_prices)
    assert result.metrics is not None
    assert result.metrics.n_steps == 10


def test_compute_metrics_empty_trades():
    m = compute_metrics(
        initial_cash=1000.0,
        portfolio_values=[1000.0, 1010.0, 1005.0],
        rewards=[0.01, -0.005],
        trades=[],
    )
    assert m.n_transactions == 0
    assert m.final_pnl == pytest.approx(5.0)
