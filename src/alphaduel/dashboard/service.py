"""Dashboard logic: build a market, run strategies, and produce Plotly figures.

This module is intentionally free of any Streamlit dependency so it can be unit
tested and reused headlessly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..baselines.policies import build_policy
from ..data.dataset import MarketData
from ..eval.backtest import BacktestResult, run_backtest

_PALETTE = [
    "#2dd4bf", "#f59e0b", "#60a5fa", "#f472b6", "#a3e635",
    "#c084fc", "#fb7185", "#34d399", "#facc15", "#38bdf8",
]

_METRIC_LABELS = {
    "total_return": "Total Return",
    "cagr": "CAGR",
    "volatility": "Volatility",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "max_drawdown": "Max Drawdown",
    "hit_rate": "Hit Rate",
    "avg_turnover": "Avg Turnover",
}


@dataclass
class RunConfig:
    budget: float = 100_000.0
    cost_bps: float = 5.0
    max_weight: float = 0.30
    rebalance_every: int = 1
    train_end: str = "2021-12-31"
    val_end: str = "2022-12-31"
    split: str = "test"


def build_market(
    source: str,
    *,
    tickers: list[str] | None = None,
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    n_days: int = 2600,
    n_assets: int = 10,
    seed: int = 42,
) -> MarketData:
    """Build a :class:`MarketData` from either synthetic or live data."""
    if source == "synthetic":
        return MarketData.from_synthetic(n_days=n_days, n_assets=n_assets, seed=seed)

    from ..config import Config, DataConfig

    cfg = Config(
        seed=seed,
        data=DataConfig(tickers=tickers or DataConfig().tickers, start=start, end=end),
    )
    return MarketData.from_config(cfg)


def run_strategies(
    market: MarketData,
    selected: dict[str, dict],
    cfg: RunConfig,
) -> dict[str, BacktestResult]:
    """Run each selected strategy over the chosen split; return name -> result."""
    lo, hi = market.split_indices(cfg.train_end, cfg.val_end)[cfg.split]
    results: dict[str, BacktestResult] = {}
    for name, params in selected.items():
        policy = build_policy(name, cfg.max_weight, **params)
        results[name] = run_backtest(
            policy,
            market,
            lo,
            hi,
            budget=cfg.budget,
            cost_bps=cfg.cost_bps,
            max_weight=cfg.max_weight,
            rebalance_every=cfg.rebalance_every,
        )
    return results


def metrics_table(results: dict[str, BacktestResult]) -> pd.DataFrame:
    rows = {name: res.metrics for name, res in results.items()}
    df = pd.DataFrame(rows).T
    cols = [c for c in _METRIC_LABELS if c in df.columns]
    df = df[cols].rename(columns=_METRIC_LABELS)
    return df


def _equity_dates(market: MarketData, res: BacktestResult) -> pd.DatetimeIndex:
    if res.value_index.size == 0:
        return pd.DatetimeIndex([])
    start = market.dates[res.decision_index[0]]
    return pd.DatetimeIndex([start]).append(market.dates[res.value_index])


def _color(i: int) -> str:
    return _PALETTE[i % len(_PALETTE)]


def equity_figure(market: MarketData, results: dict[str, BacktestResult]) -> go.Figure:
    fig = go.Figure()
    for i, (name, res) in enumerate(results.items()):
        fig.add_trace(
            go.Scatter(
                x=_equity_dates(market, res),
                y=res.equity_curve,
                name=name,
                mode="lines",
                line=dict(color=_color(i), width=2),
            )
        )
    fig.update_layout(
        title="Portfolio value (out-of-sample)",
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_dark",
        hovermode="x unified",
    )
    return fig


def drawdown_figure(market: MarketData, results: dict[str, BacktestResult]) -> go.Figure:
    fig = go.Figure()
    for i, (name, res) in enumerate(results.items()):
        equity = res.equity_curve
        peak = np.maximum.accumulate(equity)
        dd = equity / peak - 1.0
        fig.add_trace(
            go.Scatter(
                x=_equity_dates(market, res),
                y=dd,
                name=name,
                mode="lines",
                line=dict(color=_color(i), width=1.5),
            )
        )
    fig.update_layout(
        title="Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
        template="plotly_dark",
        hovermode="x unified",
    )
    return fig


def allocation_figure(market: MarketData, res: BacktestResult) -> go.Figure:
    """Stacked area of per-asset weights (plus cash) over time for one strategy."""
    fig = go.Figure()
    if res.weights.size == 0:
        return fig

    dates = market.dates[res.value_index]
    for j, ticker in enumerate(market.tickers):
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=res.weights[:, j],
                name=ticker,
                mode="lines",
                stackgroup="alloc",
                line=dict(width=0.5, color=_color(j)),
            )
        )
    cash = 1.0 - res.weights.sum(axis=1)
    fig.add_trace(
        go.Scatter(
            x=dates, y=cash, name="cash", mode="lines",
            stackgroup="alloc", line=dict(width=0.5, color="#64748b"),
        )
    )
    fig.update_layout(
        title=f"Allocation over time — {res.name}",
        xaxis_title="Date",
        yaxis_title="Weight",
        yaxis_range=[0, 1],
        template="plotly_dark",
        hovermode="x unified",
    )
    return fig


def rolling_sharpe_figure(
    market: MarketData,
    results: dict[str, BacktestResult],
    *,
    window: int = 63,
    periods_per_year: float = 252.0,
) -> go.Figure:
    fig = go.Figure()
    for i, (name, res) in enumerate(results.items()):
        r = pd.Series(res.step_returns)
        if len(r) <= window:
            continue
        roll = (
            r.rolling(window).mean() / r.rolling(window).std() * np.sqrt(periods_per_year)
        )
        fig.add_trace(
            go.Scatter(
                x=market.dates[res.value_index],
                y=roll.to_numpy(),
                name=name,
                mode="lines",
                line=dict(color=_color(i), width=1.5),
            )
        )
    fig.update_layout(
        title=f"Rolling Sharpe ({window}-step)",
        xaxis_title="Date",
        yaxis_title="Sharpe",
        template="plotly_dark",
        hovermode="x unified",
    )
    return fig


def returns_hist_figure(results: dict[str, BacktestResult]) -> go.Figure:
    fig = go.Figure()
    for i, (name, res) in enumerate(results.items()):
        fig.add_trace(
            go.Histogram(
                x=res.step_returns,
                name=name,
                opacity=0.6,
                marker_color=_color(i),
                nbinsx=60,
            )
        )
    fig.update_layout(
        title="Distribution of step returns",
        xaxis_title="Step return",
        yaxis_title="Count",
        barmode="overlay",
        template="plotly_dark",
    )
    return fig
