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

# Warm palette that harmonizes with the Anthropic cream/terracotta theme.
_PALETTE = [
    "#bb5a38", "#3d3a2a", "#c99a2e", "#4a6fa5", "#6b8e6b",
    "#a34a3f", "#8a6d3b", "#7a5c7a", "#4c8c8c", "#b07d4a",
]
_CASH_COLOR = "#b8b5a8"
_TEXT = "#3d3a2a"
_GRID = "rgba(61, 58, 42, 0.12)"
# Sequential (weight) and diverging (buy/sell) colorscales, on-theme.
# Zero maps to the cream background so "no position"/"hold" blends in.
_WEIGHT_SCALE = [[0.0, "#f4f3ed"], [1.0, "#bb5a38"]]
_TRADE_SCALE = [[0.0, "#a34a3f"], [0.5, "#f4f3ed"], [1.0, "#6b8e6b"]]

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


def _finalize(fig: go.Figure) -> go.Figure:
    """Apply the shared warm/light look with transparent backgrounds so the
    charts blend into the Anthropic cream theme (rendered with ``theme=None``)."""
    fig.update_layout(
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(color=_TEXT),
        legend=dict(bgcolor="rgba(0, 0, 0, 0)"),
        margin=dict(l=50, r=20, t=50, b=40),
    )
    fig.update_xaxes(gridcolor=_GRID, zerolinecolor=_GRID, linecolor=_GRID)
    fig.update_yaxes(gridcolor=_GRID, zerolinecolor=_GRID, linecolor=_GRID)
    return fig


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
        hovermode="x unified",
    )
    return _finalize(fig)


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
        hovermode="x unified",
    )
    return _finalize(fig)


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
            stackgroup="alloc", line=dict(width=0.5, color=_CASH_COLOR),
        )
    )
    fig.update_layout(
        title=f"Allocation over time — {res.name}",
        xaxis_title="Date",
        yaxis_title="Weight",
        yaxis_range=[0, 1],
        hovermode="x unified",
    )
    return _finalize(fig)


def allocation_heatmap_figure(market: MarketData, res: BacktestResult) -> go.Figure:
    """Heatmap of per-asset position weights over time (state / holdings)."""
    fig = go.Figure()
    if res.weights.size == 0:
        return _finalize(fig)

    fig.add_trace(
        go.Heatmap(
            x=market.dates[res.value_index],
            y=market.tickers,
            z=res.weights.T,  # [K, T]
            colorscale=_WEIGHT_SCALE,
            zmin=0.0,
            colorbar=dict(title="weight", tickformat=".0%"),
            hovertemplate="%{y}<br>%{x|%Y-%m-%d}<br>weight = %{z:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Position weights over time — {res.name}",
        xaxis_title="Date",
        yaxis_title="Asset",
    )
    return _finalize(fig)


def trades_heatmap_figure(market: MarketData, res: BacktestResult) -> go.Figure:
    """Diverging heatmap of the actual actions: Δ-weight per asset per step
    (green = buy, red = sell, blank = hold)."""
    fig = go.Figure()
    if res.weights.size == 0:
        return _finalize(fig)

    deltas = np.vstack([res.weights[0], np.diff(res.weights, axis=0)])
    zmax = float(np.abs(deltas).max()) or 1.0
    fig.add_trace(
        go.Heatmap(
            x=market.dates[res.value_index],
            y=market.tickers,
            z=deltas.T,  # [K, T]
            colorscale=_TRADE_SCALE,
            zmid=0.0,
            zmin=-zmax,
            zmax=zmax,
            colorbar=dict(title="Δ weight", tickformat="+.0%"),
            hovertemplate="%{y}<br>%{x|%Y-%m-%d}<br>Δ = %{z:+.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Trades (buy / sell) over time — {res.name}",
        xaxis_title="Date",
        yaxis_title="Asset",
    )
    return _finalize(fig)


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
        hovermode="x unified",
    )
    return _finalize(fig)


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
    )
    return _finalize(fig)
