"""Plotly charts for evaluation trajectories."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from alphaduel.evaluation.metrics import EvalResult

# Align with .streamlit/config.toml warm theme.
_INK = "#3d3a2a"
_ACCENT = "#bb5a38"
_TEAL = "#2a9d8f"
_GOLD = "#c4a35a"
_MUTED = "#8a8674"
_GRID = "#d3d2ca"
_PAPER = "#f4f3ed"
_PLOT = "#ecebe3"

_PALETTE = [_ACCENT, _TEAL, "#1f4e79", _GOLD, "#6b4f3a", "#4a6fa5", "#8b5e3c"]


def _layout(fig: go.Figure, *, title: str | None = None, height: int = 360) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=_INK)) if title else None,
        paper_bgcolor=_PAPER,
        plot_bgcolor=_PLOT,
        font=dict(color=_INK, family="Manrope, sans-serif"),
        margin=dict(l=48, r=24, t=48 if title else 24, b=40),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor=_GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=_GRID, zeroline=False)
    return fig


def equity_chart(result: EvalResult, *, title: str | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.dates,
            y=result.portfolio_values,
            mode="lines",
            name="Portfolio",
            line=dict(color=_ACCENT, width=2.2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.dates,
            y=result.cash,
            mode="lines",
            name="Cash",
            line=dict(color=_MUTED, width=1.4, dash="dot"),
        )
    )
    if result.metrics is not None:
        fig.add_hline(
            y=result.metrics.initial_cash,
            line_dash="dash",
            line_color=_MUTED,
            annotation_text="initial cash",
            annotation_position="bottom right",
        )
    return _layout(fig, title=title or f"{result.name} — equity", height=380)


def drawdown_chart(result: EvalResult) -> go.Figure:
    values = result.portfolio_values
    if not values:
        return _layout(go.Figure(), title="Drawdown")
    peak = values[0]
    dd: list[float] = []
    for v in values:
        peak = max(peak, v)
        dd.append(0.0 if peak <= 0 else (peak - v) / peak)
    fig = go.Figure(
        go.Scatter(
            x=result.dates,
            y=dd,
            fill="tozeroy",
            mode="lines",
            name="Drawdown",
            line=dict(color=_ACCENT, width=1.5),
            fillcolor="rgba(187, 90, 56, 0.25)",
        )
    )
    fig.update_yaxes(tickformat=".0%")
    return _layout(fig, title="Drawdown", height=280)


def holdings_chart(result: EvalResult) -> go.Figure:
    frame = result.holdings_frame()
    fig = go.Figure()
    if frame.empty:
        return _layout(fig, title="Holdings")
    for i, ticker in enumerate(result.tickers):
        fig.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame[ticker],
                mode="lines",
                name=ticker,
                line=dict(color=_PALETTE[i % len(_PALETTE)], width=1.8),
                stackgroup="one",
            )
        )
    return _layout(fig, title="Holdings (shares)", height=320)


def trades_scatter(result: EvalResult) -> go.Figure:
    trades = result.trades_frame()
    fig = go.Figure()
    if trades.empty:
        return _layout(fig, title="Trades")
    buys = trades[trades["side"] == "buy"]
    sells = trades[trades["side"] == "sell"]
    if not buys.empty:
        fig.add_trace(
            go.Scatter(
                x=buys["date"],
                y=buys["ticker"],
                mode="markers",
                name="Buy",
                marker=dict(symbol="triangle-up", size=12, color=_TEAL),
                text=[f"{r.shares} @ {r.price:.2f}" for r in buys.itertuples()],
                hovertemplate="%{y}<br>%{text}<extra>buy</extra>",
            )
        )
    if not sells.empty:
        fig.add_trace(
            go.Scatter(
                x=sells["date"],
                y=sells["ticker"],
                mode="markers",
                name="Sell",
                marker=dict(symbol="triangle-down", size=12, color=_ACCENT),
                text=[f"{r.shares} @ {r.price:.2f}" for r in sells.itertuples()],
                hovertemplate="%{y}<br>%{text}<extra>sell</extra>",
            )
        )
    return _layout(fig, title="Trades", height=300)


def comparison_equity(results: Sequence[EvalResult]) -> go.Figure:
    fig = go.Figure()
    for i, result in enumerate(results):
        fig.add_trace(
            go.Scatter(
                x=result.dates,
                y=result.portfolio_values,
                mode="lines",
                name=result.name,
                line=dict(color=_PALETTE[i % len(_PALETTE)], width=2),
            )
        )
    return _layout(fig, title="Equity comparison", height=420)


def episode_dashboard(result: EvalResult) -> go.Figure:
    """Compact multi-panel snapshot for the live page."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.08,
        subplot_titles=("Portfolio value", "Daily PnL"),
    )
    fig.add_trace(
        go.Scatter(
            x=result.dates,
            y=result.portfolio_values,
            mode="lines",
            name="Portfolio",
            line=dict(color=_ACCENT, width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=result.dates,
            y=result.pnls,
            name="PnL",
            marker_color=[_TEAL if v >= 0 else _ACCENT for v in result.pnls],
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        paper_bgcolor=_PAPER,
        plot_bgcolor=_PLOT,
        font=dict(color=_INK, family="Manrope, sans-serif"),
        height=520,
        showlegend=False,
        margin=dict(l=48, r=24, t=40, b=40),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor=_GRID)
    fig.update_yaxes(showgrid=True, gridcolor=_GRID)
    return fig


def metrics_bar(comparison: pd.DataFrame, *, metric: str = "total_return") -> go.Figure:
    if comparison.empty or metric not in comparison.columns:
        return _layout(go.Figure(), title=metric)
    fig = go.Figure(
        go.Bar(
            x=comparison["strategy"],
            y=comparison[metric],
            marker_color=_ACCENT,
            text=[f"{v:.2%}" if "return" in metric or "drawdown" in metric or "hit" in metric else f"{v:.2f}"
                  for v in comparison[metric]],
            textposition="outside",
        )
    )
    return _layout(fig, title=metric.replace("_", " ").title(), height=340)
