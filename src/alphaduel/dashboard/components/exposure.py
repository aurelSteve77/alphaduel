"""Exposure component: how one agent allocates capital over the replay episode."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from alphaduel.dashboard.components import theme
from alphaduel.dashboard.engine import BenchmarkResult


def render(result: BenchmarkResult, agent_name: str) -> None:
    rec = result.agents[agent_name]
    st.subheader("Allocation over time")

    if result.mode == "multi_asset" and rec.weights.shape[1] > 1:
        _stacked_weights(result.symbols, rec.weights)
    else:
        _single_exposure(rec.exposure)


def _stacked_weights(symbols: list[str], weights: np.ndarray) -> None:
    colors = theme.color_map(symbols)
    steps = list(range(len(weights)))
    fig = go.Figure()
    for i, sym in enumerate(symbols):
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=weights[:, i],
                name=sym,
                stackgroup="w",
                line={"width": 0.5, "color": colors[sym]},
                hovertemplate="%{y:.2%}<extra>" + sym + "</extra>",
            )
        )
    cash = np.clip(1.0 - weights.sum(axis=1), 0.0, 1.0)
    fig.add_trace(
        go.Scatter(x=steps, y=cash, name="Cash", stackgroup="w",
                   line={"width": 0.5, "color": "#BAB0AC"})
    )
    fig.update_layout(
        height=360, margin={"l": 10, "r": 10, "t": 10, "b": 10},
        yaxis_title="Portfolio weight", xaxis_title="Step",
        yaxis={"range": [0, 1]}, legend={"orientation": "h", "y": -0.2},
    )
    st.plotly_chart(fig, use_container_width=True)


def _single_exposure(exposure: np.ndarray) -> None:
    steps = list(range(len(exposure)))
    fig = go.Figure(
        go.Scatter(x=steps, y=exposure, fill="tozeroy", line={"color": "#4C78A8"},
                   hovertemplate="%{y:.2%}<extra>invested</extra>")
    )
    fig.update_layout(
        height=360, margin={"l": 10, "r": 10, "t": 10, "b": 10},
        yaxis_title="Invested fraction", xaxis_title="Step", yaxis={"range": [0, 1]},
    )
    st.plotly_chart(fig, use_container_width=True)
