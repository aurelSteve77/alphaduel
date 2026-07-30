"""Drawdown component: underwater curves for all agents (replay episode)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from alphaduel.dashboard.components import theme
from alphaduel.dashboard.engine import BenchmarkResult


def _underwater(equity: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(equity)
    return equity / peak - 1.0


def render(result: BenchmarkResult) -> None:
    st.subheader("Drawdown")
    st.caption("Percent below the running peak.")

    colors = theme.color_map(list(result.agents))
    fig = go.Figure()
    for name, rec in result.agents.items():
        dd = _underwater(rec.equity)
        fig.add_trace(
            go.Scatter(
                x=list(range(len(dd))),
                y=dd,
                name=theme.label(name),
                line={"color": colors[name], "width": 1.5},
                hovertemplate="%{y:.2%}<extra>" + theme.label(name) + "</extra>",
            )
        )
    fig.update_layout(
        height=320, margin={"l": 10, "r": 10, "t": 10, "b": 10},
        yaxis_title="Drawdown", xaxis_title="Step", yaxis={"tickformat": ".0%"},
        legend={"orientation": "h", "y": -0.25}, hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
