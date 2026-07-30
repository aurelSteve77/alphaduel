"""Equity-curve component: one normalized line per agent (replay episode)."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from alphaduel.dashboard.components import theme
from alphaduel.dashboard.engine import BenchmarkResult


def render(result: BenchmarkResult) -> None:
    st.subheader("Equity curves")
    st.caption("Normalized to 1.0 at episode start (single replay episode).")

    colors = theme.color_map(list(result.agents))
    fig = go.Figure()
    for name, rec in result.agents.items():
        equity = rec.equity / rec.equity[0]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(equity))),
                y=equity,
                name=theme.label(name),
                line={"color": colors[name], "width": 2},
                hovertemplate="%{y:.3f}<extra>" + theme.label(name) + "</extra>",
            )
        )
    fig.update_layout(
        height=420,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis_title="Step",
        yaxis_title="Growth of $1",
        legend={"orientation": "h", "y": -0.2},
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
