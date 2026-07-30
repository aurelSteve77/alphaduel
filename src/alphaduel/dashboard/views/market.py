"""Market page: preview the synthetic (mock) price paths and features."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from alphaduel.dashboard import mock_data, state
from alphaduel.dashboard.components import theme


def render() -> None:
    st.title("Market — mock data")
    st.caption("Synthetic geometric-Brownian-motion prices; no real quotes are downloaded.")

    params = state.ensure_params()

    if params["mode"] == "multi_asset":
        panel = mock_data.make_multi_panel(
            params["n_steps"], params["n_assets"], params["drift"], params["vol"], params["seed"]
        )
        symbols = panel.symbols
        close = panel.close
    else:
        panel = mock_data.make_single_panel(
            params["n_steps"], params["drift"], params["vol"], params["seed"]
        )
        symbols = ["ASSET"]
        close = panel.close.reshape(-1, 1)

    colors = theme.color_map(symbols)
    fig = go.Figure()
    for i, sym in enumerate(symbols):
        series = close[:, i] / close[0, i]
        fig.add_trace(
            go.Scatter(
                x=panel.timestamps,
                y=series,
                name=sym,
                line={"color": colors[sym], "width": 1.6},
                hovertemplate="%{y:.2f}<extra>" + sym + "</extra>",
            )
        )
    fig.update_layout(
        height=440,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        yaxis_title="Growth of $1",
        xaxis_title="Date",
        legend={"orientation": "h", "y": -0.2},
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Assets", len(symbols))
    c2.metric("History (days)", panel.n_steps)
    c3.metric("Features / asset", panel.n_features)
    st.caption("Features: " + ", ".join(panel.feature_names))
