"""Market page: preview real cached price paths and features."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from alphaduel.dashboard import engine, real_data, state
from alphaduel.dashboard.components import theme


def render() -> None:
    st.title("Market — real data")
    st.caption("yfinance prices from the Parquet cache (same path as `alphaduel run`).")

    params = state.ensure_params()

    try:
        panel, symbols = engine.get_panel(params)
    except Exception as exc:  # noqa: BLE001 — surface cache/download errors in the UI
        st.error(
            f"Could not load market data: {exc}\n\n"
            "Open the **Data** page to download a universe, or run:\n"
            "`uv run alphaduel download -c configs/experiment/p5_genportfolio.yaml`"
        )
        return

    close = panel.close if params["mode"] == "multi_asset" else panel.close.reshape(-1, 1)

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
    st.caption("Universe config: " + str(real_data.config_path_for(params["mode"]).name))
