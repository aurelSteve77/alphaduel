"""Overview page: cross-agent comparison (equity, leaderboard, drawdown)."""

from __future__ import annotations

import streamlit as st

from alphaduel.dashboard import state
from alphaduel.dashboard.components import drawdown, equity_curve, metrics_table


def render() -> None:
    st.title("Overview")
    result = state.get_result()
    if result is None:
        st.warning("Select at least one agent on the Configure page to run the benchmark.")
        return

    left, right = st.columns([3, 2])
    with left:
        equity_curve.render(result)
    with right:
        metrics_table.render(result)
    drawdown.render(result)
