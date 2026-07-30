"""Agent-detail page: drill into one agent's allocation, trades and reasoning."""

from __future__ import annotations

import streamlit as st

from alphaduel.dashboard import state
from alphaduel.dashboard.components import exposure, theme, thoughts, transactions


def render() -> None:
    st.title("Agent detail")
    result = state.get_result()
    if result is None:
        st.warning("Select at least one agent on the Configure page to run the benchmark.")
        return

    names = list(result.agents)
    agent_name = st.selectbox("Agent", names, format_func=theme.label)
    rec = result.agents[agent_name]

    m = rec.metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total return", theme.pct(m["total_return"]))
    c2.metric("Sharpe", theme.num(m["sharpe"]))
    c3.metric("Max drawdown", theme.pct(m["max_drawdown"]))
    c4.metric("# trades", theme.num(m["n_transactions"], 0))

    exposure.render(result, agent_name)
    transactions.render(result, agent_name)
    thoughts.render(result, agent_name)
