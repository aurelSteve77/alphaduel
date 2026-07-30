"""Transactions component: per-step fills / costs table + summary for one agent."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from alphaduel.dashboard.engine import BenchmarkResult


def render(result: BenchmarkResult, agent_name: str) -> None:
    rec = result.agents[agent_name]
    st.subheader("Transactions")

    n_trades = int((rec.fills != 0).sum())
    total_cost = float(rec.costs.sum())
    turnover = float(np.abs(np.diff(rec.exposure, prepend=0.0)).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Trades", f"{n_trades}")
    c2.metric("Total cost", f"${total_cost:,.0f}")
    c3.metric("Turnover", f"{turnover:.2f}x")

    active = np.flatnonzero(rec.fills != 0)
    if active.size == 0:
        st.info("No trades in this episode.")
        return
    df = pd.DataFrame(
        {
            "step": active,
            "fill": rec.fills[active],
            "cost": rec.costs[active],
            "equity": rec.equity[1:][active],
        }
    )
    st.dataframe(
        df.style.format({"cost": "${:,.2f}", "equity": "${:,.0f}"}),
        use_container_width=True,
        height=280,
    )
