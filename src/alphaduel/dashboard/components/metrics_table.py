"""Metrics leaderboard component: aggregate stats per agent (mean over episodes)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from alphaduel.dashboard.engine import BenchmarkResult

_COLUMNS = {
    "total_return": "Total return",
    "annualized_return": "Ann. return",
    "volatility": "Ann. vol",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "max_drawdown": "Max DD",
    "calmar": "Calmar",
    "deflated_sharpe": "Deflated Sharpe",
    "n_transactions": "# trades",
}
_PCT_ROWS = {"Total return", "Ann. return", "Ann. vol", "Max DD"}


def render(result: BenchmarkResult) -> None:
    st.subheader("Leaderboard")
    st.caption("Mean across episodes. Deflated Sharpe accounts for multiple agents tested.")

    rows = {}
    for name, rec in result.agents.items():
        rows[name] = {label: rec.metrics.get(key, 0.0) for key, label in _COLUMNS.items()}
    df = pd.DataFrame(rows).T
    df.index.name = "agent"

    styler = df.style.format(
        {c: "{:.2%}" if c in _PCT_ROWS else "{:.2f}" for c in df.columns}
    )
    if "Sharpe" in df.columns and len(df) > 1:
        styler = styler.background_gradient(subset=["Sharpe"], cmap="Greens")
    st.dataframe(styler, use_container_width=True)
