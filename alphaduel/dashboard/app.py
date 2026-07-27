"""alphaduel strategy lab — Streamlit entrypoint."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="alphaduel",
    page_icon="α",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = {
    "Live run": "pages/1_Live_Run.py",
    "Benchmark": "pages/2_Benchmark.py",
}

st.title("alphaduel")
st.markdown(
    """
Shared portfolio environment for LLM and baseline strategies.

Use the sidebar pages to:

1. **Live run** — step through one strategy episode, watch equity, trades and rationales update.
2. **Benchmark** — run several strategies on the same window and compare metrics / equity curves.

Under the hood this UI calls the same evaluation harness as
`uv run alphaduel eval --strategy …` (`StrategyRunConfig` → `evaluate` / `iter_episode`).
"""
)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Live run")
    st.write("Single-agent trajectory with buy/sell markers, holdings and LLM rationale.")
    st.page_link("pages/1_Live_Run.py", label="Open live run", icon="▶")
with col2:
    st.subheader("Benchmark")
    st.write("Side-by-side metrics and overlayed equity for multiple strategies.")
    st.page_link("pages/2_Benchmark.py", label="Open benchmark", icon="📊")

st.divider()
presets = sorted(p.stem for p in (Path(__file__).resolve().parents[2] / "configs" / "strategies").glob("*.yaml"))
st.caption("Strategy presets: " + (", ".join(presets) if presets else "(none)"))
