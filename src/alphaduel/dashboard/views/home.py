"""Landing page: app-style hero, feature cards and quick links."""

from __future__ import annotations

import streamlit as st

from alphaduel.dashboard import state


def _link(page_key: str, label: str, icon: str) -> None:
    page = st.session_state.get("_pages", {}).get(page_key)
    if page is not None:
        st.page_link(page, label=label, icon=icon, use_container_width=True)


def render() -> None:
    params = state.ensure_params()

    st.title("📈 AlphaDuel")
    st.subheader("Benchmark RL, LLM and generative agents on one trading environment.")
    st.write(
        "A reproducible arena where quantitative and language-model agents trade the same "
        "market under identical costs, execution rules and risk-adjusted rewards. "
        "Everything below runs on **real market data** (yfinance, Parquet-cached). "
        "Baselines, GenPortfolio and **LLM Vanilla** (Ollama) share the same environment."
    )

    st.divider()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1.container(border=True):
        st.markdown("### ⚙️ Configure")
        st.caption("Design the experiment: universe, costs and reward.")
        _link("configure", "Open configuration", "⚙️")
    with c2.container(border=True):
        st.markdown("### 📦 Data")
        st.caption("Download prices / macro and manage the Parquet cache.")
        _link("data", "Manage data", "📦")
    with c3.container(border=True):
        st.markdown("### 🎬 Live run")
        st.caption("Pick an agent, tune it, and watch its decisions step by step.")
        _link("live", "Launch a live run", "🎬")
    with c4.container(border=True):
        st.markdown("### 🏁 Evaluate")
        st.caption("Roster many agents (incl. multiple LLMs) and crown a winner.")
        _link("evaluate", "Open evaluate", "🏁")
    with c5.container(border=True):
        st.markdown("### 📊 Overview")
        st.caption("Quick benchmark from Configure: equity, drawdown, leaderboard.")
        _link("overview", "Open overview", "📊")

    st.divider()

    m = params
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Universe", "Multi-asset" if m["mode"] == "multi_asset" else "Single asset")
    k2.metric("Assets", m["n_assets"] if m["mode"] == "multi_asset" else 1)
    k3.metric("Agents selected", len(m["agent_names"]))
    k4.metric("Episode length", f"{m['episode_length']}d")
