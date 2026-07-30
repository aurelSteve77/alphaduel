"""Configuration page: edit the shared experiment params in session_state."""

from __future__ import annotations

import streamlit as st

from alphaduel.dashboard import engine, state
from alphaduel.dashboard.components import theme


def render() -> None:
    params = dict(state.ensure_params())
    st.title("⚙️ Configure experiment")
    st.caption("Settings here drive the Live run, Overview, Agent detail and Market pages.")

    presets = state.load_presets()
    with st.container(border=True):
        st.markdown("**Presets**")
        pc1, pc2 = st.columns([3, 1])
        preset_name = pc1.selectbox("Load a preset", list(presets), key="cfg_preset_load")
        if pc2.button("Load", use_container_width=True):
            st.session_state["params"] = dict(presets[preset_name])
            st.toast(f"Loaded preset '{preset_name}'.")
            st.rerun()

    st.subheader("Universe")

    mode_label = st.radio(
        "Type", ["Single asset", "Multi-asset portfolio"],
        index=0 if params["mode"] == "single_asset" else 1,
        horizontal=True,
    )
    mode = "multi_asset" if mode_label.startswith("Multi") else "single_asset"

    n_assets = params["n_assets"]
    if mode == "multi_asset":
        n_assets = st.slider("Number of assets", 2, 8, int(n_assets))

    available = engine.agents_for(mode)
    prev = [a for a in params["agent_names"] if a in available] or available
    agent_names = st.multiselect(
        "Agents to benchmark", options=available, default=prev, format_func=theme.label
    )

    st.subheader("Market regime")
    c1, c2, c3 = st.columns(3)
    n_steps = c1.slider("History (days)", 250, 1500, int(params["n_steps"]), step=50)
    drift = c2.slider("Daily drift (bps)", -5.0, 10.0, params["drift"] * 1e4, step=0.5) / 1e4
    vol = c3.slider("Daily volatility (%)", 0.5, 3.0, params["vol"] * 100, step=0.1) / 100.0

    st.subheader("Episodes")
    c4, c5, c6 = st.columns(3)
    episode_length = c4.slider("Episode length (days)", 30, 250, int(params["episode_length"]),
                               step=10)
    n_episodes = c5.slider("Episodes (for stats)", 1, 50, int(params["n_episodes"]))
    seed = c6.number_input("Seed", 0, 9999, int(params["seed"]))

    st.subheader("Costs & reward")
    c7, c8, c9, c10 = st.columns(4)
    initial_cash = c7.number_input("Initial cash", 1_000.0, value=float(params["initial_cash"]),
                                   step=1_000.0)
    commission_bps = c8.slider("Commission (bps)", 0.0, 10.0, float(params["commission_bps"]),
                               step=0.5)
    half_spread_bps = c9.slider("Half spread (bps)", 0.0, 10.0, float(params["half_spread_bps"]),
                                step=0.5)
    reward_kind = c10.selectbox(
        "Reward", ["log_return", "differential_sharpe", "terminal_pnl"],
        index=["log_return", "differential_sharpe", "terminal_pnl"].index(params["reward_kind"]),
    )

    st.session_state["params"] = {
        "mode": mode,
        "agent_names": tuple(agent_names),
        "n_assets": int(n_assets),
        "n_steps": int(n_steps),
        "episode_length": int(episode_length),
        "n_episodes": int(n_episodes),
        "seed": int(seed),
        "initial_cash": float(initial_cash),
        "commission_bps": float(commission_bps),
        "half_spread_bps": float(half_spread_bps),
        "reward_kind": reward_kind,
        "drift": float(drift),
        "vol": float(vol),
    }

    st.divider()
    st.subheader("Save as preset")
    sc1, sc2 = st.columns([3, 1])
    new_name = sc1.text_input(
        "Preset name", placeholder="e.g. My tech basket", label_visibility="collapsed"
    )
    if sc2.button("💾 Save", use_container_width=True):
        if new_name.strip():
            state.save_preset(new_name.strip(), st.session_state["params"])
            st.toast(f"Saved preset '{new_name.strip()}'.")
            st.rerun()
        else:
            st.warning("Enter a name for the preset.")

    dc1, dc2 = st.columns([3, 1])
    to_delete = dc1.selectbox("Delete a preset", list(presets), key="cfg_preset_delete")
    if dc2.button("🗑️ Delete", use_container_width=True):
        state.delete_preset(to_delete)
        st.toast(f"Deleted preset '{to_delete}'.")
        st.rerun()

    st.divider()
    if not agent_names:
        st.warning("Select at least one agent to enable the benchmark pages.")
    else:
        st.success(f"Configured {len(agent_names)} agent(s) on a {mode_label.lower()} universe.")
    overview = st.session_state.get("_pages", {}).get("overview")
    if overview is not None:
        st.page_link(overview, label="Go to Overview", icon="📊")

