"""Agent parameter widgets: render per-agent controls and return a params dict."""

from __future__ import annotations

import streamlit as st

_LOOKBACK_AGENTS = {"momentum", "mean_reversion", "inverse_volatility"}


def render(agent_name: str, key: str = "cfg") -> dict:
    """Render the configurable parameters for ``agent_name`` and return their values."""
    params: dict = {}
    kp = f"{key}_{agent_name}"

    if agent_name in _LOOKBACK_AGENTS:
        params["lookback"] = st.slider("Lookback (days)", 5, 60, 20, key=f"{kp}_lb")

    elif agent_name == "volatility_target":
        params["target_vol"] = st.slider(
            "Target volatility (annualized)", 0.05, 0.40, 0.15, step=0.01, key=f"{kp}_tv"
        )
        params["lookback"] = st.slider("Lookback (days)", 5, 60, 20, key=f"{kp}_lb")

    elif agent_name in ("random", "random_weights"):
        params["seed"] = int(st.number_input("Agent seed", 0, 9999, 0, key=f"{kp}_seed"))

    elif agent_name == "genportfolio":
        params["max_position_weight"] = st.slider(
            "Max position weight", 0.05, 1.0, 0.20, step=0.05, key=f"{kp}_mpw"
        )
        c1, c2 = st.columns(2)
        params["n_feature_buckets"] = c1.slider("Feature buckets", 4, 64, 16, key=f"{kp}_fb")
        params["n_weight_buckets"] = c2.slider("Weight buckets", 4, 20, 10, key=f"{kp}_wb")

    else:
        st.caption("This agent has no tunable parameters.")

    return params
