"""Agent parameter widgets: render per-agent controls and return a params dict."""

from __future__ import annotations

import streamlit as st

from alphaduel.dashboard import engine
from alphaduel.prompts import prompt_manager

_LOOKBACK_AGENTS = {"momentum", "mean_reversion", "inverse_volatility"}


def render(agent_name: str, key: str = "cfg", defaults: dict | None = None) -> dict:
    """Render the configurable parameters for ``agent_name`` and return their values."""
    params: dict = {}
    kp = f"{key}_{agent_name}"
    defaults = defaults or {}

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

    elif agent_name in engine.LLM_AGENTS:
        params.update(_render_llm(kp, defaults))

    else:
        st.caption("This agent has no tunable parameters.")

    return params


def render_llm_settings(defaults: dict | None = None, key: str = "llm_global") -> dict:
    """Global LLM settings used by Configure / Overview benchmarks."""
    return _render_llm(key, defaults or engine.default_llm_params())


def _render_llm(kp: str, defaults: dict) -> dict:
    st.caption(
        "Requires `uv sync --extra llm` and a running Ollama daemon with the model pulled. "
        "Reasoning traces are off by default."
    )
    c1, c2 = st.columns(2)
    provider = c1.selectbox(
        "Provider",
        ["ollama"],
        index=0,
        key=f"{kp}_provider",
    )
    model = c2.text_input(
        "Model",
        value=str(defaults.get("llm_model", defaults.get("model", "qwen3.5:2b"))),
        key=f"{kp}_model",
    )
    c3, c4 = st.columns(2)
    temperature = c3.slider(
        "Temperature",
        0.0,
        1.0,
        float(defaults.get("llm_temperature", defaults.get("temperature", 0.0))),
        step=0.05,
        key=f"{kp}_temp",
    )
    reasoning = c4.checkbox(
        "Enable reasoning traces",
        value=bool(defaults.get("llm_reasoning", defaults.get("reasoning", False))),
        key=f"{kp}_reason",
    )
    use_memory = st.checkbox(
        "Use conversation memory",
        value=bool(defaults.get("llm_use_memory", defaults.get("use_memory", False))),
        key=f"{kp}_mem",
    )
    max_memory_turns = st.slider(
        "Max memory turns",
        1,
        20,
        int(defaults.get("llm_max_memory_turns", defaults.get("max_memory_turns", 8))),
        key=f"{kp}_mem_n",
    )

    prompt_keys = prompt_manager.keys() or ["base_agent_system", "base_agent_user"]
    system_default = defaults.get("llm_system_prompt_key", "base_agent_system")
    user_default = defaults.get("llm_user_prompt_key", "base_agent_user")
    system_opts = prompt_keys if system_default in prompt_keys else [system_default, *prompt_keys]
    user_opts = prompt_keys if user_default in prompt_keys else [user_default, *prompt_keys]
    pc1, pc2 = st.columns(2)
    system_prompt_key = pc1.selectbox(
        "System prompt",
        system_opts,
        index=system_opts.index(system_default) if system_default in system_opts else 0,
        key=f"{kp}_sys",
    )
    user_prompt_key = pc2.selectbox(
        "User prompt",
        user_opts,
        index=user_opts.index(user_default) if user_default in user_opts else 0,
        key=f"{kp}_usr",
    )

    return {
        "llm": {
            "provider": provider,
            "model": model.strip() or "qwen3.5:2b",
            "temperature": float(temperature),
            "reasoning": bool(reasoning),
        },
        "use_memory": bool(use_memory),
        "max_memory_turns": int(max_memory_turns),
        "system_prompt_key": system_prompt_key,
        "user_prompt_key": user_prompt_key,
    }


def flatten_llm_params(agent_params: dict) -> dict:
    """Convert live-page nested LLM params into dashboard flat keys."""
    llm = agent_params.get("llm") or {}
    return {
        "llm_provider": llm.get("provider", "ollama"),
        "llm_model": llm.get("model", "qwen3.5:2b"),
        "llm_temperature": float(llm.get("temperature", 0.0)),
        "llm_reasoning": bool(llm.get("reasoning", False)),
        "llm_use_memory": bool(agent_params.get("use_memory", False)),
        "llm_max_memory_turns": int(agent_params.get("max_memory_turns", 8)),
        "llm_system_prompt_key": agent_params.get("system_prompt_key", "base_agent_system"),
        "llm_user_prompt_key": agent_params.get("user_prompt_key", "base_agent_user"),
    }
