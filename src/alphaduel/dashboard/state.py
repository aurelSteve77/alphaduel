"""Shared dashboard state: experiment params in session_state + cached benchmark run."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from alphaduel.dashboard import engine

_PRESET_PATH = Path(".alphaduel") / "presets.json"


def ensure_params() -> dict:
    """Seed default experiment params on first load and return the current params."""
    if "params" not in st.session_state:
        st.session_state["params"] = engine.default_params()
    else:
        # Migrate older mock-era sessions / presets toward real data + LLM defaults.
        params = st.session_state["params"]
        params.setdefault("use_mock", False)
        params.pop("drift", None)
        params.pop("vol", None)
        for key, value in engine.default_llm_params().items():
            params.setdefault(key, value)
    return st.session_state["params"]


@st.cache_data(show_spinner="Running benchmark (LLM agents call Ollama when selected)...")
def _run(params: dict) -> engine.BenchmarkResult:
    return engine.run_benchmark(**params)


def get_result() -> engine.BenchmarkResult | None:
    """Cached benchmark result for the current params, or ``None`` if unconfigured."""
    params = st.session_state.get("params")
    if not params or not params.get("agent_names"):
        return None
    return _run(params)


# --------------------------------------------------------------------- presets


def _coerce(params: dict) -> dict:
    params = dict(params)
    if "agent_names" in params:
        params["agent_names"] = tuple(params["agent_names"])
    params.setdefault("use_mock", False)
    params.pop("drift", None)
    params.pop("vol", None)
    for key, value in engine.default_llm_params().items():
        params.setdefault(key, value)
    return params


def _builtin_presets() -> dict[str, dict]:
    balanced = engine.default_params()
    momentum = engine.default_params()
    momentum.update(
        mode="single_asset",
        agent_names=tuple(engine.baseline_agents("single_asset")),
        n_assets=1,
        episode_length=90,
    )
    stress = engine.default_params()
    stress.update(commission_bps=5.0, half_spread_bps=6.0, episode_length=90)
    llm_live = engine.default_params()
    llm_live.update(
        agent_names=("llm_vanilla",),
        n_episodes=3,
        episode_length=40,
        llm_use_memory=False,
        llm_reasoning=False,
    )
    return {
        "Balanced multi-asset": balanced,
        "Single-asset trend": momentum,
        "High-cost stress test": stress,
        "LLM vanilla (short)": llm_live,
    }


def _persist(presets: dict[str, dict]) -> None:
    out = {}
    for name, params in presets.items():
        q = dict(params)
        q["agent_names"] = list(q.get("agent_names", []))
        out[name] = q
    try:
        _PRESET_PATH.parent.mkdir(parents=True, exist_ok=True)
        _PRESET_PATH.write_text(json.dumps(out, indent=2))
    except OSError:
        pass


def load_presets() -> dict[str, dict]:
    """Return the saved presets (built-ins on first run), cached in session_state."""
    if "presets" in st.session_state:
        return st.session_state["presets"]
    if _PRESET_PATH.exists():
        try:
            raw = json.loads(_PRESET_PATH.read_text())
            presets = {name: _coerce(p) for name, p in raw.items()}
        except (json.JSONDecodeError, OSError):
            presets = dict(_builtin_presets())
    else:
        presets = dict(_builtin_presets())
        _persist(presets)
    st.session_state["presets"] = presets
    return presets


def save_preset(name: str, params: dict) -> None:
    presets = load_presets()
    presets[name] = _coerce(dict(params))
    st.session_state["presets"] = presets
    _persist(presets)


def delete_preset(name: str) -> None:
    presets = load_presets()
    presets.pop(name, None)
    st.session_state["presets"] = presets
    _persist(presets)

