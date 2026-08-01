"""Configuration page: edit the shared experiment params in session_state."""

from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from alphaduel.dashboard import engine, real_data, state
from alphaduel.dashboard.components import agent_config, theme


def _parse_extra_tickers(text: str) -> list[str]:
    return [s.strip().upper() for s in text.replace(";", ",").split(",") if s.strip()]


def _as_date(value: str | date | None, fallback: date) -> date:
    if value is None:
        return fallback
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _render_symbol_picker(mode: str, params: dict) -> tuple[str, ...]:
    """Return the selected ticker tuple for the current mode."""
    catalog = real_data.symbol_catalog()
    prev = [str(s).upper() for s in params.get("symbols") or ()]

    if mode == "single_asset":
        options = list(catalog)
        custom_label = "Custom ticker…"
        if custom_label not in options:
            options = [*options, custom_label]
        default = prev[0] if prev else (catalog[0] if catalog else "AAPL")
        if default not in options and default != custom_label:
            options = [default, *options]
        index = options.index(default) if default in options else 0
        choice = st.selectbox("Stock", options, index=index, key="cfg_single_symbol")
        if choice == custom_label:
            custom = st.text_input(
                "Ticker",
                value=default if default not in catalog else "",
                placeholder="e.g. META",
                key="cfg_single_custom",
            ).strip().upper()
            if not custom:
                st.warning("Enter a ticker symbol.")
                return (default if default != custom_label else "AAPL",)
            return (custom,)
        return (choice,)

    # Multi-asset: one comma-separated ticker field (catalog + any custom names).
    default_multi = prev or list(
        engine.default_symbols("multi_asset", int(params.get("n_assets", 4)))
    )
    text = st.text_input(
        "Stocks (comma-separated)",
        value=", ".join(default_multi),
        key="cfg_multi_symbols",
        help=(
            "Any yfinance tickers. Catalog examples: "
            + ", ".join(catalog[:8])
            + ("…" if len(catalog) > 8 else "")
        ),
        placeholder="e.g. MSFT, JNJ, XOM, JPM, PG, CAT, ASML, SHEL, TM, BHP",
    )
    merged = _parse_extra_tickers(text)
    if len(merged) < 2:
        st.warning("Enter at least two tickers for multi-asset mode.")
        fallback = list(engine.default_symbols("multi_asset", 4))
        return tuple(merged) if len(merged) >= 2 else tuple(fallback)
    st.caption(f"{len(merged)} assets: " + ", ".join(merged))
    return tuple(merged)


def render() -> None:
    params = dict(state.ensure_params())
    st.title("⚙️ Configure experiment")
    st.caption(
        "Settings here drive Live run, Evaluate, Overview, Agent detail and Market. "
        "Use **Evaluate** to compare multiple LLM Vanilla configs side by side."
    )

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

    symbols = _render_symbol_picker(mode, params)
    n_assets = len(symbols)

    available = engine.agents_for(mode)
    prev = [a for a in params["agent_names"] if a in available] or engine.baseline_agents(mode)
    agent_names = st.multiselect(
        "Agents to benchmark", options=available, default=prev, format_func=theme.label
    )
    if any(a in engine.LLM_AGENTS for a in agent_names):
        st.info(
            "LLM Vanilla needs `uv sync --extra llm` plus a provider key / Ollama. "
            "Benchmarks with it are slower — prefer Live run or Evaluate for trials."
        )

    st.subheader("Market window")
    yaml_start, yaml_end = real_data.default_date_range(mode)
    default_start = _as_date(params.get("start_date"), date.fromisoformat(yaml_start))
    default_end = _as_date(params.get("end_date"), date.fromisoformat(yaml_end))
    min_date = date(1990, 1, 1)
    max_date = date.today() + timedelta(days=1)

    d1, d2 = st.columns(2)
    start_d = d1.date_input(
        "Start date",
        value=default_start,
        min_value=min_date,
        max_value=max_date,
        key="cfg_start_date",
    )
    end_d = d2.date_input(
        "End date",
        value=default_end,
        min_value=min_date,
        max_value=max_date,
        key="cfg_end_date",
    )
    if end_d <= start_d:
        st.warning("End date must be after start date.")
        end_d = start_d + timedelta(days=1)

    limit_tail = st.checkbox(
        "Also limit to the last N trading days within this range",
        value=bool(params.get("limit_n_steps", True)),
        key="cfg_limit_n_steps",
        help="If unchecked, the full start→end window is used (subject to available data).",
    )
    if limit_tail:
        n_steps = st.slider(
            "Max trading days (tail of the range)",
            250,
            3000,
            int(params.get("n_steps") or 500),
            step=50,
            key="cfg_n_steps",
        )
    else:
        n_steps = None
        st.caption("Using the full date range (no N-day tail).")

    st.caption(
        "Prices come from yfinance (cached under `data_cache/`). "
        "Changing dates/symbols may trigger a download on first use. "
        "Or use the **Data** page to refresh the cache."
    )

    st.subheader("Episodes")
    c4, c5, c6 = st.columns(3)
    episode_length = c4.slider(
        "Episode length (days)", 30, 250, int(params["episode_length"]), step=10
    )
    n_episodes = c5.slider("Episodes (for stats)", 1, 50, int(params["n_episodes"]))
    seed = c6.number_input("Seed", 0, 9999, int(params["seed"]))

    st.subheader("Costs & reward")
    c7, c8, c9, c10 = st.columns(4)
    initial_cash = c7.number_input(
        "Initial cash", 1_000.0, value=float(params["initial_cash"]), step=1_000.0
    )
    commission_bps = c8.slider(
        "Commission (bps)", 0.0, 10.0, float(params["commission_bps"]), step=0.5
    )
    half_spread_bps = c9.slider(
        "Half spread (bps)", 0.0, 10.0, float(params["half_spread_bps"]), step=0.5
    )
    reward_kind = c10.selectbox(
        "Reward",
        ["log_return", "differential_sharpe", "terminal_pnl"],
        index=["log_return", "differential_sharpe", "terminal_pnl"].index(params["reward_kind"]),
    )

    llm_flat = {k: params.get(k, v) for k, v in engine.default_llm_params().items()}
    if any(a in engine.LLM_AGENTS for a in agent_names) or st.checkbox(
        "Show LLM settings",
        value=any(a in engine.LLM_AGENTS for a in agent_names),
        key="cfg_show_llm",
    ):
        st.subheader("LLM (Vanilla)")
        nested = agent_config.render_llm_settings(llm_flat, key="cfg_llm")
        llm_flat = agent_config.flatten_llm_params(nested)

    st.session_state["params"] = {
        "mode": mode,
        "agent_names": tuple(agent_names),
        "symbols": tuple(symbols),
        "n_assets": int(n_assets),
        "start_date": start_d.isoformat(),
        "end_date": end_d.isoformat(),
        "limit_n_steps": bool(limit_tail),
        "n_steps": int(n_steps) if n_steps is not None else None,
        "episode_length": int(episode_length),
        "n_episodes": int(n_episodes),
        "seed": int(seed),
        "initial_cash": float(initial_cash),
        "commission_bps": float(commission_bps),
        "half_spread_bps": float(half_spread_bps),
        "reward_kind": reward_kind,
        "use_mock": False,
        **llm_flat,
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
    elif mode == "multi_asset" and len(symbols) < 2:
        st.warning("Pick at least two stocks for multi-asset mode.")
    else:
        st.success(
            f"Configured {len(agent_names)} agent(s) on "
            f"{', '.join(symbols)} · {start_d.isoformat()} → {end_d.isoformat()}."
        )
    overview = st.session_state.get("_pages", {}).get("overview")
    if overview is not None:
        st.page_link(overview, label="Go to Overview", icon="📊")
