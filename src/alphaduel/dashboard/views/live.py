"""Live-run page: pick + configure one agent and stream its decisions step by step."""

from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from alphaduel.dashboard import engine, state
from alphaduel.dashboard.components import agent_config, theme
from alphaduel.evaluation.metrics import compute_metrics

_SPEEDS = {"Slow": 0.20, "Normal": 0.08, "Fast": 0.02, "Instant": 0.0}


def render() -> None:
    default_params = state.ensure_params()
    presets = state.load_presets()

    st.title("🎬 Live run")

    options = ["Current configuration", *presets.keys()]
    choice = st.selectbox("Configuration preset", options, key="live_preset")
    run_params = default_params if choice == "Current configuration" else presets[choice]
    mode = run_params["mode"]

    st.caption(
        f"{choice} · {'multi-asset' if mode == 'multi_asset' else 'single-asset'} · "
        f"{run_params['episode_length']}d episode · reward = {run_params['reward_kind']} · "
        f"{'cash' if mode == 'single_asset' else str(run_params['n_assets']) + ' assets'}"
    )

    top = st.columns([2, 1, 1])
    agents = engine.agents_for(mode)
    agent_name = top[0].selectbox("Agent", agents, format_func=theme.label)
    speed = top[1].select_slider("Speed", options=list(_SPEEDS), value="Normal")
    ep_seed = int(top[2].number_input("Episode seed", 0, 9999, int(run_params["seed"])))

    with st.expander("Agent parameters", expanded=True):
        agent_params = agent_config.render(agent_name, key="live")

    if st.button("🚀 Launch run", type="primary", use_container_width=True):
        _launch(run_params, mode, agent_name, agent_params, ep_seed, _SPEEDS[speed])

    run = st.session_state.get("last_live_run")
    if run is not None:
        st.divider()
        _render_evaluation(run)


def _launch(run_params, mode, agent_name, agent_params, ep_seed, delay) -> None:
    env, agent, n_assets, symbols = engine.make_live_session(run_params, agent_name, agent_params)

    st.subheader("Live decisions")
    kpi_ph = st.empty()
    chart_ph = st.empty()
    cols = st.columns([3, 2])
    alloc_ph = cols[0].empty()
    decision_ph = cols[1].empty()

    step_state: dict = {}
    for step_state in engine.stream_episode(env, agent, n_assets, seed=ep_seed):
        _render_step(step_state, symbols, run_params, mode, kpi_ph, chart_ph, alloc_ph, decision_ph)
        if delay:
            time.sleep(delay)

    st.session_state["last_live_run"] = {
        "agent": agent_name,
        "mode": mode,
        "symbols": symbols,
        "params": dict(run_params),
        "equity": np.asarray(step_state["equity"], dtype=float),
        "rewards": np.asarray(step_state["rewards"], dtype=float),
        "fills": np.asarray(step_state["fills"], dtype=int),
        "costs": np.asarray(step_state["costs"], dtype=float),
        "exposure": np.asarray(step_state["exposure"], dtype=float),
        "weights": np.vstack(step_state["weights_hist"]),
        "prices": np.vstack(step_state["prices"]),
    }
    st.success("Episode complete — see the evaluation below.")


def _render_step(s, symbols, params, mode, kpi_ph, chart_ph, alloc_ph, decision_ph) -> None:
    equity = np.asarray(s["equity"], dtype=float)
    step = s["step"]
    total = params["episode_length"]
    total_ret = equity[-1] / equity[0] - 1.0
    invested = s["exposure"][-1] if s["exposure"] else 0.0
    run_sharpe = compute_metrics(equity, 0, 0.0)["sharpe"] if equity.size > 2 else 0.0

    with kpi_ph.container():
        a, b, c, d = st.columns(4)
        a.metric("Step", f"{step} / {total}")
        b.metric("Equity", f"${equity[-1]:,.0f}", f"{total_ret:+.2%}")
        c.metric("Invested", f"{invested:.0%}")
        d.metric("Sharpe (run)", f"{run_sharpe:.2f}")

    _equity_vs_market(equity, np.vstack(s["prices"]), step, chart_ph)

    weights = np.asarray(s["current_weights"], dtype=float).ravel()
    _allocation(weights, symbols, mode, step, alloc_ph)
    _decision(weights, symbols, mode, s["thought"], decision_ph)


def _equity_vs_market(equity, prices, step, chart_ph) -> None:
    market = prices.mean(axis=1)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(y=equity / equity[0], name="Agent", line={"color": "#168740", "width": 2.2})
    )
    fig.add_trace(
        go.Scatter(
            y=market / market[0], name="Market (equal weight)",
            line={"color": "#9d755d", "width": 1.5, "dash": "dot"},
        )
    )
    fig.update_layout(
        height=340, margin={"l": 10, "r": 10, "t": 10, "b": 10},
        yaxis_title="Growth of $1", xaxis_title="Step",
        legend={"orientation": "h", "y": -0.2}, hovermode="x unified",
    )
    chart_ph.plotly_chart(fig, use_container_width=True, key=f"live_eq_{step}")


def _allocation(weights, symbols, mode, step, alloc_ph) -> None:
    with alloc_ph.container():
        st.markdown("**Current allocation**")
        if mode == "multi_asset":
            colors = theme.color_map(symbols)
            fig = go.Figure(
                go.Bar(x=symbols, y=weights, marker_color=[colors[s] for s in symbols])
            )
            fig.update_layout(
                height=240, margin={"l": 10, "r": 10, "t": 10, "b": 10},
                yaxis={"range": [0, 1], "tickformat": ".0%"},
            )
            st.plotly_chart(fig, use_container_width=True, key=f"live_alloc_{step}")
        else:
            st.progress(min(float(weights[0]), 1.0), text=f"Invested {float(weights[0]):.0%}")


def _decision(weights, symbols, mode, thought, decision_ph) -> None:
    with decision_ph.container():
        st.markdown("**Latest decision**")
        if mode == "multi_asset":
            order = np.argsort(weights)[::-1][:5]
            lines = [f"- **{symbols[i]}** → {weights[i]:.1%}" for i in order if weights[i] > 1e-4]
            st.markdown("\n".join(lines) if lines else "_All cash._")
        else:
            st.metric("Target weight", f"{float(weights[0]):.0%}")
        if thought:
            st.caption(thought)


def _render_evaluation(run: dict) -> None:
    equity = run["equity"]
    fills = run["fills"]
    n_tx = int((fills != 0).sum())
    metrics = compute_metrics(equity, n_tx, float(run["rewards"].sum()))
    turnover = float(np.abs(np.diff(run["exposure"], prepend=0.0)).sum())

    st.subheader(f"Evaluation — {theme.label(run['agent'])}")
    r1 = st.columns(5)
    r1[0].metric("Total return", theme.pct(metrics["total_return"]))
    r1[1].metric("Ann. return", theme.pct(metrics["annualized_return"]))
    r1[2].metric("Ann. volatility", theme.pct(metrics["volatility"]))
    r1[3].metric("Sharpe", theme.num(metrics["sharpe"]))
    r1[4].metric("Sortino", theme.num(metrics["sortino"]))

    r2 = st.columns(5)
    r2[0].metric("Max drawdown", theme.pct(metrics["max_drawdown"]))
    r2[1].metric("Calmar", theme.num(metrics["calmar"]))
    r2[2].metric("Deflated Sharpe", theme.num(metrics["deflated_sharpe"]))
    r2[3].metric("Trades", f"{n_tx}")
    r2[4].metric("Total cost", f"${float(run['costs'].sum()):,.0f}")

    left, right = st.columns([3, 2])
    with left:
        _drawdown_chart(equity)
    with right:
        st.markdown("**Final allocation**")
        _final_allocation(run)
        st.caption(f"Turnover: {turnover:.2f}x · Final equity: ${equity[-1]:,.0f}")


def _drawdown_chart(equity) -> None:
    peak = np.maximum.accumulate(equity)
    underwater = equity / peak - 1.0
    fig = go.Figure(
        go.Scatter(y=underwater, fill="tozeroy", line={"color": "#e45756", "width": 1.4})
    )
    fig.update_layout(
        height=260, margin={"l": 10, "r": 10, "t": 10, "b": 10},
        yaxis={"tickformat": ".0%"}, yaxis_title="Drawdown", xaxis_title="Step",
    )
    st.plotly_chart(fig, use_container_width=True, key="eval_dd")


def _final_allocation(run: dict) -> None:
    weights = run["weights"][-1]
    if run["mode"] == "multi_asset":
        colors = theme.color_map(run["symbols"])
        fig = go.Figure(
            go.Bar(
                x=run["symbols"], y=weights,
                marker_color=[colors[s] for s in run["symbols"]],
            )
        )
        fig.update_layout(
            height=240, margin={"l": 10, "r": 10, "t": 10, "b": 10},
            yaxis={"range": [0, 1], "tickformat": ".0%"},
        )
        st.plotly_chart(fig, use_container_width=True, key="eval_alloc")
    else:
        st.progress(min(float(weights[0]), 1.0), text=f"Invested {float(weights[0]):.0%}")

