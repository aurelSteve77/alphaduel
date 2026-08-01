"""Evaluate page: roster many agents (incl. multiple LLM Vanilla configs) and compare them."""

from __future__ import annotations

import uuid

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from alphaduel.dashboard import engine, state
from alphaduel.dashboard.components import agent_config, drawdown, equity_curve, metrics_table, theme


def _ensure_roster() -> list[dict]:
    if "eval_roster" not in st.session_state:
        # Start with baselines from the shared Configure params (no LLMs by default).
        params = state.ensure_params()
        mode = params.get("mode", "multi_asset")
        names = [
            a
            for a in params.get("agent_names", ())
            if a in engine.agents_for(mode) and a not in engine.LLM_AGENTS
        ] or engine.baseline_agents(mode)
        st.session_state["eval_roster"] = [
            {
                "id": uuid.uuid4().hex[:8],
                "label": theme.label(name),
                "agent": name,
                "params": {},
            }
            for name in names
        ]
    return st.session_state["eval_roster"]


def _unique_label(base: str, roster: list[dict], skip_id: str | None = None) -> str:
    existing = {
        c["label"] for c in roster if skip_id is None or c["id"] != skip_id
    }
    if base not in existing:
        return base
    n = 2
    while f"{base} #{n}" in existing:
        n += 1
    return f"{base} #{n}"


def _summary_row(contestant: dict) -> str:
    agent = contestant["agent"]
    params = contestant.get("params") or {}
    if agent in engine.LLM_AGENTS:
        llm = params.get("llm") or {}
        bits = [llm.get("provider", "?"), llm.get("model", "?")]
        if params.get("use_memory"):
            bits.append("memory")
        return " · ".join(str(b) for b in bits)
    if not params:
        return "defaults"
    return ", ".join(f"{k}={v}" for k, v in list(params.items())[:3])


def _render_roster(roster: list[dict]) -> None:
    st.subheader("Contestants")
    if not roster:
        st.info("Add at least one agent below.")
        return

    for c in list(roster):
        cols = st.columns([3, 2, 3, 1])
        cols[0].markdown(f"**{c['label']}**")
        cols[1].caption(theme.label(c["agent"]))
        cols[2].caption(_summary_row(c))
        if cols[3].button("Remove", key=f"eval_rm_{c['id']}", use_container_width=True):
            st.session_state["eval_roster"] = [x for x in roster if x["id"] != c["id"]]
            st.rerun()


def _render_add_form(roster: list[dict], mode: str) -> None:
    st.subheader("Add contestant")
    available = engine.agents_for(mode)
    agent = st.selectbox(
        "Agent type",
        available,
        format_func=theme.label,
        key="eval_add_agent",
    )

    with st.expander("Parameters", expanded=agent in engine.LLM_AGENTS):
        params = agent_config.render(agent, key="eval_add", defaults=engine.default_llm_params())

    suggested = engine.default_contestant_label(agent, params)
    if agent not in engine.LLM_AGENTS:
        suggested = theme.label(agent)
    label = st.text_input("Display name", value=suggested, key="eval_add_label")

    c1, c2, c3 = st.columns(3)
    if c1.button("Add to roster", type="primary", use_container_width=True):
        final_label = _unique_label(label.strip() or suggested, roster)
        roster.append(
            {
                "id": uuid.uuid4().hex[:8],
                "label": final_label,
                "agent": agent,
                "params": params,
            }
        )
        st.session_state["eval_roster"] = roster
        st.toast(f"Added {final_label}")
        st.rerun()

    if c2.button("Add all baselines", use_container_width=True):
        for name in engine.baseline_agents(mode):
            base = theme.label(name)
            if any(x["agent"] == name and not x.get("params") for x in roster):
                continue
            roster.append(
                {
                    "id": uuid.uuid4().hex[:8],
                    "label": _unique_label(base, roster),
                    "agent": name,
                    "params": {},
                }
            )
        st.session_state["eval_roster"] = roster
        st.rerun()

    if c3.button("Clear roster", use_container_width=True):
        st.session_state["eval_roster"] = []
        st.rerun()


def _episode_metric_chart(result: engine.BenchmarkResult, metric: str = "sharpe") -> None:
    st.subheader(f"Per-episode {metric.replace('_', ' ')}")
    st.caption("Distribution across episodes (mean is what the leaderboard uses).")
    fig = go.Figure()
    colors = theme.color_map(list(result.agents))
    for name, rec in result.agents.items():
        if not rec.episode_metrics:
            continue
        ys = [m.get(metric, 0.0) for m in rec.episode_metrics]
        fig.add_trace(
            go.Box(
                y=ys,
                name=theme.label(name),
                marker_color=colors[name],
                boxmean=True,
            )
        )
    fig.update_layout(
        height=360,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        showlegend=False,
        yaxis_title=metric.replace("_", " ").title(),
    )
    st.plotly_chart(fig, use_container_width=True)


def _winner_banner(result: engine.BenchmarkResult) -> None:
    if not result.agents:
        return
    ranked = sorted(
        result.agents.items(),
        key=lambda kv: kv[1].metrics.get("sharpe", float("-inf")),
        reverse=True,
    )
    best_name, best = ranked[0]
    st.success(
        f"Top by mean Sharpe: **{theme.label(best_name)}** "
        f"(Sharpe {best.metrics.get('sharpe', 0):.2f}, "
        f"return {best.metrics.get('total_return', 0):.2%}, "
        f"max DD {best.metrics.get('max_drawdown', 0):.2%})"
    )


def render() -> None:
    params = state.ensure_params()
    roster = _ensure_roster()

    st.title("Evaluate")
    st.caption(
        "Build a roster of agents — including multiple LLM Vanilla instances with different "
        "providers/models — then run many episodes and compare who wins."
    )

    mode = params["mode"]
    st.info(
        f"Market setup comes from **Configure**: "
        f"{'multi-asset' if mode == 'multi_asset' else 'single-asset'} "
        f"({', '.join(params.get('symbols') or [])}), "
        f"{params.get('start_date', '?')} → {params.get('end_date', '?')}, "
        f"cash ${params['initial_cash']:,.0f}."
    )

    left, right = st.columns([1, 1])
    with left:
        _render_roster(roster)
    with right:
        _render_add_form(roster, mode)

    st.divider()
    st.subheader("Run")
    r1, r2, r3, r4 = st.columns(4)
    n_episodes = r1.slider(
        "Episodes",
        1,
        50,
        int(params.get("n_episodes", 20)),
        key="eval_n_episodes",
        help="Mean metrics are averaged across these episodes.",
    )
    episode_length = r2.slider(
        "Episode length (days)",
        30,
        250,
        int(params.get("episode_length", 120)),
        step=10,
        key="eval_episode_length",
    )
    seed = r3.number_input("Seed", 0, 9999, int(params.get("seed", 7)), key="eval_seed")
    max_workers = r4.slider(
        "Parallel workers",
        1,
        engine.EVAL_MAX_WORKERS,
        engine.EVAL_MAX_WORKERS,
        key="eval_max_workers",
        help=f"Contestants (incl. LLM calls) run concurrently, capped at {engine.EVAL_MAX_WORKERS}.",
    )

    save_llm_dataset = st.checkbox(
        "Save LLM trajectories for SFT",
        value=True,
        key="eval_save_llm_dataset",
        help=(
            "For each LLM contestant, write per-episode JSON (messages, state, actions, "
            "rewards) plus sft.jsonl under datasets/llm_sft/<run_id>/."
        ),
    )

    n_llm = sum(1 for c in roster if c["agent"] in engine.LLM_AGENTS)
    if n_llm:
        st.warning(
            f"{n_llm} LLM contestant(s) — up to {max_workers} run in parallel. "
            "Start with few episodes / short length while iterating."
        )
        if save_llm_dataset:
            st.caption(
                "SFT datasets will be written to `datasets/llm_sft/<run_id>/` "
                "(episode_XXX.json + sft.jsonl per LLM agent)."
            )

    run = st.button(
        "Run evaluation",
        type="primary",
        disabled=not roster,
        use_container_width=True,
    )

    if run:
        progress_bar = st.progress(0.0, text="Starting…")
        status_slot = st.empty()

        def _format_agent_rows(agents: list[dict] | None) -> pd.DataFrame:
            rows = []
            for a in agents or []:
                status = a.get("status", "queued")
                if status == "running":
                    where = (
                        f"episode {a.get('episode', 0)}/{a.get('n_episodes', 0)} · "
                        f"step {a.get('step', 0)}/{a.get('episode_length', 0)}"
                    )
                elif status == "training":
                    where = a.get("detail") or "training"
                elif status == "done":
                    where = "finished"
                elif status == "error":
                    where = a.get("detail") or "error"
                else:
                    where = "waiting for a worker"
                rows.append(
                    {
                        "agent": a.get("label", ""),
                        "status": status,
                        "where": where,
                    }
                )
            return pd.DataFrame(rows)

        def _on_progress(frac: float, msg: str, agents=None) -> None:
            progress_bar.progress(min(max(frac, 0.0), 1.0), text=msg)
            df = _format_agent_rows(agents)
            if not df.empty:
                status_slot.dataframe(df, use_container_width=True, hide_index=True)

        try:
            result = engine.run_evaluation(
                mode=mode,
                contestants=[
                    {"label": c["label"], "agent": c["agent"], "params": c.get("params") or {}}
                    for c in roster
                ],
                n_assets=int(params["n_assets"]),
                n_steps=int(params["n_steps"]) if params.get("n_steps") is not None else None,
                episode_length=int(episode_length),
                n_episodes=int(n_episodes),
                seed=int(seed),
                initial_cash=float(params["initial_cash"]),
                commission_bps=float(params["commission_bps"]),
                half_spread_bps=float(params["half_spread_bps"]),
                reward_kind=str(params["reward_kind"]),
                use_mock=bool(params.get("use_mock", False)),
                progress=_on_progress,
                max_workers=int(max_workers),
                save_llm_dataset=bool(save_llm_dataset),
                symbols=tuple(params.get("symbols") or ()),
                start_date=params.get("start_date"),
                end_date=params.get("end_date"),
            )
        except Exception as exc:  # noqa: BLE001 — surface LLM / data errors in the UI
            progress_bar.empty()
            st.error(f"Evaluation failed: {exc}")
            return

        progress_bar.empty()
        status_slot.empty()
        st.session_state["eval_result"] = result
        st.session_state["eval_meta"] = {
            "n_episodes": int(n_episodes),
            "episode_length": int(episode_length),
            "seed": int(seed),
            "dataset_dir": result.dataset_dir,
            "roster": [
                {"label": c["label"], "agent": c["agent"], "summary": _summary_row(c)}
                for c in roster
            ],
        }
        if result.dataset_dir:
            st.success(f"LLM SFT dataset saved to `{result.dataset_dir}`")
        st.toast("Evaluation complete")
        st.rerun()

    result = st.session_state.get("eval_result")
    meta = st.session_state.get("eval_meta")
    if result is None:
        st.caption("Results appear here after you run an evaluation.")
        return

    st.divider()
    st.subheader("Results")
    if meta:
        st.caption(
            f"{meta['n_episodes']} episodes × {meta['episode_length']}d · seed {meta['seed']}"
        )
        if meta.get("dataset_dir"):
            st.info(f"LLM SFT dataset: `{meta['dataset_dir']}`")
        with st.expander("Roster used"):
            st.dataframe(
                pd.DataFrame(meta["roster"]),
                use_container_width=True,
                hide_index=True,
            )

    _winner_banner(result)
    c_left, c_right = st.columns([3, 2])
    with c_left:
        equity_curve.render(result)
    with c_right:
        metrics_table.render(result)
    drawdown.render(result)
    _episode_metric_chart(result, metric="sharpe")
