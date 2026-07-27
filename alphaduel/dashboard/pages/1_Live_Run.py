"""Live single-strategy episode with streaming charts."""

from __future__ import annotations

import streamlit as st

from alphaduel.configuration import Configuration
from alphaduel.dashboard import charts, components, service
from alphaduel.evaluation.config import StrategyRunConfig
from alphaduel.utils.singleton import Singleton

if "cfg_bootstrapped" not in st.session_state:
    Singleton._instances.pop(Configuration, None)
    st.session_state.cfg_bootstrapped = True

cfg = Configuration()
presets = service.list_strategy_presets()
kinds = service.list_strategy_kinds()
universe = service.default_tickers()
default_tickers = universe[:5] or ["AAPL", "MSFT", "NVDA"]

st.title("Live run")
st.caption("Step through an episode and watch portfolio, trades and rationale update.")

with st.sidebar:
    st.header("Run setup")
    source = st.radio("Config source", ["Preset YAML", "Custom"], horizontal=True)
    preset = None
    if source == "Preset YAML":
        preset = st.selectbox("Preset", presets) if presets else None

    kind = st.selectbox(
        "Strategy kind",
        kinds,
        index=kinds.index("buy_and_hold") if "buy_and_hold" in kinds else 0,
        disabled=source == "Preset YAML",
    )

    st.subheader("Environment")
    tickers = st.multiselect(
        "Tickers",
        options=sorted(set(universe) | set(default_tickers)),
        default=default_tickers,
    )
    start = st.text_input("Start", str(cfg.get("env.start", "2024-01-01")))
    end = st.text_input("End", str(cfg.get("env.end", "2025-12-31")))
    n_days = st.number_input("n_days", min_value=2, max_value=2520, value=int(cfg.get("env.n_days", 60)))
    initial_cash = st.number_input(
        "Initial cash",
        min_value=100.0,
        value=float(cfg.get("env.initial_cash", 10_000.0)),
        step=1000.0,
    )
    fee = st.number_input(
        "Transaction fee",
        min_value=0.0,
        max_value=0.05,
        value=float(cfg.get("env.transaction_fee", 0.001)),
        format="%.4f",
    )
    max_shares = st.number_input(
        "Max shares / day", min_value=1, value=int(cfg.get("env.max_shares", 10))
    )
    seed = st.number_input("Seed", min_value=0, value=int(cfg.get("seed", 77)))
    include_news = st.toggle("Include news", value=bool(cfg.get("env.include_news", False)))

    policy: dict = {}
    show_llm = kind == "llm_policy" or (
        source == "Preset YAML" and preset == "llm_policy"
    )
    if show_llm:
        st.subheader("LLM")
        policy["model"] = st.text_input("Model", str(cfg.get("llm.model", "qwen3.5:2b")))
        policy["temperature"] = st.slider(
            "Temperature", 0.0, 1.5, float(cfg.get("llm.temperature", 0.0)), 0.05
        )
        policy["reasoning"] = st.toggle(
            "Reasoning", value=bool(cfg.get("llm.reasoning", False))
        )
        max_hist = cfg.get("llm.max_history", 10)
        policy["max_history"] = st.number_input(
            "Max history turns",
            min_value=0,
            value=int(max_hist if max_hist is not None else 10),
        )

    run_live = st.button("Run live", type="primary", use_container_width=True)
    run_fast = st.button("Run (no live updates)", use_container_width=True)


def make_run() -> StrategyRunConfig:
    selected_kind = kind
    selected_policy = dict(policy)
    name = f"live_{kind}"
    if source == "Preset YAML" and preset:
        base = StrategyRunConfig.from_yaml(preset)
        selected_kind = base.kind
        selected_policy = {**base.policy, **policy}
        name = base.name
    return service.build_run_config(
        name=name,
        kind=selected_kind,
        tickers=tickers,
        start=start,
        end=end,
        initial_cash=float(initial_cash),
        transaction_fee=float(fee),
        n_days=int(n_days),
        max_shares=int(max_shares),
        include_news=include_news,
        seed=int(seed),
        policy=selected_policy,
    )


if not tickers:
    st.warning("Select at least one ticker.")
    st.stop()

status = st.empty()
metrics_box = st.empty()
chart_box = st.empty()
step_box = st.empty()
final_box = st.container()

if run_live or run_fast:
    run = make_run()
    try:
        with st.spinner("Loading prices…"):
            prices = service.load_prices(tickers, start, end)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load prices: {exc}")
        st.stop()

    if run_fast:
        status.info(f"Running `{run.name}` ({run.kind})…")
        result = service.run_config(run, prices=prices)
        status.success("Done.")
        with final_box:
            components.render_result_panels(result, key_prefix="fast")
    else:
        status.info(f"Live: `{run.name}` ({run.kind})")
        result = None
        event = None
        progress = st.progress(0.0, text="Starting…")
        for event, result in service.iter_live(run, prices=prices):
            frac = min(1.0, event.step / max(1, int(n_days)))
            progress.progress(frac, text=f"Step {event.step} · {event.decision_date}")
            # Text-only live updates — plotly keys cannot be reused within one run.
            metrics_box.markdown(
                f"**Value** `{event.portfolio_value:,.2f}` · "
                f"**Cash** `{event.cash:,.2f}` · "
                f"**PnL** `{event.pnl:+,.2f}` · "
                f"**Step** `{event.step}`"
            )
            step_box.empty()
            with step_box.container():
                components.render_live_step(event, list(result.tickers))

        progress.progress(1.0, text="Complete")
        status.success("Episode finished.")
        if result is not None:
            # Single chart render after the loop (unique keys via key_prefix).
            chart_box.plotly_chart(
                charts.episode_dashboard(result),
                width="stretch",
                key="live_summary_chart",
            )
            with final_box:
                st.subheader("Full report")
                components.render_result_panels(result, key_prefix="live_final")
else:
    st.info("Configure the run in the sidebar, then click **Run live**.")
