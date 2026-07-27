"""Compare multiple strategies on a shared evaluation window."""

from __future__ import annotations

import streamlit as st

from alphaduel.configuration import Configuration
from alphaduel.dashboard import charts, service
from alphaduel.evaluation.config import StrategyRunConfig
from alphaduel.utils.singleton import Singleton

if "cfg_bootstrapped" not in st.session_state:
    Singleton._instances.pop(Configuration, None)
    st.session_state.cfg_bootstrapped = True

cfg = Configuration()
presets = service.list_strategy_presets()
universe = service.default_tickers()
default_tickers = universe[:5] or ["AAPL", "MSFT", "NVDA"]

st.title("Benchmark")
st.caption("Run several strategies on the same market window and compare results.")

with st.sidebar:
    st.header("Shared window")
    selected = st.multiselect(
        "Strategies",
        options=presets or service.list_strategy_kinds(),
        default=[p for p in ("buy_and_hold", "random", "hold") if p in (presets or service.list_strategy_kinds())][:2],
    )
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
    include_llm = st.toggle(
        "Include llm_policy (needs Ollama)",
        value=False,
        help="LLM runs are slow; leave off for quick baseline benches.",
    )
    run_btn = st.button("Run benchmark", type="primary", use_container_width=True)

if not selected:
    st.warning("Pick at least one strategy.")
    st.stop()
if not tickers:
    st.warning("Select at least one ticker.")
    st.stop()

strategies = list(selected)
if include_llm and "llm_policy" not in strategies and "llm_policy" in (presets or []):
    strategies.append("llm_policy")


def build_runs() -> list[StrategyRunConfig]:
    runs: list[StrategyRunConfig] = []
    for name in strategies:
        if name in presets:
            base = StrategyRunConfig.from_yaml(name)
            kind = base.kind
            policy = dict(base.policy)
            label = base.name
        else:
            kind = name
            policy = {}
            label = name
        runs.append(
            service.build_run_config(
                name=label,
                kind=kind,
                tickers=tickers,
                start=start,
                end=end,
                initial_cash=float(initial_cash),
                transaction_fee=float(fee),
                n_days=int(n_days),
                max_shares=int(max_shares),
                seed=int(seed),
                policy=policy,
            )
        )
    return runs


if run_btn:
    try:
        with st.spinner("Loading shared price panel…"):
            prices = service.load_prices(tickers, start, end)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load prices: {exc}")
        st.stop()

    runs = build_runs()
    results = []
    progress = st.progress(0.0, text="Benchmarking…")
    for i, run in enumerate(runs):
        progress.progress(i / max(1, len(runs)), text=f"Running {run.name} ({run.kind})…")
        try:
            results.append(service.run_config(run, prices=prices))
        except Exception as exc:  # noqa: BLE001
            st.error(f"`{run.name}` failed: {exc}")
    progress.progress(1.0, text="Done")

    if not results:
        st.warning("No successful runs.")
        st.stop()

    table = service.metrics_comparison_table(results)
    st.subheader("Metrics")
    display = table.copy()
    if not display.empty:
        display["total_return"] = display["total_return"].map(lambda v: f"{v:.2%}")
        display["max_drawdown"] = display["max_drawdown"].map(lambda v: f"{v:.2%}")
        display["hit_rate"] = display["hit_rate"].map(lambda v: f"{v:.1%}")
        display["sharpe"] = display["sharpe"].map(
            lambda v: "n/a" if v is None or (isinstance(v, float) and v != v) else f"{v:.2f}"
        )
        display["final_value"] = display["final_value"].map(lambda v: f"{v:,.2f}")
        display["final_pnl"] = display["final_pnl"].map(lambda v: f"{v:+,.2f}")
        display["total_fees"] = display["total_fees"].map(lambda v: f"{v:,.2f}")
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.subheader("Equity")
    st.plotly_chart(
        charts.comparison_equity(results),
        width="stretch",
        key="bench_equity",
    )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            charts.metrics_bar(table, metric="total_return"),
            width="stretch",
            key="bench_return_bar",
        )
    with right:
        st.plotly_chart(
            charts.metrics_bar(table, metric="max_drawdown"),
            width="stretch",
            key="bench_dd_bar",
        )

    st.subheader("Per-strategy detail")
    for result in results:
        with st.expander(f"{result.name} ({result.kind})", expanded=False):
            from alphaduel.dashboard import components

            components.render_result_panels(result, key_prefix=f"bench_{result.name}")
else:
    st.info("Choose strategies and a shared window, then click **Run benchmark**.")
