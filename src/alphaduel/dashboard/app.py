"""Streamlit dashboard for alphaduel.

Run with:

    uv run alphaduel-dashboard
    # or: uv run streamlit run src/alphaduel/dashboard/app.py
"""

from __future__ import annotations

import streamlit as st

from alphaduel.baselines.policies import STRATEGY_SPECS
from alphaduel.dashboard import service
from alphaduel.dashboard.service import RunConfig

st.set_page_config(page_title="alphaduel — strategy lab", layout="wide")

_INT_PARAMS = {"lookback", "top_n", "bottom_n", "window", "seed"}


def _sidebar() -> tuple[str, dict, dict, RunConfig]:
    st.sidebar.title("alphaduel")
    st.sidebar.caption("Reasoning LLM agents vs quant models — strategy lab")

    source = st.sidebar.radio("Data source", ["synthetic", "live"], horizontal=True)
    market_kwargs: dict = {"source": source}
    if source == "synthetic":
        market_kwargs["n_assets"] = st.sidebar.slider("Assets", 3, 15, 10)
        market_kwargs["n_days"] = st.sidebar.slider("Trading days", 800, 3500, 2600, step=100)
        market_kwargs["seed"] = st.sidebar.number_input("Seed", value=42, step=1)
    else:
        tickers = st.sidebar.text_input(
            "Tickers (comma-separated)", "AAPL,MSFT,JNJ,JPM,XOM,PG,KO,WMT,HD,CVX"
        )
        market_kwargs["tickers"] = [t.strip().upper() for t in tickers.split(",") if t.strip()]

    st.sidebar.divider()
    st.sidebar.subheader("Environment")
    cfg = RunConfig(
        budget=float(st.sidebar.number_input("Budget", value=100_000, step=10_000)),
        cost_bps=float(st.sidebar.slider("Cost (bps)", 0.0, 30.0, 5.0, step=0.5)),
        max_weight=float(st.sidebar.slider("Max weight / asset", 0.05, 1.0, 0.30, step=0.05)),
        rebalance_every=int(st.sidebar.slider("Rebalance every (days)", 1, 21, 1)),
        split=st.sidebar.selectbox("Split", ["test", "val", "train"], index=0),
    )

    st.sidebar.divider()
    st.sidebar.subheader("Strategies")
    selected: dict[str, dict] = {}
    for name, defaults in STRATEGY_SPECS.items():
        on = st.sidebar.checkbox(name, value=name in {"equal_weight", "momentum", "inverse_vol"})
        if not on:
            continue
        params: dict = {}
        for pname, pdefault in defaults.items():
            if pname in _INT_PARAMS:
                params[pname] = st.sidebar.number_input(
                    f"{name}.{pname}", value=int(pdefault), step=1, key=f"{name}.{pname}"
                )
        selected[name] = params

    return source, market_kwargs, selected, cfg


def main() -> None:
    source, market_kwargs, selected, cfg = _sidebar()

    st.title("Strategy lab")
    if not selected:
        st.info("Select at least one strategy in the sidebar.")
        return

    try:
        market = service.build_market(**market_kwargs)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load market data: {exc}")
        return

    results = service.run_strategies(market, selected, cfg)
    ppy = 252.0 / cfg.rebalance_every

    lo, hi = market.split_indices(cfg.train_end, cfg.val_end)[cfg.split]
    st.caption(
        f"{source} data · {market.n_assets} assets · {cfg.split} split · "
        f"{market.dates[lo].date()} → {market.dates[hi - 1].date()} ({hi - lo} steps)"
    )

    st.subheader("Performance metrics")
    table = service.metrics_table(results)
    st.dataframe(
        table.style.format(
            {
                "Total Return": "{:.1%}", "CAGR": "{:.1%}", "Volatility": "{:.1%}",
                "Sharpe": "{:.2f}", "Sortino": "{:.2f}", "Max Drawdown": "{:.1%}",
                "Hit Rate": "{:.1%}", "Avg Turnover": "{:.3f}",
            }
        ),
        width="stretch",
    )

    left, right = st.columns(2)
    left.plotly_chart(service.equity_figure(market, results), width="stretch", theme=None)
    right.plotly_chart(service.drawdown_figure(market, results), width="stretch", theme=None)

    left2, right2 = st.columns(2)
    left2.plotly_chart(
        service.rolling_sharpe_figure(market, results, periods_per_year=ppy),
        width="stretch",
        theme=None,
    )
    right2.plotly_chart(service.returns_hist_figure(results), width="stretch", theme=None)

    st.subheader("Policy actions over time")
    ctrl1, ctrl2 = st.columns(2)
    choice = ctrl1.selectbox("Strategy", list(results.keys()))
    view = ctrl2.radio(
        "View",
        ["Weights heatmap", "Trades (buy/sell)", "Stacked area"],
        horizontal=True,
    )
    res = results[choice]
    if view == "Weights heatmap":
        fig = service.allocation_heatmap_figure(market, res)
    elif view == "Trades (buy/sell)":
        fig = service.trades_heatmap_figure(market, res)
    else:
        fig = service.allocation_figure(market, res)
    st.plotly_chart(fig, width="stretch", theme=None)


main()
