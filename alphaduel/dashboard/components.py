"""Reusable Streamlit widgets for evaluation views."""

from __future__ import annotations

import streamlit as st

from alphaduel.dashboard import charts
from alphaduel.evaluation.metrics import EvalResult
from alphaduel.evaluation.runner import StepEvent


def render_metrics(result: EvalResult) -> None:
    m = result.metrics
    if m is None:
        st.info("Metrics not ready yet.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio", f"{m.final_value:,.2f}", f"{m.final_pnl:+,.2f}")
    c2.metric("Return", f"{m.total_return:.2%}")
    c3.metric("Max DD", f"{m.max_drawdown:.2%}")
    sharpe = f"{m.sharpe:.2f}" if m.sharpe is not None else "n/a"
    c4.metric("Sharpe", sharpe)
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Hit rate", f"{m.hit_rate:.1%}")
    c6.metric("Trades", f"{m.n_transactions}")
    c7.metric("Fees", f"{m.total_fees:,.2f}")
    c8.metric("Steps", f"{m.n_steps}")


def render_live_step(event: StepEvent, tickers: list[str]) -> None:
    """Compact text snapshot for in-loop live updates (no keyed widgets)."""
    action_bits = [
        f"{t}: {a:+d}" for t, a in zip(tickers, event.action, strict=False) if a != 0
    ]
    action_txt = ", ".join(action_bits) if action_bits else "hold"
    lines = [
        f"**Step {event.step}** · decided `{event.decision_date}` · marked `{event.mark_date}`",
        f"Value `{event.portfolio_value:,.2f}` · Cash `{event.cash:,.2f}` · "
        f"PnL `{event.pnl:+,.2f}` · Fees `{event.fees:,.2f}`",
        f"**Action:** {action_txt}",
    ]
    if event.trades:
        legs = ", ".join(
            f"{t.side} {t.shares} {t.ticker} @ {t.price:.2f}" for t in event.trades
        )
        lines.append(f"**Executed:** {legs}")
    st.markdown("\n\n".join(lines))
    if event.rationale:
        st.markdown("**Rationale**")
        st.markdown(event.rationale)


def render_result_panels(result: EvalResult, *, key_prefix: str = "report") -> None:
    """Render metrics + charts + tables. ``key_prefix`` must be unique per page section."""
    prefix = f"{key_prefix}_{result.name}_{result.kind}".replace(" ", "_")
    render_metrics(result)
    st.plotly_chart(
        charts.episode_dashboard(result),
        width="stretch",
        key=f"{prefix}_episode",
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            charts.drawdown_chart(result),
            width="stretch",
            key=f"{prefix}_drawdown",
        )
    with right:
        st.plotly_chart(
            charts.holdings_chart(result),
            width="stretch",
            key=f"{prefix}_holdings",
        )
    st.plotly_chart(
        charts.trades_scatter(result),
        width="stretch",
        key=f"{prefix}_trades",
    )

    tabs = st.tabs(["Trades", "Decisions", "Equity", "Holdings"])
    with tabs[0]:
        st.dataframe(
            result.trades_frame(),
            width="stretch",
            hide_index=True,
            key=f"{prefix}_trades_df",
        )
    with tabs[1]:
        st.dataframe(
            result.decisions_frame(),
            width="stretch",
            hide_index=True,
            key=f"{prefix}_decisions_df",
        )
    with tabs[2]:
        st.dataframe(
            result.equity_frame(),
            width="stretch",
            hide_index=True,
            key=f"{prefix}_equity_df",
        )
    with tabs[3]:
        st.dataframe(
            result.holdings_frame(),
            width="stretch",
            hide_index=True,
            key=f"{prefix}_holdings_df",
        )
