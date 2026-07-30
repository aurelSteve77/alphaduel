"""Thoughts component: step-by-step agent reasoning (LLM / generative agents)."""

from __future__ import annotations

import streamlit as st

from alphaduel.dashboard.engine import BenchmarkResult


def render(result: BenchmarkResult, agent_name: str) -> None:
    rec = result.agents[agent_name]
    if not any(t for t in rec.thoughts):
        return

    st.subheader("Agent reasoning")
    n = len(rec.thoughts)
    step = st.slider("Step", 0, max(n - 1, 0), 0, key=f"thoughts_{agent_name}")
    if step < len(rec.parse_oks) and rec.parse_oks[step] is False:
        st.warning("Parse failed on this step — positions held (do-nothing).")
    if step < len(rec.llm_actions) and rec.llm_actions[step] is not None:
        st.caption("Parsed actions")
        st.code(str(rec.llm_actions[step]), language="json")
    thought = rec.thoughts[step]
    if thought:
        st.code(thought, language="text")
    else:
        st.caption("No reasoning recorded for this step.")
