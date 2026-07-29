"""AlphaDuel dashboard (P4 placeholder).

Planned: pick an experiment/agent, replay an episode live, and show portfolio value,
actions, PnL curve, rewards, and (for LLM agents) the agent's thoughts per step.
Run with: ``uv run alphaduel dashboard`` (requires the `dashboard` extra).
"""

from __future__ import annotations


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="AlphaDuel", layout="wide")
    st.title("AlphaDuel — agent benchmark dashboard")
    st.info(
        "Dashboard is a Phase 4 deliverable. Planned panels: equity curve, actions, "
        "rewards, drawdown, transaction log, and LLM 'thoughts'. See SPEC.md §3."
    )


if __name__ == "__main__":
    main()
