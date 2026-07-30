"""AlphaDuel dashboard entrypoint (multipage via ``st.navigation``).

Streamlit runs this file directly. It sets up shared experiment state, builds the grouped
navigation, and routes to the selected page. Uses real cached market data (yfinance).
"""

from __future__ import annotations

import streamlit as st

from alphaduel.dashboard import state
from alphaduel.dashboard.views import agent_detail, configure, data, home, live, market, overview


def _sidebar_status() -> None:
    params = st.session_state.get("params", {})
    if not params:
        return
    universe = "Multi-asset" if params["mode"] == "multi_asset" else "Single asset"
    st.sidebar.divider()
    st.sidebar.caption("Current experiment")
    st.sidebar.write(f"**{universe}** · {len(params['agent_names'])} agents")
    st.sidebar.caption(f"{params['episode_length']}d episodes · seed {params['seed']}")


def main() -> None:
    st.set_page_config(page_title="AlphaDuel", page_icon="📈", layout="wide")
    state.ensure_params()

    pages = {
        "home": st.Page(home.render, title="Home", icon="🏠", url_path="home", default=True),
        "configure": st.Page(
            configure.render, title="Configure", icon="⚙️", url_path="configure"
        ),
        "data": st.Page(data.render, title="Data", icon="📦", url_path="data"),
        "live": st.Page(live.render, title="Live run", icon="🎬", url_path="live"),
        "overview": st.Page(overview.render, title="Overview", icon="📊", url_path="overview"),
        "agent": st.Page(agent_detail.render, title="Agent detail", icon="🔍", url_path="agent"),
        "market": st.Page(market.render, title="Market", icon="📈", url_path="market"),
    }
    st.session_state["_pages"] = pages

    nav = st.navigation(
        {
            "": [pages["home"]],
            "Setup": [pages["configure"], pages["data"], pages["live"]],
            "Analyze": [pages["overview"], pages["agent"], pages["market"]],
        }
    )
    _sidebar_status()
    nav.run()


if __name__ == "__main__":
    main()

