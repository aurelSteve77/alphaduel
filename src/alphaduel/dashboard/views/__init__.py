"""Dashboard pages/views (used by ``st.navigation`` in ``app.py``)."""

from __future__ import annotations

from alphaduel.dashboard.views import (
    agent_detail,
    configure,
    home,
    live,
    market,
    overview,
)

__all__ = ["agent_detail", "configure", "home", "live", "market", "overview"]
