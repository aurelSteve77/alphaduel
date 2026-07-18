"""Interactive dashboard for exploring strategies on the shared substrate.

``service`` holds the pure logic (running strategies + building Plotly figures);
``app`` is the Streamlit UI. The split keeps the logic importable and testable
without a Streamlit runtime.
"""

from __future__ import annotations
