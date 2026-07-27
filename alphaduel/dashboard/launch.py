"""Launch the Streamlit dashboard."""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``alphaduel-dashboard`` / ``python -m alphaduel.dashboard``."""
    try:
        from streamlit.web import cli as stcli
    except ImportError as exc:  # pragma: no cover
        print(
            "Streamlit is required. Install with: uv sync --extra dashboard",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    app = Path(__file__).resolve().parent / "app.py"
    args = argv if argv is not None else sys.argv[1:]
    sys.argv = ["streamlit", "run", str(app), *args]
    stcli.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
