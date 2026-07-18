"""Launcher for the Streamlit dashboard (the ``alphaduel-dashboard`` script)."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    try:
        from streamlit.web import cli as stcli
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SystemExit(
            "Streamlit is not installed. Install the app extra: uv sync --extra app"
        ) from exc

    app_path = str(Path(__file__).with_name("app.py"))
    sys.argv = ["streamlit", "run", app_path, *sys.argv[1:]]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
