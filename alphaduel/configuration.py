"""Project-wide configuration loaded from ``configs/project.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from alphaduel.utils.singleton import Singleton

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "project.yaml"


class Configuration(metaclass=Singleton):
    """Singleton wrapper around the shared project YAML config.

    Usage::

        Configuration().get("seed")
        Configuration().get("data.universe")
        Configuration().get("missing.key", default=None)
    """

    def __init__(self, path: Path | str | None = None) -> None:
        config_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with config_path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        self._data: dict[str, Any] = data if isinstance(data, dict) else {}
        self._path = config_path

    @property
    def path(self) -> Path:
        """Absolute path of the loaded config file."""
        return self._path

    def get(self, key: str, default: Any = None) -> Any:
        """Return a value by dotted key path (e.g. ``"a.b.c"``).

        Returns ``default`` if any segment along the path is missing or if an
        intermediate value is not a mapping.
        """
        current: Any = self._data
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    def as_dict(self) -> dict[str, Any]:
        """Return a shallow copy of the raw config mapping."""
        return dict(self._data)
