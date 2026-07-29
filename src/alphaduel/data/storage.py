"""Content-addressed Parquet cache for downloaded data.

Downloads are expensive/flaky (esp. yfinance), so every fetch is cached by a key
derived from the query. The cache is the boundary between the network and the rest
of the system, keeping runs reproducible and offline-friendly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


class ParquetCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _key(self, namespace: str, params: dict) -> Path:
        digest = hashlib.sha256(
            json.dumps(params, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        ns_dir = self.root / namespace
        ns_dir.mkdir(parents=True, exist_ok=True)
        return ns_dir / f"{digest}.parquet"

    def get(self, namespace: str, params: dict) -> pd.DataFrame | None:
        path = self._key(namespace, params)
        if path.exists():
            return pd.read_parquet(path)
        return None

    def put(self, namespace: str, params: dict, df: pd.DataFrame) -> Path:
        path = self._key(namespace, params)
        df.to_parquet(path, index=True)
        return path
