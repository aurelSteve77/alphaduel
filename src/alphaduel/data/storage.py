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

    def delete(self, namespace: str, params: dict) -> bool:
        """Remove one cached entry. Returns True if a file was deleted."""
        path = self._key(namespace, params)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear_namespace(self, namespace: str) -> int:
        """Delete every parquet under ``namespace``. Returns number of files removed."""
        ns_dir = self.root / namespace
        if not ns_dir.exists():
            return 0
        removed = 0
        for path in ns_dir.glob("*.parquet"):
            path.unlink()
            removed += 1
        return removed

    def clear_all(self) -> int:
        """Delete every parquet under the cache root. Returns number of files removed."""
        removed = 0
        for path in self.root.glob("*/*.parquet"):
            path.unlink()
            removed += 1
        return removed

    def list_entries(self) -> list[dict]:
        """Summarize cached parquet files for UI / diagnostics."""
        entries: list[dict] = []
        if not self.root.exists():
            return entries
        for path in sorted(self.root.glob("*/*.parquet")):
            stat = path.stat()
            info: dict = {
                "namespace": path.parent.name,
                "file": path.name,
                "path": str(path),
                "bytes": stat.st_size,
                "modified": pd.Timestamp(stat.st_mtime, unit="s", tz="UTC"),
                "rows": None,
                "symbols": None,
                "start": None,
                "end": None,
            }
            try:
                df = pd.read_parquet(path)
                info["rows"] = len(df)
                if "symbol" in df.columns:
                    info["symbols"] = sorted(df["symbol"].astype(str).unique().tolist())
                ts_col = "timestamp" if "timestamp" in df.columns else None
                if ts_col is None and isinstance(df.index, pd.DatetimeIndex):
                    ts = df.index
                elif ts_col is not None:
                    ts = pd.to_datetime(df[ts_col], utc=True)
                else:
                    ts = None
                if ts is not None and len(ts) > 0:
                    info["start"] = pd.Timestamp(ts.min())
                    info["end"] = pd.Timestamp(ts.max())
            except Exception:  # noqa: BLE001 — corrupt cache rows still show as files
                pass
            entries.append(info)
        return entries
