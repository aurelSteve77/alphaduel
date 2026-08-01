"""Load real market panels for the dashboard via the experiment configs + Parquet cache.

Uses the same ``pipeline.build_panel`` path as ``alphaduel run`` (yfinance / FRED).
"""

from __future__ import annotations

from pathlib import Path

from alphaduel.config.loader import load_experiment_config
from alphaduel.config.schema import Secrets
from alphaduel.features.store import MarketPanel, MultiAssetPanel
from alphaduel.pipeline import build_panel

_ROOT = Path(__file__).resolve().parents[3]
_CONFIGS = {
    "single_asset": _ROOT / "configs" / "experiment" / "p0_mvp.yaml",
    "multi_asset": _ROOT / "configs" / "experiment" / "p5_genportfolio.yaml",
}


def config_path_for(mode: str) -> Path:
    try:
        return _CONFIGS[mode]
    except KeyError as exc:
        raise ValueError(f"Unknown dashboard mode: {mode!r}") from exc


def universe_symbols(mode: str) -> list[str]:
    """Return the configured symbol list for a mode (without downloading)."""
    cfg = load_experiment_config(config_path_for(mode))
    return list(cfg.data.symbols)


def symbol_catalog() -> list[str]:
    """Union of single- and multi-asset YAML universes for the Configure picker."""
    seen: list[str] = []
    for mode in ("multi_asset", "single_asset"):
        for sym in universe_symbols(mode):
            if sym not in seen:
                seen.append(sym)
    return seen


def default_date_range(mode: str) -> tuple[str, str]:
    """ISO date strings ``(start, end)`` from the mode's experiment YAML."""
    cfg = load_experiment_config(config_path_for(mode))
    return cfg.data.start.isoformat(), cfg.data.end.isoformat()


def _tail(panel: MarketPanel | MultiAssetPanel, n_steps: int | None):
    if n_steps is None or panel.n_steps <= n_steps:
        return panel
    start = panel.n_steps - n_steps
    if isinstance(panel, MultiAssetPanel):
        return MultiAssetPanel(
            timestamps=panel.timestamps[start:],
            symbols=list(panel.symbols),
            open=panel.open[start:],
            close=panel.close[start:],
            features=panel.features[start:],
            feature_names=list(panel.feature_names),
        )
    return MarketPanel(
        timestamps=panel.timestamps[start:],
        open=panel.open[start:],
        close=panel.close[start:],
        features=panel.features[start:],
        feature_names=list(panel.feature_names),
        symbol=getattr(panel, "symbol", "ASSET"),
    )


def load_panel(
    mode: str,
    n_assets: int = 4,
    n_steps: int | None = 500,
    symbols: list[str] | tuple[str, ...] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> tuple[MarketPanel | MultiAssetPanel, list[str]]:
    """Build a real-data panel for the dashboard.

    Downloads into the Parquet cache on first use when data is missing.
    ``symbols`` / ``start`` / ``end`` override the YAML universe and date window.
    ``n_steps`` then keeps only the last N bars of that window (if set).
    """
    from datetime import date

    cfg = load_experiment_config(config_path_for(mode))
    if symbols:
        syms = [str(s).strip().upper() for s in symbols if str(s).strip()]
        if not syms:
            raise ValueError("At least one symbol is required")
        if mode == "single_asset":
            syms = syms[:1]
        elif len(syms) < 2:
            raise ValueError("Multi-asset mode requires at least 2 symbols")
    elif mode == "multi_asset":
        syms = list(cfg.data.symbols[: max(1, n_assets)])
    else:
        syms = list(cfg.data.symbols[:1])

    data_updates: dict = {"symbols": syms}
    if start:
        data_updates["start"] = date.fromisoformat(str(start)[:10])
    if end:
        data_updates["end"] = date.fromisoformat(str(end)[:10])
    if data_updates.get("start") and data_updates.get("end"):
        if data_updates["end"] <= data_updates["start"]:
            raise ValueError("end date must be after start date")

    cfg = cfg.model_copy(update={"data": cfg.data.model_copy(update=data_updates)})

    panel = build_panel(cfg, Secrets())
    panel = _tail(panel, n_steps)
    if isinstance(panel, MultiAssetPanel):
        return panel, list(panel.symbols)
    return panel, list(panel.symbols)
