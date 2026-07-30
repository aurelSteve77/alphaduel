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
) -> tuple[MarketPanel | MultiAssetPanel, list[str]]:
    """Build a real-data panel for the dashboard.

    Downloads into the Parquet cache on first use when data is missing.
    """
    cfg = load_experiment_config(config_path_for(mode))
    if mode == "multi_asset":
        symbols = list(cfg.data.symbols[: max(1, n_assets)])
    else:
        symbols = list(cfg.data.symbols[:1])
    cfg = cfg.model_copy(update={"data": cfg.data.model_copy(update={"symbols": symbols})})

    panel = build_panel(cfg, Secrets())
    panel = _tail(panel, n_steps)
    if isinstance(panel, MultiAssetPanel):
        return panel, list(panel.symbols)
    return panel, list(panel.symbols)
