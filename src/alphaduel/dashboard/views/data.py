"""Data page: inspect the Parquet cache, download universes, and clear stale entries."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from alphaduel.config.loader import load_experiment_config
from alphaduel.config.schema import DataConfig, ExperimentConfig, MacroConfig, Secrets
from alphaduel.dashboard import real_data
from alphaduel.data.macro import FredMacroSource
from alphaduel.data.prices import YahooPriceSource
from alphaduel.data.storage import ParquetCache

_PRESETS = {
    "Single asset (p0_mvp)": "single_asset",
    "Multi-asset (p5_genportfolio)": "multi_asset",
    "Custom": "custom",
}

_DEFAULT_MACRO = ["VIXCLS", "DGS10", "DGS2", "CPIAUCSL"]


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _cache() -> ParquetCache:
    return ParquetCache(Secrets().data_dir)


def _entries_frame(entries: list[dict]) -> pd.DataFrame:
    if not entries:
        return pd.DataFrame(
            columns=["namespace", "file", "size", "rows", "symbols", "start", "end", "modified"]
        )
    rows = []
    for e in entries:
        symbols = e.get("symbols") or []
        rows.append(
            {
                "namespace": e["namespace"],
                "file": e["file"],
                "size": _fmt_bytes(int(e["bytes"])),
                "rows": e.get("rows"),
                "symbols": ", ".join(symbols) if symbols else "—",
                "start": e["start"].date().isoformat() if e.get("start") is not None else "—",
                "end": e["end"].date().isoformat() if e.get("end") is not None else "—",
                "modified": e["modified"].strftime("%Y-%m-%d %H:%M") if e.get("modified") else "—",
            }
        )
    return pd.DataFrame(rows)


def _resolve_config(
    preset_key: str,
    symbols: list[str],
    start: date,
    end: date,
) -> ExperimentConfig:
    mode = _PRESETS[preset_key]
    if mode == "custom":
        if not symbols:
            raise ValueError("Add at least one symbol for a custom download.")
        return ExperimentConfig(
            name="dashboard_custom",
            data=DataConfig(
                symbols=symbols,
                start=start,
                end=end,
                macro=MacroConfig(series=list(_DEFAULT_MACRO)),
            ),
        )
    cfg = load_experiment_config(real_data.config_path_for(mode))
    data_update = {
        "symbols": symbols or list(cfg.data.symbols),
        "start": start,
        "end": end,
    }
    return cfg.model_copy(update={"data": cfg.data.model_copy(update=data_update)})


def _run_download(cfg: ExperimentConfig, *, include_macro: bool, force: bool) -> None:
    secrets = Secrets()
    cache = ParquetCache(secrets.data_dir)
    YahooPriceSource(cache).fetch(
        cfg.data.symbols, cfg.data.start, cfg.data.end, force=force
    )
    if include_macro:
        series = list(cfg.data.macro.series) or list(_DEFAULT_MACRO)
        FredMacroSource(cache, secrets.fred_api_key, series).fetch(
            cfg.data.symbols, cfg.data.start, cfg.data.end, force=force
        )


def render() -> None:
    st.title("📦 Data")
    st.caption(
        "Manage the Parquet cache used by the dashboard and CLI "
        "(`ALPHADUEL_DATA_DIR`, default `./data_cache`)."
    )

    secrets = Secrets()
    cache = _cache()
    entries = cache.list_entries()
    total_bytes = sum(int(e["bytes"]) for e in entries)
    price_n = sum(1 for e in entries if e["namespace"] == "prices_yfinance")
    macro_n = sum(1 for e in entries if e["namespace"] == "macro_fred")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cache dir", Path(secrets.data_dir).resolve().name)
    m2.metric("Total size", _fmt_bytes(total_bytes))
    m3.metric("Price files", price_n)
    m4.metric("Macro files", macro_n)

    if secrets.fred_api_key:
        st.success("FRED_API_KEY is set — macro downloads are available.")
    else:
        st.warning(
            "FRED_API_KEY is missing in `.env` — price downloads still work; macro will fail."
        )

    st.divider()
    st.subheader("Download")

    preset = st.selectbox("Universe preset", list(_PRESETS))
    mode = _PRESETS[preset]

    if mode == "custom":
        default_symbols = real_data.universe_symbols("multi_asset")
        default_start = date(2015, 1, 1)
        default_end = date(2023, 12, 31)
    else:
        cfg0 = load_experiment_config(real_data.config_path_for(mode))
        default_symbols = list(cfg0.data.symbols)
        default_start = cfg0.data.start
        default_end = cfg0.data.end

    symbols_text = st.text_input(
        "Symbols (comma-separated)",
        value=", ".join(default_symbols),
        help="Override the preset universe if needed.",
    )
    symbols = [s.strip().upper() for s in symbols_text.split(",") if s.strip()]

    c1, c2, c3 = st.columns(3)
    start = c1.date_input("Start", value=default_start)
    end = c2.date_input("End", value=default_end)
    include_macro = c3.checkbox("Also fetch FRED macro", value=True)
    force = st.checkbox("Force re-download (overwrite cache)", value=False)

    if st.button("⬇️ Download", type="primary", use_container_width=True):
        try:
            if end <= start:
                raise ValueError("End date must be after start date.")
            cfg = _resolve_config(preset, symbols, start, end)
            with st.spinner(f"Downloading {len(cfg.data.symbols)} symbol(s)…"):
                _run_download(cfg, include_macro=include_macro, force=force)
            st.cache_data.clear()
            st.success(
                f"Cached {', '.join(cfg.data.symbols)} "
                f"({cfg.data.start} → {cfg.data.end})"
                + (" + FRED macro" if include_macro else "")
            )
            st.rerun()
        except Exception as exc:  # noqa: BLE001 — show download errors in the UI
            st.error(str(exc))

    st.divider()
    st.subheader("Cached datasets")

    frame = _entries_frame(entries)
    if frame.empty:
        st.info("Cache is empty. Download a universe above to get started.")
    else:
        st.dataframe(frame, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Clear cache")
    st.caption("Deletes Parquet files only. Re-download to restore.")

    b1, b2, b3 = st.columns(3)
    if b1.button("Clear prices", use_container_width=True):
        n = cache.clear_namespace("prices_yfinance")
        st.cache_data.clear()
        st.toast(f"Removed {n} price file(s).")
        st.rerun()
    if b2.button("Clear macro", use_container_width=True):
        n = cache.clear_namespace("macro_fred")
        st.cache_data.clear()
        st.toast(f"Removed {n} macro file(s).")
        st.rerun()
    if b3.button("Clear all", type="primary", use_container_width=True):
        n = cache.clear_all()
        st.cache_data.clear()
        st.toast(f"Removed {n} file(s).")
        st.rerun()

    st.caption(
        f"Cache root: `{Path(secrets.data_dir).resolve()}` · refreshed {datetime.now():%H:%M:%S}"
    )
