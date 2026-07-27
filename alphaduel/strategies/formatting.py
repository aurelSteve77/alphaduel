"""Convert Gymnasium trading observations into prompt-friendly payloads."""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_float(value: Any) -> float:
    if isinstance(value, np.ndarray):
        return float(value.reshape(-1)[0])
    return float(value)


def format_holdings(tickers: list[str], holdings: np.ndarray | list[float]) -> dict[str, float]:
    arr = np.asarray(holdings, dtype=np.float64).reshape(-1)
    return {ticker: float(arr[i]) for i, ticker in enumerate(tickers)}


def format_features(
    tickers: list[str],
    feature_names: list[str],
    features: np.ndarray,
    *,
    precision: int = 4,
) -> dict[str, dict[str, float]]:
    """Turn the ``[n_assets, n_features]`` matrix into a nested dict for Jinja."""
    matrix = np.asarray(features, dtype=np.float64)
    out: dict[str, dict[str, float]] = {}
    for i, ticker in enumerate(tickers):
        out[ticker] = {
            name: round(float(matrix[i, j]), precision)
            for j, name in enumerate(feature_names)
        }
    return out


def format_news(news: dict[str, Any] | None) -> dict[str, str] | None:
    """Collapse per-ticker news blobs into short strings for the user prompt."""
    if not news:
        return None
    rendered: dict[str, str] = {}
    for ticker, payload in news.items():
        if not isinstance(payload, dict):
            rendered[str(ticker)] = str(payload)
            continue
        count = int(payload.get("news_count", 0) or 0)
        headlines = str(payload.get("headlines", "") or "").strip()
        if count <= 0 and not headlines:
            continue
        snippet = headlines if headlines else "(no headline)"
        # Keep prompts bounded.
        if len(snippet) > 400:
            snippet = snippet[:397] + "..."
        rendered[str(ticker)] = f"[{count}] {snippet}"
    return rendered or None


def build_prompt_payload(
    obs: dict[str, np.ndarray],
    info: dict[str, Any],
    *,
    tickers: list[str],
    feature_names: list[str],
    max_shares: int,
    allow_short: bool,
) -> dict[str, Any]:
    """Assemble the Jinja payload for ``llm_policy`` system + user prompts."""
    info_tickers = [str(t) for t in info.get("tickers", tickers)]
    names = [str(n) for n in info.get("feature_names", feature_names)]
    holdings = obs.get("holdings", info.get("holdings", np.zeros(len(info_tickers))))
    return {
        "date": str(info.get("date", "")),
        "tickers": info_tickers,
        "max_shares": int(max_shares),
        "allow_short": bool(allow_short),
        "cash": round(_as_float(obs.get("cash", info.get("cash", 0.0))), 2),
        "portfolio_value": round(
            _as_float(obs.get("portfolio_value", info.get("portfolio_value", 0.0))), 2
        ),
        "holdings": format_holdings(info_tickers, holdings),
        "features": format_features(info_tickers, names, obs["features"]),
        "news": format_news(info.get("news")),
        "step": int(info.get("step", 0)),
    }
