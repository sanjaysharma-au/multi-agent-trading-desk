"""Optional per-ticker metadata lookup (slow; use for hits only)."""

from __future__ import annotations

import logging

import yfinance as yf

log = logging.getLogger(__name__)


def enrich(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
    except Exception as exc:
        log.warning("info lookup failed for %s: %s", ticker, exc)
        return {}
    return {
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": info.get("marketCap"),
    }
