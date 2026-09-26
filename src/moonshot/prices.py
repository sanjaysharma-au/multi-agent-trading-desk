"""Batched yfinance price download with a per-ticker parquet cache."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterator

import pandas as pd
import yfinance as yf

COLUMNS = ["Close", "Adj Close", "Volume"]
log = logging.getLogger(__name__)


def _cache_path(cache_dir: Path, ticker: str, lookback: str) -> Path:
    return cache_dir / lookback / f"{ticker}.parquet"


def _is_fresh(path: Path, max_age_days: float) -> bool:
    return path.exists() and time.time() - path.stat().st_mtime < max_age_days * 86400


def _download(tickers: list[str], lookback: str, retries: int = 3) -> pd.DataFrame | None:
    for attempt in range(retries):
        try:
            return yf.download(
                tickers,
                period=lookback,
                auto_adjust=False,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception as exc:  # yfinance raises assorted errors on throttling
            log.warning("download failed (attempt %d): %s", attempt + 1, exc)
            time.sleep(5 * (attempt + 1))
    return None


def iter_prices(
    tickers: list[str],
    cache_dir: Path,
    lookback: str = "10y",
    max_age_days: float = 1,
    chunk_size: int = 200,
    pause: float = 1.0,
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield (ticker, DataFrame[Close, Adj Close, Volume]) for each ticker with data."""
    stale = []
    for t in tickers:
        path = _cache_path(cache_dir, t, lookback)
        if _is_fresh(path, max_age_days):
            yield t, pd.read_parquet(path)
        else:
            stale.append(t)

    for start in range(0, len(stale), chunk_size):
        chunk = stale[start : start + chunk_size]
        raw = _download(chunk, lookback)
        if raw is None or raw.empty:
            continue
        for t in chunk:
            if t not in raw.columns.get_level_values(0):
                continue
            df = raw[t]
            if not set(COLUMNS).issubset(df.columns):
                continue
            df = df[COLUMNS].dropna(subset=["Close"])
            if df.empty:
                continue
            path = _cache_path(cache_dir, t, lookback)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path)
            yield t, df
        if start + chunk_size < len(stale):
            time.sleep(pause)
