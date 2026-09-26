"""Pure multibagger detection logic (no I/O)."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def parse_period(text: str) -> int:
    """Convert '3y', '18m', '6w' or '100d' to a number of trading days."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([ymwd])\s*", text.lower())
    if not m:
        raise ValueError(f"invalid period {text!r}; use e.g. 3y, 18m, 6w, 100d")
    n, unit = float(m.group(1)), m.group(2)
    per_unit = {"y": TRADING_DAYS_PER_YEAR, "m": TRADING_DAYS_PER_YEAR / 12, "w": 5, "d": 1}[unit]
    return max(1, round(n * per_unit))


def find_best_run(
    df: pd.DataFrame,
    multiple: float = 5.0,
    window_days: int = 756,
    min_price: float = 1.0,
    min_dollar_vol: float = 100_000,
    smooth_days: int = 5,
) -> dict | None:
    """Find the largest rise within `window_days` for one ticker.

    `df` needs a DatetimeIndex and columns `Close`, `Adj Close`, `Volume`.
    Returns a result dict if the best rise is >= `multiple`, else None.
    """
    df = df.dropna(subset=["Adj Close", "Close"])
    if len(df) < 2:
        return None

    # Rolling median removes one-day bad ticks / fake spikes.
    p = df["Adj Close"].rolling(smooth_days, min_periods=1).median() if smooth_days > 1 else df["Adj Close"]
    p = p.where(p > 0)

    # Forward max over the next `window_days` rows (excluding today).
    fwd_max = p[::-1].rolling(window_days, min_periods=1).max()[::-1].shift(-1)
    ratio = fwd_max / p

    dollar_vol = (df["Close"] * df["Volume"].fillna(0)).rolling(20, min_periods=5).mean()
    eligible = (df["Close"] >= min_price) & (dollar_vol >= min_dollar_vol)
    ratio = ratio.where(eligible)

    if ratio.notna().sum() == 0:
        return None
    start = ratio.idxmax()
    best = float(ratio[start])
    if not np.isfinite(best) or best < multiple:
        return None

    i = p.index.get_loc(start)
    window = p.iloc[i + 1 : i + 1 + window_days]
    peak = window.idxmax()
    start_price = float(p.iloc[i])
    peak_price = float(window[peak])
    crossed = window[window >= start_price * multiple]
    last_price = float(p.iloc[-1])

    return {
        "start_date": start.date(),
        "start_price": round(start_price, 4),
        "peak_date": peak.date(),
        "peak_price": round(peak_price, 4),
        "multiple": round(best, 2),
        "trading_days_to_peak": p.index.get_loc(peak) - i,
        "threshold_crossed_date": crossed.index[0].date() if len(crossed) else None,
        "last_price": round(last_price, 4),
        "last_vs_peak_pct": round((last_price / peak_price - 1) * 100, 1),
        "current_multiple": round(last_price / start_price, 2),
    }
