"""US-listed common stock universe from NASDAQ Trader symbol directory files."""

from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/"
EXCHANGES = {"A": "NYSE American", "N": "NYSE", "P": "NYSE Arca", "Z": "Cboe BZX", "V": "IEX"}
# Non-common-stock securities, matched against the security name.
EXCLUDE_NAME = r"\b(?:warrants?|units?|rights?|preferred|notes? due|debentures?|subordinated|trust preferred)\b"


def _fetch(name: str, cache_dir: Path, max_age_days: float) -> pd.DataFrame:
    path = cache_dir / name
    if not path.exists() or time.time() - path.stat().st_mtime > max_age_days * 86400:
        resp = requests.get(BASE_URL + name, timeout=30)
        resp.raise_for_status()
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(resp.text)
    df = pd.read_csv(io.StringIO(path.read_text()), sep="|", dtype=str)
    # Last row is "File Creation Time: ..."
    return df[~df.iloc[:, 0].str.startswith("File Creation Time", na=True)]


def load_universe(cache_dir: Path, max_age_days: float = 1) -> pd.DataFrame:
    """Return DataFrame[ticker, name, exchange] of US common stocks (yfinance symbols)."""
    nasdaq = _fetch("nasdaqlisted.txt", cache_dir, max_age_days)
    nasdaq = nasdaq[(nasdaq["Test Issue"] == "N") & (nasdaq["ETF"] == "N")]
    nasdaq = pd.DataFrame({"symbol": nasdaq["Symbol"], "name": nasdaq["Security Name"], "exchange": "NASDAQ"})

    other = _fetch("otherlisted.txt", cache_dir, max_age_days)
    other = other[(other["Test Issue"] == "N") & (other["ETF"] == "N")]
    other = pd.DataFrame(
        {"symbol": other["ACT Symbol"], "name": other["Security Name"], "exchange": other["Exchange"].map(EXCHANGES)}
    )

    df = pd.concat([nasdaq, other], ignore_index=True).dropna(subset=["symbol"])
    df = df[~df["symbol"].str.contains(r"[$^#]", regex=True)]
    df = df[~df["name"].fillna("").str.contains(EXCLUDE_NAME, case=False, regex=True)]
    df["ticker"] = df["symbol"].str.replace(".", "-", regex=False)
    return df.drop_duplicates("ticker")[["ticker", "name", "exchange"]].sort_values("ticker").reset_index(drop=True)
