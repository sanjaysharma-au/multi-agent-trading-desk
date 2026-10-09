"""Map a company's SEC SIC code to a sector ETF and load that ETF's weekly price series."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

DEFAULT_ETF = "SPY"

# Longest matching prefix wins.
SIC_TO_ETF = {
    "737": "IGV", "7370": "IGV", "3674": "SOXX", "3672": "SOXX", "3570": "XLK", "3576": "XLK", "3661": "XLK",
    "3663": "XLK", "3669": "XLK", "3823": "XLK", "3825": "XLK", "3827": "XLK", "3829": "XLK",
    "1311": "XLE", "1381": "XLE", "1382": "XLE", "1389": "XLE", "2911": "XLE", "3533": "XLE", "4922": "XLE",
    "1040": "GDX", "10": "XLB", "12": "XLB", "13": "XLB", "24": "XLB", "26": "XLB", "28": "XLB", "33": "XLB",
    "6798": "XLRE", "65": "XLRE", "60": "XLF", "61": "XLF", "62": "XLF", "63": "XLF", "64": "XLF", "67": "XLF",
    "35": "XLI", "36": "XLI", "37": "XLI", "40": "XLI", "41": "XLI", "42": "XLI", "44": "XLI", "45": "XLI",
    "47": "XLI", "73": "XLI", "87": "XLI", "15": "XLI", "16": "XLI", "17": "XLI",
    "7310": "XLC", "7311": "XLC", "7312": "XLC", "7900": "XLC", "7990": "XLC", "78": "XLC", "48": "XLC",
    "49": "XLU", "5912": "XLP", "20": "XLP", "21": "XLP", "01": "XLP", "02": "XLP", "54": "XLP",
    "50": "XLY", "51": "XLY", "52": "XLY", "53": "XLY", "55": "XLY", "56": "XLY", "57": "XLY", "58": "XLY",
    "59": "XLY", "70": "XLY", "72": "XLY", "75": "XLY", "79": "XLY", "22": "XLY", "23": "XLY", "25": "XLY",
    "30": "XLY", "39": "XLY",
    "38": "XLV", "80": "XLV", "3841": "XLV", "3826": "XLK",
}


def etf_for_sic(sic: str | None) -> str:
    sic = sic or ""
    for n in range(len(sic), 0, -1):
        if sic[:n] in SIC_TO_ETF:
            return SIC_TO_ETF[sic[:n]]
    return DEFAULT_ETF


def load_etf_series(etf: str, cache_dir: Path, max_age_days: float = 1) -> dict | None:
    """Weekly adjusted closes in the same {"d": days since epoch, "p": prices} shape as a stock's series."""
    path = cache_dir / f"{etf}.parquet"
    if not path.exists() or time.time() - path.stat().st_mtime > max_age_days * 86400:
        try:
            df = yf.download(etf, period="max", auto_adjust=False, progress=False)
            s = df["Adj Close"].squeeze().dropna()
            if s.empty:
                raise ValueError("no data")
            cache_dir.mkdir(parents=True, exist_ok=True)
            s.rename("Adj Close").to_frame().to_parquet(path)
        except Exception as exc:
            log.warning("sector ETF download failed for %s: %s", etf, exc)
            if not path.exists():
                return None
    s = pd.read_parquet(path)["Adj Close"].dropna().resample("W-FRI").last().dropna()
    if s.empty:
        return None
    days = (s.index - pd.Timestamp("1970-01-01")).days
    return {"d": [int(d) for d in days], "p": [float(f"{v:.5g}") for v in s]}
