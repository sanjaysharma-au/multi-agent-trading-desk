"""Price-to-sales and price-to-free-cash-flow history, one row per earnings announcement."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from .fundamentals import _get_json, cik_map, load_cashflow, load_fundamentals
from .inflection import merge_financials

log = logging.getLogger(__name__)

SHARES_CONCEPT = "WeightedAverageNumberOfDilutedSharesOutstanding"


def _diluted_shares(facts: dict) -> pd.Series:
    """As-reported diluted share count per quarter end. Q4 is not reported on its own, so the
    fiscal-year average stands in for it."""
    rows = facts.get("facts", {}).get("us-gaap", {}).get(SHARES_CONCEPT, {}).get("units", {}).get("shares", [])
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    df["start"], df["end"], df["filed"] = (pd.to_datetime(df[c]) for c in ("start", "end", "filed"))
    df["days"] = (df["end"] - df["start"]).dt.days
    df = df.sort_values("filed")
    quarterly = df[df["days"].between(80, 100)].drop_duplicates("end").set_index("end")["val"]
    annual = df[df["days"].between(350, 380)].drop_duplicates("end").set_index("end")["val"]
    return pd.concat([quarterly, annual[~annual.index.isin(quarterly.index)]]).sort_index().astype(float)


def _splits(ticker: str) -> pd.Series:
    """Raises if the lookup fails: a missing split history would silently misprice every quarter."""
    s = yf.Ticker(ticker).splits
    if s is None:
        raise RuntimeError(f"split lookup returned nothing for {ticker}")
    s.index = s.index.tz_localize(None).astype("datetime64[ns]")
    return s


def _closes(ticker: str, prices_dir: Path) -> pd.Series:
    paths = sorted(prices_dir.glob(f"*/{ticker}.parquet"), key=lambda p: p.stat().st_mtime)
    if not paths:
        return pd.Series(dtype=float)
    return pd.read_parquet(paths[-1])["Close"].dropna()


def build_valuation(ticker: str, data_dir: Path, ciks: dict[str, int] | None = None) -> dict | None:
    """Market cap = the last close before the results were announced, put back into the units of
    that time (undoing later splits), times that quarter's as-reported diluted shares."""
    sec_dir = data_dir / "sec"
    ciks = ciks if ciks is not None else cik_map(sec_dir)
    cik = ciks.get(ticker.upper())
    fnd = load_fundamentals(ticker, sec_dir, ciks=ciks)
    closes = _closes(ticker, data_dir / "prices")
    if cik is None or fnd is None or fnd.empty or closes.empty:
        return None
    facts = _get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", sec_dir / f"facts_{cik}.json", 7)
    shares = _diluted_shares(facts or {})
    if shares.empty:
        return None

    df = merge_financials(fnd, load_cashflow(ticker, sec_dir, ciks=ciks)).merge(
        fnd[["period_end", "announced"]], on="period_end", how="left")
    df["ttm_revenue"] = df["revenue"].rolling(4).sum()
    df["ttm_fcf"] = df["free_cash_flow"].rolling(4).sum()
    df = df.dropna(subset=["announced", "ttm_revenue"])
    splits = _splits(ticker)

    rows = []
    for r in df.itertuples():
        before = closes[closes.index < r.announced]
        sh = shares[shares.index <= r.period_end]
        if before.empty or sh.empty or r.period_end - sh.index[-1] > pd.Timedelta(days=120):
            continue
        close = float(before.iloc[-1])
        later = splits[splits.index > before.index[-1]]
        price_then = close * float(later.prod()) if len(later) else close
        mcap = price_then * float(sh.iloc[-1])
        fcf = r.ttm_fcf if pd.notna(r.ttm_fcf) else None
        rows.append({
            "announced": f"{r.announced:%Y-%m-%d}", "period_end": f"{r.period_end:%Y-%m-%d}",
            "mcap": round(mcap), "ttm_revenue": round(r.ttm_revenue),
            "ps": round(mcap / r.ttm_revenue, 2) if r.ttm_revenue > 0 else None,
            "p_fcf": round(mcap / fcf, 1) if fcf and fcf > 0 else None,
        })
    if not rows:
        return None
    return {"ticker": ticker.upper(), "rows": rows}


def valuation_path(ticker: str, data_dir: Path) -> Path:
    return data_dir / "valuation" / f"{ticker.upper()}.json"


def write_valuation(ticker: str, data_dir: Path) -> Path:
    result = build_valuation(ticker, data_dir) or {"ticker": ticker.upper(), "rows": [], "skipped": "no SEC or price data"}
    path = valuation_path(ticker, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))
    return path


def load_valuation(ticker: str, data_dir: Path) -> dict | None:
    path = valuation_path(ticker, data_dir)
    return json.loads(path.read_text()) if path.exists() else None
