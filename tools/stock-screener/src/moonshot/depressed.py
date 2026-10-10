"""Screen for stocks trading far below their highs whose business fundamentals are still sound."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .fundamentals import cik_map, load_cashflow, load_fundamentals
from .inflection import merge_financials
from .prices import iter_prices
from .sector import etf_for_sic, sic_for_ticker
from .valuation import load_valuation, write_valuation

log = logging.getLogger(__name__)

PHARMA_SIC_PREFIXES = ("283", "8731")
TRADING_DAYS_PER_YEAR = 252
MAX_QUARTER_AGE_DAYS = 200
MIN_QUARTERS = 8


def price_metrics(df: pd.DataFrame) -> dict | None:
    """Drawdowns from the 52-week and 3-year highs of adjusted close, plus liquidity."""
    adj = df["Adj Close"].dropna()
    if len(adj) < TRADING_DAYS_PER_YEAR + 8:
        return None
    last = float(adj.iloc[-1])
    return {
        "price": round(float(df["Close"].dropna().iloc[-1]), 2),
        "dd_52w_pct": round((1 - last / float(adj.iloc[-TRADING_DAYS_PER_YEAR:].max())) * 100, 1),
        "dd_3y_pct": round((1 - last / float(adj.iloc[-3 * TRADING_DAYS_PER_YEAR:].max())) * 100, 1),
        "dollar_volume": float((df["Close"] * df["Volume"].fillna(0)).iloc[-20:].mean()),
        "price_date": adj.index[-1].date(),
    }


def fundamentals_snapshot(merged: pd.DataFrame, today: pd.Timestamp) -> dict | None:
    """Trailing-twelve-month revenue, growth, profit and free cash flow from quarterly financials."""
    m = merged.sort_values("period_end")
    if len(m) < MIN_QUARTERS or (today - m["period_end"].iloc[-1]).days > MAX_QUARTER_AGE_DAYS:
        return None
    last4, prior4 = m.tail(4), m.iloc[-8:-4]
    if last4["free_cash_flow"].isna().any() or last4["revenue"].isna().any() or prior4["revenue"].isna().any():
        return None
    revenue, prior_revenue = float(last4["revenue"].sum()), float(prior4["revenue"].sum())
    if revenue <= 0 or prior_revenue <= 0:
        return None
    net_income, fcf = float(last4["net_income"].sum()), float(last4["free_cash_flow"].sum())
    return {
        "ttm_revenue": revenue,
        "revenue_growth_pct": round((revenue / prior_revenue - 1) * 100, 1),
        "ttm_net_income": net_income,
        "ttm_free_cash_flow": fcf,
        "net_margin_pct": round(net_income / revenue * 100, 1),
        "fcf_margin_pct": round(fcf / revenue * 100, 1),
        "latest_quarter_end": f"{m['period_end'].iloc[-1]:%Y-%m-%d}",
    }


def run_screen(
    universe: pd.DataFrame, data_dir: Path, min_drawdown: float = 40.0, min_price: float = 3.0,
    min_dollar_volume: float = 1_000_000, min_growth: float = 10.0, max_age_days: float = 1,
    include_pharma: bool = False,
) -> pd.DataFrame:
    sec_dir = data_dir / "sec"
    ciks = cik_map(sec_dir)
    meta = universe.set_index("ticker")
    today = pd.Timestamp.today().normalize()
    candidates, seen = [], set()
    prices_dir = data_dir / "prices"

    def consider(ticker: str, df: pd.DataFrame) -> None:
        seen.add(ticker)
        m = price_metrics(df)
        if m and m["dd_52w_pct"] >= min_drawdown and m["price"] >= min_price and m["dollar_volume"] >= min_dollar_volume:
            candidates.append((ticker, m))

    tickers = universe["ticker"].tolist()
    for ticker, df in iter_prices(tickers, prices_dir, "10y", max_age_days):
        consider(ticker, df)
    cached = {t for t in tickers if (prices_dir / "10y" / f"{t}.parquet").exists()}
    failed = sorted(cached - seen)
    if failed:
        print(f"{len(failed)} tickers failed to download; retrying once...")
        time.sleep(30)
        for ticker, df in iter_prices(failed, prices_dir, "10y", max_age_days):
            consider(ticker, df)
    still_failed = sorted(cached - seen)
    for ticker in still_failed:
        consider(ticker, pd.read_parquet(prices_dir / "10y" / f"{ticker}.parquet"))
    if still_failed:
        print(f"{len(still_failed)} tickers could not be refreshed and use older cached prices: "
              f"{', '.join(still_failed[:20])}{' ...' if len(still_failed) > 20 else ''}")
    print(f"{len(candidates)} stocks at least {min_drawdown:g}% below their 52-week high; checking fundamentals...")

    rows = []
    for ticker, m in candidates:
        sic = sic_for_ticker(ticker, sec_dir, ciks)
        if not include_pharma and sic.startswith(PHARMA_SIC_PREFIXES):
            continue
        try:
            fnd = load_fundamentals(ticker, sec_dir, 7, ciks)
            if fnd is None or fnd.empty:
                continue
            snap = fundamentals_snapshot(merge_financials(fnd, load_cashflow(ticker, sec_dir, 7, ciks)), today)
        except Exception as exc:
            log.warning("fundamentals failed for %s: %s", ticker, exc)
            continue
        if not snap or snap["ttm_net_income"] <= 0 or snap["ttm_free_cash_flow"] <= 0 or snap["revenue_growth_pct"] < min_growth:
            continue
        rows.append({"ticker": ticker, "name": meta.at[ticker, "name"], "exchange": meta.at[ticker, "exchange"],
                     "sic": sic, "sector_etf": etf_for_sic(sic), **m, **snap})
    df = pd.DataFrame(rows)
    return df.sort_values("dd_52w_pct", ascending=False).reset_index(drop=True) if not df.empty else df


def add_valuation_columns(df: pd.DataFrame, data_dir: Path, refresh: bool = True) -> pd.DataFrame:
    """Price-to-sales now, and how it compares with the median of the last three years."""
    now, vs_median = [], []
    for ticker in df["ticker"]:
        try:
            if refresh or load_valuation(ticker, data_dir) is None:
                write_valuation(ticker, data_dir)
        except Exception as exc:
            log.warning("valuation failed for %s: %s", ticker, exc)
        rows = (load_valuation(ticker, data_dir) or {}).get("rows", [])
        cutoff = f"{pd.Timestamp.today() - pd.Timedelta(days=3 * 365):%Y-%m-%d}"
        history = [r["ps"] for r in rows if r["announced"] >= cutoff and r["ps"]]
        ps = (load_valuation(ticker, data_dir) or {}).get("ps_current") if rows else None
        now.append(ps)
        vs_median.append(round((ps / float(np.median(history)) - 1) * 100, 1) if ps and len(history) >= 4 else None)
    df = df.copy()
    df["ps_now"], df["ps_vs_3y_median_pct"] = now, vs_median
    return df


def write_params(csv_path: Path, params: dict) -> None:
    csv_path.with_suffix(".params.json").write_text(json.dumps(params))


def load_params(csv_path: Path) -> dict:
    path = csv_path.with_suffix(".params.json")
    return json.loads(path.read_text()) if path.exists() else {}
