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
from .sector import etf_for_sic, load_etf_series, sic_for_ticker
from .valuation import _closes, load_valuation, write_valuation

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
    high_52w, high_3y = float(adj.iloc[-TRADING_DAYS_PER_YEAR:].max()), float(adj.iloc[-3 * TRADING_DAYS_PER_YEAR:].max())
    if last <= 0 or high_52w <= 0 or high_3y <= 0:
        return None
    return {
        "price": round(float(df["Close"].dropna().iloc[-1]), 2),
        "dd_52w_pct": round((1 - last / high_52w) * 100, 1),
        "dd_3y_pct": round((1 - last / high_3y) * 100, 1),
        "dollar_volume": float((df["Close"] * df["Volume"].fillna(0)).iloc[-20:].mean()),
        "price_date": adj.index[-1].date(),
    }


FORWARD_MONTHS = {"ret_6m": 6, "ret_12m": 12, "ret_24m": 24}


def forward_returns(closes: pd.Series, as_of: pd.Timestamp) -> dict:
    """Percent change from the close on or before `as_of` to 6, 12 and 24 months later and to the
    latest close; None where the horizon has not been reached yet."""
    base_series = closes[closes.index <= as_of]
    if base_series.empty:
        return {key: None for key in [*FORWARD_MONTHS, "ret_now"]}
    base = float(base_series.iloc[-1])
    out = {}
    for key, months in FORWARD_MONTHS.items():
        target = as_of + pd.DateOffset(months=months)
        later = closes[closes.index >= target]
        out[key] = round((float(later.iloc[0]) / base - 1) * 100, 1) if len(later) and later.index[0] - target < pd.Timedelta(days=10) else None
    out["ret_now"] = round((float(closes.iloc[-1]) / base - 1) * 100, 1)
    return out


def benchmark_returns(data_dir: Path, as_of: pd.Timestamp, etf: str = "SPY") -> dict:
    load_etf_series(etf, data_dir / "sector")
    path = data_dir / "sector" / f"{etf}.parquet"
    if not path.exists():
        return {}
    return forward_returns(pd.read_parquet(path)["Adj Close"].dropna(), as_of)


def merged_as_of(fnd: pd.DataFrame, cashflow: pd.DataFrame | None, as_of: pd.Timestamp) -> pd.DataFrame:
    """Quarterly financials limited to what had been announced by `as_of`."""
    merged = merge_financials(fnd, cashflow).merge(fnd[["period_end", "announced"]], on="period_end", how="left")
    return merged[merged["announced"] <= as_of].drop(columns="announced")


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
    include_pharma: bool = False, as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """With `as_of`, the same screen run on that date using only prices and filings available then.
    The universe is today's listings, so companies delisted since are missing."""
    sec_dir = data_dir / "sec"
    ciks = cik_map(sec_dir)
    meta = universe.set_index("ticker")
    today = as_of if as_of is not None else pd.Timestamp.today().normalize()
    candidates, seen = [], set()
    prices_dir = data_dir / "prices"

    def consider(ticker: str, df: pd.DataFrame) -> None:
        seen.add(ticker)
        full = df
        if as_of is not None:
            df = df[df.index <= as_of]
        m = price_metrics(df)
        if m and m["dd_52w_pct"] >= min_drawdown and m["price"] >= min_price and m["dollar_volume"] >= min_dollar_volume:
            candidates.append((ticker, m, full["Adj Close"].dropna()))

    tickers = universe["ticker"].tolist()
    if as_of is not None:
        max_age_days = 36500
    for ticker, df in iter_prices(tickers, prices_dir, "10y", max_age_days):
        consider(ticker, df)
    cached = {t for t in tickers if (prices_dir / "10y" / f"{t}.parquet").exists()}
    failed = sorted(cached - seen) if as_of is None else []
    if failed:
        print(f"{len(failed)} tickers failed to download; retrying once...")
        time.sleep(30)
        for ticker, df in iter_prices(failed, prices_dir, "10y", max_age_days):
            consider(ticker, df)
    still_failed = sorted(cached - seen) if as_of is None else []
    for ticker in still_failed:
        consider(ticker, pd.read_parquet(prices_dir / "10y" / f"{ticker}.parquet"))
    if still_failed:
        print(f"{len(still_failed)} tickers could not be refreshed and use older cached prices: "
              f"{', '.join(still_failed[:20])}{' ...' if len(still_failed) > 20 else ''}")
    when = f" on {as_of:%Y-%m-%d}" if as_of is not None else ""
    print(f"{len(candidates)} stocks at least {min_drawdown:g}% below their 52-week high{when}; checking fundamentals...")

    rows = []
    for ticker, m, closes in candidates:
        sic = sic_for_ticker(ticker, sec_dir, ciks)
        if not include_pharma and sic.startswith(PHARMA_SIC_PREFIXES):
            continue
        try:
            fnd = load_fundamentals(ticker, sec_dir, 7, ciks)
            if fnd is None or fnd.empty:
                continue
            snap = fundamentals_snapshot(merged_as_of(fnd, load_cashflow(ticker, sec_dir, 7, ciks), today), today)
        except Exception as exc:
            log.warning("fundamentals failed for %s: %s", ticker, exc)
            continue
        if not snap or snap["ttm_net_income"] <= 0 or snap["ttm_free_cash_flow"] <= 0 or snap["revenue_growth_pct"] < min_growth:
            continue
        extra = {}
        if as_of is not None:
            etf = etf_for_sic(sic)
            bench = benchmark_returns(data_dir, as_of, etf)
            extra = {"as_of": f"{as_of:%Y-%m-%d}", **forward_returns(closes, as_of)}
            extra["ret_12m_vs_sector"] = (round(extra["ret_12m"] - bench["ret_12m"], 1)
                                          if extra["ret_12m"] is not None and bench.get("ret_12m") is not None else None)
        rows.append({"ticker": ticker, "name": meta.at[ticker, "name"], "exchange": meta.at[ticker, "exchange"],
                     "sic": sic, "sector_etf": etf_for_sic(sic), **m, **snap, **extra})
    df = pd.DataFrame(rows)
    return df.sort_values("dd_52w_pct", ascending=False).reset_index(drop=True) if not df.empty else df


def ps_as_of(rows: list[dict], closes: pd.Series, as_of: pd.Timestamp) -> tuple[float | None, float | None]:
    """Price-to-sales on `as_of`, and its percent difference from the median of the three years before."""
    known = [r for r in rows if r["announced"] <= f"{as_of:%Y-%m-%d}" and r["ps"]]
    if not known:
        return None, None
    last = known[-1]
    before = closes[closes.index < pd.Timestamp(last["announced"])]
    at = closes[closes.index <= as_of]
    if before.empty or at.empty or before.iloc[-1] <= 0:
        return None, None
    ps = last["ps"] * float(at.iloc[-1]) / float(before.iloc[-1])
    cutoff = f"{as_of - pd.Timedelta(days=3 * 365):%Y-%m-%d}"
    history = [r["ps"] for r in known if r["announced"] >= cutoff]
    return round(ps, 2), (round((ps / float(np.median(history)) - 1) * 100, 1) if len(history) >= 4 else None)


def add_valuation_columns(df: pd.DataFrame, data_dir: Path, refresh: bool = True, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Price-to-sales now (or on `as_of`), and how it compares with the median of the last three years."""
    now, vs_median = [], []
    for ticker in df["ticker"]:
        try:
            if refresh or load_valuation(ticker, data_dir) is None:
                write_valuation(ticker, data_dir)
        except Exception as exc:
            log.warning("valuation failed for %s: %s", ticker, exc)
        valuation = load_valuation(ticker, data_dir) or {}
        rows = valuation.get("rows", [])
        if as_of is not None:
            ps, rel = ps_as_of(rows, _closes(ticker, data_dir / "prices"), as_of) if rows else (None, None)
            now.append(ps)
            vs_median.append(rel)
            continue
        cutoff = f"{pd.Timestamp.today() - pd.Timedelta(days=3 * 365):%Y-%m-%d}"
        history = [r["ps"] for r in rows if r["announced"] >= cutoff and r["ps"]]
        ps = valuation.get("ps_current") if rows else None
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
