"""Quarterly fundamentals from SEC EDGAR, dated by when results were announced.

Values come from the XBRL companyfacts API. Each quarter is placed on the date of the
first 8-K with item 2.02 ("Results of Operations") filed after the quarter ended, i.e.
the earnings release. If no such 8-K exists, the 10-Q/10-K filing date is used instead.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

USER_AGENT = os.environ.get("MOONSHOT_SEC_USER_AGENT", "moonshot-screener research-tool")
REVENUE_CONCEPTS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
]
EPS_CONCEPTS = ["EarningsPerShareDiluted", "EarningsPerShareBasic"]
NET_INCOME_CONCEPTS = ["NetIncomeLoss", "ProfitLoss"]
ANNOUNCE_WINDOW_DAYS = 100  # 8-K 2.02 must come within this many days of quarter end
_last_request = 0.0


def sec_get(url: str) -> requests.Response | None:
    """Public alias of `_get` for other modules that need a rate-limited, SEC-friendly GET."""
    return _get(url)


def _get(url: str) -> requests.Response | None:
    global _last_request
    time.sleep(max(0.0, 0.12 - (time.time() - _last_request)))  # SEC allows 10 req/s
    _last_request = time.time()
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    except requests.RequestException as exc:
        log.warning("SEC request failed %s: %s", url, exc)
        return None
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp


def _get_json(url: str, path: Path, max_age_days: float) -> dict | None:
    if path.exists() and time.time() - path.stat().st_mtime < max_age_days * 86400:
        return json.loads(path.read_text())
    resp = _get(url)
    if resp is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(resp.text)
    return resp.json()


def cik_map(cache_dir: Path, max_age_days: float = 7) -> dict[str, int]:
    data = _get_json("https://www.sec.gov/files/company_tickers.json", cache_dir / "company_tickers.json", max_age_days)
    return {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in (data or {}).values()}


def filing_index_url(cik: int, accn: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn.replace('-', '')}/{accn}-index.htm"


def exhibit_links(cik: int, accn: str, cache_dir: Path) -> dict:
    """Links to a filing's press release (EX-99.1) and extra commentary (EX-99.2), if any.

    Filings never change, so results are cached permanently. Returns {} if the filing is unavailable.
    """
    path = cache_dir / "exhibits" / f"{accn}.json"
    if path.exists():
        return json.loads(path.read_text())
    index = filing_index_url(cik, accn)
    resp = _get(index)
    links = {}
    if resp is not None:
        for href, kind in re.findall(r'<a href="([^"]+)">[^<]*</a></td>\s*<td[^>]*>\s*(EX-99\.?0?[12])\b', resp.text):
            key = "release" if kind.replace(".0", ".").endswith(".1") or kind == "EX-991" else "commentary"
            links.setdefault(key, "https://www.sec.gov" + href if href.startswith("/") else href)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(links))
    return links


def _announcement_dates(cik: int, cache_dir: Path, max_age_days: float) -> pd.DataFrame:
    """Filing date and accession number of every 8-K with item 2.02, including older paginated files."""
    sub = _get_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", cache_dir / f"sub_{cik}.json", max_age_days)
    if not sub:
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "accn": pd.Series(dtype=str)})
    blocks = [sub["filings"]["recent"]]
    for f in sub["filings"].get("files", []):
        older = _get_json(f"https://data.sec.gov/submissions/{f['name']}", cache_dir / f["name"], max_age_days * 30)
        if older:
            blocks.append(older)
    rows = []
    for b in blocks:
        for form, date, items, accn in zip(b["form"], b["filingDate"], b.get("items", [""] * len(b["form"])),
                                           b["accessionNumber"]):
            if form == "8-K" and "2.02" in (items or ""):
                rows.append((pd.Timestamp(date), accn))
    df = pd.DataFrame(rows, columns=["date", "accn"])
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def _quarterly(facts: dict, concepts: list[str], unit: str) -> pd.DataFrame:
    """Quarterly values (period end, value, first filed) merged across concepts.

    Q4 is rarely reported on its own, so it is derived as the fiscal year total minus Q1-Q3.
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    merged: dict[pd.Timestamp, tuple[float, pd.Timestamp, str | None]] = {}
    for concept in concepts:
        rows = gaap.get(concept, {}).get("units", {}).get(unit, [])
        if not rows:
            continue
        df = pd.DataFrame(rows)
        if "start" not in df:
            continue
        df["start"], df["end"], df["filed"] = (pd.to_datetime(df[c]) for c in ("start", "end", "filed"))
        df["days"] = (df["end"] - df["start"]).dt.days
        # Earliest filing wins: that is what the market saw at the announcement.
        df = df.sort_values("filed")
        q = df[df["days"].between(80, 100)].drop_duplicates("end")
        a = df[df["days"].between(350, 380)].drop_duplicates("end")
        found = {r.end: (float(r.val), r.filed, getattr(r, "accn", None)) for r in q.itertuples()}
        for r in a.itertuples():
            if r.end in found:
                continue
            inside = q[(q["start"] >= r.start - pd.Timedelta(days=5)) & (q["end"] < r.end)]
            if len(inside) == 3:
                found[r.end] = (float(r.val) - inside["val"].sum(), r.filed, getattr(r, "accn", None))
        for end, v in found.items():
            merged.setdefault(end, v)
    if not merged:
        return pd.DataFrame(columns=["period_end", "value", "filed", "accn"])
    out = pd.DataFrame([(e, *v) for e, v in merged.items()], columns=["period_end", "value", "filed", "accn"])
    return out.sort_values("period_end").reset_index(drop=True)


def _quarterize_cumulative(facts: dict, concepts: list[str], unit: str) -> pd.DataFrame:
    """Quarterly values for a concept that may be reported cumulative year-to-date.

    Cash-flow-statement items (operating cash flow, capex, ...) are almost always tagged this way:
    a 10-Q's "duration" fact runs from the fiscal year's start to the current quarter's end, not
    just over the quarter itself (Q2's `start` is January 1, not April 1). This recovers each
    quarter's own value by grouping facts that share a `start` date (the same fiscal year) and
    differencing consecutive `end` dates. A company that instead tags genuinely discrete quarters
    (each with its own distinct `start`) forms singleton groups, so nothing is differenced and the
    reported value is used as-is - the same code path handles both conventions.
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    merged: dict[pd.Timestamp, tuple[float, pd.Timestamp]] = {}
    for concept in concepts:
        rows = gaap.get(concept, {}).get("units", {}).get(unit, [])
        if not rows:
            continue
        df = pd.DataFrame(rows)
        if "start" not in df:
            continue
        df["start"], df["end"], df["filed"] = (pd.to_datetime(df[c]) for c in ("start", "end", "filed"))
        # Earliest filing per (start, end) wins: that is what the market saw at the time.
        df = df.sort_values("filed").drop_duplicates(["start", "end"])
        for _, group in df.groupby("start"):
            group = group.sort_values("end")
            prior_val = 0.0
            prior_end = None
            for r in group.itertuples():
                # Skip an out-of-order or duplicate-length checkpoint (bad/restated data).
                if prior_end is not None and r.end <= prior_end:
                    continue
                merged.setdefault(r.end, (float(r.val) - prior_val, r.filed))
                prior_val, prior_end = float(r.val), r.end
    if not merged:
        return pd.DataFrame(columns=["period_end", "value", "filed"])
    out = pd.DataFrame([(e, *v) for e, v in merged.items()], columns=["period_end", "value", "filed"])
    return out.sort_values("period_end").reset_index(drop=True)


OPERATING_CF_CONCEPTS = [
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
]
CAPEX_CONCEPTS = [
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
    "PaymentsForCapitalImprovements",
]


def load_cashflow(ticker: str, cache_dir: Path, max_age_days: float = 7,
                  ciks: dict[str, int] | None = None) -> pd.DataFrame | None:
    """Quarterly operating cash flow, capex and free cash flow (= CFO - capex).

    Reuses the same cached companyfacts JSON as `load_fundamentals` - no extra SEC request.
    """
    ciks = ciks if ciks is not None else cik_map(cache_dir, max_age_days)
    cik = ciks.get(ticker.upper())
    if cik is None:
        return None
    facts = _get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", cache_dir / f"facts_{cik}.json", max_age_days)
    if not facts:
        return None
    cfo = _quarterize_cumulative(facts, OPERATING_CF_CONCEPTS, "USD").rename(columns={"value": "operating_cf"})
    capex = _quarterize_cumulative(facts, CAPEX_CONCEPTS, "USD").rename(columns={"value": "capex"})
    df = cfo.drop(columns="filed").merge(capex.drop(columns="filed"), on="period_end", how="outer").sort_values("period_end")
    if df.empty:
        return None
    # Capex is reported as a positive cash outflow; free cash flow nets it against operating cash flow.
    df["free_cash_flow"] = df["operating_cf"] - df["capex"].fillna(0)
    return df.reset_index(drop=True)


def _yoy(df: pd.DataFrame, col: str) -> pd.Series:
    """Growth vs the quarter ending ~1 year earlier (None if missing or base <= 0)."""
    s = df.set_index("period_end")[col]
    out = []
    for end, v in s.items():
        prior = s[(s.index >= end - pd.Timedelta(days=380)) & (s.index <= end - pd.Timedelta(days=350))]
        base = prior.iloc[0] if len(prior) else None
        out.append((v / base - 1) * 100 if base and base > 0 and pd.notna(v) else None)
    return pd.Series(out, index=df.index)


def load_fundamentals(ticker: str, cache_dir: Path, max_age_days: float = 7, ciks: dict[str, int] | None = None) -> pd.DataFrame | None:
    """Quarterly revenue, EPS and net income with announcement dates, or None if unavailable."""
    ciks = ciks if ciks is not None else cik_map(cache_dir, max_age_days)
    cik = ciks.get(ticker.upper())
    if cik is None:
        return None
    facts = _get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", cache_dir / f"facts_{cik}.json", max_age_days)
    if not facts:
        return None

    parts = []
    for name, concepts, unit in [("revenue", REVENUE_CONCEPTS, "USD"), ("eps", EPS_CONCEPTS, "USD/shares"),
                                 ("net_income", NET_INCOME_CONCEPTS, "USD")]:
        q = _quarterly(facts, concepts, unit)
        parts.append(q.rename(columns={"value": name, "filed": f"filed_{name}", "accn": f"accn_{name}"}).set_index("period_end"))
    df = pd.concat(parts, axis=1).sort_index()
    if df.empty:
        return None
    df.index.name = "period_end"
    df = df.reset_index()
    # The first 10-Q/10-K that reported any of the metrics.
    filed_cols = [c for c in df.columns if c.startswith("filed_")]
    filed = df[filed_cols].apply(pd.to_datetime)
    df["filed"] = filed.min(axis=1)
    df["filed_accn"] = [
        df.at[i, "accn_" + filed.loc[i].idxmin().removeprefix("filed_")] if filed.loc[i].notna().any() else None
        for i in df.index
    ]
    df = df.drop(columns=filed_cols + [c for c in df.columns if c.startswith("accn_")])

    ann = _announcement_dates(cik, cache_dir, max_age_days)
    dates, sources, accns = [], [], []
    for r in df.itertuples():
        hit = ann[(ann["date"] > r.period_end) & (ann["date"] <= r.period_end + pd.Timedelta(days=ANNOUNCE_WINDOW_DAYS))]
        # An 8-K later than the 10-Q/10-K is not the original release.
        if len(hit) and (pd.isna(r.filed) or hit["date"].iloc[0] <= r.filed):
            dates.append(hit["date"].iloc[0]); sources.append("8-K"); accns.append(hit["accn"].iloc[0])
        else:
            dates.append(r.filed); sources.append("10-Q/10-K"); accns.append(r.filed_accn)
    df["announced"], df["announced_source"], df["accession"] = dates, sources, accns
    df["cik"] = cik
    df["revenue_yoy_pct"] = _yoy(df, "revenue")
    return df[["period_end", "announced", "announced_source", "accession", "cik",
               "revenue", "revenue_yoy_pct", "eps", "net_income"]]
