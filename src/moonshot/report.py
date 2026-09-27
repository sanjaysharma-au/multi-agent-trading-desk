"""Build a self-contained interactive HTML report from a scan CSV and cached prices."""

from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from .fundamentals import cik_map, exhibit_links, filing_index_url, load_fundamentals

log = logging.getLogger(__name__)

TEMPLATE = Path(__file__).with_name("report_template.html")


def latest_csv(results_dir: Path) -> Path | None:
    files = sorted(results_dir.glob("multibaggers_*.csv"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _weekly_series(ticker: str, prices_dir: Path) -> dict | None:
    """Weekly adjusted closes from the freshest cached parquet for `ticker`."""
    paths = sorted(prices_dir.glob(f"*/{ticker}.parquet"), key=lambda p: p.stat().st_mtime)
    if not paths:
        return None
    s = pd.read_parquet(paths[-1])["Adj Close"].dropna()
    s = s.resample("W-FRI").last().dropna()
    if s.empty:
        return None
    days = (s.index - pd.Timestamp("1970-01-01")).days
    return {"d": [int(d) for d in days], "p": [float(f"{v:.5g}") for v in s]}


def _days(s: pd.Series) -> list[int]:
    return [int(d) for d in (pd.to_datetime(s) - pd.Timestamp("1970-01-01")).dt.days]


def _num(s: pd.Series, digits: int = 5) -> list[float | None]:
    return [float(f"{v:.{digits}g}") if pd.notna(v) else None for v in s]


def _links(r, sec_dir: Path) -> dict:
    """Press release links for an 8-K announcement; just the filing for a 10-Q/10-K fallback."""
    if not isinstance(r.accession, str):
        return {}
    if r.announced_source == "8-K":
        return exhibit_links(r.cik, r.accession, sec_dir)
    return {"filing": filing_index_url(r.cik, r.accession)}


def _fundamentals_payload(tickers: list[str], data_dir: Path, since: dict[str, pd.Timestamp]) -> dict:
    """Quarterly fundamentals and press-release links for the given tickers only, keyed by ticker.

    Links are resolved only for quarters announced on or after `since[ticker]` (the chart's first date).
    """
    sec_dir = data_dir / "sec"
    ciks = cik_map(sec_dir)
    out = {}
    for n, t in enumerate(tickers, start=1):
        if n % 25 == 0:
            print(f"  ...fundamentals {n}/{len(tickers)} (first run fetches press-release links, ~40/stock)", file=sys.stderr)
        try:
            f = load_fundamentals(t, sec_dir, ciks=ciks)
            if f is None or f.empty:
                continue
            f = f.dropna(subset=["announced"])
            if t in since:
                f = f[f["announced"] >= since[t] - pd.Timedelta(days=100)]
            if f.empty:
                continue
            links = []
            for r in f.itertuples():
                try:
                    links.append(_links(r, sec_dir))
                except Exception as exc:
                    log.warning("could not resolve links for %s %s: %s", t, r.accession, exc)
                    links.append({})
            out[t] = {
                "a": _days(f["announced"]),
                "pe": _days(f["period_end"]),
                "src": f["announced_source"].tolist(),
                "revenue": _num(f["revenue"]),
                "revenue_yoy_pct": _num(f["revenue_yoy_pct"], 4),
                "eps": _num(f["eps"], 4),
                "net_income": _num(f["net_income"]),
                "links": links,
            }
        except Exception as exc:
            log.warning("fundamentals failed for %s: %s", t, exc)
    return out


def _merge_scores(df: pd.DataFrame, csv_path: Path, scores_csv: Path | None) -> pd.DataFrame:
    """Left-join `moonshot rate` output (ticker, score, rationale, model) onto the scan results."""
    path = scores_csv or csv_path.with_name(f"{csv_path.stem}_scores.csv")
    if not path.exists():
        return df
    scores = pd.read_csv(path)[["ticker", "score", "rationale", "model"]].rename(
        columns={"score": "nemotron_score", "rationale": "nemotron_rationale", "model": "nemotron_model"}
    )
    print(f"Merging {len(scores)} fundamentals scores from {path}", file=sys.stderr)
    return df.merge(scores, on="ticker", how="left")


def build_report(csv_path: Path, data_dir: Path, multiple: float, out: Path, fundamentals_top: int = 100,
                 scores_csv: Path | None = None) -> Path:
    """`fundamentals_top`: fetch SEC data for the top N rows of the CSV (0 disables)."""
    df = pd.read_csv(csv_path)
    df = _merge_scores(df, csv_path, scores_csv)
    series = {}
    since = {}
    for t in df["ticker"]:
        s = _weekly_series(t, data_dir / "prices")
        if s:
            series[t] = s
            since[t] = pd.Timestamp("1970-01-01") + pd.Timedelta(days=s["d"][0])
    fund = {}
    if fundamentals_top > 0:
        top = df["ticker"].head(fundamentals_top).tolist()
        print(f"Loading SEC fundamentals for the top {len(top)} of {len(df)} multibaggers...", file=sys.stderr)
        fund = _fundamentals_payload(top, data_dir, since)
    payload = {
        "rows": json.loads(df.to_json(orient="records")),
        "series": series,
        "fundamentals": fund,
        "fundamentals_top": fundamentals_top,
        "multiple": multiple,
        "source": csv_path.name,
        "generated": f"{dt.date.today():%Y-%m-%d}",
    }
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("/*DATA*/null", data)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out
