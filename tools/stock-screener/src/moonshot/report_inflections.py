"""Build a self-contained interactive HTML report for `moonshot inflect` crossing/failure candidates."""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

from .fundamentals import cik_map
from .report import _fundamentals_payload, _weekly_series
from .sector import etf_for_sic, load_etf_series

TEMPLATE = Path(__file__).with_name("inflections_template.html")

# multi-agent-trading-desk/data/earnings_call_scores, from
# multi-agent-trading-desk/tools/stock-screener/src/moonshot/report_inflections.py
DEFAULT_MOONSHOT_DIR = Path(__file__).resolve().parents[4] / "data" / "earnings_call_scores"

# SEC SIC industry group 283 (drugs: 2833-2836) plus 8731 (commercial biological research).
PHARMA_SIC_PREFIXES = ("283", "8731")


def _sic(ticker: str, sec_dir: Path, ciks: dict[str, int]) -> str:
    path = sec_dir / f"sub_{ciks.get(ticker, 0)}.json"
    if not path.exists():
        return ""
    return json.loads(path.read_text()).get("sic") or ""


def _is_pharma(ticker: str, sec_dir: Path, ciks: dict[str, int]) -> bool:
    return _sic(ticker, sec_dir, ciks).startswith(PHARMA_SIC_PREFIXES)


def _drop_pharma(df: pd.DataFrame, sec_dir: Path) -> pd.DataFrame:
    if df.empty:
        return df
    ciks = cik_map(sec_dir)
    pharma = df["ticker"].map(lambda t: _is_pharma(t, sec_dir, ciks))
    if pharma.any():
        print(f"Dropping {int(pharma.sum())} pharma crossings: {', '.join(df.loc[pharma, 'ticker'])}",
              file=sys.stderr)
    return df[~pharma].reset_index(drop=True)


def _known_day(crossing_period_end: str, fund: dict | None) -> int:
    """Days since epoch when the market learned of the crossing: the results announcement date,
    falling back to the quarter end if it is unknown."""
    period_end = (pd.Timestamp(crossing_period_end) - pd.Timestamp("1970-01-01")).days
    if fund and period_end in fund["pe"]:
        return fund["a"][fund["pe"].index(period_end)]
    return period_end


def _rise_pct(series: dict | None, known_day: int) -> float | None:
    """Percent change from the last weekly close before `known_day` to the latest close."""
    if not series:
        return None
    before = [p for d, p in zip(series["d"], series["p"]) if d < known_day]
    if not before or before[-1] <= 0:
        return None
    return round((series["p"][-1] / before[-1] - 1) * 100, 1)


def _relative_pct(stock_rise: float | None, sector_rise: float | None) -> float | None:
    if stock_rise is None or sector_rise is None:
        return None
    return round(((1 + stock_rise / 100) / (1 + sector_rise / 100) - 1) * 100, 1)


def _merge_scores(df: pd.DataFrame, scores_csv: Path | None) -> pd.DataFrame:
    if not scores_csv or not scores_csv.exists():
        return df
    scores = pd.read_csv(scores_csv)[["ticker", "score", "rationale", "model"]].rename(
        columns={"score": "nemotron_score", "rationale": "nemotron_rationale", "model": "nemotron_model"}
    )
    print(f"Merging {len(scores)} scores from {scores_csv}", file=sys.stderr)
    return df.merge(scores, on="ticker", how="left")


def _moonshot_payload(tickers: list[str], moonshot_dir: Path) -> dict:
    """Per-quarter rows from the earnings-call moonshot pipeline (multi-agent-trading-desk),
    keyed by ticker: [quarter, proximity, momentum, ambition, runway, tier, stage, one_line].

    Separate from the `nemotron_score`/`nemotron_rationale` columns above, which come from
    `moonshot rate` in this project and score fundamentals only, with no stage or rationale text.
    """
    out = {}
    if not moonshot_dir.exists():
        return out
    for t in tickers:
        path = moonshot_dir / f"{t}_moonshot_v3.jsonl"
        if not path.exists():
            continue
        rows = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            tier = (r.get("proximity_tier") or "").split("_")[0] or "?"
            rows.append([r["quarter"], r["proximity"], r["momentum"], r["ambition"], r["runway"],
                         tier, r["stage"], r.get("one_line", "")])
        if rows:
            rows.sort(key=lambda row: (int(row[0][:4]), int(row[0][5])))
            out[t] = rows
    if out:
        print(f"Merging moonshot pipeline data for {len(out)} of {len(tickers)} tickers from {moonshot_dir}",
              file=sys.stderr)
    return out


def _add_rise_columns(crossings: pd.DataFrame, series: dict, fund: dict, data_dir: Path) -> None:
    sec_dir, ciks = data_dir / "sec", cik_map(data_dir / "sec")
    crossings["sector_etf"] = [etf_for_sic(_sic(t, sec_dir, ciks)) for t in crossings["ticker"]]
    etf_series = {e: load_etf_series(e, data_dir / "sector") for e in sorted(set(crossings["sector_etf"]))}
    rise, sector_rise = [], []
    for r in crossings.itertuples():
        known = _known_day(r.crossing_period_end, fund.get(r.ticker))
        rise.append(_rise_pct(series.get(r.ticker), known))
        sector_rise.append(_rise_pct(etf_series[r.sector_etf], known))
    crossings["rise_since_crossing_pct"] = rise
    crossings["sector_rise_pct"] = sector_rise
    crossings["rise_vs_sector_pct"] = [_relative_pct(a, b) for a, b in zip(rise, sector_rise)]


def build_inflections_report(
    crossings_csv: Path | None, failures_csv: Path | None, data_dir: Path, out: Path,
    crossings_scores: Path | None = None, failures_scores: Path | None = None,
    moonshot_dir: Path | None = None,
) -> Path:
    crossings = pd.DataFrame()
    failures = pd.DataFrame()
    if crossings_csv and crossings_csv.exists():
        crossings = _merge_scores(_drop_pharma(pd.read_csv(crossings_csv), data_dir / "sec"), crossings_scores)
    if failures_csv and failures_csv.exists():
        failures = _merge_scores(pd.read_csv(failures_csv), failures_scores)

    all_tickers = sorted(set(crossings.get("ticker", [])) | set(failures.get("ticker", [])))
    print(f"Building price series and fundamentals for {len(all_tickers)} tickers...", file=sys.stderr)
    series, since = {}, {}
    for t in all_tickers:
        s = _weekly_series(t, data_dir / "prices")
        if s:
            series[t] = s
            since[t] = pd.Timestamp("1970-01-01") + pd.Timedelta(days=s["d"][0])
    fund = _fundamentals_payload(all_tickers, data_dir, since)
    moonshot = _moonshot_payload(all_tickers, moonshot_dir or DEFAULT_MOONSHOT_DIR)
    if not crossings.empty:
        _add_rise_columns(crossings, series, fund, data_dir)

    payload = {
        "crossings": json.loads(crossings.to_json(orient="records")),
        "failures": json.loads(failures.to_json(orient="records")),
        "series": series,
        "fundamentals": fund,
        "moonshot": moonshot,
        "generated": f"{dt.date.today():%Y-%m-%d}",
    }
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("/*DATA*/null", data)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out
