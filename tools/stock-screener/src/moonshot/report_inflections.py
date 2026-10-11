"""Build a self-contained interactive HTML report for `moonshot inflect` crossing/failure candidates."""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

from .fundamentals import cik_map
from .report import _fundamentals_payload, _weekly_series, rise_pct
from .depressed import load_params
from .research import load_research
from .sector import etf_for_sic, load_etf_series, sic_for_ticker
from .valuation import load_valuation

TEMPLATE = Path(__file__).with_name("inflections_template.html")

# multi-agent-trading-desk/data/earnings_call_scores, from
# multi-agent-trading-desk/tools/stock-screener/src/moonshot/report_inflections.py
DEFAULT_MOONSHOT_DIR = Path(__file__).resolve().parents[4] / "data" / "earnings_call_scores"

# SEC SIC industry group 283 (drugs: 2833-2836) plus 8731 (commercial biological research).
PHARMA_SIC_PREFIXES = ("283", "8731")


def _is_pharma(ticker: str, sec_dir: Path, ciks: dict[str, int]) -> bool:
    return sic_for_ticker(ticker, sec_dir, ciks).startswith(PHARMA_SIC_PREFIXES)


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
    crossings["sector_etf"] = [etf_for_sic(sic_for_ticker(t, sec_dir, ciks)) for t in crossings["ticker"]]
    etf_series = {e: load_etf_series(e, data_dir / "sector") for e in sorted(set(crossings["sector_etf"]))}
    rise, sector_rise = [], []
    for r in crossings.itertuples():
        known = _known_day(r.crossing_period_end, fund.get(r.ticker))
        rise.append(rise_pct(series.get(r.ticker), known))
        sector_rise.append(rise_pct(etf_series[r.sector_etf], known))
    crossings["rise_since_crossing_pct"] = rise
    crossings["sector_rise_pct"] = sector_rise
    crossings["rise_vs_sector_pct"] = [_relative_pct(a, b) for a, b in zip(rise, sector_rise)]
    ps_then, ps_now = [], []
    for r in crossings.itertuples():
        valuation = load_valuation(r.ticker, data_dir) or {}
        rows = valuation.get("rows", [])
        at = next((v for v in rows if v["period_end"] >= str(r.crossing_period_end)[:10]), None)
        ps_then.append(at["ps"] if at else None)
        ps_now.append(valuation.get("ps_current") or (rows[-1]["ps"] if rows else None))
    crossings["ps_at_crossing"] = ps_then
    crossings["ps_now"] = ps_now
    crossings["ps_change_pct"] = [
        round((b / a - 1) * 100, 1) if a and b else None for a, b in zip(ps_then, ps_now)
    ]


def _load_snapshots(current_csv: Path) -> list[dict]:
    """Historical runs of the depressed screen (`moonshot depressed --as-of`) saved beside the current CSV."""
    out = []
    for path in sorted(current_csv.parent.glob("depressed_asof_*.csv")):
        params = load_params(path)
        rows = json.loads(pd.read_csv(path).to_json(orient="records"))
        if rows:
            out.append({"key": path.stem.removeprefix("depressed_asof_")[:4], "as_of": params.get("as_of"),
                        "params": params, "rows": rows})
    return out


def build_inflections_report(
    crossings_csv: Path | None, failures_csv: Path | None, data_dir: Path, out: Path,
    crossings_scores: Path | None = None, failures_scores: Path | None = None,
    moonshot_dir: Path | None = None, page: str = "inflections",
) -> Path:
    crossings = pd.DataFrame()
    failures = pd.DataFrame()
    if crossings_csv and crossings_csv.exists():
        listed = pd.read_csv(crossings_csv)
        if page == "inflections":
            listed = _drop_pharma(listed, data_dir / "sec")
        crossings = _merge_scores(listed, crossings_scores)
    if failures_csv and failures_csv.exists():
        failures = _merge_scores(pd.read_csv(failures_csv), failures_scores)

    snapshots = _load_snapshots(crossings_csv) if page == "depressed" and crossings_csv else []
    snapshot_tickers = {t for snap in snapshots for t in (r["ticker"] for r in snap["rows"])}
    all_tickers = sorted(set(crossings.get("ticker", [])) | set(failures.get("ticker", [])) | snapshot_tickers)
    print(f"Building price series and fundamentals for {len(all_tickers)} tickers...", file=sys.stderr)
    series, since = {}, {}
    for t in all_tickers:
        s = _weekly_series(t, data_dir / "prices")
        if s:
            series[t] = s
            since[t] = pd.Timestamp("1970-01-01") + pd.Timedelta(days=s["d"][0])
    fund = _fundamentals_payload(all_tickers, data_dir, since)
    moonshot = _moonshot_payload(all_tickers, moonshot_dir or DEFAULT_MOONSHOT_DIR)
    if not crossings.empty and page == "inflections":
        _add_rise_columns(crossings, series, fund, data_dir)

    spy = load_etf_series("SPY", data_dir / "sector")
    research = {t: r for t in all_tickers if (r := load_research(t, data_dir)) and "skipped" not in r}
    payload = {
        "page": page,
        "benchmark": {"name": "S&P 500 (SPY)", **spy} if spy else {},
        "params": load_params(crossings_csv) if page == "depressed" and crossings_csv else {},
        "snapshots": snapshots,
        "research": research,
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
