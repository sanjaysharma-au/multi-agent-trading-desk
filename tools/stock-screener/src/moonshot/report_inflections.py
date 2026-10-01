"""Build a self-contained interactive HTML report for `moonshot inflect` crossing/failure candidates."""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

from .report import _fundamentals_payload, _weekly_series

TEMPLATE = Path(__file__).with_name("inflections_template.html")

# multi-agent-trading-desk/data/earnings_call_scores, from
# multi-agent-trading-desk/tools/stock-screener/src/moonshot/report_inflections.py
DEFAULT_MOONSHOT_DIR = Path(__file__).resolve().parents[4] / "data" / "earnings_call_scores"


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


def build_inflections_report(
    crossings_csv: Path | None, failures_csv: Path | None, data_dir: Path, out: Path,
    crossings_scores: Path | None = None, failures_scores: Path | None = None,
    moonshot_dir: Path | None = None,
) -> Path:
    crossings = pd.DataFrame()
    failures = pd.DataFrame()
    if crossings_csv and crossings_csv.exists():
        crossings = _merge_scores(pd.read_csv(crossings_csv), crossings_scores)
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
