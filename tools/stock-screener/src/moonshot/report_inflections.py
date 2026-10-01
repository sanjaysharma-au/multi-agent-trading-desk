"""Build a self-contained interactive HTML report for `moonshot inflect` crossing/failure candidates."""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

from .report import _fundamentals_payload, _weekly_series

TEMPLATE = Path(__file__).with_name("inflections_template.html")


def _merge_scores(df: pd.DataFrame, scores_csv: Path | None) -> pd.DataFrame:
    if not scores_csv or not scores_csv.exists():
        return df
    scores = pd.read_csv(scores_csv)[["ticker", "score", "rationale", "model"]].rename(
        columns={"score": "nemotron_score", "rationale": "nemotron_rationale", "model": "nemotron_model"}
    )
    print(f"Merging {len(scores)} scores from {scores_csv}", file=sys.stderr)
    return df.merge(scores, on="ticker", how="left")


def build_inflections_report(
    crossings_csv: Path | None, failures_csv: Path | None, data_dir: Path, out: Path,
    crossings_scores: Path | None = None, failures_scores: Path | None = None,
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

    payload = {
        "crossings": json.loads(crossings.to_json(orient="records")),
        "failures": json.loads(failures.to_json(orient="records")),
        "series": series,
        "fundamentals": fund,
        "generated": f"{dt.date.today():%Y-%m-%d}",
    }
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("/*DATA*/null", data)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out
