"""Build a self-contained interactive HTML report from a scan CSV and cached prices."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

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


def build_report(csv_path: Path, data_dir: Path, multiple: float, out: Path) -> Path:
    df = pd.read_csv(csv_path)
    series = {}
    for t in df["ticker"]:
        s = _weekly_series(t, data_dir / "prices")
        if s:
            series[t] = s
    payload = {
        "rows": json.loads(df.to_json(orient="records")),
        "series": series,
        "multiple": multiple,
        "source": csv_path.name,
        "generated": f"{dt.date.today():%Y-%m-%d}",
    }
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("/*DATA*/null", data)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out
