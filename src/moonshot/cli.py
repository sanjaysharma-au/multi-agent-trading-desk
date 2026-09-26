"""Command-line entrypoint: `moonshot scan`."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import webbrowser
from pathlib import Path

import pandas as pd

from .enrich import enrich
from .multibagger import find_best_run, parse_period
from .prices import iter_prices
from .report import build_report, latest_csv
from .universe import load_universe


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="moonshot", description="Find US stocks that were multibaggers.")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="scan for past multibaggers")
    scan.add_argument("--multiple", type=float, default=5.0, help="minimum rise, e.g. 5 = 5x (default 5)")
    scan.add_argument("--window", default="3y", help="max time for the rise: 3y, 18m, 100d (default 3y)")
    scan.add_argument("--lookback", default="10y", help="yfinance history period: 5y, 10y, max (default 10y)")
    scan.add_argument("--min-price", type=float, default=1.0, help="min raw price at run start (default 1)")
    scan.add_argument("--min-dollar-volume", type=float, default=100_000, help="min 20d avg $ volume at start")
    scan.add_argument("--smooth-days", type=int, default=5, help="rolling-median smoothing window (default 5)")
    scan.add_argument("--tickers", help="comma-separated tickers instead of the full US universe")
    scan.add_argument("--limit", type=int, help="only scan the first N universe tickers")
    scan.add_argument("--enrich", action="store_true", help="add sector/industry/market cap for hits")
    scan.add_argument("--max-age", type=float, default=1, help="cache max age in days (default 1)")
    scan.add_argument("--data-dir", type=Path, default=Path("data"), help="cache directory (default data/)")
    scan.add_argument("--out", type=Path, help="output CSV (default results/multibaggers_<date>.csv)")
    scan.add_argument("--top", type=int, default=25, help="rows to print (default 25)")
    scan.add_argument("--report", action="store_true", help="also build the HTML report and open it")

    report = sub.add_parser("report", help="build an interactive HTML report from a scan CSV")
    report.add_argument("csv", nargs="?", type=Path, help="scan CSV (default: latest results/multibaggers_*.csv)")
    report.add_argument("--multiple", type=float, default=5.0, help="threshold used in the scan (default 5)")
    report.add_argument("--data-dir", type=Path, default=Path("data"), help="cache directory (default data/)")
    report.add_argument("--out", type=Path, help="output HTML (default: CSV path with .html)")
    report.add_argument("--no-open", action="store_true", help="don't open the report in a browser")
    return parser


def scan(args: argparse.Namespace) -> int:
    window_days = parse_period(args.window)

    if args.tickers:
        universe = pd.DataFrame({"ticker": [t.strip().upper() for t in args.tickers.split(",") if t.strip()]})
        universe["name"] = None
        universe["exchange"] = None
    else:
        universe = load_universe(args.data_dir / "universe", args.max_age)
    if args.limit:
        universe = universe.head(args.limit)
    meta = universe.set_index("ticker")
    tickers = universe["ticker"].tolist()
    print(f"Scanning {len(tickers)} tickers for >= {args.multiple:g}x within {args.window} "
          f"({window_days} trading days), lookback {args.lookback}", file=sys.stderr)

    rows = []
    for n, (ticker, df) in enumerate(
        iter_prices(tickers, args.data_dir / "prices", args.lookback, args.max_age), start=1
    ):
        if n % 250 == 0:
            print(f"  ...{n}/{len(tickers)} processed, {len(rows)} hits", file=sys.stderr)
        run = find_best_run(
            df, args.multiple, window_days, args.min_price, args.min_dollar_volume, args.smooth_days
        )
        if run:
            rows.append({"ticker": ticker, "name": meta.at[ticker, "name"], "exchange": meta.at[ticker, "exchange"], **run})

    if not rows:
        print("No multibaggers found.", file=sys.stderr)
        return 0

    result = pd.DataFrame(rows).sort_values("multiple", ascending=False).reset_index(drop=True)
    if args.enrich:
        print(f"Enriching {len(result)} hits...", file=sys.stderr)
        extra = pd.DataFrame([enrich(t) for t in result["ticker"]])
        result = pd.concat([result, extra], axis=1)

    out = args.out or Path("results") / f"multibaggers_{dt.date.today():%Y%m%d}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)

    cols = ["ticker", "multiple", "start_date", "peak_date", "trading_days_to_peak", "current_multiple",
            "last_vs_peak_pct"]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(result[cols].head(args.top).to_string(index=False))
    print(f"\n{len(result)} multibaggers written to {out}", file=sys.stderr)
    if args.report:
        _write_report(out, args.data_dir, args.multiple, out.with_suffix(".html"), open_browser=True)
    return 0


def _write_report(csv: Path, data_dir: Path, multiple: float, out: Path, open_browser: bool) -> None:
    path = build_report(csv, data_dir, multiple, out)
    print(f"Report written to {path}", file=sys.stderr)
    if open_browser:
        webbrowser.open(path.resolve().as_uri())


def report(args: argparse.Namespace) -> int:
    csv = args.csv or latest_csv(Path("results"))
    if not csv or not csv.exists():
        print("No scan CSV found; run `moonshot scan` first or pass a CSV path.", file=sys.stderr)
        return 1
    _write_report(csv, args.data_dir, args.multiple, args.out or csv.with_suffix(".html"), not args.no_open)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    if args.command == "scan":
        return scan(args)
    if args.command == "report":
        return report(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
