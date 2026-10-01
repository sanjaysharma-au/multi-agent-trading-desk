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
from .fundamentals import cik_map, load_cashflow, load_fundamentals
from .inflection import find_crossing, find_failure, merge_financials
from .report_inflections import build_inflections_report
from .multibagger import find_best_run, parse_period
from .prices import iter_prices
from .report import build_report, latest_csv
from .scoring import DEFAULT_MODEL, Checkpoint, load_api_key, run as run_scoring, status_lines, write_scores_csv
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
    scan.add_argument("--max-jump", type=float, default=10.0,
                      help="split the price history at any overnight move bigger than this factor "
                           "(bankruptcy-emergence splices etc.); a run can never cross that split (default 10)")
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
    report.add_argument("--fundamentals-top", type=int, default=100,
                        help="fetch SEC fundamentals for the top N rows of the CSV (default 100; 0 = skip)")
    report.add_argument("--no-fundamentals", action="store_true", help="same as --fundamentals-top 0")
    report.add_argument("--scores", type=Path,
                        help="`moonshot rate` output CSV to merge in (default: <csv-stem>_scores.csv next to the scan CSV)")

    rate = sub.add_parser("rate", help="score the top multibaggers' SEC filings with an LLM, blind to price")
    rate.add_argument("csv", nargs="?", type=Path, help="scan CSV (default: latest results/multibaggers_*.csv)")
    rate.add_argument("--top", type=int, default=100, help="score the top N rows by multiple (default 100)")
    rate.add_argument("--model", default=DEFAULT_MODEL, help=f"NVIDIA API model id (default {DEFAULT_MODEL})")
    rate.add_argument("--api-key", help="NVIDIA API key (default: $NEMOTRON_API_KEY / $NVIDIA_API_KEY / .env)")
    rate.add_argument("--data-dir", type=Path, default=Path("data"), help="cache directory (default data/)")
    rate.add_argument("--checkpoint", type=Path, help="checkpoint file (default data/scores/<csv-stem>.json)")
    rate.add_argument("--releases", type=int, default=4, help="recent press releases to include per ticker (default 4)")
    rate.add_argument("--max-attempts", type=int, default=5,
                      help="give up on a ticker after this many failed attempts across runs (default 5)")
    rate.add_argument("--api-retries", type=int, default=3, help="retries within one API call (default 3)")
    rate.add_argument("--sleep", type=float, default=1.0, help="seconds to sleep between tickers (default 1)")
    rate.add_argument("--status", action="store_true", help="print progress and exit without scoring")
    rate.add_argument("--restart", action="store_true", help="wipe the checkpoint and start over")
    rate.add_argument("--retry-failed", action="store_true", help="reset failed tickers to pending first")
    rate.add_argument("--out", type=Path, help="output CSV of scores (default results/<csv-stem>_scores.csv)")

    inflect = sub.add_parser(
        "inflect",
        help="find companies proving their business model for the first time (a Palantir-shaped crossing)",
    )
    inflect.add_argument("--tickers", help="comma-separated tickers instead of the full US universe")
    inflect.add_argument("--limit", type=int, help="only scan the first N universe tickers")
    inflect.add_argument("--min-pre-quarters", type=int, default=4,
                         help="quarters of history to look at before a candidate crossing (default 4)")
    inflect.add_argument("--min-pre-unprofitable-frac", type=float, default=0.75,
                         help="fraction of those quarters that must be pre-profit (default 0.75)")
    inflect.add_argument("--min-post-quarters", type=int, default=2,
                         help="quarters since the crossing required to call it durable (default 2)")
    inflect.add_argument("--max-wobble-quarters", type=int, default=1,
                         help="quarters after the crossing allowed to dip to GAAP-profit-only (default 1)")
    inflect.add_argument("--mode", choices=["crossing", "failure", "both"], default="crossing",
                         help="crossing = Palantir-shaped success, failure = Canoo-shaped, both (default crossing)")
    inflect.add_argument("--failure-min-quarters", type=int, default=8,
                         help="quarters of history required before calling a failure (default 8)")
    inflect.add_argument("--failure-max-revenue-to-burn", type=float, default=0.1,
                         help="cumulative revenue / cumulative losses must stay below this (default 0.1)")
    inflect.add_argument("--max-age", type=float, default=1, help="universe cache max age in days (default 1)")
    inflect.add_argument("--sec-max-age", type=float, default=7, help="SEC data cache max age in days (default 7)")
    inflect.add_argument("--data-dir", type=Path, default=Path("data"), help="cache directory (default data/)")
    inflect.add_argument("--out", type=Path, help="output CSV (default results/inflections_<date>.csv)")
    inflect.add_argument("--top", type=int, default=25, help="rows to print (default 25)")

    ireport = sub.add_parser("inflect-report", help="build an interactive HTML report from `moonshot inflect` output")
    ireport.add_argument("--crossings", type=Path, help="crossings CSV (default: latest results/inflections_*.csv, excluding _failures)")
    ireport.add_argument("--failures", type=Path, help="failures CSV (default: latest results/inflections_*_failures.csv)")
    ireport.add_argument("--crossings-scores", type=Path, help="`moonshot rate` output CSV for the crossings")
    ireport.add_argument("--failures-scores", type=Path, help="`moonshot rate` output CSV for the failures")
    ireport.add_argument("--data-dir", type=Path, default=Path("data"), help="cache directory (default data/)")
    ireport.add_argument("--out", type=Path, default=Path("results/inflections_report.html"), help="output HTML")
    ireport.add_argument("--no-open", action="store_true", help="don't open the report in a browser")
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
            df, args.multiple, window_days, args.min_price, args.min_dollar_volume, args.smooth_days,
            args.max_jump,
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


def _write_report(csv: Path, data_dir: Path, multiple: float, out: Path, open_browser: bool,
                  fundamentals_top: int = 100, scores_csv: Path | None = None) -> None:
    path = build_report(csv, data_dir, multiple, out, fundamentals_top, scores_csv)
    print(f"Report written to {path}", file=sys.stderr)
    if open_browser:
        webbrowser.open(path.resolve().as_uri())


def report(args: argparse.Namespace) -> int:
    csv = args.csv or latest_csv(Path("results"))
    if not csv or not csv.exists():
        print("No scan CSV found; run `moonshot scan` first or pass a CSV path.", file=sys.stderr)
        return 1
    _write_report(csv, args.data_dir, args.multiple, args.out or csv.with_suffix(".html"), not args.no_open,
                  0 if args.no_fundamentals else args.fundamentals_top, args.scores)
    return 0


def _row_tickers(df: pd.DataFrame) -> list[tuple[str, str | None, str | None, str | None]]:
    def col(name: str) -> pd.Series:
        return df[name] if name in df.columns else pd.Series([None] * len(df))

    return list(zip(df["ticker"], col("name"), col("sector"), col("industry")))


def rate(args: argparse.Namespace) -> int:
    csv = args.csv or latest_csv(Path("results"))
    if not csv or not csv.exists():
        print("No scan CSV found; run `moonshot scan` first or pass a CSV path.", file=sys.stderr)
        return 1
    df = pd.read_csv(csv).head(args.top)
    tickers = _row_tickers(df)

    checkpoint_path = args.checkpoint or Path("data/scores") / f"{csv.stem}.json"
    if args.restart:
        Checkpoint(checkpoint_path).clear()
    checkpoint = Checkpoint(checkpoint_path)
    if args.retry_failed:
        n = checkpoint.reset_failed()
        print(f"Reset {n} failed ticker(s) to pending.", file=sys.stderr)

    if args.status:
        print(f"Checkpoint: {checkpoint_path}", file=sys.stderr)
        for line in status_lines(tickers, checkpoint):
            print(line, file=sys.stderr)
        return 0

    api_key = load_api_key(args.api_key)
    if not api_key:
        print("No NVIDIA API key found. Pass --api-key, set NEMOTRON_API_KEY, or add it to a .env file "
              "(NEMOTRON_API_KEY=...).", file=sys.stderr)
        return 1

    print(f"Scoring {len(tickers)} tickers with {args.model} (checkpoint: {checkpoint_path})", file=sys.stderr)
    run_scoring(tickers, args.data_dir / "sec", checkpoint, api_key, args.model, args.max_attempts,
               args.releases, args.sleep, args.api_retries, log_line=lambda s: print(s, file=sys.stderr))

    out = args.out or Path("results") / f"{csv.stem}_scores.csv"
    n = write_scores_csv(checkpoint, out)
    print(f"\n{n} scores written to {out}" if n else "\nNo scores to write yet.", file=sys.stderr)
    for line in status_lines(tickers, checkpoint):
        print(line, file=sys.stderr)
    return 0


def inflect(args: argparse.Namespace) -> int:
    if args.tickers:
        universe = pd.DataFrame({"ticker": [t.strip().upper() for t in args.tickers.split(",") if t.strip()]})
        universe["name"] = None
    else:
        universe = load_universe(args.data_dir / "universe", args.max_age)
    if args.limit:
        universe = universe.head(args.limit)
    meta = universe.set_index("ticker")
    tickers = universe["ticker"].tolist()
    do_crossing, do_failure = args.mode in ("crossing", "both"), args.mode in ("failure", "both")
    what = {"crossing": "a Palantir-shaped crossing", "failure": "a Canoo-shaped failure",
           "both": "a Palantir-shaped crossing and a Canoo-shaped failure"}[args.mode]
    print(f"Checking {len(tickers)} tickers for {what} "
          f"(needs SEC XBRL data; foreign filers and companies without it are skipped)", file=sys.stderr)

    sec_dir = args.data_dir / "sec"
    ciks = cik_map(sec_dir, args.sec_max_age)
    crossings, failures = [], []
    for n, ticker in enumerate(tickers, start=1):
        if n % 250 == 0:
            print(f"  ...{n}/{len(tickers)} checked, {len(crossings)} crossings, {len(failures)} failures",
                  file=sys.stderr)
        try:
            fnd = load_fundamentals(ticker, sec_dir, args.sec_max_age, ciks)
            if fnd is None or fnd.empty:
                continue
            cf = load_cashflow(ticker, sec_dir, args.sec_max_age, ciks)
            merged = merge_financials(fnd, cf)
            if do_crossing:
                hit = find_crossing(merged, args.min_pre_quarters, args.min_post_quarters,
                                    args.max_wobble_quarters, args.min_pre_unprofitable_frac)
                if hit:
                    crossings.append({"ticker": ticker, "name": meta.at[ticker, "name"], **hit})
            if do_failure:
                hit = find_failure(merged, args.failure_min_quarters, args.failure_max_revenue_to_burn)
                if hit:
                    failures.append({"ticker": ticker, "name": meta.at[ticker, "name"], **hit})
        except Exception as exc:
            logging.getLogger(__name__).warning("inflection check failed for %s: %s", ticker, exc)

    stem = f"inflections_{dt.date.today():%Y%m%d}"
    if do_crossing:
        _write_inflection_csv(crossings, "quarters_since_crossing", args.out or Path("results") / f"{stem}.csv",
                              ["ticker", "crossing_period_end", "quarters_since_crossing",
                               "quarters_of_prior_history", "wobble_quarters", "latest_revenue",
                               "latest_net_income", "latest_free_cash_flow"],
                              "crossings (sustained profit + FCF)", args.top)
    if do_failure:
        # In --mode both, crossings and failures have different schemas and can't share one file,
        # so --out (if given) applies only when failure is the sole mode being run.
        failure_out = args.out if args.mode == "failure" and args.out else Path("results") / f"{stem}_failures.csv"
        _write_inflection_csv(failures, "revenue_to_burn_ratio", failure_out,
                              ["ticker", "quarters_of_history", "quarters_with_revenue",
                               "cumulative_revenue", "cumulative_losses", "revenue_to_burn_ratio",
                               "latest_net_income", "latest_free_cash_flow"],
                              "failures (never proved the model)", args.top)
    return 0


def _write_inflection_csv(rows: list[dict], sort_key: str, out: Path, cols: list[str], label: str, top: int) -> None:
    if not rows:
        print(f"No {label} found.", file=sys.stderr)
        return
    result = pd.DataFrame(rows).sort_values(sort_key).reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(f"\n{label}:")
        print(result[cols].head(top).to_string(index=False))
    print(f"{len(result)} {label} written to {out}", file=sys.stderr)


def _latest_inflections_csv(want_failures: bool) -> Path | None:
    files = sorted(Path("results").glob("inflections_*.csv"), key=lambda p: p.stat().st_mtime)
    files = [f for f in files if ("_failures" in f.stem) == want_failures]
    return files[-1] if files else None


def inflect_report(args: argparse.Namespace) -> int:
    crossings = args.crossings or _latest_inflections_csv(want_failures=False)
    failures = args.failures or _latest_inflections_csv(want_failures=True)
    if not crossings and not failures:
        print("No inflections CSV found; run `moonshot inflect` first, or pass --crossings/--failures.",
              file=sys.stderr)
        return 1
    print(f"Using crossings: {crossings or '(none)'}", file=sys.stderr)
    print(f"Using failures: {failures or '(none)'}", file=sys.stderr)
    path = build_inflections_report(crossings, failures, args.data_dir, args.out,
                                    args.crossings_scores, args.failures_scores)
    print(f"Report written to {path}", file=sys.stderr)
    if not args.no_open:
        webbrowser.open(path.resolve().as_uri())
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    if args.command == "scan":
        return scan(args)
    if args.command == "report":
        return report(args)
    if args.command == "rate":
        return rate(args)
    if args.command == "inflect":
        return inflect(args)
    if args.command == "inflect-report":
        return inflect_report(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
