import argparse
import statistics
import sys
from datetime import datetime
from pathlib import Path

from agents.model_designer import (
    CONTRACT_VERSION,
    CONTRACT_VERSION_DAYTRADE,
    CONTRACT_VERSION_EARNINGS,
    CONTRACT_VERSION_OVERNIGHT,
    CONTRACT_VERSION_SWING,
    generate_model_script,
    DEFAULT_BACKEND,
)
from wfo.run_iteration import DAILY_DATA_DIR, DATA_DIR, EARNINGS_DIR, ITERATIONS_DIR, NEWS_DIR, run_iteration
from wfo.schema import IterationResults, load_results

MAX_ITERATIONS = 10
CUMULATIVE_ROI_ACCEPT_THRESHOLD = 4.00
DEFAULT_TRAIN_MONTHS = 3
DEFAULT_PREDICT_MONTHS = 1
DEFAULT_GAP_DAYS = 2
TARGET_MAX_TRADES_PER_MONTH = 100
TARGET_MAX_TRADES_PER_MONTH_SWING = 15
TARGET_MAX_TRADES_PER_MONTH_DAYTRADE = 21
TARGET_MAX_TRADES_PER_MONTH_OVERNIGHT = 21
TARGET_MAX_TRADES_PER_MONTH_EARNINGS = 10


def chain_roi(period_rois: list[float]) -> float:
    equity = 1.0
    for roi in period_rois:
        equity *= 1 + roi
    return equity - 1


def sharpe_ratio(period_returns: list[float], periods_per_year: float) -> float:
    if len(period_returns) < 2 or periods_per_year <= 0:
        return 0.0
    mean = statistics.fmean(period_returns)
    stdev = statistics.stdev(period_returns)
    if stdev == 0:
        return 0.0
    return (mean / stdev) * (periods_per_year ** 0.5)


def chained_max_drawdown(period_returns: list[float]) -> float:
    equity = peak = 1.0
    worst = 0.0
    for r in period_returns:
        equity *= 1 + r
        peak = max(peak, equity)
        worst = min(worst, (equity - peak) / peak)
    return worst


def diagnose(results: IterationResults) -> dict:
    if not results.windows:
        return {
            "avg_roi": 0.0, "median_roi": 0.0, "cumulative_roi": 0.0, "cumulative_net_roi": -1.0,
            "avg_f1": 0.0, "avg_buy_hold_roi": 0.0, "cumulative_buy_hold_roi": 0.0,
            "total_trades": 0, "trades_per_month": 0.0, "num_windows": 0, "collapsed_windows": 0,
            "sharpe_net": 0.0, "sharpe_buy_hold": 0.0,
            "worst_window_drawdown": 0.0, "chained_max_drawdown": 0.0, "chained_max_drawdown_buy_hold": 0.0,
        }
    rois = [w.roi for w in results.windows]
    net_rois = [w.net_roi for w in results.windows]
    f1s = [w.f1 for w in results.windows]
    bh_rois = [w.buy_hold_roi for w in results.windows]
    collapsed = sum(1 for w in results.windows if w.f1 == 0.0)
    total_trades = sum(w.num_trades for w in results.windows)
    total_months = len(results.windows) * results.predict_months
    periods_per_year = 12 / results.predict_months if results.predict_months else 0
    return {
        "avg_roi": statistics.fmean(rois),
        "median_roi": statistics.median(rois),
        "cumulative_roi": chain_roi(rois),
        "cumulative_net_roi": chain_roi(net_rois),
        "avg_f1": statistics.fmean(f1s),
        "avg_buy_hold_roi": statistics.fmean(bh_rois),
        "cumulative_buy_hold_roi": chain_roi(bh_rois),
        "total_trades": total_trades,
        "trades_per_month": total_trades / total_months if total_months else 0.0,
        "num_windows": len(results.windows),
        "collapsed_windows": collapsed,
        # Risk-adjusted metrics: a partially-invested, selective strategy can
        # legitimately "lose" on raw cumulative ROI against buy&hold during a
        # strong bull run while still being a genuinely better risk/reward
        # proposition — these are what actually judge that.
        "sharpe_net": sharpe_ratio(net_rois, periods_per_year),
        "sharpe_buy_hold": sharpe_ratio(bh_rois, periods_per_year),
        "worst_window_drawdown": min(w.max_drawdown for w in results.windows),
        "chained_max_drawdown": chained_max_drawdown(net_rois),
        "chained_max_drawdown_buy_hold": chained_max_drawdown(bh_rois),
    }


LONG_ONLY_DIRECTIVE = (
    "This strategy must be LONG-ONLY: never take short positions. Set the module-level "
    "`LONG_ONLY = True` and apply it by clipping the position to `max(position, 0)` before "
    "computing strategy_returns, net_roi, and num_trades, per the contract's long_only support."
)


def build_instructions(
    best_approach: str | None,
    best_diag: dict | None,
    last_approach: str | None,
    last_diag: dict | None,
    style: str = "intraday",
    last_failure: str | None = None,
    avoid_approaches: list[str] | None = None,
) -> str:
    target_max_trades = {
        "swing": TARGET_MAX_TRADES_PER_MONTH_SWING,
        "daytrade": TARGET_MAX_TRADES_PER_MONTH_DAYTRADE,
        "overnight": TARGET_MAX_TRADES_PER_MONTH_OVERNIGHT,
        "earnings": TARGET_MAX_TRADES_PER_MONTH_EARNINGS,
    }.get(style, TARGET_MAX_TRADES_PER_MONTH)
    style_desc = {
        "swing": "swing-trading daily-bar signal prediction (positions held days to a couple of weeks)",
        "daytrade": "day-trade (open-to-close) daily-bar signal prediction — decide at the open, "
                    "always exit at that same day's close, no overnight holding",
        "overnight": "overnight (close-to-open) daily-bar signal prediction — decide at today's close, "
                     "always exit at the next day's open, holding overnight",
        "earnings": "post-earnings-announcement-drift (PEAD) signal prediction, pooled across multiple "
                    "tickers' earnings events — decide at the reaction-day close (first close after the "
                    "report is public), hold for a fixed number of trading days",
    }.get(style, "intraday minute-bar signal prediction")

    failure_warning = ""
    if last_failure is not None:
        failure_warning = (
            f"WARNING: your most recent attempt FAILED and produced no usable results: "
            f"\"{last_failure}\". If this was a '0 usable windows' failure, it almost always means "
            "added filtering/validation logic (e.g. a volatility-regime filter, an internal "
            "train/validation split, a stricter minimum-row threshold) ate far more rows than "
            "expected relative to how few rows a several-week predict window actually has after "
            "indicator warmup — double-check your row-count math against realistic window sizes "
            "before adding new filtering steps. Do not repeat whatever caused this. "
        )

    diversity_note = ""
    if avoid_approaches:
        listed = " | ".join(avoid_approaches[-6:])
        diversity_note = (
            "Approaches already tried in this session — do NOT repeat any of these, or a close "
            f"variant of one; pick a meaningfully different model type, feature set, or prediction "
            f"framing than every one of them: {listed}. "
        )

    if best_diag is None:
        return (
            f"{LONG_ONLY_DIRECTIVE} {failure_warning}{diversity_note}Design an initial model for "
            f"{style_desc} under this long-only constraint. Pick your own feature set, model, and "
            "prediction framing."
        )

    parts = [LONG_ONLY_DIRECTIVE]
    if failure_warning:
        parts.append(failure_warning)
    parts.append(
        f"BEST RESULT SO FAR ({best_approach}): cumulative NET ROI={best_diag['cumulative_net_roi']:+.3f} "
        f"(gross {best_diag['cumulative_roi']:+.3f} vs buy-and-hold {best_diag['cumulative_buy_hold_roi']:+.3f}), "
        f"avg F1={best_diag['avg_f1']:.3f}, at {best_diag['trades_per_month']:.0f} trades/month. Your job is "
        "to beat this specifically — build on what's working rather than abandoning it for something "
        "unrelated unless you have a specific reason to believe it will do better."
    )

    is_regression = last_diag is not best_diag and last_diag["cumulative_net_roi"] < best_diag["cumulative_net_roi"]
    if is_regression:
        parts.append(
            f"Your most recent attempt ({last_approach}) REGRESSED from the best: net ROI "
            f"{last_diag['cumulative_net_roi']:+.3f} at {last_diag['trades_per_month']:.0f} trades/month. "
            "Don't continue further in that direction — return toward the best result's approach and "
            "trade frequency, and refine from there instead of drifting further away from it."
        )

    if last_diag["cumulative_roi"] <= last_diag["cumulative_buy_hold_roi"]:
        parts.append(
            "The most recent attempt did not beat simply buying and holding the stock over the same "
            "periods — that's the real bar to clear, not just positive ROI."
        )
    if last_diag["cumulative_net_roi"] < last_diag["cumulative_roi"] - 0.05:
        parts.append(
            f"Note: the most recent attempt's net-of-fees cumulative ROI "
            f"({last_diag['cumulative_net_roi']:+.3f}) is meaningfully worse than gross "
            f"({last_diag['cumulative_roi']:+.3f}) — it took {last_diag['trades_per_month']:.0f} "
            f"trades/month. At a flat $1/trade fee, anything much above {target_max_trades} "
            "trades/month tends to get dominated by fees almost regardless of edge. Don't just trade "
            f"'a bit less' — substantially lengthen the horizon (e.g. 4-8x longer) and/or raise the "
            f"confidence/magnitude threshold so the flat/no-trade case fires much more often, aiming "
            f"for roughly {target_max_trades // 2}-{target_max_trades} "
            "trades/month, not several hundred."
        )
    if last_diag["collapsed_windows"] > last_diag["num_windows"] / 2:
        parts.append(
            "More than half of the most recent attempt's windows collapsed to a constant prediction — "
            "address class imbalance or add stronger discriminative features rather than just retuning."
        )
    parts.append(
        "Improve on the BEST result above: aim for higher and more consistent NET ROI across windows "
        "(not just gross), a higher CUMULATIVE (compounded) return across the whole walk-forward "
        "stretch, and a trade frequency close to what's already working unless you have good reason "
        "to move it."
    )
    return " ".join(parts)


def print_iteration_report(
    iteration_num: int, max_iterations: int, results: IterationResults, diag: dict, prev_diag: dict | None
):
    print(f"\n[iteration {iteration_num}/{max_iterations}]  ticker: {results.ticker}")
    print(f"  approach: {results.approach}")
    print(f"  prediction target: {results.prediction_target}")
    if prev_diag is not None:
        print(f"  cumulative ROI:     {prev_diag['cumulative_roi']:+.3f} -> {diag['cumulative_roi']:+.3f}  (cumulative buy&hold: {diag['cumulative_buy_hold_roi']:+.3f})")
        print(f"  cumulative net ROI: {prev_diag['cumulative_net_roi']:+.3f} -> {diag['cumulative_net_roi']:+.3f}  ($1/trade fee, $10k capital)")
        print(f"  avg ROI:            {prev_diag['avg_roi']:+.3f} -> {diag['avg_roi']:+.3f}  (avg buy&hold: {diag['avg_buy_hold_roi']:+.3f})")
        print(f"  avg F1:             {prev_diag['avg_f1']:.3f} -> {diag['avg_f1']:.3f}")
    else:
        print(f"  cumulative ROI:     {diag['cumulative_roi']:+.3f}  (cumulative buy&hold: {diag['cumulative_buy_hold_roi']:+.3f})")
        print(f"  cumulative net ROI: {diag['cumulative_net_roi']:+.3f}  ($1/trade fee, $10k capital)")
        print(f"  avg ROI:            {diag['avg_roi']:+.3f}  (avg buy&hold: {diag['avg_buy_hold_roi']:+.3f})")
        print(f"  avg F1:             {diag['avg_f1']:.3f}")
    print(f"  windows: {diag['num_windows']} ({diag['collapsed_windows']} collapsed to constant prediction)")
    print(f"  total trades: {diag['total_trades']}  ({diag['trades_per_month']:.0f}/month)")
    print(
        f"  Sharpe (net):       {diag['sharpe_net']:+.2f}  (buy&hold: {diag['sharpe_buy_hold']:+.2f})"
    )
    print(
        f"  max drawdown:       worst window {diag['worst_window_drawdown']:+.1%}  |  "
        f"chained {diag['chained_max_drawdown']:+.1%}  (buy&hold chained: {diag['chained_max_drawdown_buy_hold']:+.1%})"
    )
    for w in results.windows:
        print(
            f"    window {w.index:2d}  f1={w.f1:.3f}  acc={w.accuracy:.3f}  trades={w.num_trades:4d}  "
            f"roi={w.roi:+.3f}  net_roi={w.net_roi:+.3f}  buy&hold={w.buy_hold_roi:+.3f}  maxdd={w.max_drawdown:+.3f}"
        )


def run_loop(
    tickers: list[str],
    train_months: int,
    predict_months: int,
    gap_days: int,
    holdout_months: int = 0,
    holdout_tickers: list[str] | None = None,
    style: str = "intraday",
    independent: bool = False,
    accept_threshold: float = CUMULATIVE_ROI_ACCEPT_THRESHOLD,
    max_iterations: int = MAX_ITERATIONS,
    backend: str = DEFAULT_BACKEND,
) -> None:
    sys.stdout.reconfigure(line_buffering=True)
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    holdout_tickers = holdout_tickers or []
    last_approach, last_diag = None, None
    best_approach, best_diag = None, None
    best_script_path = None

    if independent:
        print("Independent mode: each iteration is a fresh, unguided design — no feedback from "
              "prior attempts, to avoid adaptively curve-fitting to this specific dataset.")

    data_dir = DAILY_DATA_DIR if style in ("swing", "daytrade", "overnight", "earnings") else DATA_DIR
    contract_version = {
        "swing": CONTRACT_VERSION_SWING,
        "daytrade": CONTRACT_VERSION_DAYTRADE,
        "overnight": CONTRACT_VERSION_OVERNIGHT,
        "earnings": CONTRACT_VERSION_EARNINGS,
    }.get(style, CONTRACT_VERSION)

    print(f"Style: {style}  (data dir: {data_dir})")
    if style == "earnings":
        print(f"Pooled search tickers (every iteration uses all of them together): {', '.join(tickers)}")
    else:
        print(f"Rotating search tickers: {', '.join(tickers)}")
    if holdout_months:
        print(f"Also reserving the trailing {holdout_months} month(s) of each search ticker as a "
              "time-based blind holdout.")
    if holdout_tickers:
        print(f"Reserving entire tickers never used in search: {', '.join(holdout_tickers)} "
              "(full history available for a large-sample blind test afterward).")

    last_failure = None
    tried_approaches: list[str] = []
    for i in range(1, max_iterations + 1):
        ticker = ",".join(tickers) if style == "earnings" else tickers[(i - 1) % len(tickers)]
        if independent:
            # Deliberately no PERFORMANCE feedback from prior iterations — each
            # design is unguided by results, so the search can't adaptively
            # curve-fit to this specific dataset's noise across rounds.
            # `last_failure` (an engineering-correctness signal, not a
            # performance one) and the list of approaches already tried (so
            # the model doesn't just repeat itself with no memory) are the
            # only things carried forward.
            instructions = build_instructions(
                None, None, None, None, style=style, last_failure=last_failure,
                avoid_approaches=tried_approaches,
            )
        else:
            instructions = build_instructions(
                best_approach, best_diag, last_approach, last_diag, style=style, last_failure=last_failure
            )
        tag_label = "POOLED" if style == "earnings" else ticker
        tag = f"{run_id}_{tag_label}_iter{i}"

        try:
            script_path = generate_model_script(instructions, iteration_tag=tag, style=style, backend=backend)
        except Exception as e:
            last_failure = f"generation error: {e}"
            print(f"\n[iteration {i}/{max_iterations}] ticker: {ticker}  FAILED: {last_failure}")
            continue

        ok, outcome = run_iteration(
            script_path, ticker, train_months, predict_months, gap_days,
            iteration_id=tag, contract_version=contract_version, holdout_months=holdout_months,
            data_dir=data_dir, news_dir=NEWS_DIR if style == "intraday" else None,
            earnings_dir=EARNINGS_DIR if style == "earnings" else None,
        )
        if not ok:
            last_failure = outcome
            print(f"\n[iteration {i}/{max_iterations}] ticker: {ticker}  FAILED: {outcome}")
            continue

        last_failure = None
        results = load_results(ITERATIONS_DIR / outcome / "results.json")
        new_diag = diagnose(results)
        print_iteration_report(i, max_iterations, results, new_diag, last_diag)

        last_approach, last_diag = results.approach, new_diag
        tried_approaches.append(results.approach)
        if best_diag is None or new_diag["cumulative_net_roi"] > best_diag["cumulative_net_roi"]:
            best_approach, best_diag = last_approach, last_diag
            best_script_path = script_path
            print(
                f"  ** new best (net ROI {best_diag['cumulative_net_roi']:+.3f}, "
                f"Sharpe {best_diag['sharpe_net']:+.2f} vs buy&hold {best_diag['sharpe_buy_hold']:+.2f}) **"
            )

        if new_diag["cumulative_net_roi"] >= accept_threshold:
            print(
                f"\nACCEPTED at iteration {i}: cumulative NET ROI {new_diag['cumulative_net_roi']:.3f} "
                f">= threshold {accept_threshold} "
                f"(gross was {new_diag['cumulative_roi']:.3f})"
            )
            _print_holdout_hint(
                holdout_months, holdout_tickers, best_script_path, train_months, predict_months, gap_days, data_dir
            )
            return

    _print_holdout_hint(
        holdout_months, holdout_tickers, best_script_path, train_months, predict_months, gap_days, data_dir
    )
    print(
        f"\nIteration cap ({max_iterations}) reached without meeting cumulative NET ROI threshold "
        f"{accept_threshold}. Best found: net ROI {best_diag['cumulative_net_roi']:+.3f}, "
        f"Sharpe {best_diag['sharpe_net']:+.2f} (buy&hold Sharpe {best_diag['sharpe_buy_hold']:+.2f}) "
        f"({best_approach})"
    )


def _print_holdout_hint(
    holdout_months, holdout_tickers, best_script_path, train_months, predict_months, gap_days, data_dir
):
    if best_script_path is None:
        return
    if holdout_months:
        print(
            f"\nTo blind-test the best model against its reserved {holdout_months}-month time holdout, "
            f"rerun it with --holdout-months 0 and inspect the trailing windows:\n"
            f"  PYTHONPATH=src python -m wfo.run_iteration {best_script_path} --ticker <search ticker> "
            f"--train-months {train_months} --predict-months {predict_months} --gap-days {gap_days} "
            f"--holdout-months 0 --data-dir {data_dir}"
        )
    for ht in holdout_tickers:
        print(
            f"\nTo blind-test the best model against the fully-reserved ticker {ht} (full history, "
            f"never seen during search):\n"
            f"  PYTHONPATH=src python -m wfo.run_iteration {best_script_path} --ticker {ht} "
            f"--train-months {train_months} --predict-months {predict_months} --gap-days {gap_days} "
            f"--holdout-months 0 --data-dir {data_dir}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", required=True, help="Comma-separated tickers to rotate through during search, e.g. AAPL,NVDA,AMZN (for --style earnings, these are POOLED together every iteration instead of rotated)")
    parser.add_argument("--holdout-tickers", default="", help="Comma-separated tickers to reserve entirely (never used in search), e.g. MSFT,SPY")
    parser.add_argument("--style", choices=["intraday", "swing", "daytrade", "overnight", "earnings"], default="intraday", help="intraday = minute bars, short horizons; swing = daily bars, multi-day holding periods; daytrade = daily bars, enter at open exit at close same day; overnight = daily bars, enter at close exit at next open; earnings = post-earnings-drift, daily bars, events pooled across all --tickers")
    parser.add_argument("--train-months", type=int, default=None, help="Defaults: 3 for intraday, 10 for swing/daytrade, 60 for earnings")
    parser.add_argument("--predict-months", type=int, default=None, help="Defaults: 1 for intraday, 2 for swing/daytrade, 24 for earnings")
    parser.add_argument("--gap-days", type=int, default=None, help="Defaults: 2 for intraday, 5 for swing/daytrade, 10 for earnings")
    parser.add_argument(
        "--holdout-months", type=int, default=0,
        help="Additionally reserve this many trailing months from every search ticker's WFO windows.",
    )
    parser.add_argument(
        "--independent", action="store_true",
        help="Each iteration is a fresh, unguided design with no feedback from prior attempts — "
             "avoids the search adaptively curve-fitting to this dataset across rounds.",
    )
    parser.add_argument(
        "--accept-threshold", type=float, default=CUMULATIVE_ROI_ACCEPT_THRESHOLD,
        help=f"Cumulative NET ROI that stops the loop early (default {CUMULATIVE_ROI_ACCEPT_THRESHOLD}).",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=MAX_ITERATIONS,
        help=f"Number of iterations to run (default {MAX_ITERATIONS}).",
    )
    parser.add_argument(
        "--backend", choices=["nemotron", "claude"], default=DEFAULT_BACKEND,
        help=f"LLM backend to use (default {DEFAULT_BACKEND}).",
    )
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    holdout_tickers = [t.strip() for t in args.holdout_tickers.split(",") if t.strip()]

    if args.style == "earnings":
        train_months = args.train_months if args.train_months is not None else 60
        predict_months = args.predict_months if args.predict_months is not None else 24
        gap_days = args.gap_days if args.gap_days is not None else 10
    elif args.style in ("swing", "daytrade", "overnight"):
        train_months = args.train_months if args.train_months is not None else 10
        predict_months = args.predict_months if args.predict_months is not None else 2
        gap_days = args.gap_days if args.gap_days is not None else 5
    else:
        train_months = args.train_months if args.train_months is not None else DEFAULT_TRAIN_MONTHS
        predict_months = args.predict_months if args.predict_months is not None else DEFAULT_PREDICT_MONTHS
        gap_days = args.gap_days if args.gap_days is not None else DEFAULT_GAP_DAYS

    run_loop(
        tickers, train_months, predict_months, gap_days,
        args.holdout_months, holdout_tickers, style=args.style,
        independent=args.independent, accept_threshold=args.accept_threshold,
        max_iterations=args.max_iterations, backend=args.backend,
    )


if __name__ == "__main__":
    main()
