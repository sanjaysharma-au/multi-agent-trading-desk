import argparse
import pickle
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

from wfo.schema import IterationResults, WindowResult, write_results
from wfo.windows import walk_forward_windows

# ---------------------------------------------------------------------------
# LONG-ONLY, pooled PEAD strategy using a kernel Support Vector Classifier
# (RBF-kernel SVC with probability estimates, standardized features) to
# classify whether an earnings event's forward return will be positive.
#
# Feature set (all knowable by reaction-day close, all causal/local to the
# ticker's own price+earnings history):
#   - surprise_pct            : reported EPS surprise (%)
#   - surprise_delta          : change in surprise_pct vs THIS TICKER'S OWN
#                                previous reported quarter (causal
#                                quarter-over-quarter surprise acceleration/
#                                deceleration signal) -- distinct from a
#                                surprise z-score, this captures the *trend*
#                                in beat/miss magnitude across consecutive
#                                reports.
#   - gap_return              : reaction_open / prior_close - 1
#   - reaction_return         : reaction_close / prior_close - 1
#   - pre_runup_10d           : 10-day pre-earnings price run-up
#   - vol_buildup_ratio       : ratio of the ticker's average daily volume in
#                                the 5 trading days immediately BEFORE the
#                                earnings report vs. its average volume over
#                                the preceding 20-day window -- an
#                                "anticipation" volume buildup signal (does
#                                unusual pre-earnings trading activity show
#                                up ahead of the report), distinct from a
#                                reaction-day volume ratio/z-score.
#
# Prediction framing is DELIBERATELY NOT "raw probability -> position": the
# SVC's own predict_proba is used to bucket predict-set events into the
# quintile calibration bins learned from the TRAIN set's own predicted
# probabilities, and each event is sized by that bin's TRAIN-set empirical
# historical hit-rate (win rate) rather than by the raw model probability
# directly. This is a calibrated, bucketed conviction-sizing scheme: an event
# only gets a nonzero long position when its bucket's own track record on the
# training data cleared 50% hit rate, scaled by how far above 50% it sits.
# The classifier's ordinary 0/1 predict() is used only for f1/accuracy.
#
# LONG_ONLY = True: positions are clipped to max(position, 0) -- the
# strategy only ever goes long or sits in cash, never shorts.
# ---------------------------------------------------------------------------

HOLDING_DAYS = 8
LABEL_RETURN_THRESHOLD = 0.0

PRE_RUNUP_DAYS = 10
VOL_SHORT_DAYS = 5
VOL_LONG_DAYS = 20
MIN_PRE_HISTORY = max(PRE_RUNUP_DAYS + 1, VOL_SHORT_DAYS + VOL_LONG_DAYS)

N_CALIBRATION_BINS = 5

FEATURE_COLS = [
    "surprise_pct",
    "surprise_delta",
    "gap_return",
    "reaction_return",
    "pre_runup_10d",
    "vol_buildup_ratio",
]

MIN_TRAIN_EVENTS = 20
MIN_PREDICT_EVENTS = 5
STARTING_CAPITAL_USD = 10_000
COMMISSION_PER_TRADE_USD = 1.0
LONG_ONLY = True


def load_price_data(data_dir: Path, ticker: str) -> pd.DataFrame:
    df = pd.read_csv(data_dir / f"{ticker}.csv")
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.date
    return df.sort_values("date").reset_index(drop=True)


def load_earnings_data(earnings_dir: Path, ticker: str) -> pd.DataFrame:
    path = earnings_dir / f"{ticker}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["earnings_date", "report_time", "surprise_pct"])
    df = pd.read_csv(path)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"]).dt.date
    df = df.sort_values("earnings_date").reset_index(drop=True)
    # Causal quarter-over-quarter surprise delta: uses only THIS ticker's own
    # prior reported surprise, known well before the current report.
    df["prev_surprise_pct"] = df["surprise_pct"].shift(1)
    return df


def build_events(prices: pd.DataFrame, earnings: pd.DataFrame, ticker: str) -> pd.DataFrame:
    dates = prices["date"].tolist()
    date_idx = {d: i for i, d in enumerate(dates)}
    rows = []

    for _, ev in earnings.iterrows():
        if pd.isna(ev.get("prev_surprise_pct")):
            continue  # need a prior earnings report for this ticker to compute surprise_delta

        ed = ev["earnings_date"]
        after_close = ev["report_time"] == "AMC"

        if ed in date_idx:
            base_idx = date_idx[ed]
        else:
            later = [d for d in dates if d > ed]
            if not later:
                continue
            base_idx = date_idx[later[0]]
            after_close = False

        reaction_idx = base_idx + 1 if after_close else base_idx
        if reaction_idx - MIN_PRE_HISTORY < 0:
            continue
        if reaction_idx + HOLDING_DAYS >= len(prices):
            continue

        prior_close = prices["close"].iloc[reaction_idx - 1]
        reaction_open = prices["open"].iloc[reaction_idx]
        reaction_close = prices["close"].iloc[reaction_idx]
        far_close = prices["close"].iloc[reaction_idx - 1 - PRE_RUNUP_DAYS]

        # Anticipation volume buildup: 5-day pre-earnings avg volume vs the
        # preceding 20-day avg volume, both windows strictly before the
        # reaction day (causal, local to this event).
        vol_recent = prices["volume"].iloc[reaction_idx - VOL_SHORT_DAYS: reaction_idx]
        vol_prior = prices["volume"].iloc[
            reaction_idx - VOL_SHORT_DAYS - VOL_LONG_DAYS: reaction_idx - VOL_SHORT_DAYS
        ]
        prior_mean = vol_prior.mean()
        vol_buildup_ratio = float(vol_recent.mean() / prior_mean) if prior_mean else 1.0

        future_close = prices["close"].iloc[reaction_idx + HOLDING_DAYS]

        rows.append({
            "ticker": ticker,
            "earnings_date": ed,
            "reaction_date": dates[reaction_idx],
            "surprise_pct": ev["surprise_pct"],
            "surprise_delta": ev["surprise_pct"] - ev["prev_surprise_pct"],
            "gap_return": reaction_open / prior_close - 1,
            "reaction_return": reaction_close / prior_close - 1,
            "pre_runup_10d": prior_close / far_close - 1,
            "vol_buildup_ratio": vol_buildup_ratio,
            "future_return": future_close / reaction_close - 1,
        })

    if not rows:
        return pd.DataFrame(columns=["ticker", "earnings_date", "reaction_date", *FEATURE_COLS, "future_return"])
    events = pd.DataFrame(rows)
    events["label"] = (events["future_return"] > LABEL_RETURN_THRESHOLD).astype("Int64")
    return events


def prepare_window_slice(events: pd.DataFrame, start, end) -> pd.DataFrame:
    mask = (events["reaction_date"] >= start) & (events["reaction_date"] < end)
    return events.loc[mask].dropna(subset=[*FEATURE_COLS, "label", "future_return"]).reset_index(drop=True)


def buy_hold_roi(price_data: dict, start, end) -> float:
    # Equal-weighted average buy&hold across the pooled tickers over the same
    # calendar window, as the closest analogue of "do nothing" for a
    # multi-ticker strategy.
    rois = []
    for prices in price_data.values():
        mask = (prices["date"] >= start) & (prices["date"] < end)
        raw = prices.loc[mask]
        if len(raw) >= 2:
            rois.append(float(raw["close"].iloc[-1] / raw["close"].iloc[0] - 1))
    return float(np.mean(rois)) if rois else 0.0


def compute_calibration_bins(train_probs: np.ndarray, train_labels: np.ndarray, n_bins: int = N_CALIBRATION_BINS):
    """Learn quantile bin edges from TRAIN predicted probabilities, and the
    TRAIN-set empirical positive-label rate ("win rate") within each bin."""
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(train_probs, quantiles))
    if len(edges) < 3:
        edges = np.array([-np.inf, np.inf])
    else:
        edges = edges.copy()
        edges[0] = -np.inf
        edges[-1] = np.inf

    inner_edges = edges[1:-1]
    train_bins = np.digitize(train_probs, inner_edges, right=True)
    n_actual_bins = len(edges) - 1
    win_rates = np.full(n_actual_bins, float(train_labels.mean()))
    for b in range(n_actual_bins):
        mask = train_bins == b
        if mask.sum() > 0:
            win_rates[b] = float(train_labels[mask].mean())
    return edges, win_rates


def positions_from_calibration(probs: np.ndarray, edges: np.ndarray, win_rates: np.ndarray) -> np.ndarray:
    inner_edges = edges[1:-1]
    bins = np.digitize(probs, inner_edges, right=True)
    bin_win_rates = win_rates[bins]
    # Conviction-scaled long-only position: only size up when the bucket's
    # own training-set track record clears a 50% hit rate.
    return np.clip((bin_win_rates - 0.5) * 2, 0, 1)


def backtest(predict_events: pd.DataFrame, positions: np.ndarray, long_only: bool = LONG_ONLY):
    predict_events = predict_events.sort_values("reaction_date").reset_index(drop=True)
    positions = pd.Series(positions, index=predict_events.index)
    if long_only:
        positions = positions.clip(lower=0)
    num_trades = int((positions != 0).sum())

    # Each event is already one independent, non-overlapping trade. Trades
    # from different tickers can fall on overlapping calendar dates; for
    # simplicity, like the other styles here, this backtest assumes one
    # full-capital position taken at a time, compounded in chronological
    # (reaction_date) order -- not a real multi-position portfolio allocation.
    strategy_returns = positions * predict_events["future_return"]
    equity = (1 + strategy_returns).cumprod()
    roi = float(equity.iloc[-1] - 1) if len(equity) else 0.0
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    capital = STARTING_CAPITAL_USD
    net_equity = []
    for r, pos in zip(strategy_returns, positions):
        capital = capital * (1 + r) - (COMMISSION_PER_TRADE_USD if pos != 0 else 0.0)
        net_equity.append(capital)
    net_roi = float(capital / STARTING_CAPITAL_USD - 1) if len(strategy_returns) else 0.0

    ledger = predict_events[["ticker", "reaction_date", "future_return"]].copy()
    ledger["predicted"] = positions.to_numpy()
    ledger["strategy_return"] = strategy_returns
    ledger["equity"] = equity
    ledger["net_equity_usd"] = net_equity
    return roi, net_roi, max_drawdown, num_trades, ledger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--earnings-dir", type=Path, default=Path("data/earnings"))
    parser.add_argument("--ticker", required=True, help="Comma-separated tickers to pool, e.g. AAPL,MSFT,NVDA")
    parser.add_argument("--train-months", type=int, required=True)
    parser.add_argument("--predict-months", type=int, required=True)
    parser.add_argument("--gap-days", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--holdout-months", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tickers = [t.strip() for t in args.ticker.split(",") if t.strip()]
    price_data = {t: load_price_data(args.data_dir, t) for t in tickers}
    all_events = pd.concat(
        [build_events(price_data[t], load_earnings_data(args.earnings_dir, t), t) for t in tickers],
        ignore_index=True,
    ).sort_values("reaction_date").reset_index(drop=True)

    start = max(p["date"].min() for p in price_data.values())
    full_end = min(p["date"].max() for p in price_data.values())
    end = full_end - timedelta(days=30 * args.holdout_months)
    windows = walk_forward_windows(start, end, args.train_months, args.predict_months, args.gap_days)

    window_results = []
    for w in windows:
        train = prepare_window_slice(all_events, w.train_start, w.train_end)
        predict = prepare_window_slice(all_events, w.predict_start, w.predict_end)
        if len(train) < MIN_TRAIN_EVENTS or len(predict) < MIN_PREDICT_EVENTS:
            continue
        if train["label"].nunique() < 2:
            continue

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(
                kernel="rbf", C=1.0, gamma="scale",
                probability=True, class_weight="balanced", random_state=42,
            )),
        ])
        model.fit(train[FEATURE_COLS], train["label"])

        # Classification metrics from the model's ordinary decision boundary.
        class_predictions = model.predict(predict[FEATURE_COLS])
        f1 = float(f1_score(predict["label"], class_predictions, zero_division=0))
        accuracy = float(accuracy_score(predict["label"], class_predictions))

        # Position sizing: calibrated-bucket conviction, learned from the
        # TRAIN set's own predicted-probability quintiles and their
        # train-set empirical hit rates, applied to the predict set.
        train_probs = model.predict_proba(train[FEATURE_COLS])[:, 1]
        train_labels = train["label"].to_numpy(dtype=float)
        edges, win_rates = compute_calibration_bins(train_probs, train_labels)

        predict_probs = model.predict_proba(predict[FEATURE_COLS])[:, 1]
        positions = positions_from_calibration(predict_probs, edges, win_rates)

        roi, net_roi, max_drawdown, num_trades, ledger = backtest(predict, positions)
        bh_roi = buy_hold_roi(price_data, w.predict_start, w.predict_end)

        weights_path = args.output_dir / f"window_{w.index}_weights.pkl"
        ledger_path = args.output_dir / f"window_{w.index}_ledger.csv"
        weights_path.write_bytes(pickle.dumps({
            "model": model,
            "calibration_edges": edges,
            "calibration_win_rates": win_rates,
        }))
        ledger.to_csv(ledger_path, index=False)

        window_results.append(
            WindowResult(
                index=w.index,
                train_start=w.train_start.isoformat(),
                train_end=w.train_end.isoformat(),
                predict_start=w.predict_start.isoformat(),
                predict_end=w.predict_end.isoformat(),
                f1=f1,
                accuracy=accuracy,
                roi=roi,
                net_roi=net_roi,
                max_drawdown=max_drawdown,
                buy_hold_roi=bh_roi,
                num_trades=num_trades,
                weights_path=str(weights_path),
                ledger_path=str(ledger_path),
            )
        )

    results = IterationResults(
        ticker=",".join(tickers),
        prediction_target=f"pead_forward_return_above_0_in_{HOLDING_DAYS}days_pooled",
        approach=(
            "Long-only, pooled PEAD strategy using a standardized RBF-kernel SVC (probability-calibrated) "
            "on EPS surprise%, ticker-own quarter-over-quarter surprise-delta, reaction-day gap/return, "
            "10-day pre-earnings run-up, and a pre-earnings anticipation volume-buildup ratio, pooled across "
            f"{len(tickers)} tickers' earnings events, holding {HOLDING_DAYS} trading days from the "
            "reaction-day close; positions are sized not from the raw model probability directly but from the "
            "training-set empirical hit-rate of the predicted-probability quintile bucket each event falls "
            "into, clipped to max(position, 0) so it only ever goes long or sits in cash, never shorts."
        ),
        train_months=args.train_months,
        predict_months=args.predict_months,
        gap_days=args.gap_days,
        windows=window_results,
    )
    write_results(args.output_dir / "results.json", results)


if __name__ == "__main__":
    main()
