import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from wfo.schema import IterationResults, WindowResult, write_results
from wfo.windows import walk_forward_windows

# ---------------------------------------------------------------------------
# EARNINGS (post-earnings-announcement drift / PEAD) reference model, POOLED
# across multiple tickers. This is fundamentally different from the other
# styles: instead of one row per trading day, the unit of analysis is one row
# per EARNINGS EVENT (roughly 4/ticker/year — far sparser than daily bars).
# Pooling many tickers' events into a single training set is what makes this
# style viable at all with only ~10 years of daily history.
#
# The trade: enter at the REACTION-DAY CLOSE (the first close after the
# earnings report has been public — same day for a before-market-open (BMO)
# report, next day for an after-market-close (AMC) report), hold for
# HOLDING_DAYS trading days, exit at close. Every feature below is knowable
# by that reaction-day close: the reported surprise is public, and so is the
# full reaction-day bar (gap, reaction move, volume). The only forward-looking
# piece is the label itself (the HOLDING_DAYS-ahead forward return).
#
# Because each event's features/label depend on THAT TICKER's OWN price bars
# in a small window around THAT EVENT's own date (not on the training-window
# boundary), leakage-safety here works differently than the bar-based styles:
# compute each event's row from its ticker's full price series first (this is
# inherently local to the event and cannot leak across events), THEN filter
# the resulting EVENTS TABLE to [window_start, window_end) by reaction_date
# for train/predict slicing — never the other way around.
# ---------------------------------------------------------------------------

HOLDING_DAYS = 10
LABEL_RETURN_THRESHOLD = 0.0
PRE_RUNUP_DAYS = 5
VOLUME_LOOKBACK_DAYS = 20

FEATURE_COLS = ["surprise_pct", "gap_return", "reaction_return", "pre_runup", "volume_z"]

MIN_TRAIN_EVENTS = 20
MIN_PREDICT_EVENTS = 5
STARTING_CAPITAL_USD = 10_000
COMMISSION_PER_TRADE_USD = 1.0
LONG_ONLY = False


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
    return df


def build_events(prices: pd.DataFrame, earnings: pd.DataFrame, ticker: str) -> pd.DataFrame:
    dates = prices["date"].tolist()
    date_idx = {d: i for i, d in enumerate(dates)}
    rows = []

    for _, ev in earnings.iterrows():
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
        if reaction_idx - max(PRE_RUNUP_DAYS, VOLUME_LOOKBACK_DAYS) < 0:
            continue
        if reaction_idx + HOLDING_DAYS >= len(prices):
            continue

        prior_close = prices["close"].iloc[reaction_idx - 1]
        reaction_open = prices["open"].iloc[reaction_idx]
        reaction_close = prices["close"].iloc[reaction_idx]
        pre_close = prices["close"].iloc[reaction_idx - PRE_RUNUP_DAYS]

        vol_window = prices["volume"].iloc[reaction_idx - VOLUME_LOOKBACK_DAYS:reaction_idx]
        vol_mean, vol_std = vol_window.mean(), vol_window.std()
        reaction_volume = prices["volume"].iloc[reaction_idx]

        future_close = prices["close"].iloc[reaction_idx + HOLDING_DAYS]

        rows.append({
            "ticker": ticker,
            "earnings_date": ed,
            "reaction_date": dates[reaction_idx],
            "surprise_pct": ev["surprise_pct"],
            "gap_return": reaction_open / prior_close - 1,
            "reaction_return": reaction_close / prior_close - 1,
            "pre_runup": prior_close / pre_close - 1,
            "volume_z": (reaction_volume - vol_mean) / vol_std if vol_std else 0.0,
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


def buy_hold_roi(price_data: dict[str, pd.DataFrame], start, end) -> float:
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


def backtest(predict_events: pd.DataFrame, predictions, long_only: bool = LONG_ONLY):
    predict_events = predict_events.sort_values("reaction_date").reset_index(drop=True)
    predictions = pd.Series(predictions, index=predict_events.index)
    if long_only:
        predictions = predictions.clip(lower=0)
    num_trades = int((predictions != 0).sum())

    # Each event is already one independent, non-overlapping trade (like the
    # day-trade/overnight styles). Trades from different tickers can fall on
    # overlapping calendar dates; for simplicity this backtest, like the
    # other styles here, assumes one full-capital position taken at a time,
    # compounded in chronological (reaction_date) order — not a real
    # multi-position portfolio allocation.
    strategy_returns = predictions * predict_events["future_return"]
    equity = (1 + strategy_returns).cumprod()
    roi = float(equity.iloc[-1] - 1) if len(equity) else 0.0
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    capital = STARTING_CAPITAL_USD
    net_equity = []
    for r, pred in zip(strategy_returns, predictions):
        capital = capital * (1 + r) - (COMMISSION_PER_TRADE_USD if pred != 0 else 0.0)
        net_equity.append(capital)
    net_roi = float(capital / STARTING_CAPITAL_USD - 1) if len(strategy_returns) else 0.0

    ledger = predict_events[["ticker", "reaction_date", "future_return"]].copy()
    ledger["predicted"] = predictions.to_numpy()
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

    tickers = [t.strip() for t in args.ticker.split(",") if t.strip()]
    price_data = {t: load_price_data(args.data_dir, t) for t in tickers}
    all_events = pd.concat(
        [build_events(price_data[t], load_earnings_data(args.earnings_dir, t), t) for t in tickers],
        ignore_index=True,
    ).sort_values("reaction_date").reset_index(drop=True)

    start = max(p["date"].min() for p in price_data.values())
    full_end = min(p["date"].max() for p in price_data.values())
    from datetime import timedelta
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

        model = RandomForestClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=10,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        model.fit(train[FEATURE_COLS], train["label"])
        predictions = model.predict(predict[FEATURE_COLS])

        f1 = float(f1_score(predict["label"], predictions, zero_division=0))
        accuracy = float(accuracy_score(predict["label"], predictions))
        roi, net_roi, max_drawdown, num_trades, ledger = backtest(predict, predictions)
        bh_roi = buy_hold_roi(price_data, w.predict_start, w.predict_end)

        weights_path = args.output_dir / f"window_{w.index}_weights.pkl"
        ledger_path = args.output_dir / f"window_{w.index}_ledger.csv"
        weights_path.write_bytes(pickle.dumps(model))
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
            f"RandomForestClassifier on EPS surprise%/gap/reaction-return/pre-earnings-runup/"
            f"volume-zscore, pooled across {len(tickers)} tickers' earnings events, "
            f"{HOLDING_DAYS}-day post-earnings holding period (PEAD)."
        ),
        train_months=args.train_months,
        predict_months=args.predict_months,
        gap_days=args.gap_days,
        windows=window_results,
    )
    write_results(args.output_dir / "results.json", results)


if __name__ == "__main__":
    main()
