import argparse
import pickle
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score

from wfo.schema import IterationResults, WindowResult, write_results
from wfo.windows import walk_forward_windows

# ---------------------------------------------------------------------------
# PEAD (post-earnings-announcement-drift) reference model, POOLED across
# multiple tickers, LONG-ONLY.
#
# Framing: instead of a single point-forecast classifier/regressor, this
# model fits TWO quantile-regression estimators (GradientBoostingRegressor
# with `loss="quantile"`) on the pooled earnings-event table:
#   - a MEDIAN forecaster (alpha=0.5) of the HOLDING_DAYS-ahead forward return
#   - a DOWNSIDE forecaster (alpha=LOW_QUANTILE) of the same horizon's
#     pessimistic-scenario return
#
# A trade is only taken when BOTH conditions hold (a "margin of safety" gate):
#   1. the predicted median return clears MIN_EDGE (the central forecast is
#      genuinely positive, not just noise around zero), and
#   2. the predicted downside quantile clears -MAX_DOWNSIDE_TOLERANCE (even
#      the pessimistic scenario isn't a bad loss) — this is what makes the
#      gate risk-averse rather than a simple probability/magnitude threshold.
# Position size (still long-only, capped at 1.0x capital) scales with the
# predicted median return relative to TARGET_RETURN_SCALE, so higher-conviction
# events get proportionally larger (but never levered, never short) sizing.
#
# Feature set emphasises reaction-day PRICE-IMPACT / VOLATILITY microstructure
# rather than surprise-momentum or cross-sectional ranking:
#   - surprise_pct                 : raw EPS surprise (%)
#   - eps_magnitude_signed         : sign(surprise) * log1p(|eps_reported|) —
#                                     dollar-magnitude-weighted surprise, so a
#                                     large *dollar* beat/miss counts more than
#                                     a large *percent* beat/miss alone.
#   - gap_return / reaction_return : the day's own initial reaction, as usual.
#   - amihud_illiquidity           : |reaction_return| / (reaction $-volume,
#                                     $mm) — a reaction-day Amihud price-impact
#                                     ratio; a big move on thin dollar volume
#                                     signals a less-efficiently-priced surprise.
#   - bollinger_position           : (reaction_close - trailing 20d SMA) /
#                                     (2 * trailing 20d STD) — mean-reversion
#                                     positioning of the reaction close within
#                                     its own recent price channel.
#   - atr_ratio                    : reaction-day true range / trailing 14d
#                                     ATR (computed from days strictly BEFORE
#                                     the reaction day) — a volatility-
#                                     expansion-on-the-day signal.
#
# All features are computed per-ticker from that ticker's own full price
# series in a small window around each event's own reaction-day index
# (never sliced-then-computed), then the resulting EVENTS TABLE is filtered
# by reaction_date into train/predict windows.
# ---------------------------------------------------------------------------

HOLDING_DAYS = 7
LOW_QUANTILE = 0.3
MIN_EDGE = 0.0
MAX_DOWNSIDE_TOLERANCE = 0.03
TARGET_RETURN_SCALE = 0.05
LABEL_RETURN_THRESHOLD = 0.0

SMA_WINDOW = 20
ATR_WINDOW = 14
LOOKBACK_DAYS = max(SMA_WINDOW, ATR_WINDOW) + 1  # +1 for the prior-close needed by the first TR in the window

FEATURE_COLS = [
    "surprise_pct",
    "eps_magnitude_signed",
    "gap_return",
    "reaction_return",
    "amihud_illiquidity",
    "bollinger_position",
    "atr_ratio",
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
        return pd.DataFrame(columns=["earnings_date", "report_time", "eps_estimate", "eps_reported", "surprise_pct"])
    df = pd.read_csv(path)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"]).dt.date
    return df


def build_events(prices: pd.DataFrame, earnings: pd.DataFrame, ticker: str) -> pd.DataFrame:
    dates = prices["date"].tolist()
    date_idx = {d: i for i, d in enumerate(dates)}
    closes = prices["close"].to_numpy()
    highs = prices["high"].to_numpy()
    lows = prices["low"].to_numpy()
    opens = prices["open"].to_numpy()
    volumes = prices["volume"].to_numpy()
    vwaps = prices["vwap"].to_numpy() if "vwap" in prices.columns else closes

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
        if reaction_idx - LOOKBACK_DAYS < 0:
            continue
        if reaction_idx + HOLDING_DAYS >= len(prices):
            continue

        prior_close = closes[reaction_idx - 1]
        reaction_open = opens[reaction_idx]
        reaction_close = closes[reaction_idx]
        reaction_high = highs[reaction_idx]
        reaction_low = lows[reaction_idx]
        reaction_volume = volumes[reaction_idx]
        reaction_vwap = vwaps[reaction_idx]

        gap_return = reaction_open / prior_close - 1
        reaction_return = reaction_close / prior_close - 1

        # --- Amihud-style reaction-day price-impact ratio ---
        reaction_dollar_volume_mm = max(reaction_vwap * reaction_volume, 1.0) / 1e6
        amihud_illiquidity = abs(reaction_return) / reaction_dollar_volume_mm

        # --- Bollinger-style mean-reversion position, from the 20d window BEFORE reaction day ---
        sma_window_close = closes[reaction_idx - SMA_WINDOW:reaction_idx]
        sma20 = sma_window_close.mean()
        std20 = sma_window_close.std()
        bollinger_position = (reaction_close - sma20) / (2 * std20) if std20 > 1e-9 else 0.0

        # --- ATR-ratio: reaction-day true range vs trailing 14d ATR (days strictly before reaction day) ---
        atr_win_start = reaction_idx - ATR_WINDOW
        atr_highs = highs[atr_win_start:reaction_idx]
        atr_lows = lows[atr_win_start:reaction_idx]
        atr_prior_closes = closes[atr_win_start - 1:reaction_idx - 1]
        tr_hist = np.maximum.reduce([
            atr_highs - atr_lows,
            np.abs(atr_highs - atr_prior_closes),
            np.abs(atr_lows - atr_prior_closes),
        ])
        atr14 = tr_hist.mean()
        reaction_tr = max(
            reaction_high - reaction_low,
            abs(reaction_high - prior_close),
            abs(reaction_low - prior_close),
        )
        atr_ratio = reaction_tr / atr14 if atr14 > 1e-9 else 0.0

        eps_reported = ev.get("eps_reported", np.nan)
        surprise_pct = ev["surprise_pct"]
        eps_magnitude_signed = (
            float(np.sign(surprise_pct)) * np.log1p(abs(eps_reported))
            if pd.notna(eps_reported) else 0.0
        )

        future_close = closes[reaction_idx + HOLDING_DAYS]

        rows.append({
            "ticker": ticker,
            "earnings_date": ed,
            "reaction_date": dates[reaction_idx],
            "surprise_pct": surprise_pct,
            "eps_magnitude_signed": eps_magnitude_signed,
            "gap_return": gap_return,
            "reaction_return": reaction_return,
            "amihud_illiquidity": amihud_illiquidity,
            "bollinger_position": bollinger_position,
            "atr_ratio": atr_ratio,
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


def size_positions(pred_median: np.ndarray, pred_low: np.ndarray) -> np.ndarray:
    eligible = (pred_median > MIN_EDGE) & (pred_low > -MAX_DOWNSIDE_TOLERANCE)
    raw_size = np.clip(pred_median / TARGET_RETURN_SCALE, 0.0, 1.0)
    position = np.where(eligible, raw_size, 0.0)
    # LONG_ONLY: explicit clip to non-negative, per contract (redundant here
    # since raw_size/eligible are already >= 0, but applied for consistency).
    if LONG_ONLY:
        position = np.clip(position, 0.0, None)
    return position


def backtest(predict_events: pd.DataFrame, pred_median: np.ndarray, pred_low: np.ndarray):
    predict_events = predict_events.sort_values("reaction_date").reset_index(drop=True)
    positions = size_positions(np.asarray(pred_median), np.asarray(pred_low))
    num_trades = int((positions != 0).sum())

    # Each event is already one independent, non-overlapping trade. Trades
    # from different tickers can fall on overlapping calendar dates; as with
    # the other styles here, this backtest assumes one full-capital position
    # taken at a time, compounded sequentially in reaction_date order.
    strategy_returns = positions * predict_events["future_return"].to_numpy()
    equity = np.cumprod(1 + strategy_returns)
    roi = float(equity[-1] - 1) if len(equity) else 0.0
    running_max = np.maximum.accumulate(equity) if len(equity) else np.array([])
    drawdown = (equity - running_max) / running_max if len(equity) else np.array([])
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    capital = STARTING_CAPITAL_USD
    net_equity = []
    for r, pos in zip(strategy_returns, positions):
        capital = capital * (1 + r) - (COMMISSION_PER_TRADE_USD if pos != 0 else 0.0)
        net_equity.append(capital)
    net_roi = float(capital / STARTING_CAPITAL_USD - 1) if len(strategy_returns) else 0.0

    ledger = predict_events[["ticker", "reaction_date", "future_return"]].copy()
    ledger["predicted"] = positions
    ledger["pred_median_return"] = pred_median
    ledger["pred_low_return"] = pred_low
    ledger["strategy_return"] = strategy_returns
    ledger["equity"] = equity
    ledger["net_equity_usd"] = net_equity
    return roi, net_roi, max_drawdown, num_trades, ledger, positions


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
        if train["future_return"].std() < 1e-9:
            continue

        model_median = GradientBoostingRegressor(
            loss="quantile", alpha=0.5,
            n_estimators=150, max_depth=3, min_samples_leaf=8,
            learning_rate=0.05, random_state=42,
        )
        model_low = GradientBoostingRegressor(
            loss="quantile", alpha=LOW_QUANTILE,
            n_estimators=150, max_depth=3, min_samples_leaf=8,
            learning_rate=0.05, random_state=42,
        )
        model_median.fit(train[FEATURE_COLS], train["future_return"])
        model_low.fit(train[FEATURE_COLS], train["future_return"])

        pred_median = model_median.predict(predict[FEATURE_COLS])
        pred_low = model_low.predict(predict[FEATURE_COLS])

        roi, net_roi, max_drawdown, num_trades, ledger, positions = backtest(predict, pred_median, pred_low)

        predicted_binary = (positions > 0).astype(int)
        actual_binary = predict["label"].astype(int).to_numpy()
        f1 = float(f1_score(actual_binary, predicted_binary, zero_division=0))
        accuracy = float(accuracy_score(actual_binary, predicted_binary))
        bh_roi = buy_hold_roi(price_data, w.predict_start, w.predict_end)

        weights_path = args.output_dir / f"window_{w.index}_weights.pkl"
        ledger_path = args.output_dir / f"window_{w.index}_ledger.csv"
        weights_path.write_bytes(pickle.dumps({"model_median": model_median, "model_low": model_low}))
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
        prediction_target=f"pead_quantile_gated_median_return_{HOLDING_DAYS}day_long_only_pooled",
        approach=(
            "Twin GradientBoostingRegressor quantile models (median @ alpha=0.5 and downside "
            f"@ alpha={LOW_QUANTILE}) over reaction-day price-impact/volatility microstructure features "
            "(Amihud illiquidity ratio, Bollinger mean-reversion position, ATR expansion ratio, gap/"
            "reaction returns, raw and dollar-magnitude-weighted EPS surprise), gating long-only entries "
            f"on a margin-of-safety rule (predicted median return > 0 AND predicted downside quantile > "
            f"-{MAX_DOWNSIDE_TOLERANCE:.0%}) with position size scaled to the predicted median return over a "
            f"{HOLDING_DAYS}-day post-earnings holding period, pooled across multiple tickers' earnings events."
        ),
        train_months=args.train_months,
        predict_months=args.predict_months,
        gap_days=args.gap_days,
        windows=window_results,
    )
    write_results(args.output_dir / "results.json", results)


if __name__ == "__main__":
    main()
