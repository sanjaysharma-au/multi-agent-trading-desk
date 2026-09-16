"""Build a panel dataset for melt-up prediction.

One row per (ticker, observation date), sampled every SAMPLE_EVERY trading days
across an UNBIASED random sample of the market. Features use only data up to
and including the observation date; labels look forward 252 trading days.

This is the key fix over the earlier attempt: train and evaluate on the same
population the rule would actually be deployed on, rather than training on a
hand-matched winners-vs-lookalikes set and applying it out-of-distribution.
"""
from pathlib import Path

import numpy as np
import pandas as pd

NEGATIVE_DIR = Path("/home/asdf/Source/Repos/multi-agent-trading-desk/data/negative_sample_daily")
SPY_PATH = Path("/home/asdf/Source/Repos/multi-agent-trading-desk/data/daily_aggs/SPY.csv")
OUT_PATH = Path("/tmp/meltup_panel.csv")

SAMPLE_EVERY = 21          # one observation per ticker per ~month
FWD_DAYS = 252             # forward window for the label
MIN_HISTORY = 260          # need at least ~1y of history for trailing features
MIN_PRICE = 1.0            # skip sub-$1 quotes (artifact-prone)


def load_daily(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["date", "Close"]).sort_values("date").reset_index(drop=True)


def load_spy() -> pd.DataFrame:
    df = pd.read_csv(SPY_PATH)
    # Yahoo daily bars are stamped at market-open UTC (04:00), so normalize to
    # midnight or the merge against the panel's date-only index matches nothing.
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)
    df["spy_ret_126d"] = df["close"].pct_change(126)
    df["spy_vol_63d"] = df["close"].pct_change().rolling(63).std()
    return df[["date", "spy_ret_126d", "spy_vol_63d"]]


def build_ticker_panel(df: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    if len(df) < MIN_HISTORY + FWD_DAYS // 2:
        return None

    c = df["Close"]
    h = df["High"]
    low = df["Low"]
    v = df["Volume"]
    ret1 = c.pct_change()

    f = pd.DataFrame({"date": df["date"], "ticker": ticker, "close": c})

    # --- multi-horizon momentum ---
    for n in (21, 63, 126, 252):
        f[f"ret_{n}d"] = c.pct_change(n)

    # --- position within trailing range ---
    high252 = h.rolling(252).max()
    low252 = low.rolling(252).min()
    f["drawdown_from_252d_high"] = c / high252 - 1
    f["pct_above_252d_low"] = c / low252 - 1
    high756 = h.rolling(756, min_periods=252).max()
    f["drawdown_from_3y_high"] = c / high756 - 1
    # days since the trailing 252d high (vectorized via argmax on rolling window)
    f["days_since_252d_high"] = (
        h.rolling(252).apply(lambda w: len(w) - 1 - int(np.argmax(w)), raw=True)
    )

    # --- volatility / range ---
    f["volatility_21d"] = ret1.rolling(21).std()
    f["volatility_252d"] = ret1.rolling(252).std()
    f["vol_ratio_21_252"] = f["volatility_21d"] / f["volatility_252d"]
    daily_range = (h - low) / c
    f["range_ratio_21_252"] = daily_range.rolling(21).mean() / daily_range.rolling(252).mean()

    # --- volume / liquidity ---
    f["volume_ratio_21_252"] = v.rolling(21).mean() / v.rolling(252).mean()
    f["volume_ratio_63_252"] = v.rolling(63).mean() / v.rolling(252).mean()
    f["log_dollar_volume"] = np.log1p((c * v).rolling(21).mean())
    f["log_price"] = np.log(c.clip(lower=0.01))

    # --- short-horizon behaviour ---
    f["up_day_ratio_63d"] = (ret1 > 0).rolling(63).mean()
    f["max_1d_gain_63d"] = ret1.rolling(63).max()
    f["rsi_14"] = _rsi(c, 14)

    # --- listing age ---
    f["listing_age_days"] = np.arange(len(df))

    # --- labels: best forward move over the next FWD_DAYS ---
    fwd_max = h.shift(-1).rolling(FWD_DAYS, min_periods=FWD_DAYS // 2).max().shift(-(FWD_DAYS - 1))
    f["fwd_max_multiple"] = fwd_max / c
    fwd_close = c.shift(-FWD_DAYS)
    f["fwd_252d_return"] = fwd_close / c - 1

    f = f.iloc[MIN_HISTORY::SAMPLE_EVERY]
    f = f[f["close"] >= MIN_PRICE]
    return f


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean().replace(0, np.nan)
    return 100 - (100 / (1 + gain / loss))


def main():
    spy = load_spy()
    frames = []
    files = sorted(NEGATIVE_DIR.glob("*.csv"))
    for i, path in enumerate(files):
        try:
            df = load_daily(path)
        except Exception:
            continue
        panel = build_ticker_panel(df, path.stem)
        if panel is not None and len(panel):
            frames.append(panel)
        if (i + 1) % 200 == 0:
            print(f"  processed {i+1}/{len(files)} tickers, {sum(len(x) for x in frames)} rows so far")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.merge(spy, on="date", how="left")
    panel.to_csv(OUT_PATH, index=False)

    labelled = panel.dropna(subset=["fwd_max_multiple"])
    print(f"\nPanel: {len(panel)} rows across {panel['ticker'].nunique()} tickers")
    print(f"Labelled rows (have a full forward window): {len(labelled)}")
    for thresh in (2, 3, 5, 10):
        rate = (labelled["fwd_max_multiple"] >= thresh).mean()
        print(f"  base rate of reaching {thresh}x within 252d: {rate:.2%} ({int(rate*len(labelled))} events)")


if __name__ == "__main__":
    main()
