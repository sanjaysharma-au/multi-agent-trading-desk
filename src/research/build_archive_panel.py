"""Build a SURVIVORSHIP-FREE panel from Massive's grouped-daily archive.

Yahoo only serves currently-listed names, so the earlier panel silently
excluded every company that died. Massive's archive keeps them, so a panel
built here includes the delisted names and their real outcomes.

Windows are shortened (126d trailing features, 126d forward label) because the
archive spans only ~521 trading days. The identical configuration is applied to
the Yahoo survivor-only data by build_survivor_panel.py, so that comparing the
two isolates survivorship rather than the window change.

A ticker that vanishes from the archive mid-forward-window has DELISTED - that
truncation is a real outcome, recorded via `delisted` / `fwd_days_available`,
not dropped as missing data.
"""
import glob
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

MARKET_WIDE = sorted(glob.glob("/home/asdf/Source/Repos/multi-agent-trading-desk/data/market_wide/*.csv"))
LONG_CACHE = Path("/tmp/archive_long.pkl")
OUT_PATH = "/tmp/meltup_panel_archive.csv"

TRAIL = 126
FWD_DAYS = 126
SAMPLE_EVERY = 21
MIN_PRICE = 1.0


def build_long() -> pd.DataFrame:
    if LONG_CACHE.exists():
        print("loading cached long frame")
        return pickle.loads(LONG_CACHE.read_bytes())
    frames = []
    for i, path in enumerate(MARKET_WIDE):
        d = pd.read_csv(path, usecols=["ticker", "open", "high", "low", "close", "volume"])
        d["date"] = pd.Timestamp(Path(path).stem)
        frames.append(d)
        if (i + 1) % 100 == 0:
            print(f"  read {i+1}/{len(MARKET_WIDE)} files")
    long = pd.concat(frames, ignore_index=True)
    for c in ["open", "high", "low", "close", "volume"]:
        long[c] = pd.to_numeric(long[c], errors="coerce", downcast="float")
    long = long.dropna(subset=["close"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    LONG_CACHE.write_bytes(pickle.dumps(long))
    return long


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean().replace(0, np.nan)
    return 100 - (100 / (1 + gain / loss))


def panel_for(df: pd.DataFrame, ticker: str, archive_end: pd.Timestamp) -> pd.DataFrame | None:
    if len(df) < TRAIL + 30:
        return None
    df = df.reset_index(drop=True)
    c, h, low, v = df["close"], df["high"], df["low"], df["volume"]
    ret1 = c.pct_change()

    f = pd.DataFrame({"date": df["date"], "ticker": ticker, "close": c})
    for n in (21, 63, 126):
        f[f"ret_{n}d"] = c.pct_change(n)
    high126, low126 = h.rolling(126).max(), low.rolling(126).min()
    f["drawdown_from_126d_high"] = c / high126 - 1
    f["pct_above_126d_low"] = c / low126 - 1
    f["days_since_126d_high"] = h.rolling(126).apply(lambda w: len(w) - 1 - int(np.argmax(w)), raw=True)
    f["volatility_21d"] = ret1.rolling(21).std()
    f["volatility_126d"] = ret1.rolling(126).std()
    f["vol_ratio_21_126"] = f["volatility_21d"] / f["volatility_126d"]
    dr = (h - low) / c
    f["range_ratio_21_126"] = dr.rolling(21).mean() / dr.rolling(126).mean()
    f["volume_ratio_21_126"] = v.rolling(21).mean() / v.rolling(126).mean()
    f["volume_ratio_63_126"] = v.rolling(63).mean() / v.rolling(126).mean()
    f["log_dollar_volume"] = np.log1p((c * v).rolling(21).mean())
    f["log_price"] = np.log(c.clip(lower=0.01))
    f["up_day_ratio_63d"] = (ret1 > 0).rolling(63).mean()
    f["max_1d_gain_63d"] = ret1.rolling(63).max()
    f["rsi_14"] = rsi(c, 14)
    f["listing_age_days"] = np.arange(len(df))

    # Forward outcome over the next FWD_DAYS bars THAT EXIST. If the ticker
    # delists partway through, fwd_days_available < FWD_DAYS and that is the
    # real outcome, not a missing value.
    fwd_max = h.shift(-1).rolling(FWD_DAYS, min_periods=1).max().shift(-(FWD_DAYS - 1))
    # rolling+shift loses the tail, so compute the tail explicitly
    highs = h.to_numpy()
    n = len(df)
    fwd_max_arr = np.full(n, np.nan)
    fwd_avail = np.zeros(n, dtype=int)
    fwd_last = np.full(n, np.nan)
    for i in range(n):
        seg = highs[i + 1: i + 1 + FWD_DAYS]
        fwd_avail[i] = len(seg)
        if len(seg):
            fwd_max_arr[i] = seg.max()
            fwd_last[i] = c.to_numpy()[min(i + FWD_DAYS, n - 1)]
    f["fwd_max_multiple"] = fwd_max_arr / c
    f["fwd_days_available"] = fwd_avail
    f["fwd_last_close"] = fwd_last
    f["fwd_return"] = fwd_last / c - 1

    last_date = df["date"].iloc[-1]
    f["ticker_last_date"] = last_date
    f["delisted"] = last_date < archive_end - pd.Timedelta(days=14)

    f = f.iloc[TRAIL::SAMPLE_EVERY]
    f = f[(f["close"] >= MIN_PRICE) & (f["fwd_days_available"] > 0)]
    return f


def main():
    long = build_long()
    archive_end = long["date"].max()
    print(f"archive: {long['date'].min().date()} .. {archive_end.date()}, "
          f"{long['ticker'].nunique()} tickers, {len(long):,} rows\n")

    frames = []
    for i, (ticker, g) in enumerate(long.groupby("ticker", sort=False)):
        p = panel_for(g, ticker, archive_end)
        if p is not None and len(p):
            frames.append(p)
        if (i + 1) % 2000 == 0:
            print(f"  {i+1} tickers processed, {sum(len(x) for x in frames):,} rows")

    panel = pd.concat(frames, ignore_index=True)

    spy = long[long["ticker"] == "SPY"][["date", "close"]].sort_values("date")
    spy["spy_ret_126d"] = spy["close"].pct_change(126)
    spy["spy_vol_63d"] = spy["close"].pct_change().rolling(63).std()
    panel = panel.merge(spy[["date", "spy_ret_126d", "spy_vol_63d"]], on="date", how="left")
    panel.to_csv(OUT_PATH, index=False)

    full = panel[panel["fwd_days_available"] >= FWD_DAYS]
    print(f"\nPanel: {len(panel):,} rows, {panel['ticker'].nunique()} tickers")
    print(f"  rows from tickers that later delisted: {panel['delisted'].mean():.1%}")
    print(f"  rows with a truncated forward window (delisted mid-window): "
          f"{(panel['fwd_days_available'] < FWD_DAYS).mean():.1%}")
    print(f"  base rate of touching 2x within {FWD_DAYS}d (full windows only): "
          f"{(full['fwd_max_multiple'] >= 2).mean():.2%}")


if __name__ == "__main__":
    main()
