"""Walk-forward the melt-up screener across multiple market regimes.

Each fold trains only on data whose forward label window closes before the
fold starts (embargoed), scores the fold, takes the top 2%, and runs the
gap-aware triple-barrier simulation. If the edge only exists in the 2023-25
speculative-smallcap regime, it will show up here.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

PANEL = "/tmp/meltup_panel.csv"
DAILY_DIR = Path("/home/asdf/Source/Repos/multi-agent-trading-desk/data/negative_sample_daily")

TARGET_MULTIPLE, TOP_PCT = 2.0, 2.0
TP_PCT, SL_PCT, MAX_HOLD = 1.00, 0.50, 252
EMBARGO_DAYS = 400
SLIPPAGE = 0.005

FEATURES = [
    "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "drawdown_from_252d_high", "pct_above_252d_low", "drawdown_from_3y_high",
    "days_since_252d_high",
    "volatility_21d", "volatility_252d", "vol_ratio_21_252", "range_ratio_21_252",
    "volume_ratio_21_252", "volume_ratio_63_252", "log_dollar_volume", "log_price",
    "up_day_ratio_63d", "max_1d_gain_63d", "rsi_14", "listing_age_days",
    "spy_ret_126d", "spy_vol_63d",
]

_cache: dict[str, pd.DataFrame] = {}


def load_daily(t: str) -> pd.DataFrame:
    if t not in _cache:
        df = pd.read_csv(DAILY_DIR / f"{t}.csv")
        df = df.rename(columns={df.columns[0]: "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
        for c in ["Open", "High", "Low", "Close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        _cache[t] = df.dropna(subset=["date", "Close"]).sort_values("date").reset_index(drop=True)
    return _cache[t]


def simulate(t: str, entry_date: pd.Timestamp) -> float | None:
    df = load_daily(t)
    pos = df.index[df["date"] == entry_date]
    if len(pos) == 0:
        return None
    i0 = int(pos[0])
    entry = df["Close"].iloc[i0]
    if entry <= 0:
        return None
    tp, sl = entry * (1 + TP_PCT), entry * (1 - SL_PCT)
    path = df.iloc[i0 + 1: i0 + 1 + MAX_HOLD]
    cost = 2 * SLIPPAGE + 0.0002
    for _, bar in path.iterrows():
        if bar["Low"] <= sl:
            return min(sl, bar["Open"]) / entry - 1 - cost
        if bar["High"] >= tp:
            return max(tp, bar["Open"]) / entry - 1 - cost
    if len(path) == 0:
        return None
    return path["Close"].iloc[-1] / entry - 1 - cost


def main():
    df = pd.read_csv(PANEL, parse_dates=["date"]).dropna(subset=["fwd_max_multiple"])
    df["label"] = (df["fwd_max_multiple"] >= TARGET_MULTIPLE).astype(int)

    folds = [(pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y+1}-01-01")) for y in range(2020, 2026)]
    print(f"{'fold':>12} {'train n':>8} {'test n':>7} {'base':>7} {'prec@2%':>8} {'lift':>6} "
          f"{'trades':>7} {'mean/trade':>11} {'win':>6}")

    all_rets = []
    for start, end in folds:
        train = df[df["date"] < start - pd.Timedelta(days=EMBARGO_DAYS)]
        test = df[(df["date"] >= start) & (df["date"] < end)]
        if len(train) < 3000 or len(test) < 300 or train["label"].nunique() < 2:
            print(f"{start.year:>12} skipped (insufficient data)")
            continue

        model = HistGradientBoostingClassifier(
            max_depth=4, max_iter=300, learning_rate=0.05,
            min_samples_leaf=50, l2_regularization=1.0, random_state=42,
        )
        model.fit(train[FEATURES], train["label"])
        probs = model.predict_proba(test[FEATURES])[:, 1]
        t = test.assign(prob=probs)

        k = max(1, int(len(t) * TOP_PCT / 100))
        top = t.nlargest(k, "prob")
        base = t["label"].mean()
        prec = top["label"].mean()

        rets = [r for r in (simulate(row["ticker"], row["date"]) for _, row in top.iterrows()) if r is not None]
        all_rets.extend(rets)
        mean_ret = np.mean(rets) if rets else np.nan
        win = np.mean([r > 0 for r in rets]) if rets else np.nan

        print(f"{start.year:>12} {len(train):>8} {len(test):>7} {base:>6.1%} {prec:>7.1%} "
              f"{prec/base if base else np.nan:>6.2f} {len(rets):>7} {mean_ret:>10.1%} {win:>5.0%}")

    if all_rets:
        a = np.array(all_rets)
        print(f"\nPooled across all folds: {len(a)} trades, mean {a.mean():+.1%}, "
              f"median {np.median(a):+.1%}, win {(a>0).mean():.1%}, mean/std {a.mean()/a.std():.2f}")
        print(f"Fraction of folds with positive mean: see per-year rows above")


if __name__ == "__main__":
    main()
