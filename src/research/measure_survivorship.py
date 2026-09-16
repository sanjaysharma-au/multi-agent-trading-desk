"""Measure survivorship bias with a controlled within-archive experiment.

Same data source, same windows, same dates. The ONLY difference between the two
arms is whether companies that later died are present:

  FULL      - every ticker in Massive's archive (survivorship-free truth)
  SURVIVORS - only tickers still trading at archive end (what Yahoo shows you)

The gap between the two arms IS the survivorship bias, measured rather than
estimated.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

PANEL = "/tmp/meltup_panel_archive.csv"
LONG_CACHE = Path("/tmp/archive_long.pkl")

FWD_DAYS = 126
TOP_PCT = 2.0
TP_PCT, SL_PCT = 1.00, 0.50
SLIPPAGE = 0.005
DELIST_HAIRCUT = 0.50   # pessimistic: delisted stub assumed worth half its last print

FEATURES = [
    "ret_21d", "ret_63d", "ret_126d",
    "drawdown_from_126d_high", "pct_above_126d_low", "days_since_126d_high",
    "volatility_21d", "volatility_126d", "vol_ratio_21_126", "range_ratio_21_126",
    "volume_ratio_21_126", "volume_ratio_63_126", "log_dollar_volume", "log_price",
    "up_day_ratio_63d", "max_1d_gain_63d", "rsi_14", "listing_age_days",
    "spy_ret_126d", "spy_vol_63d",
]


def build_paths(long: pd.DataFrame) -> dict:
    paths = {}
    for t, g in long.groupby("ticker", sort=False):
        g = g.sort_values("date")
        paths[t] = (
            g["date"].to_numpy(),
            g["open"].to_numpy(dtype=float),
            g["high"].to_numpy(dtype=float),
            g["low"].to_numpy(dtype=float),
            g["close"].to_numpy(dtype=float),
        )
    return paths


def simulate(paths: dict, ticker: str, entry_date, pessimistic: bool) -> float | None:
    rec = paths.get(ticker)
    if rec is None:
        return None
    dates, o, h, l, c = rec
    idx = np.searchsorted(dates, np.datetime64(entry_date))
    if idx >= len(dates) or dates[idx] != np.datetime64(entry_date):
        return None
    entry = c[idx]
    if entry <= 0:
        return None
    tp, sl = entry * (1 + TP_PCT), entry * (1 - SL_PCT)
    cost = 2 * SLIPPAGE + 0.0002

    end = min(idx + 1 + FWD_DAYS, len(dates))
    for j in range(idx + 1, end):
        if l[j] <= sl:
            return min(sl, o[j]) / entry - 1 - cost
        if h[j] >= tp:
            return max(tp, o[j]) / entry - 1 - cost

    if end - (idx + 1) < FWD_DAYS:
        # Ran out of data inside the window: the ticker DELISTED while held.
        stub = c[end - 1] * (DELIST_HAIRCUT if pessimistic else 1.0)
        return stub / entry - 1 - cost
    return c[end - 1] / entry - 1 - cost


def run_arm(panel: pd.DataFrame, paths: dict, label: str, pessimistic: bool) -> dict:
    # A short forward window means one of two very different things: the ticker
    # DELISTED (a real, knowable outcome) or we simply ran off the end of the
    # archive (censored - outcome not yet known). Keep the former, drop the
    # latter, or censoring gets scored as if it were performance.
    complete = panel["fwd_days_available"] >= FWD_DAYS
    genuinely_delisted = panel["delisted"] & (panel["fwd_days_available"] > 0)
    df = panel[complete | genuinely_delisted].copy()
    df["label"] = (df["fwd_max_multiple"] >= 2.0).astype(int)

    # The archive yields only ~12 months of usable observation dates (126d
    # trailing + 126d forward consumes most of its 2-year span), so a purged
    # time split leaves no training data. Split by TICKER instead: no ticker
    # appears in both halves, and both arms get the identical split, so the
    # DELTA between arms still cleanly isolates survivorship. Absolute levels
    # here are not an out-of-time estimate - the Yahoo walk-forward covers that.
    tickers = np.sort(df["ticker"].unique())
    rng = np.random.default_rng(7)
    test_tickers = set(rng.choice(tickers, size=int(len(tickers) * 0.4), replace=False))
    train = df[~df["ticker"].isin(test_tickers)]
    test = df[df["ticker"].isin(test_tickers)]
    if len(train) < 2000 or len(test) < 500 or train["label"].nunique() < 2:
        print(f"{label}: insufficient data (train={len(train)}, test={len(test)})")
        return {}

    model = HistGradientBoostingClassifier(
        max_depth=4, max_iter=300, learning_rate=0.05,
        min_samples_leaf=50, l2_regularization=1.0, random_state=42,
    )
    model.fit(train[FEATURES], train["label"])
    probs = model.predict_proba(test[FEATURES])[:, 1]
    t = test.assign(prob=probs)

    k = max(1, int(len(t) * TOP_PCT / 100))
    top = t.nlargest(k, "prob")
    base, prec = t["label"].mean(), top["label"].mean()
    auc = roc_auc_score(t["label"], probs)

    rets, delisted_hits = [], 0
    for _, r in top.iterrows():
        v = simulate(paths, r["ticker"], r["date"], pessimistic)
        if v is None:
            continue
        rets.append(v)
        if r["fwd_days_available"] < FWD_DAYS:
            delisted_hits += 1
    a = np.array(rets)

    print(f"\n{label}")
    print(f"  universe {df['ticker'].nunique():>5} tickers | train {len(train):>6} | test {len(test):>6}")
    print(f"  AUC {auc:.3f} | base {base:.2%} | precision@{TOP_PCT}% {prec:.1%} | lift {prec/base:.2f}x")
    print(f"  trades {len(a):>4} | mean {a.mean():+.1%} | median {np.median(a):+.1%} | "
          f"win {(a>0).mean():.1%} | mean/std {a.mean()/a.std():.2f}")
    print(f"  picks that delisted while held: {delisted_hits} ({delisted_hits/max(len(a),1):.1%})")
    return {"mean": a.mean(), "median": float(np.median(a)), "win": (a > 0).mean(),
            "lift": prec / base, "auc": auc, "n": len(a), "delisted_pct": delisted_hits / max(len(a), 1)}


def main():
    panel = pd.read_csv(PANEL, parse_dates=["date", "ticker_last_date"])
    long = pickle.loads(LONG_CACHE.read_bytes())
    paths = build_paths(long)

    print(f"panel {len(panel):,} rows | {panel['ticker'].nunique()} tickers | "
          f"{panel['delisted'].mean():.1%} of rows from later-delisted names")

    survivors = panel[~panel["delisted"]]

    print("\n" + "=" * 72)
    print("ARM 1 - SURVIVORS ONLY (what a Yahoo-sourced backtest sees)")
    print("=" * 72)
    s = run_arm(survivors, paths, "survivors only", pessimistic=False)

    print("\n" + "=" * 72)
    print("ARM 2 - FULL UNIVERSE (survivorship-free, delisted marked at last print)")
    print("=" * 72)
    f1 = run_arm(panel, paths, "full universe, delisted at last print", pessimistic=False)

    print("\n" + "=" * 72)
    print(f"ARM 3 - FULL UNIVERSE, pessimistic ({DELIST_HAIRCUT:.0%} haircut on delisted stubs)")
    print("=" * 72)
    f2 = run_arm(panel, paths, "full universe, delisted haircut", pessimistic=True)

    if s and f1:
        print("\n" + "=" * 72)
        print("MEASURED SURVIVORSHIP BIAS")
        print("=" * 72)
        print(f"  mean return per trade: {s['mean']:+.1%} (survivors) vs {f1['mean']:+.1%} (full) "
              f"-> bias {(s['mean']-f1['mean'])*100:+.1f} pp")
        if f2:
            print(f"  vs pessimistic full:   {s['mean']:+.1%} vs {f2['mean']:+.1%} "
                  f"-> bias {(s['mean']-f2['mean'])*100:+.1f} pp")
        print(f"  lift:  {s['lift']:.2f}x (survivors) vs {f1['lift']:.2f}x (full)")


if __name__ == "__main__":
    main()
