"""Reproduce the archive screener's full-universe top-2% picks and save
per-trade outcomes (ticker, entry date, outcome type, return) to disk, so the
news probe can look up each trade's pre-entry sentiment/keywords and compare
by outcome. Same model/split as measure_survivorship.py ARM 2 -- just adding
a CSV dump this time.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

PANEL = "/tmp/meltup_panel_archive.csv"
LONG_CACHE = Path("/tmp/archive_long.pkl")
OUT = "/tmp/screener_trades_with_outcomes.csv"

FWD_DAYS = 126
TOP_PCT = 2.0
TP_PCT, SL_PCT = 1.00, 0.50
SLIPPAGE = 0.005

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
        paths[t] = (g["date"].to_numpy(), g["open"].to_numpy(dtype=float),
                    g["high"].to_numpy(dtype=float), g["low"].to_numpy(dtype=float),
                    g["close"].to_numpy(dtype=float))
    return paths


def simulate(paths: dict, ticker: str, entry_date, delisted: bool) -> dict | None:
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
            return {"outcome": "stop", "ret": min(sl, o[j]) / entry - 1 - cost}
        if h[j] >= tp:
            return {"outcome": "target", "ret": max(tp, o[j]) / entry - 1 - cost}
    if end - (idx + 1) < FWD_DAYS and delisted:
        return {"outcome": "delisted_timeout", "ret": c[end - 1] / entry - 1 - cost}
    return {"outcome": "timeout", "ret": c[end - 1] / entry - 1 - cost}


def main():
    panel = pd.read_csv(PANEL, parse_dates=["date", "ticker_last_date"])
    long = pickle.loads(LONG_CACHE.read_bytes())
    paths = build_paths(long)

    complete = panel["fwd_days_available"] >= FWD_DAYS
    genuinely_delisted = panel["delisted"] & (panel["fwd_days_available"] > 0)
    df = panel[complete | genuinely_delisted].copy()
    df["label"] = (df["fwd_max_multiple"] >= 2.0).astype(int)

    tickers = np.sort(df["ticker"].unique())
    rng = np.random.default_rng(7)
    test_tickers = set(rng.choice(tickers, size=int(len(tickers) * 0.4), replace=False))
    train = df[~df["ticker"].isin(test_tickers)]
    test = df[df["ticker"].isin(test_tickers)]

    model = HistGradientBoostingClassifier(
        max_depth=4, max_iter=300, learning_rate=0.05,
        min_samples_leaf=50, l2_regularization=1.0, random_state=42,
    )
    model.fit(train[FEATURES], train["label"])
    probs = model.predict_proba(test[FEATURES])[:, 1]
    t = test.assign(prob=probs)

    k = max(1, int(len(t) * TOP_PCT / 100))
    top = t.nlargest(k, "prob")

    rows = []
    for _, r in top.iterrows():
        sim = simulate(paths, r["ticker"], r["date"], bool(r["delisted"]))
        if sim is None:
            continue
        rows.append({"ticker": r["ticker"], "entry_date": r["date"].date().isoformat(),
                      "outcome": sim["outcome"], "ret": sim["ret"], "prob": r["prob"],
                      "delisted_flag": bool(r["delisted"])})

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"{len(out)} trades saved to {OUT}")
    print(out["outcome"].value_counts().to_string())


if __name__ == "__main__":
    main()
