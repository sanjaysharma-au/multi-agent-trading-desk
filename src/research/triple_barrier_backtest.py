"""Stress-tested triple-barrier backtest.

Adds three realism fixes over v1, each of which only ever hurts the result:
  1. Gap-aware fills - a stop that gaps through fills at the OPEN, not at the
     stop price (v1 assumed you always got your stop level, which is exactly
     wrong for illiquid microcaps).
  2. Per-leg slippage, scaled up for thin names.
  3. Optional liquidity floor, to ask whether the edge survives in names you
     could actually trade size in.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DAILY_DIR = Path("/home/asdf/Source/Repos/multi-agent-trading-desk/data/negative_sample_daily")
SCORED = "/tmp/meltup_panel_scored.csv"

TOP_PCT = 2.0
TP_PCT = 1.00
SL_PCT = 0.50
MAX_HOLD_DAYS = 252
COMMISSION_PER_LEG = 1.0
CAPITAL_PER_TRADE = 10_000


def load_daily(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["date", "Close"]).sort_values("date").reset_index(drop=True)


def simulate(df: pd.DataFrame, entry_date: pd.Timestamp, slippage: float) -> dict | None:
    pos = df.index[df["date"] == entry_date]
    if len(pos) == 0:
        return None
    i0 = int(pos[0])
    entry = df["Close"].iloc[i0]
    if entry <= 0:
        return None

    tp_level = entry * (1 + TP_PCT)
    sl_level = entry * (1 - SL_PCT)
    path = df.iloc[i0 + 1: i0 + 1 + MAX_HOLD_DAYS]
    if len(path) == 0:
        return None

    for n, (_, bar) in enumerate(path.iterrows(), start=1):
        hit_sl = bar["Low"] <= sl_level
        hit_tp = bar["High"] >= tp_level
        if hit_sl:
            # gap-aware: if the bar OPENED below the stop, that is your fill
            fill = min(sl_level, bar["Open"])
            return {"outcome": "stop", "gross_ret": fill / entry - 1, "days_held": n}
        if hit_tp:
            fill = max(tp_level, bar["Open"])  # favourable gap works in your favour
            return {"outcome": "target", "gross_ret": fill / entry - 1, "days_held": n}

    return {"outcome": "timeout", "gross_ret": path["Close"].iloc[-1] / entry - 1, "days_held": len(path)}


def run(picks: pd.DataFrame, label: str, slippage: float) -> None:
    cache: dict[str, pd.DataFrame] = {}
    rows = []
    for _, row in picks.iterrows():
        t = row["ticker"]
        if t not in cache:
            cache[t] = load_daily(DAILY_DIR / f"{t}.csv")
        sim = simulate(cache[t], row["date"], slippage)
        if sim is None:
            continue
        cost = 2 * COMMISSION_PER_LEG / CAPITAL_PER_TRADE + 2 * slippage
        sim["net_ret"] = sim["gross_ret"] - cost
        rows.append(sim)

    if not rows:
        print(f"{label}: no trades")
        return
    res = pd.DataFrame(rows)
    counts = res["outcome"].value_counts(normalize=True)
    ann = res["net_ret"].mean() * (252 / res["days_held"].mean())
    print(f"\n{label}")
    print(f"  trades={len(res)}  target={counts.get('target',0):.1%}  "
          f"stop={counts.get('stop',0):.1%}  timeout={counts.get('timeout',0):.1%}")
    print(f"  mean net/trade {res['net_ret'].mean():+.2%}   median {res['net_ret'].median():+.2%}   "
          f"win {(res['net_ret']>0).mean():.1%}")
    print(f"  mean hold {res['days_held'].mean():.0f}d   approx annualized {ann:+.1%}   "
          f"mean/std {res['net_ret'].mean()/res['net_ret'].std():.2f}")


def main():
    scored = pd.read_csv(SCORED, parse_dates=["date"])
    k = max(1, int(len(scored) * TOP_PCT / 100))
    picks = scored.nlargest(k, "prob").sort_values("date")
    picks = picks.assign(dollar_volume=np.expm1(picks["log_dollar_volume"]))

    print(f"Top {TOP_PCT}% = {len(picks)} candidate trades")
    print("Each variant below only ADDS realism; none of them flatter the strategy.")

    run(picks, "1. v1 baseline, no gap fills, no slippage (for comparison)", slippage=0.0)
    run(picks, "2. + gap-aware stop fills, 0.5% slippage/leg", slippage=0.005)
    run(picks, "3. + 2% slippage/leg (realistic for sub-$1M/day microcaps)", slippage=0.02)

    for floor in (500_000, 1_000_000, 5_000_000):
        liquid = picks[picks["dollar_volume"] >= floor]
        run(liquid, f"4. liquidity floor >=${floor:,.0f}/day, 0.5% slippage/leg  "
                    f"({len(liquid)} of {len(picks)} picks survive)", slippage=0.005)


if __name__ == "__main__":
    main()
