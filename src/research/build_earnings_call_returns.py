import argparse

import pandas as pd

HORIZONS = [1, 5, 21, 63, 126, 252]


def load_prices(price_path: str) -> pd.DataFrame:
    prices = pd.read_csv(price_path)
    prices["date"] = pd.to_datetime(prices["timestamp"], unit="ms").dt.normalize()
    prices = prices.sort_values("date").reset_index(drop=True)
    return prices[["date", "close"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    args = parser.parse_args()
    ticker = args.ticker

    prices = load_prices(f"data/daily_aggs/{ticker}.csv")
    earnings = pd.read_csv(f"data/earnings/{ticker}.csv", parse_dates=["earnings_date"])
    earnings["earnings_date"] = earnings["earnings_date"].dt.normalize()
    earnings = earnings[earnings["earnings_date"] >= prices["date"].min()].copy()

    dates = prices["date"].to_numpy()
    closes = prices["close"].to_numpy()

    rows = []
    for _, r in earnings.iterrows():
        edate = r["earnings_date"]
        idx = dates.searchsorted(edate.to_numpy())
        is_amc = str(r["report_time"]).strip().upper() == "AMC"
        reaction_idx = idx + 1 if is_amc else idx
        pre_idx = reaction_idx - 1
        if pre_idx < 0 or reaction_idx >= len(dates):
            continue

        row = {
            "earnings_date": edate.date().isoformat(),
            "quarter": _quarter_label(edate, is_amc),
            "report_time": r["report_time"],
            "eps_surprise_pct": r["surprise_pct"],
            "pre_close": closes[pre_idx],
        }
        for h in HORIZONS:
            target_idx = reaction_idx + h - 1
            if target_idx < len(dates):
                row[f"fwd_ret_{h}d"] = closes[target_idx] / closes[pre_idx] - 1
            else:
                row[f"fwd_ret_{h}d"] = None
        rows.append(row)

    out_path = f"data/earnings_call_scores/{ticker}_returns.csv"
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print(f"{len(out)} rows written to {out_path}")
    print(out.head())


def _quarter_label(earnings_date: pd.Timestamp, is_amc: bool) -> str:
    month = earnings_date.month
    year = earnings_date.year
    if month in (1, 2, 3):
        return f"{year - 1}Q4"
    if month in (4, 5, 6):
        return f"{year}Q1"
    if month in (7, 8, 9):
        return f"{year}Q2"
    return f"{year}Q3"


if __name__ == "__main__":
    main()
