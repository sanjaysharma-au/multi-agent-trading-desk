from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM"]
OUTPUT_DIR = Path("data/earnings")
EARNINGS_LOOKUP_LIMIT = 60


def fetch_ticker(ticker: str):
    dates = yf.Ticker(ticker).get_earnings_dates(limit=EARNINGS_LOOKUP_LIMIT)
    if dates is None or dates.empty:
        print(f"[{ticker}] no earnings data")
        return

    dates = dates.dropna(subset=["Reported EPS", "Surprise(%)"]).sort_index()
    rows = []
    for ts, row in dates.iterrows():
        # yfinance timestamps the report ~16:00 for an after-close release and
        # earlier (commonly ~07:00-09:00) for a before-open release. There is
        # no explicit BMO/AMC flag, so this threshold is the standard proxy.
        report_time = "AMC" if ts.hour >= 12 else "BMO"
        rows.append({
            "earnings_date": ts.date().isoformat(),
            "report_time": report_time,
            "eps_estimate": row["EPS Estimate"],
            "eps_reported": row["Reported EPS"],
            "surprise_pct": row["Surprise(%)"],
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / f"{ticker}.csv", index=False)
    print(f"[{ticker}] wrote {len(out)} earnings events")


def main():
    for ticker in TICKERS:
        fetch_ticker(ticker)


if __name__ == "__main__":
    main()
