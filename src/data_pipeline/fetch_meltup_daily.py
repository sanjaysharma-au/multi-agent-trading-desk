import time
from pathlib import Path

import pandas as pd
import yfinance as yf

CANDIDATES_PATH = Path("data/meltup_candidates_filtered.csv")
OUTPUT_DIR = Path("data/meltup_daily")
PERIOD = "10y"
INTERVAL = "1d"
BATCH_SIZE = 40
DOWNLOAD_THREADS = 5
SECONDS_BETWEEN_BATCHES = 10


def main():
    tickers = pd.read_csv(CANDIDATES_PATH)["ticker"].tolist()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching full daily history for {len(tickers)} melt-up candidates")

    written, failed = 0, []
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i: i + BATCH_SIZE]
        data = yf.download(
            tickers=batch, period=PERIOD, interval=INTERVAL, group_by="ticker",
            threads=DOWNLOAD_THREADS, progress=False, auto_adjust=False,
        )
        for ticker in batch:
            try:
                df = data[ticker].dropna(subset=["Close"])
            except (KeyError, TypeError):
                failed.append(ticker)
                continue
            if df.empty:
                failed.append(ticker)
                continue
            df.to_csv(OUTPUT_DIR / f"{ticker}.csv")
            written += 1
        print(f"batch {i}-{i+len(batch)}: {written} written so far, {len(failed)} failed so far")
        if i + BATCH_SIZE < len(tickers):
            time.sleep(SECONDS_BETWEEN_BATCHES)

    print(f"\nDone: {written} written, {len(failed)} failed: {failed}")


if __name__ == "__main__":
    main()
