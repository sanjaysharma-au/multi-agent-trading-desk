import time
from pathlib import Path

import pandas as pd
import yfinance as yf

UNIVERSE_SOURCE = Path("data/market_wide")
OUTPUT_PATH = Path("data/meltup_candidates.csv")
FAILED_TICKERS_PATH = Path("data/meltup_screen_failed_tickers.txt")
PERIOD = "10y"
INTERVAL = "3mo"
BATCH_SIZE = 50
DOWNLOAD_THREADS = 5
SECONDS_BETWEEN_BATCHES = 8
RETRY_PASSES = 2
SECONDS_BEFORE_RETRY = 30
LOOKAHEAD_QUARTERS = 4  # ~1 year
MULTIPLE_THRESHOLD = 10.0
MIN_BASE_PRICE = 1.0  # excludes sub-$1 tickers, where tiny denominators produce artifactual multiples


def load_universe() -> list[str]:
    latest = sorted(UNIVERSE_SOURCE.glob("*.csv"))[-1]
    tickers = pd.read_csv(latest)["ticker"].dropna().unique().tolist()
    return [t.replace(".", "-") for t in tickers if isinstance(t, str) and t.strip()]


def scan_batch(batch: list[str]) -> tuple[list[dict], list[str]]:
    data = yf.download(
        tickers=batch, period=PERIOD, interval=INTERVAL, group_by="ticker",
        threads=DOWNLOAD_THREADS, progress=False, auto_adjust=False,
    )
    hits = []
    missing = []
    for ticker in batch:
        try:
            df = data[ticker].dropna(subset=["Close", "High"])
        except (KeyError, TypeError):
            missing.append(ticker)
            continue
        if len(df) < LOOKAHEAD_QUARTERS + 1:
            # Ambiguous: could be a genuinely short-lived listing, or a
            # rate-limited response that came back mostly empty. Treat as
            # missing so a retry pass gets a fair second look.
            missing.append(ticker)
            continue

        closes = df["Close"].to_numpy()
        highs = df["High"].to_numpy()
        dates = df.index.to_list()

        best_multiple, best_start, best_peak = 0.0, None, None
        for i in range(len(df) - LOOKAHEAD_QUARTERS):
            base = closes[i]
            if base < MIN_BASE_PRICE:
                continue
            window_high = highs[i + 1: i + 1 + LOOKAHEAD_QUARTERS].max()
            multiple = window_high / base
            if multiple > best_multiple:
                best_multiple, best_start, best_peak = multiple, dates[i], dates[i + 1 + int(highs[i + 1: i + 1 + LOOKAHEAD_QUARTERS].argmax())]

        if best_multiple >= MULTIPLE_THRESHOLD:
            hits.append({
                "ticker": ticker,
                "multiple": round(float(best_multiple), 1),
                "base_quarter": best_start.date().isoformat(),
                "peak_quarter": best_peak.date().isoformat(),
            })
    return hits, missing


def scan_all(tickers: list[str]) -> tuple[list[dict], list[str]]:
    all_hits = []
    all_missing = []
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i: i + BATCH_SIZE]
        try:
            hits, missing = scan_batch(batch)
        except Exception as e:
            print(f"batch {i}-{i+len(batch)} failed outright: {e}")
            all_missing.extend(batch)
            continue
        all_hits.extend(hits)
        all_missing.extend(missing)
        print(f"batch {i}-{i+len(batch)}: {len(hits)} hits, {len(missing)} missing/failed "
              f"(totals: {len(all_hits)} hits, {len(all_missing)} missing)")
        if i + BATCH_SIZE < len(tickers):
            time.sleep(SECONDS_BETWEEN_BATCHES)
    return all_hits, all_missing


def main():
    tickers = load_universe()
    print(f"Screening {len(tickers)} tickers for >= {MULTIPLE_THRESHOLD}x moves within ~1 year "
          f"(quarterly-resolution, {PERIOD} lookback)")

    all_hits, missing = scan_all(tickers)

    for retry_round in range(1, RETRY_PASSES + 1):
        if not missing:
            break
        print(f"\nRetry pass {retry_round}/{RETRY_PASSES}: re-checking {len(missing)} "
              f"missing/failed tickers after a {SECONDS_BEFORE_RETRY}s cooldown "
              "(most 'no data found' responses are actually rate-limiting, not real delistings)")
        time.sleep(SECONDS_BEFORE_RETRY)
        hits, missing = scan_all(missing)
        all_hits.extend(hits)

    if missing:
        FAILED_TICKERS_PATH.write_text("\n".join(sorted(missing)))
        print(f"\n{len(missing)} tickers still had no data after {RETRY_PASSES} retries "
              f"(likely genuinely delisted/illiquid) -> {FAILED_TICKERS_PATH}")

    out = pd.DataFrame(all_hits).sort_values("multiple", ascending=False)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\n{len(out)} candidates written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
