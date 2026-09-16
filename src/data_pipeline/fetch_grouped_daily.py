import json
import os
import time
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv
from massive import RESTClient

load_dotenv()

OUTPUT_DIR = Path("data/market_wide")
YEARS_BACK = 2
SECONDS_BETWEEN_CALLS = 13
MAX_RETRIES = 6
CSV_HEADER = ["ticker", "open", "high", "low", "close", "volume", "vwap", "transactions"]


def trading_day_candidates(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() < 5:  # skip Sat/Sun; holidays just return an empty/short result, wasted but harmless
            yield d
        d += timedelta(days=1)


def fetch_day(client: RESTClient, day: date) -> list[dict] | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.get_grouped_daily_aggs(day.isoformat(), adjusted=True, raw=True)
            data = json.loads(resp.data.decode())
            return data.get("results", [])
        except Exception as e:
            if "before end of day" in str(e):
                print(f"[{day}] not available yet (too recent for EOD-only data on this plan) -- skipping")
                return None
            if "NOT_AUTHORIZED" in str(e):
                raise
            if attempt == MAX_RETRIES:
                print(f"[{day}] giving up after {MAX_RETRIES} attempts: {e}")
                return None
            wait = SECONDS_BETWEEN_CALLS * attempt
            time.sleep(wait)


def write_day(day: date, results: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{day.isoformat()}.csv"
    with path.open("w", newline="") as f:
        f.write(",".join(CSV_HEADER) + "\n")
        for r in results:
            f.write(
                f"{r.get('T','')},{r.get('o','')},{r.get('h','')},{r.get('l','')},"
                f"{r.get('c','')},{r.get('v','')},{r.get('vw','')},{r.get('n','')}\n"
            )


def main():
    client = RESTClient(os.environ["MASSIVE_API_KEY"])
    today = date.today()
    start = today - timedelta(days=365 * YEARS_BACK)

    days = list(trading_day_candidates(start, today))
    todo = [d for d in days if not (OUTPUT_DIR / f"{d.isoformat()}.csv").exists()]
    print(f"{len(days)} candidate trading days, {len(todo)} remaining to fetch")

    for i, day in enumerate(todo):
        results = fetch_day(client, day)
        if results is not None:
            write_day(day, results)
            print(f"[{day}] {len(results)} tickers")
        if i < len(todo) - 1:
            time.sleep(SECONDS_BETWEEN_CALLS)


if __name__ == "__main__":
    main()
