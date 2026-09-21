"""Bulk-fetch Form 4 (insider transaction) filings by date range.

Mirrors fetch_grouped_daily.py's pattern (one file per day, resumable, skip if
already fetched) so the local archive can be interrupted and resumed safely.

Unlike grouped-daily (one call = the whole market for that day), Form 4 volume
can exceed 1000 filings on a single busy day, so each day may need several
pages. Pages are fetched with the SAME inter-call delay as days -- the plain
iterator's silent auto-pagination (no delay between pages) is what caused
repeated 429s during manual probing; this script paces every single HTTP call,
page or day, identically.
"""
import json
import os
import time
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv
from massive import RESTClient

load_dotenv()

OUTPUT_DIR = Path("data/form4")
YEARS_BACK = 2
SECONDS_BETWEEN_CALLS = 13
MAX_RETRIES = 6
PAGE_LIMIT = 1000

KEEP_FIELDS = [
    "issuer_name", "issuer_trading_symbol", "tickers", "issuer_cik",
    "owner_cik", "owner_name", "is_officer", "is_director", "is_ten_percent_owner",
    "officer_title", "security_type", "transaction_code", "transaction_date",
    "filing_date", "transaction_shares", "transaction_price_per_share",
    "transaction_value", "transaction_acquired_disposed",
]


def trading_day_candidates(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def fetch_page(client: RESTClient, day: date, next_url: str | None) -> dict | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if next_url is None:
                resp = client.list_stocks_filings_form_4(
                    filing_date=day.isoformat(), limit=PAGE_LIMIT, raw=True,
                )
            else:
                # next_url from the API is already a full absolute URL; fetch it
                # directly via the client's own connection pool rather than
                # re-deriving a relative path (which BadResponse'd on the /vX
                # prefix mismatch in earlier attempts).
                headers = client._concat_headers({})
                resp = client.client.request("GET", next_url, headers=headers)
            return json.loads(resp.data.decode())
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"[{day}] page fetch giving up: {e}")
                return None
            time.sleep(SECONDS_BETWEEN_CALLS * attempt)


def fetch_day(client: RESTClient, day: date) -> list[dict] | None:
    all_results = []
    next_url = None
    while True:
        data = fetch_page(client, day, next_url)
        if data is None:
            return all_results if all_results else None
        all_results.extend(data.get("results", []))
        next_url = data.get("next_url")
        if not next_url:
            break
        time.sleep(SECONDS_BETWEEN_CALLS)  # pace the NEXT page exactly like a new call
    return all_results


def write_day(day: date, results: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{day.isoformat()}.jsonl"
    with path.open("w") as f:
        for r in results:
            trimmed = {k: r.get(k) for k in KEEP_FIELDS}
            f.write(json.dumps(trimmed) + "\n")


def main():
    client = RESTClient(os.environ["MASSIVE_API_KEY"])
    today = date.today()
    start = today - timedelta(days=365 * YEARS_BACK)

    days = list(trading_day_candidates(start, today))
    todo = [d for d in days if not (OUTPUT_DIR / f"{d.isoformat()}.jsonl").exists()]
    print(f"{len(days)} candidate trading days, {len(todo)} remaining to fetch")

    for i, day in enumerate(todo):
        results = fetch_day(client, day)
        if results is not None:
            write_day(day, results)
            print(f"[{day}] {len(results)} Form 4 filings")
        if i < len(todo) - 1:
            time.sleep(SECONDS_BETWEEN_CALLS)


if __name__ == "__main__":
    main()
