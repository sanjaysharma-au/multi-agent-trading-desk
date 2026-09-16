"""Does pre-entry news distinguish the screener's winners from its losers?

Stratified sample of the 978 archive-screener picks: ALL stop/delisted-timeout
trades (the losses we want to filter out) plus a matched-size sample of target
trades (the wins). For each, pulls the 180 days of news strictly BEFORE entry
(no look-ahead) and computes sentiment + distress/upside keyword rates.
"""
import os
import time
from datetime import timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from massive import RESTClient

REPO = "/home/asdf/Source/Repos/multi-agent-trading-desk"
load_dotenv(f"{REPO}/.env")

TRADES = "/tmp/screener_trades_with_outcomes.csv"
OUT = "/tmp/screener_trades_news.csv"
LOOKBACK_DAYS = 180
SECONDS_BETWEEN_CALLS = 13
MAX_RETRIES = 4
SEED = 11

SENTIMENT_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
DISTRESS_KEYWORDS = [
    "reverse split", "going concern", "nasdaq", "compliance", "delisting",
    "offering", "dilution", "bankruptcy", "chapter 11", "restructuring",
    "default", "class action", "investigation", "lawsuit", "fraud",
]
UPSIDE_KEYWORDS = [
    "fda", "approval", "contract", "award", "partnership", "acquisition",
    "record revenue", "beats", "breakthrough", "patent", "launch", "expansion",
]


def fetch(client: RESTClient, ticker: str, start: str, end: str):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return list(client.list_ticker_news(
                ticker=ticker, published_utc_gte=start, published_utc_lte=end,
                limit=1000, sort="published_utc", order="asc",
            ))
        except Exception:
            if attempt == MAX_RETRIES:
                return None
            time.sleep(SECONDS_BETWEEN_CALLS * attempt)


def summarise_window(arts, ticker: str, window_start: pd.Timestamp, window_end: pd.Timestamp) -> dict:
    scores, distress, upside, n = [], 0, 0, 0
    for a in arts:
        pub = pd.to_datetime(a.published_utc, utc=True).tz_localize(None)
        if not (window_start <= pub <= window_end):
            continue
        n += 1
        for ins in (a.insights or []):
            if ins.ticker == ticker and ins.sentiment in SENTIMENT_SCORE:
                scores.append(SENTIMENT_SCORE[ins.sentiment])
        text = f"{a.title or ''} {a.description or ''}".lower()
        if any(k in text for k in DISTRESS_KEYWORDS):
            distress += 1
        if any(k in text for k in UPSIDE_KEYWORDS):
            upside += 1
    return {
        "articles": n,
        "mean_sentiment": sum(scores) / len(scores) if scores else np.nan,
        "distress_rate": distress / n if n else np.nan,
        "upside_rate": upside / n if n else np.nan,
        "has_distress_news": distress > 0,
    }


def main():
    trades = pd.read_csv(TRADES, parse_dates=["entry_date"])

    losers = trades[trades["outcome"].isin(["stop", "delisted_timeout"])]
    rng = np.random.default_rng(SEED)
    winners = trades[trades["outcome"] == "target"].sample(
        n=min(len(losers), (trades["outcome"] == "target").sum()), random_state=SEED
    )
    sample = pd.concat([losers, winners]).reset_index(drop=True)
    unique_tickers = sample["ticker"].unique()
    print(f"probing {len(sample)} trades across {len(unique_tickers)} UNIQUE tickers "
          f"({len(losers)} losers: stop/delisted, {len(winners)} matched winners: target)")
    print("one API call per ticker (covering its full needed span), sliced locally per trade\n")

    client = RESTClient(os.environ["MASSIVE_API_KEY"])
    rows = []
    for ti, ticker in enumerate(unique_tickers):
        trades = sample[sample["ticker"] == ticker]
        span_start = (trades["entry_date"].min() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        span_end = trades["entry_date"].max().strftime("%Y-%m-%d")
        arts = fetch(client, ticker, span_start, span_end)

        for _, row in trades.iterrows():
            entry = row["entry_date"]
            if arts is None:
                continue
            r = summarise_window(arts, ticker, entry - timedelta(days=LOOKBACK_DAYS), entry)
            r.update({"ticker": ticker, "entry_date": entry.date().isoformat(),
                      "outcome": row["outcome"], "ret": row["ret"]})
            rows.append(r)
            ms = f"{r['mean_sentiment']:+.2f}" if not np.isnan(r["mean_sentiment"]) else "n/a "
            dr = f"{r['distress_rate']:.0%}" if not np.isnan(r["distress_rate"]) else "n/a"
            print(f"  {ticker:<6} {row['outcome']:<17} n={r['articles']:>3} sent={ms} distress={dr}")

        if ti < len(unique_tickers) - 1:
            time.sleep(SECONDS_BETWEEN_CALLS)
        if (ti + 1) % 25 == 0:
            pd.DataFrame(rows).to_csv(OUT, index=False)  # periodic checkpoint
            print(f"  --- checkpoint: {ti+1}/{len(unique_tickers)} tickers, {len(rows)} trades ---")

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    print(f"\n=== Coverage & signal by outcome ({len(out)} probed) ===")
    out["loser"] = out["outcome"].isin(["stop", "delisted_timeout"])
    for grp, sub in out.groupby("loser"):
        label = "LOSERS (stop/delisted)" if grp else "WINNERS (target)"
        cov = (sub["articles"] > 0).mean()
        has_news = sub.dropna(subset=["distress_rate"])
        print(f"  {label:<24} n={len(sub):>3}  coverage={cov:.0%}  "
              f"mean_sentiment={has_news['mean_sentiment'].mean():+.3f}  "
              f"distress_rate={has_news['distress_rate'].mean():.1%}  "
              f"upside_rate={has_news['upside_rate'].mean():.1%}  "
              f"any_distress_news={sub['has_distress_news'].mean():.1%}")


if __name__ == "__main__":
    main()
