"""Event-aligned news probe: melt-up precursors vs pre-delisting distress.

Fixes two flaws in the first probe:
  1. Windows are aligned to EACH ticker's own event date (melt-up launch, or
     last trading day before delisting), not a fixed calendar year. Probing
     2025 news for a stock that melted up in 2020 measures its quiet years.
  2. The window ENDS at the event and looks strictly BACKWARD, so we cannot
     pick up post-collapse artifacts - notably the class-action law-firm
     press releases ("Shareholders Who Lost Money") that flood out AFTER a
     crash and would otherwise look brilliantly predictive while being
     entirely unusable ex ante.
"""
import os
import time
from datetime import timedelta

import pandas as pd
from dotenv import load_dotenv
from massive import RESTClient

REPO = "/home/asdf/Source/Repos/multi-agent-trading-desk"
load_dotenv(f"{REPO}/.env")

N_PER_GROUP = 12
LOOKBACK_DAYS = 180
SECONDS_BETWEEN_CALLS = 13
MAX_RETRIES = 4
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
                limit=100, sort="published_utc", order="asc",
            ))
        except Exception:
            if attempt == MAX_RETRIES:
                return None
            time.sleep(SECONDS_BETWEEN_CALLS * attempt)


def summarise(arts, ticker: str) -> dict:
    scores, distress, upside = [], 0, 0
    for a in arts:
        for ins in (a.insights or []):
            if ins.ticker == ticker and ins.sentiment in SENTIMENT_SCORE:
                scores.append(SENTIMENT_SCORE[ins.sentiment])
        text = f"{a.title or ''} {a.description or ''}".lower()
        if any(k in text for k in DISTRESS_KEYWORDS):
            distress += 1
        if any(k in text for k in UPSIDE_KEYWORDS):
            upside += 1
    n = len(arts)
    return {
        "articles": n,
        "mean_sentiment": sum(scores) / len(scores) if scores else None,
        "pct_negative": sum(1 for s in scores if s < 0) / len(scores) if scores else None,
        "distress_rate": distress / n if n else None,
        "upside_rate": upside / n if n else None,
    }


def main():
    mu = pd.read_csv("/tmp/meltup_precursor_features.csv")
    mu["event_date"] = pd.to_datetime(mu["meltup_start_date"])
    mu = mu[mu["event_date"] >= "2019-06-01"]           # news+sentiment coverage era
    meltups = mu[["ticker", "event_date"]].head(N_PER_GROUP).values.tolist()

    prof = pd.read_csv("/tmp/delisted_profile.csv")
    dy = prof[(prof["stopped_trading"]) & (prof["drawdown_at_end"] <= -0.7)].copy()
    dy["event_date"] = pd.to_datetime(dy["last_seen"])
    dying = dy[["ticker", "event_date"]].head(N_PER_GROUP).values.tolist()

    sample = [("meltup", t, d) for t, d in meltups] + [("dying", t, d) for t, d in dying]
    print(f"probing {len(sample)} tickers, {LOOKBACK_DAYS}d window ending at each ticker's OWN event\n")

    client = RESTClient(os.environ["MASSIVE_API_KEY"])
    rows = []
    for i, (group, ticker, event_date) in enumerate(sample):
        start = (event_date - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        end = event_date.strftime("%Y-%m-%d")
        arts = fetch(client, ticker, start, end)
        if arts is None:
            print(f"  [{group:>6}] {ticker:<6} FETCH FAILED")
        else:
            r = summarise(arts, ticker)
            r.update({"group": group, "ticker": ticker, "event_date": end})
            rows.append(r)
            ms = f"{r['mean_sentiment']:+.2f}" if r["mean_sentiment"] is not None else "  n/a"
            dr = f"{r['distress_rate']:.0%}" if r["distress_rate"] is not None else " n/a"
            ur = f"{r['upside_rate']:.0%}" if r["upside_rate"] is not None else " n/a"
            print(f"  [{group:>6}] {ticker:<6} ->{end}  n={r['articles']:>3}  "
                  f"sent={ms}  distress={dr}  upside={ur}")
        if i < len(sample) - 1:
            time.sleep(SECONDS_BETWEEN_CALLS)

    df = pd.DataFrame(rows)
    df.to_csv("/tmp/news_probe_event_aligned.csv", index=False)

    print("\n=== Pre-event comparison (strictly backward-looking) ===")
    for g, sub in df.groupby("group"):
        cov = (sub["articles"] > 0).mean()
        withs = sub.dropna(subset=["mean_sentiment"])
        print(f"  {g:>6}: coverage {cov:.0%} | median articles {sub['articles'].median():.0f} | "
              f"mean sentiment {withs['mean_sentiment'].mean():+.3f} | "
              f"distress {sub['distress_rate'].mean():.1%} | upside {sub['upside_rate'].mean():.1%}")


if __name__ == "__main__":
    main()
