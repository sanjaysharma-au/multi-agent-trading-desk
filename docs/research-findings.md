# Research findings log

Dated entries on strategy ideas investigated via the signal-generation loop,
independent of whether they led anywhere — so a later session doesn't
re-spend time re-discovering the same dead end.

---

## 2026-09-17 — News/sentiment as a melt-up-screener distress filter

**Idea (user hypothesis):** stocks that melt up and stocks that go under should
read differently in the news beforehand — use that to filter the melt-up
screener's losing picks (see entry below), specifically the 9.8% that delist
while held and the 199-of-978 that hit the stop.

**Probe 1 — extremes, event-aligned, backward-looking only** (window ends at
each ticker's own event, so no post-collapse artifacts like class-action
press releases leak in): confirmed coverage is usable — 75% of pre-delisting
names have news, sentiment averaged neutral (+0.001), distress-keyword rate
54.7% (reverse split / going concern / Nasdaq compliance / dilution).
**But 0 of 12 melt-up tickers had ANY news in the 180 days before their own
launch** — including well-covered 2024-25 names (AXTI, ORBS, DRUG), not just
thin-archive-era ones (GME, RIOT in 2020). These microcaps are genuinely
uncovered before they move; the press shows up after. A feature null for the
entire positive class cannot help a "find melt-ups" screener.

**Probe 2 — reframed as a loss filter, tested on the screener's own trades**
(not extremes): sampled 236 losing trades (stop/delisted-timeout) + 236
matched winning trades (hit +100% target) from the archive screener's actual
2025-26 picks, fetched each ticker's news ONCE (not per-trade — cut API calls
from 472 to 303 by re-slicing one wider window locally per trade) covering the
180 days before each entry.

Result: directionally consistent but **not statistically significant**.
Distress rate 28.0% (losers) vs 21.5% (winners) — real gap, but t-test
p=0.115. Correlation between distress_rate and actual trade return: **-0.10**
(~1% of variance). Return by has-any-distress-news: 34.8% vs 34.2% mean,
identical median — no discrimination at all on that cruder metric.

**Why the extreme-case signal doesn't transfer:** the "loser" bucket inside an
already-selected screener population is mostly ordinary failed reversals
(a beaten-down stock that just kept falling), not going-concern situations.
The dichotomy is sharp at the tails (true melt-up vs true bankruptcy) and
muddy in the middle — which is exactly where the screener operates.

**Conclusion:** the underlying intuition is directionally correct and
confirmed twice now, but keyword-matching + Massive's off-the-shelf sentiment
does not produce a usable filter for this screener as-is. Not deployed.

**If revisited:** the open question is statistical power, not direction —
p=0.115 on n≈145/group is "trending, not there." A larger sample (the probe
covered only 303 of the screener's ~700 unique picked tickers) might sharpen
this rather than reverse it. Also worth trying: a learned model over the raw
sentiment/keyword features (like the price-based screener itself) rather than
a single hand-picked keyword list and a threshold test.

---

## 2026-09-16 — Melt-up screener (10x movers), cross-sectional ML

**Idea:** find stocks that rose 10x+ within a year, characterise their prior
behaviour, and turn it into a tradeable screen.

**Pipeline** (`src/research/`, plus `src/data_pipeline/screen_meltups.py`,
`fetch_meltup_daily.py`, `fetch_grouped_daily.py`):
Massive grouped-daily supplies the *universe* (~16k tickers, one call per
trading day); Yahoo supplies deep per-ticker *history*. A quarterly-resolution
screen over 10y found 463 candidates -> 393 after a >=$1 base-price filter that
removes sub-penny artifacts (e.g. PPCB's bogus 242,433x).

**Dead ends worth not repeating:**
- *Descriptive precursor study alone is useless.* 94.6% of melt-ups launch from
  a sharp prior decline (median -40% over the preceding 60d), but that is close
  to tautological — a trough is preceded by a fall — and has no control group.
- *Naive threshold rule loses money.* Buying every stock with >=60% drawdown +
  >=120 days since high + still falling: median return negative at EVERY horizon
  (-2.8% at 30d to -21.8% at 252d), win rate <45%, 10x hit rate 0.1%.
- *Training on winners vs hand-matched lookalikes does not transfer.* That setup
  scored AUC 0.89, but scoring the real population with it produced NO monotonic
  relationship between probability and return — the top decile was WORSE than
  the bottom. Classic selection-on-the-outcome trap.
- *Micro-structure precursors add nothing.* Up-day ratio, close-location-value,
  new-low counts are identical across classes. Only volatility contraction at
  the trough differed (0.97 vs 1.73) and it did not improve AUC (0.884 vs 0.887).

**What actually worked** — three fixes together:
1. Train *in-distribution*: a panel of (ticker, date) observations across an
   unbiased universe, labelled by realised forward outcome — not a curated
   matched set.
2. A *learnable* target: "touches 2x within a year" (~9% base rate), not 10x
   (0.07%, unlearnable).
3. Recognise the target is a PATH statistic, so it must be traded with a
   standing take-profit. Buy-and-holding the picks loses money (median -29%)
   while a +100% TP / -50% SL triple-barrier is profitable on the same picks.

**Results:** ranking skill is strong and regime-robust — 3.3x to 7.0x lift over
base rate in ALL SIX walk-forward folds (2020-2025), strongest (7.0x) in the
2022 bear market. Pooled triple-barrier P&L: 974 trades, mean **+21.6%/trade**,
median **-10.8%**, win 47%, mean/std 0.28. Edge survives gap-aware stop fills,
2% slippage, and *improves* under liquidity floors (stop rate falls 39% -> 27.5%
as liquidity rises), arguing against a microcap-illiquidity artifact.

**Survivorship, measured not estimated** (`measure_survivorship.py`): Yahoo drops
delisted names, so a controlled within-archive experiment compared survivors-only
against the full Massive universe (11,349 tickers, 1,289 later delisted). Bias is
**4.3pp realistic / 6.2pp pessimistic per trade (~10-15% proportional)** — about
3x smaller than a crude sensitivity estimate suggested, because only 9.8% of picks
delist while held, the -50% stop catches most dying names first, and most
delistings are acquisitions of healthy companies (median $13.54, only -18.4% off
peak) rather than bankruptcies. Ranking skill barely moves (lift 7.45x -> 6.81x).

**Honest bottom line:** survivorship-corrected ~**+18-19% per trade**. Real,
but fat-tailed and high-variance: median trade loses money, 2 of 6 years were
negative, per-trade mean/std ~0.28. This is a portfolio-construction problem
(many small positions, strict stops), not a prediction problem. Not deployed.

**Known gaps if revisited:** no short-interest or fundamental/solvency data (the
most obvious missing dimension); Massive archive limited to 2y so the
survivorship measurement uses a ticker-disjoint split over one favourable
17-month window; regime dependence unexplained (2021 and 2024 lost money).

---

## 2026-09-16 — Post-earnings-announcement drift (PEAD), earnings style

**Idea:** stocks that beat/miss EPS estimates tend to keep drifting in that
direction for 1-2 weeks after the report (a well-documented academic
anomaly). Built a new `earnings` style in the orchestrator
(`src/wfo/example_earnings_model.py`, `CONTRACT_VERSION_EARNINGS`) — event-
driven rather than bar-driven, pooling earnings events across multiple
tickers to get enough sample size (earnings only happen ~4x/ticker/year).

**What we found:**
- The effect is visible in raw descriptive stats: on NVDA/GOOGL/AMZN/META,
  post-earnings 5-day drift is clearly split by beat vs. miss direction.
- Walk-forward search over a pool of 7 then 11 large/mega-cap tickers found
  several models with strong **in-pool** results — best: net ROI +99% to
  +126%, Sharpe ~1.0-1.55 (vs. buy&hold Sharpe ~1.1-1.15), positive in 6-7
  of 7 out-of-sample WFO windows, low drawdown. Saved at
  `models/best/earnings_pead_svc_20260916.py` and
  `models/best/earnings_pead_quantilegate_20260916.py`.
- **Generalization test (the deciding result):** trained fresh on the full
  pool, tested on tickers never seen in any search or training —
  TSLA/V/XOM/KO (7-ticker pool) and DIS/PFE/CSCO/WMT (11-ticker pool).
  Both pools generalized poorly: accuracy landed at 40-58% (essentially
  coin-flip) and net ROI was small and inconsistent (roughly -4% to +20%,
  mostly single digits), across all 8 held-out tickers in both trials.
  Broadening the training pool from 7 to 11 diverse-sector tickers did not
  improve this.

**Conclusion:** the in-pool result is real but narrow — it reflects
patterns specific to the training tickers' own decade of history (mega-cap
tech + one bank, correlated earnings dynamics, shared bull-market backdrop),
not a portable PEAD signal. Two independent training pools Ă— 4 fresh
tickers each = 8 generalization checks, none showing a real edge. Stopped
here; not pursuing further variants of this approach.

**If revisited:** the harness (`--style earnings`, pooled-ticker event
construction, holdout-ticker test pattern) is reusable — reference
`src/wfo/example_earnings_model.py` and the held-out-ticker test scripts
this investigation used (ad hoc, not checked in) if picking this back up
with a different feature set or a much larger/more diverse training pool.

---

## 2026-09-15 — News sentiment features, intraday style

**Idea:** add Massive's per-article news sentiment (positive/neutral/
negative) as features for intraday models, joined leakage-safely via
`wfo.timeutils.attach_sentiment_features` (backward merge_asof so a bar only
ever sees news published at or before its own timestamp).

**What we found:** in a 24-iteration independent search across 8 tickers,
only 2 of 22 valid iterations chose to use sentiment as a feature at all;
those 2 averaged much better (+9.8% net ROI, Sharpe +1.37) than the 20
technicals-only iterations (-7.9% net ROI, Sharpe -0.46) — but n=2 is far
too small to call that a real effect rather than those two also happening
to land on better technical setups independently.

**Conclusion:** inconclusive, not negative. The sample was too small
because the search wasn't forced to compare sentiment vs. no-sentiment
head-to-head — it was left to the LLM's discretion, which rarely reached
for it. Paused (not disproven) per user decision on 2026-09-15. Sentiment
support stays in the codebase (`src/data_pipeline/fetch_news_sentiment.py`,
`attach_sentiment_features`, intraday contract `v7-sentiment`) but unused by
default.

**If revisited:** force an explicit A/B split in `build_instructions()`
(alternate iterations between "must use sentiment" / "must not") to get a
fair, adequately-sized sample on each side, rather than relying on the
model's own discretion.
