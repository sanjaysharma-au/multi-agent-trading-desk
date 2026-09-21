# Research findings log

Dated entries on strategy ideas investigated via the signal-generation loop,
independent of whether they led anywhere — so a later session doesn't
re-spend time re-discovering the same dead end.

---

## 2026-09-18 — Earnings-call trajectory tracking + stance/price-target tools

**Reframing, after the ML-predictor idea above was falsified:** stop trying
to make these transcripts predict returns statistically, and instead build
decision-support tools for human speculation -- a quarter-over-quarter
"is this company's guidance getting more or less honest" tracker, and a
per-transcript bullish/neutral/bearish stance with a grounded price-move
estimate. Neither claims statistical validity; both are meant to be read the
way a human reads a sell-side analyst note.

**Trajectory tool** (`src/agents/earnings_trajectory.py`,
`src/research/{run_trajectory_batch,extract_trajectory_verdicts}.py`): feeds
the PRIOR quarter's specific extracted guidance claims (from the existing
guidance.md) alongside the CURRENT quarter's full transcript, and asks the
model to check each prior claim as DELIVERED / MISSED / REVISED / NOT
ADDRESSED, then issue an improving/stable/deteriorating verdict. This is
fundamentally more trustworthy than a return-prediction score because it's
checking a fact (was a specific promise kept), not forecasting anything.
Spot-checked against the real Tesla Q2->Q3 2019 transition: correctly caught
that management stopped mentioning its own 25-30% gross-margin target and its
self-set "Battery Day" date (both prominent the prior quarter) rather than
addressing either -- exactly the kind of quiet omission this tool is meant to
surface, that a return-blind reader of the Q3 call alone would likely miss.

**Ran across TSLA's full 39 consecutive quarter-pairs (2016-2026).** Tally:
**23 deteriorating, 14 stable, only 2 improving** (both in 2018, right after
the Model 3 "production hell" turnaround) -- and a live 6-quarter unbroken
deteriorating streak running from 2025Q2 through 2026Q2, the longest such run
in the dataset.

**Important tension, not glossed over:** guidance-credibility trajectory and
actual stock performance have clearly NOT moved together for Tesla -- the
stock delivered enormous returns over a decade where "deteriorating"
outnumbered "improving" more than 11-to-1. Consistent with the correlation
finding above (guidance credibility didn't survive as a general predictor):
this tool measures something real and coherent (is management's forecasting
discipline getting more or less honest over time), but that is demonstrably
not the same thing as "is this a good stock to own." Use it for what it
actually measures, not as a disguised price signal.

**Stance/price-target tool** (`src/agents/earnings_stance.py`): reads a
synthesis report plus the actual stock price at the time of the call, and
outputs bullish/neutral/bearish plus bull/base/bear 12-month % move scenarios
grounded in specific claims from the synthesis (not valuation multiples --
deliberately simple to avoid fabricating numbers the model has no basis for).
Includes an explicit instruction not to use hindsight knowledge of what
actually happened to the company, since a stance/price-target is much more
directly "the answer" a backtest would check than an abstract credibility
score was.

**Single spot-check outcome, illustrative not statistical:** on the same Oct
2019 Tesla call, the tool produced a well-reasoned NEUTRAL stance with a bull
case of +55% over 12 months. The stock's actual return was +736% -- more than
13x past even the most bullish scenario generated. Not a reasoning failure
(the read of the call itself was sound and appropriately hedged given what
was actually said) -- Tesla's 2020 move was driven by COVID-era retail mania,
S&P 500 inclusion speculation, and a broad growth-stock re-rating, none of
which any single earnings call could contain. This is a hard ceiling on what
a transcript-only tool can ever forecast, independent of how good the
reasoning is: treat price-target output as "a reasoned scenario spread given
only this document," never as a real forecast with statistical coverage.

**Status:** both tools work as designed and are useful analyst aids for
manual speculation. Not deployed as automated signals -- that's the whole
point of this reframing. If revisited: run the trajectory tool on HYLN (or
any other ticker) for a second data point on whether the "credibility and
price diverge" pattern is Tesla-specific or general; the stance tool's
single-call-hindsight-blindness caveat matters more for a backtest sample
than for genuine live/prospective use, where no hindsight problem exists at
all.

---

## 2026-09-18 — Multi-agent earnings-call analysis as a return predictor

**Idea:** analyze earnings-call transcripts through several independent
specialist lenses (tone/sentiment, guidance credibility, competitive
positioning, financial health), synthesize the four reads explicitly
surfacing agreement/conflict rather than averaging them, then test whether
any of it predicts forward returns.

**Pipeline built** (`src/agents/earnings_call_analyst.py`,
`src/research/{score_earnings_synthesis,build_earnings_call_returns,
correlate_earnings_scores}.py`): four `claude -p` specialist calls run in
parallel per transcript, a fifth call synthesizes them, a Haiku call extracts
5 structured 0-100 scores from the synthesis (tone_confidence,
guidance_credibility, competitive_strength, financial_health, conflict_level)
so results become correlatable rather than only qualitative. Transcripts
sourced by scraping stockanalysis.com's per-company transcript index (no
bulk transcript API exists) -- see "known gaps" below for what that implies
about scaling this beyond a couple of tickers.

**Qualitative pipeline output is genuinely strong.** Spot-checked against the
real Oct 2019 Tesla call: every claim in every report was grounded in an
actual, verifiable quote (not hallucinated), the guidance-credibility agent
correctly flagged Musk's live on-call redefinition of "feature-complete" as a
quiet walk-back and caught that a "Cybertruck production constrained this
quarter" remark was chronologically impossible (truck wasn't unveiled yet),
and the synthesis surfaced four genuine, unresolved tensions (e.g. confident
tone vs. weak evidentiary support on the same superlative claims) instead of
averaging them into a single score. This part of the idea works as designed.

**Quantitative test: built full historical panels for two tickers and
correlated the 5 scores against 6 forward-return horizons plus EPS surprise.**
- **TSLA (n=40 quarters, 2016-2026):** `guidance_credibility` correlated with
  the 1-year forward return at r=+0.39 (Pearson, p<0.05) and r=+0.41
  (Spearman, p<0.05) -- consistent across both measures, and it beat
  EPS-surprise-alone at the same horizon (r=-0.02). Builds from ~0 at short
  horizons to its peak at 252d, which is qualitatively sensible (evidence-backed
  vs. aspirational guidance shows up over a year, not next-day).
- **HYLN (n=23 quarters, 2020-2026), run specifically to test whether the TSLA
  result generalizes:** the same relationship appeared to replicate at first
  glance (r=+0.35) but Pearson/Spearman diverged sharply (+0.35 vs +0.15) --
  the classic signature of an outlier-driven correlation, not a real one.
  Confirmed directly: removing just 2 of 20 quarters (two extreme return
  outliers, +378% and +143% in one year -- HYLN's stock is meme-stock-volatile)
  flips the sign entirely, r=+0.35 -> r=-0.15. **The TSLA finding does not
  survive a second ticker.**

**What the two-ticker test DID establish, consistently:** raw EPS surprise
alone carries ~zero 1-year predictive signal in both names (r=-0.02 TSLA,
r=-0.09 HYLN) -- this part replicated cleanly, it's just that the proposed
fix (qualitative guidance-credibility score) turned out to be a Tesla-specific
artifact rather than a general one. Plausible reason: Tesla's guidance has a
uniquely legible "aspirational vs. evidence-backed" split under one specific,
highly-quotable CEO; most companies' guidance language may not carry the same
signal at all.

**Conclusion: not a usable signal, but a well-executed negative result** --
same shape as the sentiment and insider-trading findings above: plausible
hypothesis, working pipeline, doesn't survive rigorous testing. Filed rather
than pursued further. Not deployed.

**Known gaps / scaling reality if revisited:**
- No bulk transcript API exists; every ticker requires manually locating and
  scraping a transcript-index page (stockanalysis.com worked well for both
  TSLA and HYLN) then per-quarter scraping with pagination-safe extraction.
  This is a real one-time backfill cost per ticker (~5-10 min engineering +
  scraping time), not something that scales to hundreds of tickers casually.
- The melt-up screener's universe (overwhelmingly microcaps) mismatches this
  tool's natural fit -- most microcaps don't host formal analyst earnings
  calls with transcripts at all. This tool fits the PEAD-style large-cap
  universe (`fetch_earnings_calendar.py`'s tickers), not the melt-up universe
  (see the melt-up screener entry below for that universe's characteristics).
- The guidance-credibility score partly draws on the underlying model's
  general/training-data knowledge of a company's known track record (by
  design, for judging plausibility against past promises) -- appropriate for
  backtesting already-known history, but means this specific mechanism won't
  carry the same advantage when analyzing a brand-new call after the model's
  training cutoff.
- If revisited, a third ticker with a genuinely different guidance style
  (neither Musk-style grandiose promises nor HYLN's SPAC-hype volatility)
  would be needed before concluding anything either way -- two tickers proved
  enough to falsify the hypothesis but were not intended as, and are not, a
  general-purpose validation.

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
