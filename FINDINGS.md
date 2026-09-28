# Findings: building the moonshot-screener pipeline

Written for: another chat/agent (possibly backed by Nemotron) being handed this project to continue
or reproduce the work. This documents what was built, and — more importantly — every non-obvious
data bug that was found and fixed along the way, since those are the expensive, hard-won part of
this project and the easiest thing to silently re-introduce.

## What this project does

Two independent screeners over the US stock market, plus an LLM-based qualitative layer:

1. **`moonshot scan`** — finds stocks that already rose N× within a time window, purely from price
   history (Yahoo Finance via `yfinance`). Backward-looking.
2. **`moonshot inflect`** — finds companies whose *SEC filings* (not price) show a "Palantir-shaped"
   event: a first-time, sustained crossing from unprofitable into GAAP profit + positive free cash
   flow (`--mode crossing`), or the mirror-image "Canoo-shaped" failure to ever prove the business
   model (`--mode failure`). Independent of stock price entirely.
3. **`moonshot rate`** — sends a company's SEC filings (and only its filings — never price, never
   outcome) to an NVIDIA Nemotron model for a qualitative "fundamental trajectory" score from -100
   to +100.

Two hard design constraints, both explicit user requirements, both load-bearing for the whole
project's validity:
- **Blind to price.** The LLM never sees the ticker's stock price, its multiple, or market cap
  (market cap = price × shares, so it's excluded too). The point is to judge the *business*, not
  react to a price chart it already recognizes.
- **Blind to outcome/hindsight.** The LLM is never shown what happened to a company *after* the
  period its filing data covers — no bankruptcy 8-Ks, no delisting notices. If you show it the
  outcome, you're not testing a signal, you're just building an expensive way to look up something
  you already told the model. This constraint is why the fix for finding #5 below had to be "give
  the model a better-computed summary of data it already had," not "just show it the bankruptcy."

## Key files (what to show another agent to reproduce this)

| File | What's in it |
|---|---|
| `README.md` | Full usage docs for every command, written as the fixes landed |
| `src/moonshot/fundamentals.py` | SEC EDGAR fetching, the two XBRL quarterization strategies (see finding #1) |
| `src/moonshot/multibagger.py` | Price-based run detection, including the bankruptcy-splice segmentation (finding #2) |
| `src/moonshot/inflection.py` | The crossing/failure detectors — read the docstrings, they explain every guard and why it exists |
| `src/moonshot/scoring.py` | The Nemotron prompt and API integration — read `SYSTEM_PROMPT` and `format_lifetime_totals` |
| `src/moonshot/cli.py` | How everything is wired together as commands |
| `tests/test_inflection.py`, `tests/test_scoring.py`, `tests/test_multibagger.py` | Every bug below has a regression test named after the real-world case that caught it |

For just understanding the *bugs* (this document) without reading code, this file alone is enough.
For reproducing or extending the work, start with `inflection.py` and its tests — that's where most
of the hard-won correctness lives.

## The five real bugs found (in the order discovered)

### 1. SEC cash-flow figures are usually cumulative year-to-date, not per-quarter

**Symptom:** a generic "get quarterly values from XBRL" function (written for revenue/EPS, which
are usually tagged as discrete quarters) silently dropped most operating-cash-flow and capex data.

**Root cause:** a company's Q2 10-Q duration fact for `NetCashProvidedByUsedInOperatingActivities`
often has `start = January 1` (not April 1) — it's the first-half total, not Q2 alone. Q3's fact is
often the nine-month total. This is standard SEC/XBRL practice for cash-flow-statement line items,
unlike income-statement items which are more often tagged per-quarter directly.

**Fix:** `_quarterize_cumulative()` in `fundamentals.py` groups facts by shared `start` date (same
fiscal year) and differences consecutive `end` dates to recover each quarter's own value. A company
that instead tags genuinely discrete quarters (each with its own distinct `start`) forms singleton
groups, so nothing is differenced and the raw value is used as-is — one code path handles both
conventions without needing to know in advance which one a filer uses.

### 2. Bankruptcy-emergence price splices create fake multi-hundred-x "multibaggers"

**Symptom:** the price-based scanner reported some absurd multiples (Chord Energy at 1,989×, Amplify
Energy at 152×) that weren't real gains anyone could have held.

**Root cause:** when a company emerges from Chapter 11, old equity is often cancelled and new shares
issued to creditors. Price-data vendors sometimes splice the pre- and post-emergence prices into one
continuous series under the same ticker, even though they're not the same security. A "5,000%
gain" spanning that splice isn't a real return.

**Fix:** `find_best_run()` in `multibagger.py` splits a ticker's price history into segments at any
overnight move bigger than `max_jump` (default 10×) in the *raw* adjusted close. A qualifying run can
never start in one segment and end in another. (A subtler version of the same bug: a start day's
5-day rolling median could straddle the jump and produce a smoothed "price" that never actually
traded — fixed by the same segmentation, since smoothing now never crosses a segment boundary either.)

### 3. Banks/REITs/insurers use different revenue XBRL tags → false "Canoo-shaped failures"

**Symptom:** `find_failure` (the "never proved the business model" detector) flagged Goldman Sachs,
Truist, Annaly Capital, and ~175 other clearly-profitable financial companies as failures.

**Root cause:** this pipeline's revenue concept list (`Revenues`,
`RevenueFromContractWithCustomerExcludingAssessedTax`, etc.) is standard for operating companies, but
banks/REITs/insurers report their top line as interest or premium income under different XBRL
concepts entirely. Those companies showed "$0 revenue in every quarter, ever" in this pipeline's
data — not because they had none, but because the pipeline wasn't reading the right concept. Since
`find_failure`'s core signal is `cumulative revenue / cumulative losses`, a $0 numerator against any
historical loss quarter (even one from a one-off event years earlier) produced a spurious near-zero
ratio for an obviously healthy company.

**Fix:** two new guards in `find_failure`: (a) skip entirely if the company has *zero* quarters of
any recorded revenue across its whole history — that's a sign of a tagging gap, not evidence of a
real zero-revenue company (a real failure like Canoo still had 5 of 23 quarters with *some* revenue,
so this doesn't exclude genuine cases); (b) skip if net income has been positive on balance over the
trailing `min_quarters` window — a company cannot sustain real profitability with genuinely zero
revenue, so recent positive net income overriding a bad ratio is itself evidence of a tagging gap.

### 4. The *same* tagging-gap bug also broke `find_crossing`, not just `find_failure`

**Symptom:** HCA Healthcare, Realty Income (O), Extra Space Storage (EXR), and Terreno Realty (TRNO)
all showed a "first-time profit + free-cash-flow crossing" on the *identical* date, 2017-03-31. Four
unrelated large-caps crossing on the same day is a tell for a shared artifact, not four independent
business-model proofs.

**Root cause:** same as #3, but on the crossing side. HCA's hospital-revenue concept wasn't in this
pipeline's list, so its revenue read as null for years — during which HCA actually earned **$13.7
billion in cumulative net income**. When HCA's data started using a concept the pipeline recognizes
(plausibly coinciding with ASC 606 revenue-recognition adoption industry-wide around 2017-2018), it
looked exactly like a textbook stage-0-to-stage-3 crossing, even though the company had been hugely
profitable the entire time.

**Fix:** `find_crossing` now also requires that cumulative net income across *every* quarter before
the candidate crossing point is negative — a real "unproven business" candidate must have actually
lost money overall pre-crossing, not just have an artifact-zero revenue reading while secretly
profitable. (Sanity check this preserved: Charter Communications (CHTR) still correctly crosses —
it has real, always-tagged revenue and ~20 quarters of genuine losses before turning profitable in
2016, a real leveraged-scale-up story, not a data artifact.)

### 5. Nemotron mistook "a dying company cutting costs" for "a turnaround"

**Symptom:** scoring Canoo (GOEV) and Lordstown Motors (RIDE) — both of which later went bankrupt —
using only their pre-bankruptcy quarterly numbers (no hindsight, per the design constraint above)
returned +60 and +35, both rationales calling it a "clear turnaround."

**Root cause:** a company that is genuinely running out of money often shows *shrinking losses* in
its last few quarters, simply because it's slashing spending to survive. That looks, quarter to
quarter, identical to a business genuinely improving its unit economics. A trend-only read (is the
recent trajectory improving?) can't tell these apart — and an earlier version of the pure structural
`find_failure` detector had the *same* flaw before it was redesigned around cumulative ratios instead
of trend (see the git history / the `find_failure` docstring for that earlier design mistake).

**Fix:** every prompt to Nemotron now includes a computed `Lifetime totals` line — cumulative revenue
as a percentage of cumulative losses, the exact same ratio `find_failure` uses — plus an explicit
system-prompt rule (see `SYSTEM_PROMPT` in `scoring.py`) telling the model that narrowing losses
funded by cost-cutting, with revenue still flat, is not the same as a real turnaround. Critically,
this is *not new information* — it's a summary of the same historical numbers already in the prompt,
computed with no outside knowledge and no outcome leak. After the fix, using the identical
pre-bankruptcy data, GOEV scored -20 and RIDE scored -40, both citing the near-zero ratio by name in
their own rationale.

## The meta-lesson

Every one of these bugs was found the same way: something looked *statistically implausible*
(four large-caps crossing on the identical date; a Fortune-500 bank flagged as near-bankruptcy; a
company that later filed Chapter 7 scoring +60), not by reading code and spotting a typo. If you're
extending this pipeline, the highest-value habit is the same one that found all five: when a result
looks too clean, too coincidental, or too surprising for the story to make sense, check the raw
underlying data for that specific case before trusting the aggregate output.
