# moonshot

Finds US stocks that **were** multibaggers: stocks that rose at least *N*x within a time window at some point in their price history. It uses free data from yfinance and the NASDAQ Trader symbol directory.

## Setup
```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
```

## Usage
```bash
moonshot scan                                   # full US universe, 5x within 3y, 10y lookback
moonshot scan --multiple 10 --window 5y --lookback max
moonshot scan --tickers NVDA,SMCI,AAPL          # specific tickers
moonshot scan --limit 300 --enrich              # first 300 tickers, add sector/industry/market cap
```
The top rows print to the terminal. The full ranked list is written to `results/multibaggers_<date>.csv`, or to `--out`.

## Visualizing
```bash
moonshot report                 # builds an HTML page from the latest results/multibaggers_*.csv and opens it
moonshot report path/to.csv --multiple 10 --no-open
moonshot scan --report          # scan, then build and open the report
```
The report is one self-contained HTML file with no server and no internet needed. It has:
- **Summary tiles**: number of hits, median peak multiple, how many still hold the threshold, and how many are now below where the run began.
- **Scatter**: peak multiple vs. multiple today, both on log scales. Hover a dot for details and click it to select that stock.
- **Price chart**: weekly adjusted close for the selected stock on a log scale, with the run shaded and the start and peak marked.
- **Table**: sortable and searchable. Click a row to chart that stock.

It uses the cached prices in `data/prices/`, so build it on the same machine as the scan.

### Fundamentals
Below the price chart, a second panel shows each stock's quarterly **revenue, revenue YoY growth, diluted EPS or net income**. It shares the price chart's timeline.
- **Placement:** each quarter sits on the date its results were **announced**, not on the date the quarter ended. That date is the first 8-K with item 2.02 (the earnings release) filed after the quarter ended. If no 8-K is found, the 10-Q/10-K filing date is used, and the tooltip says which source applied.
- **Announcement ticks:** small marks on the price chart's baseline show every announcement.
- **Source:** SEC EDGAR, free with no API key, covering about 2009 onward. Data is fetched only for the stocks in the scan results and cached in `data/sec/` for 7 days.
- **Q4:** Q4 values are derived as the fiscal year total minus Q1–Q3. Where a figure was later restated, the first-reported value is used.
- **Press releases:** each quarter links to its earnings press release (exhibit 99.1 of the 8-K), plus CFO commentary or slides (99.2) when the company files them. Click a bar, or use the quarter table under the chart. The first report run fetches about 40 filing pages per stock, roughly 10 minutes for 90 stocks. The links are cached permanently in `data/sec/exhibits/`, so later runs are fast.
- **Gaps:** foreign filers (20-F) and companies without XBRL have no data. Use `--no-fundamentals` to skip this panel.
- **User-Agent:** the SEC asks for a descriptive User-Agent. You can set one with `MOONSHOT_SEC_USER_AGENT="yourname you@example.com"`.

## Rating fundamentals with an LLM (blind to price)
`moonshot rate` sends each stock's SEC filings to an NVIDIA Nemotron model and asks it for a
**fundamental trajectory score from -100 to 100**: 0 is a neutral starting point, +100 is a genuine
business moonshot (a real breakout in the numbers, not the stock), and -100 is filings that point to
real bankruptcy risk.
```bash
export NEMOTRON_API_KEY=nvapi-...            # or put NEMOTRON_API_KEY=... in a local .env file (already done in this repo)
moonshot rate                                # scores the top 100 of the latest scan CSV by multiple
moonshot rate results/multibaggers_full.csv --top 50
moonshot rate --status                       # progress only, no API calls, no key needed
moonshot rate                                # re-running resumes: done tickers are skipped
moonshot rate --retry-failed                 # give failed tickers another attempt run
```

**Blind to price, on purpose.** The model is only ever shown: the ticker, company name, sector, its
quarterly revenue/EPS/net-income history from SEC EDGAR, and the text of its recent earnings press
releases. It never sees the price, the multiple, the run dates, or market cap (market cap is price
times shares, so it's excluded too). The system prompt also tells the model that it may already know
this company's stock history — including if it's a famous "meme stock" — and to ignore that
completely and score only the filed numbers and text. See the prompt in
[scoring.py](src/moonshot/scoring.py).

**Blind to the outcome, too.** The model is told to judge a company exactly as an analyst reading
these filings in real time would have — no hindsight about what happened to it afterward (e.g. a
bankruptcy filing is never shown; only earnings-related 8-Ks are). This matters for distressed
companies specifically: a company that's genuinely dying can shrink its losses quarter to quarter
just by slashing spending, which looks identical on paper to a business improving its unit
economics. Real cases hit exactly this trap: Canoo (GOEV) and Lordstown Motors (RIDE), both of which
later went bankrupt, initially scored +60/+35 ("clear turnaround") when the model saw only the raw
quarterly numbers. The fix is a `Lifetime totals` line in every prompt — cumulative revenue as a
percentage of cumulative losses, computed from the same historical data already shown, no outside
information — plus an explicit rule telling the model to weigh it. After that, the same two
companies (using the same pre-outcome data) scored -20/-40, correctly citing the near-zero ratio as
the reason. See `format_lifetime_totals` in [scoring.py](src/moonshot/scoring.py).

**Checkpointed and resumable.** Every ticker's result is written to
`data/scores/<csv-stem>.json` immediately after it's scored, so a crash, a rate limit, or Ctrl-C
loses at most the one ticker in progress. Simply running `moonshot rate` again resumes: it skips
every ticker already marked done and retries failed ones, up to `--max-attempts` (default 5) tries
across runs. `moonshot rate --status` reports progress (done/failed/pending, plus each failure's
error) without making any API calls or needing a key — this is the command to check status from chat.
`--restart` wipes the checkpoint and starts over; `--retry-failed` resets failed tickers to pending
first.

Scored results are written to `results/<csv-stem>_scores.csv` (ticker, score, rationale, model,
timestamp), sorted by score. **Model:** defaults to `nvidia/nemotron-3-super-120b-a12b`, confirmed working against your key (list your
account's available models at any time with `GET https://integrate.api.nvidia.com/v1/models`). It's a
reasoning model - it thinks before answering, which is why `call_nemotron` in
[scoring.py](src/moonshot/scoring.py) requests up to 3000 tokens per call even though the answer
itself is short. Pass `--model` to use a different one from your account's catalog.

### Viewing scores in the report
Once `moonshot rate` has written `results/<csv-stem>_scores.csv`, `moonshot report` picks it up
automatically (or pass `--scores path/to.csv`) and adds:
- A **"Fundamentals score" badge** above the price chart for the selected stock (colored green/red,
  with a qualitative label - Moonshot signal / Improving / Neutral / Weakening / Distress risk) and
  its rationale below, so you can compare the price story and the fundamentals story side by side.
- A **sortable "Fund. score" column** in the table (hover a value for the full rationale).

## How a run is detected
For every day that could start a run, the scanner finds the highest price over the next `--window` trading days. It divides that high by the start price and keeps the ticker if the best ratio is at least `--multiple`.
- Prices are adjusted for splits and dividends (`Adj Close`), so reported multiples are total return.
- A `--smooth-days` rolling median (default 5) removes one-day bad ticks.
- A start day counts only if the raw price is at least `--min-price` (default $1) and the 20-day average dollar volume is at least `--min-dollar-volume` (default $100k).

Output columns:
- `start_date`, `start_price`, `peak_date`, `peak_price` (adjusted prices)
- `multiple`
- `trading_days_to_peak`
- `threshold_crossed_date`: the first day it reached N times the start price
- `last_price`
- `last_vs_peak_pct`
- `current_multiple`: the last price over the start price, which shows whether the gains held

## Caching
Price history is cached per ticker in `data/prices/<lookback>/` and reused for `--max-age` days (default 1). The first full-universe run of about 6k tickers takes a while. Later runs take seconds.

## Limitations
- **Survivorship bias**: only stocks listed today are scanned. Multibaggers that were later acquired or delisted are missed.
- yfinance is an unofficial API and is rate-limited. Data errors happen, so check extreme hits by hand.
- Recent IPOs and SPACs can show huge short-window multiples. Look at `start_date` and `trading_days_to_peak`.

## Tests
```bash
.venv/bin/pytest
```
