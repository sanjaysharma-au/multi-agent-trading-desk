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
