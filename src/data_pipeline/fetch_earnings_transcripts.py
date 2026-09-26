import argparse
import html
import re
import time
from pathlib import Path

import requests

BASE_URL = "https://stockanalysis.com"
OUTPUT_DIR = Path("data/earnings_calls")
DEFAULT_DELAY_SECONDS = 3.0
REQUEST_TIMEOUT_SECONDS = 30
MAX_ATTEMPTS = 4
RETRY_STATUS = {429, 500, 502, 503, 504}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:130.0) Gecko/20100101 Firefox/130.0",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

QUARTER_LINK_RE = re.compile(r'href="(/stocks/{ticker}/transcripts/\d+-q([1-4])-(\d{{4}})/)"')
BLOCK_SPLIT_RE = re.compile(r'<div class="border-t border-sharp pt-5')
SPEAKER_RE = re.compile(r'<div class="text-lg font-bold[^"]*">(.*?)</div>', re.DOTALL)
TITLE_RE = re.compile(r'<div class="text-sm italic text-muted">(.*?)</div>', re.DOTALL)
SENTENCE_RE = re.compile(r'<span class="transcript-sentence[^"]*"[^>]*>(.*?)</span>', re.DOTALL)
PANEL_MARKER = 'id="transcript-panel-full"'
TAG_RE = re.compile(r"<[^>]+>")


def _get(session: requests.Session, url: str) -> str:
    last_error = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
            if response.status_code == 200:
                return response.text
            last_error = RuntimeError(f"HTTP {response.status_code} for {url}")
            if response.status_code not in RETRY_STATUS:
                raise last_error
        except (requests.Timeout, requests.ConnectionError) as e:
            last_error = e
        if attempt < MAX_ATTEMPTS - 1:
            time.sleep(5 * (2 ** attempt))
    raise last_error


def _clean(fragment: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(TAG_RE.sub("", fragment))).strip()


def list_quarters(session: requests.Session, ticker: str) -> dict[str, str]:
    page = _get(session, f"{BASE_URL}/stocks/{ticker.lower()}/transcripts/")
    pattern = re.compile(QUARTER_LINK_RE.pattern.format(ticker=re.escape(ticker.lower())))
    quarters = {}
    for path, quarter, year in pattern.findall(page):
        quarters[f"{year}Q{quarter}"] = path
    return quarters


def parse_transcript(page: str) -> str:
    start = page.find(PANEL_MARKER)
    if start == -1:
        raise ValueError("transcript panel not found")
    lines = []
    for block in BLOCK_SPLIT_RE.split(page[start:])[1:]:
        speaker = SPEAKER_RE.search(block)
        sentences = [_clean(s) for s in SENTENCE_RE.findall(block)]
        sentences = [s for s in sentences if s]
        if not speaker or not sentences:
            continue
        lines.append(_clean(speaker.group(1)))
        title = TITLE_RE.search(block)
        if title:
            lines.append(_clean(title.group(1)))
        lines.extend(sentences)
    if not lines:
        raise ValueError("no transcript text extracted")
    return "\n".join(lines) + "\n"


def fetch_ticker(session: requests.Session, ticker: str, output_dir: Path, delay: float, overwrite: bool, only: set[str] | None) -> None:
    quarters = list_quarters(session, ticker)
    if not quarters:
        print(f"[{ticker}] no quarterly transcripts found")
        return
    labels = sorted(quarters)
    if only:
        labels = [l for l in labels if l in only]
    print(f"[{ticker}] {len(quarters)} quarterly transcripts listed, {len(labels)} selected")

    output_dir.mkdir(parents=True, exist_ok=True)
    for label in labels:
        out_path = output_dir / f"{ticker}_{label}.txt"
        if out_path.exists() and not overwrite:
            print(f"[{ticker} {label}] already exists, skipping")
            continue
        time.sleep(delay)
        try:
            text = parse_transcript(_get(session, BASE_URL + quarters[label]))
            out_path.write_text(text)
            print(f"[{ticker} {label}] -> {out_path} ({len(text):,} chars)")
        except Exception as e:
            print(f"[{ticker} {label}] FAILED: {type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--only", nargs="+", default=None, help="quarter labels such as 2023Q1")
    args = parser.parse_args()

    only = set(args.only) if args.only else None
    with requests.Session() as session:
        for ticker in args.tickers:
            fetch_ticker(session, ticker.upper(), args.output_dir, args.delay, args.overwrite, only)


if __name__ == "__main__":
    main()
