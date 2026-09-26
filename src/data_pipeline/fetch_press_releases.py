import argparse
import html
import os
import re
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

load_dotenv()

OUTPUT_DIR = Path("data/press_releases")
TRANSCRIPT_DIR = Path("data/earnings_calls")
DEFAULT_DELAY_SECONDS = 0.25
REQUEST_TIMEOUT_SECONDS = 30
MAX_ATTEMPTS = 4
MAX_TEXT_CHARS = 150_000
KNOWN_CIKS = {"NVDA": 1045810, "PLTR": 1321655}
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4}

HEAD_CHARS = 1500
QUARTER_PATTERNS = [
    ("ordinal_year", re.compile(r"\b(first|second|third|fourth)\s+quarter\s+(?:and\s+(?:full\s+)?(?:fiscal\s+)?(?:year\s+)?)?(?:of\s+)?(?:fiscal\s+(?:year\s+)?)?(?:fy\s*)?'?(20\d\d|\d\d)\b", re.I)),
    ("q_year", re.compile(r"\bQ([1-4])\s*(?:fy|fiscal(?:\s+year)?)?\s*'?(20\d\d|\d\d)\b", re.I)),
    ("year_ordinal", re.compile(r"\bfiscal\s+(?:year\s+)?(20\d\d)\s+(first|second|third|fourth)\s+quarter\b", re.I)),
]


def _headers() -> dict[str, str]:
    agent = os.environ.get("SEC_USER_AGENT", "").strip()
    if not agent:
        raise SystemExit(
            "SEC_USER_AGENT is not set. The SEC requires a contact in the User-Agent of automated requests: "
            "add SEC_USER_AGENT=<name and email> to .env (see .env.example)."
        )
    return {"User-Agent": agent, "Accept-Encoding": "gzip, deflate"}


def _get(session: requests.Session, url: str, delay: float) -> requests.Response:
    last_error = None
    for attempt in range(MAX_ATTEMPTS):
        time.sleep(delay)
        try:
            response = session.get(url, headers=_headers(), timeout=REQUEST_TIMEOUT_SECONDS)
            if response.status_code == 200:
                return response
            last_error = RuntimeError(f"HTTP {response.status_code} for {url}")
            if response.status_code not in {429, 500, 502, 503, 504}:
                raise last_error
        except (requests.Timeout, requests.ConnectionError) as e:
            last_error = e
        time.sleep(3 * (2 ** attempt))
    raise last_error


def find_cik(session: requests.Session, ticker: str, delay: float) -> int:
    if ticker in KNOWN_CIKS:
        return KNOWN_CIKS[ticker]
    data = _get(session, "https://www.sec.gov/files/company_tickers.json", delay).json()
    for entry in data.values():
        if entry["ticker"].upper() == ticker:
            return int(entry["cik_str"])
    raise SystemExit(f"no CIK found for {ticker}; pass --cik")


def list_earnings_filings(session: requests.Session, cik: int, delay: float) -> list[dict]:
    base = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
    main = _get(session, base, delay).json()
    blocks = [main["filings"]["recent"]]
    for older in main["filings"].get("files", []):
        blocks.append(_get(session, f"https://data.sec.gov/submissions/{older['name']}", delay).json())
    filings = []
    for block in blocks:
        for i, form in enumerate(block["form"]):
            if form == "8-K" and "2.02" in block["items"][i]:
                filings.append({"date": block["filingDate"][i], "accession": block["accessionNumber"][i]})
    return sorted(filings, key=lambda f: f["date"])


def exhibit_urls(session: requests.Session, cik: int, accession: str, delay: float) -> list[str]:
    nodash = accession.replace("-", "")
    index = _get(session, f"https://www.sec.gov/Archives/edgar/data/{cik}/{nodash}/{accession}-index.htm", delay).text
    urls = []
    for row in re.findall(r"<tr[^>]*>.*?</tr>", index, flags=re.S | re.I):
        type_match = re.search(r">\s*(EX-99(?:\.\d+)?)\s*<", row, flags=re.I)
        link = re.search(r'href="([^"]+\.(?:htm|html|txt))"', row, flags=re.I)
        if type_match and link:
            urls.append((type_match.group(1).upper(), "https://www.sec.gov" + link.group(1)))
    return [url for _, url in sorted(urls)]


def html_to_text(raw: str) -> str:
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
    text = re.sub(r"</t[dh]\s*>", " | ", text, flags=re.I)
    text = re.sub(r"</(tr|p|div|h[1-6]|li)\s*>|<br\s*/?>", "\n", text, flags=re.I)
    text = html.unescape(re.sub(r"<[^>]+>", " ", text)).replace("\xa0", " ")
    lines = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.split("|")]
        line = " | ".join(c for c in cells if c and c not in {"$", ")", "%"})
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


FORWARD_LOOKING = re.compile(r"(expect|outlook|guidance|guide|forecast|project|anticipat|target|for the upcoming)", re.I)
FORWARD_WINDOW = 45


def _label_from_match(kind: str, match: re.Match) -> str:
    first, second = match.group(1), match.group(2)
    if kind == "year_ordinal":
        first, second = second, first
    quarter = ORDINALS.get(first.lower()) if not first.isdigit() else int(first)
    year = int(second)
    year = year + 2000 if year < 100 else year
    return f"{year}Q{quarter}"


def label_from_text(text: str) -> str | None:
    head = text[:HEAD_CHARS]
    matches = []
    for kind, pattern in QUARTER_PATTERNS:
        for match in pattern.finditer(head):
            forward = bool(FORWARD_LOOKING.search(head[max(0, match.start() - FORWARD_WINDOW):match.start()]))
            matches.append((forward, match.start(), kind, match))
    if not matches:
        return None
    matches.sort(key=lambda m: (m[0], m[1]))
    _, _, kind, match = matches[0]
    return _label_from_match(kind, match)


def label_from_llm(text: str, model: str, backend: str) -> str | None:
    from agents.earnings_call_analyst import _call_llm
    from agents.moonshot_ledger import parse_json_object

    prompt = (
        "Which fiscal quarter does this earnings press release report? Use the company's own fiscal labels as "
        'written in the document. Output ONLY JSON: {"quarter": 1-4, "fiscal_year": YYYY}.\n\n' + text[:2500]
    )
    try:
        result = parse_json_object(_call_llm("You read financial documents.", prompt, model=model, backend=backend, label="press-quarter"))
        return f"{int(result['fiscal_year'])}Q{int(result['quarter'])}"
    except (ValueError, KeyError, TypeError):
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Download each quarter's earnings press release (SEC 8-K item 2.02, exhibit 99) for a ticker."
    )
    parser.add_argument("ticker")
    parser.add_argument("--cik", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--transcript-dir", type=Path, default=TRANSCRIPT_DIR)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS)
    parser.add_argument("--backend", choices=["nemotron", "claude"], default="nemotron",
                        help="model used only to read the quarter from a release whose title the patterns cannot parse")
    parser.add_argument("--model", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    ticker = args.ticker.upper()

    wanted = sorted(
        p.stem.replace(f"{ticker}_", "") for p in args.transcript_dir.glob(f"{ticker}_*Q*.txt")
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    found = {l for l in wanted if (args.output_dir / f"{ticker}_{l}.txt").exists()}

    model = args.model
    if model is None:
        from agents.earnings_call_analyst import DEFAULT_MODEL
        model = DEFAULT_MODEL if args.backend == "nemotron" else "haiku"

    with requests.Session() as session:
        cik = args.cik or find_cik(session, ticker, args.delay)
        filings = list_earnings_filings(session, cik, args.delay)
        print(f"[{ticker}] CIK {cik}: {len(filings)} earnings 8-K filings, {len(wanted)} transcript quarters wanted")

        for filing in filings:
            try:
                urls = exhibit_urls(session, cik, filing["accession"], args.delay)
                if not urls:
                    print(f"[{filing['date']}] no EX-99 exhibit, skipping")
                    continue
                texts = [html_to_text(_get(session, u, args.delay).text) for u in urls]
                text = "\n\n".join(texts)[:MAX_TEXT_CHARS]
                label = label_from_text(text) or label_from_llm(text, model, args.backend)
                if not label:
                    print(f"[{filing['date']}] could not tell which quarter this release reports, skipping")
                    continue
                if wanted and label not in wanted:
                    print(f"[{filing['date']}] {label} has no transcript, skipping")
                    continue
                out_path = args.output_dir / f"{ticker}_{label}.txt"
                if out_path.exists() and not args.overwrite and len(out_path.read_text()) >= len(text):
                    print(f"[{filing['date']}] {label} already on disk and at least as long, skipping")
                    continue
                out_path.write_text(f"{ticker} earnings release, filed {filing['date']}\n\n{text}\n")
                found.add(label)
                print(f"[{filing['date']}] {label} -> {out_path} ({len(text):,} chars)")
            except Exception as e:
                print(f"[{filing['date']}] FAILED: {type(e).__name__}: {e}")

    missing = sorted(set(wanted) - found)
    (args.output_dir / f"_unavailable_{ticker}.txt").write_text("".join(l + "\n" for l in missing))
    print(f"[{ticker}] {len(found & set(wanted))} of {len(wanted)} quarters have a press release; unavailable: {missing}")


if __name__ == "__main__":
    main()
