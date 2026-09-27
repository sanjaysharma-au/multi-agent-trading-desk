"""Score a stock's fundamental trajectory with an NVIDIA Nemotron model, blind to its stock price.

Design:
- The model only ever sees SEC-filed facts: quarterly revenue/EPS/net income and excerpts of the
  company's own earnings press releases. It is never given the price, the multiple, the run dates,
  or market cap (market cap is price x shares, so it counts as price data and is excluded too).
- The system prompt also tells the model to ignore anything it already knows about the stock's
  trading history, in case it recognizes the company from training data (e.g. a "meme stock").
- Every ticker's result is written to a checkpoint file immediately, so a crash, a rate limit, or a
  Ctrl-C loses at most the one ticker in flight. Rerunning the same command resumes automatically:
  already-scored tickers are skipped, and failed ones are retried up to `max_retries` times across
  runs.
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests

from .fundamentals import cik_map, exhibit_links, load_fundamentals, sec_get

log = logging.getLogger(__name__)

NEMOTRON_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"

SYSTEM_PROMPT = """You are a fundamental equity research analyst. You will be shown one company's \
SEC-filed quarterly results and excerpts from its own earnings press releases.

Score the company's fundamental trajectory - NOT its stock price - on a scale from -100 to 100:
  +100 = an extraordinary positive surprise: a genuine moonshot inflection in the business itself \
(e.g. revenue or profit breaking out far above its prior trend, a turn from losses to strong \
profitability, guidance raised sharply, a clear structural improvement).
   +50 = strong, sustained improvement (consistent high growth, improving margins, no red flags).
     0 = starting point / steady state - performance roughly continues its prior trend, with \
nothing that clearly breaks out or breaks down.
   -50 = meaningful deterioration (declining revenue, widening losses, weakening disclosures).
  -100 = severe distress: filings show real bankruptcy risk (going-concern language, collapsing \
revenue, deep and worsening losses, insolvency or liquidity warnings).

Rules:
- Base the score ONLY on the filings data given below: the reported numbers and the press-release \
text. Do not use any other information about the company.
- You may already know this company and how its stock performed - including if it is a well-known \
"meme stock" or had a famous rally or crash. Ignore all of that completely. Do not mention, infer, \
or let your score be influenced by the stock price, trading volume, short interest, or market \
narrative in any way. If you are not confident you can separate the two, base the score strictly \
on the numbers and text provided and say so in the rationale.
- Weigh the most recent quarters most heavily, but note the trend across the whole history given.
- Respond with ONLY a single JSON object, no other text: \
{"score": <integer from -100 to 100>, "rationale": "<2-4 sentences citing specific quarters>"}"""


# ---------- text extraction ----------

_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.S | re.I)
_BLOCK_BREAK_RE = re.compile(r"</(p|div|tr|li|h[1-6])>|<br\s*/?>", re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\xa0]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def html_to_text(raw: str) -> str:
    """Rough but dependency-free HTML-to-text: good enough for an LLM prompt, not for display."""
    raw = _SCRIPT_STYLE_RE.sub(" ", raw)
    raw = _BLOCK_BREAK_RE.sub("\n", raw)
    text = _TAG_RE.sub(" ", raw)
    text = html.unescape(text)
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return _BLANK_LINES_RE.sub("\n\n", text).strip()


def fetch_release_text(url: str, sec_dir: Path, max_chars: int = 6000) -> str | None:
    """Fetch and cache one press release's text. Filings never change, so the cache is permanent."""
    import hashlib

    cache_dir = sec_dir / "release_text"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{hashlib.sha1(url.encode()).hexdigest()}.txt"
    if not path.exists():
        resp = sec_get(url)
        if resp is None:
            return None
        path.write_text(html_to_text(resp.text))
    text = path.read_text()
    return text[:max_chars] if text else None


# ---------- prompt building ----------

def _fmt_money(v: float | None) -> str:
    if v is None or pd.isna(v):
        return "–"
    a, sign = abs(v), "-" if v < 0 else ""
    if a >= 1e9:
        return f"{sign}${a / 1e9:.2f}B"
    if a >= 1e6:
        return f"{sign}${a / 1e6:.1f}M"
    if a >= 1e3:
        return f"{sign}${a / 1e3:.0f}K"
    return f"{sign}${a:.0f}"


def format_fundamentals_table(f: pd.DataFrame) -> str:
    """Quarterly numbers only - no price, no multiple, no dates about the stock's run."""
    lines = [f"{'quarter ended':<13} {'announced':<11} {'revenue':>10} {'yoy':>8} {'eps':>7} {'net income':>11}"]
    for r in f.itertuples():
        yoy = f"{r.revenue_yoy_pct:+.1f}%" if pd.notna(r.revenue_yoy_pct) else "–"
        eps = f"${r.eps:.2f}" if pd.notna(r.eps) else "–"
        lines.append(
            f"{r.period_end.date()!s:<13} {r.announced.date()!s:<11} {_fmt_money(r.revenue):>10} "
            f"{yoy:>8} {eps:>7} {_fmt_money(r.net_income):>11}"
        )
    return "\n".join(lines)


def recent_release_texts(f: pd.DataFrame, sec_dir: Path, n: int) -> list[str]:
    """Text of the last `n` earnings press releases available (most recent last)."""
    texts = []
    recent = f.dropna(subset=["accession", "cik"])
    recent = recent[recent["announced_source"] == "8-K"].sort_values("announced").tail(n)
    for r in recent.itertuples():
        links = exhibit_links(int(r.cik), r.accession, sec_dir)
        url = links.get("release")
        if not url:
            continue
        text = fetch_release_text(url, sec_dir)
        if text:
            texts.append((r.period_end.date(), text))
    return texts


def build_user_prompt(ticker: str, name: str | None, sector: str | None, industry: str | None,
                      table_text: str, releases: list[tuple]) -> str:
    parts = [f"Ticker: {ticker}", f"Company: {name or 'unknown'}"]
    if sector:
        parts.append(f"Sector / industry: {sector} / {industry or 'unknown'}")
    parts.append("\nQuarterly fundamentals from SEC filings (most recent last):\n" + table_text)
    for period_end, text in releases:
        parts.append(f"\n--- Earnings press release for quarter ended {period_end} ---\n{text}")
    return "\n".join(parts)


# ---------- the API call ----------

_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE_RE = re.compile(r"```(?:json)?")
_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_score(content: str) -> dict:
    """Extract {"score", "rationale"} from a model reply.

    Some Nemotron variants are reasoning models that prepend a <think>...</think> block, or wrap
    the answer in a markdown code fence; both are stripped before looking for the JSON object.
    """
    cleaned = _FENCE_RE.sub("", _THINK_RE.sub("", content))
    m = _JSON_RE.search(cleaned)
    if not m:
        raise ValueError(f"no JSON object in model output: {content[:300]!r}")
    obj = json.loads(m.group(0))
    score = max(-100, min(100, int(round(float(obj["score"])))))
    return {"score": score, "rationale": str(obj.get("rationale", "")).strip()}


def call_nemotron(user_prompt: str, api_key: str, model: str = DEFAULT_MODEL,
                  timeout: float = 90, max_retries: int = 3) -> dict:
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                NEMOTRON_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_prompt}],
                    "temperature": 0.2,
                    # This is a reasoning model: its internal <think> trace shares this budget with
                    # the final JSON answer, and empirically runs to ~1-1.5k tokens on this prompt.
                    # Too low a limit truncates the reply mid-JSON before the answer is reached.
                    "max_tokens": 3000,
                },
                timeout=timeout,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = min(60, 5 * attempt)
                log.warning("Nemotron %s, waiting %ss (attempt %d/%d)", resp.status_code, wait, attempt, max_retries)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            choice = resp.json()["choices"][0]
            if choice.get("finish_reason") == "length":
                raise ValueError("response hit the token limit before finishing (increase max_tokens)")
            return parse_score(choice["message"]["content"])
        except Exception as exc:  # noqa: BLE001 - deliberately broad; caller decides what to do
            last_err = exc
            log.warning("Nemotron call failed (attempt %d/%d): %s", attempt, max_retries, exc)
            time.sleep(min(20, 3 * attempt))
    raise RuntimeError(f"Nemotron call failed after {max_retries} attempts: {last_err}")


def load_api_key(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    for var in ("NEMOTRON_API_KEY", "NVIDIA_API_KEY"):
        if os.environ.get(var):
            return os.environ[var]
    for env_path in (Path(".env"), Path(__file__).resolve().parents[2] / ".env"):
        if not env_path.exists():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            if key.strip() in ("NEMOTRON_API_KEY", "NVIDIA_API_KEY"):
                return val.strip().strip("'\"")
    return None


# ---------- checkpointing ----------

@dataclass
class Checkpoint:
    """A JSON file recording, per ticker: done/failed, its score, and how many attempts it took.

    Saved atomically (write to a temp file, then rename) after every single ticker, so an
    interruption at any point loses at most the ticker in progress. Re-running the scan skips every
    ticker already marked "done" and resumes the rest.
    """

    path: Path
    data: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.path.exists():
            self.data = json.loads(self.path.read_text())

    def status_of(self, ticker: str) -> str | None:
        return self.data.get(ticker, {}).get("status")

    def attempts_of(self, ticker: str) -> int:
        return self.data.get(ticker, {}).get("attempts", 0)

    def mark_done(self, ticker: str, score: int, rationale: str, model: str) -> None:
        self.data[ticker] = {
            "status": "done", "score": score, "rationale": rationale, "model": model,
            "attempts": self.attempts_of(ticker) + 1, "updated": pd.Timestamp.now('UTC').isoformat(),
        }
        self._save()

    def mark_failed(self, ticker: str, error: object, model: str) -> None:
        self.data[ticker] = {
            "status": "failed", "error": str(error)[:800], "model": model,
            "attempts": self.attempts_of(ticker) + 1, "updated": pd.Timestamp.now('UTC').isoformat(),
        }
        self._save()

    def reset_failed(self) -> int:
        n = 0
        for v in self.data.values():
            if v.get("status") == "failed":
                v["status"] = "pending"
                n += 1
        if n:
            self._save()
        return n

    def clear(self) -> None:
        self.data = {}
        if self.path.exists():
            self.path.unlink()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, sort_keys=True))
        tmp.replace(self.path)  # atomic on the same filesystem


# ---------- the run loop ----------

def score_ticker(ticker: str, name: str | None, sector: str | None, industry: str | None,
                 sec_dir: Path, ciks: dict[str, int], api_key: str, model: str, releases_n: int,
                 api_retries: int) -> dict:
    """Score one ticker. Raises on failure (no fundamentals, no API response, bad output)."""
    f = load_fundamentals(ticker, sec_dir, ciks=ciks)
    if f is None or f.empty:
        raise ValueError("no SEC fundamentals available for this ticker")
    table_text = format_fundamentals_table(f)
    releases = recent_release_texts(f, sec_dir, releases_n)
    prompt = build_user_prompt(ticker, name, sector, industry, table_text, releases)
    return call_nemotron(prompt, api_key, model, max_retries=api_retries)


def run(tickers: list[tuple[str, str | None, str | None, str | None]], sec_dir: Path, checkpoint: Checkpoint,
       api_key: str, model: str, max_attempts: int, releases_n: int, sleep_between: float,
       api_retries: int, log_line=print) -> None:
    """Score every ticker not already done, skipping any that have exhausted `max_attempts`."""
    ciks = cik_map(sec_dir)
    for ticker, name, sector, industry in tickers:
        if checkpoint.status_of(ticker) == "done":
            continue
        if checkpoint.status_of(ticker) == "failed" and checkpoint.attempts_of(ticker) >= max_attempts:
            log_line(f"{ticker:6s} skipped (failed {checkpoint.attempts_of(ticker)}x already)")
            continue
        try:
            result = score_ticker(ticker, name, sector, industry, sec_dir, ciks, api_key, model,
                                  releases_n, api_retries)
            checkpoint.mark_done(ticker, result["score"], result["rationale"], model)
            log_line(f"{ticker:6s} {result['score']:+4d}  {result['rationale'][:110]}")
        except Exception as exc:  # noqa: BLE001 - one ticker's failure must not stop the batch
            checkpoint.mark_failed(ticker, exc, model)
            log_line(f"{ticker:6s} FAILED  {exc}")
        time.sleep(sleep_between)


def write_scores_csv(checkpoint: Checkpoint, out: Path) -> int:
    rows = [{"ticker": t, **v} for t, v in checkpoint.data.items() if v.get("status") == "done"]
    if not rows:
        return 0
    df = pd.DataFrame(rows)[["ticker", "score", "rationale", "model", "updated"]].sort_values(
        "score", ascending=False
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return len(df)


def status_lines(tickers: list[tuple[str, ...]], checkpoint: Checkpoint) -> list[str]:
    done = [t for t, *_ in tickers if checkpoint.status_of(t) == "done"]
    failed = [t for t, *_ in tickers if checkpoint.status_of(t) == "failed"]
    pending = len(tickers) - len(done) - len(failed)
    lines = [f"{len(done)}/{len(tickers)} scored, {len(failed)} failed, {pending} pending"]
    for t in failed:
        v = checkpoint.data[t]
        lines.append(f"  FAILED {t} (attempt {v.get('attempts', '?')}): {v.get('error', '')[:150]}")
    return lines
