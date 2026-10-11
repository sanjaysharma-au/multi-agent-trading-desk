"""Sector, industry and a short business description for a ticker, cached on disk."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import yfinance as yf

log = logging.getLogger(__name__)

SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
MAX_SUMMARY_CHARS = 280


def shorten(summary: str, max_chars: int = MAX_SUMMARY_CHARS, max_sentences: int = 2) -> str:
    summary = " ".join((summary or "").split())
    if not summary:
        return ""
    out = ""
    for sentence in SENTENCE_BREAK.split(summary)[:max_sentences]:
        if out and len(out) + 1 + len(sentence) > max_chars:
            break
        out = f"{out} {sentence}".strip()
    if len(out) > max_chars:
        out = out[:max_chars].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    return out


def profile_path(ticker: str, data_dir: Path) -> Path:
    return data_dir / "profiles" / f"{ticker.upper()}.json"


def _sic_description(ticker: str, sec_dir: Path, ciks: dict[str, int]) -> str:
    path = sec_dir / f"sub_{ciks.get(ticker.upper(), 0)}.json"
    return (json.loads(path.read_text()).get("sicDescription") or "") if path.exists() else ""


def load_profile(ticker: str, data_dir: Path, ciks: dict[str, int] | None = None, max_age_days: float = 90) -> dict:
    path = profile_path(ticker, data_dir)
    if path.exists() and time.time() - path.stat().st_mtime < max_age_days * 86400:
        return json.loads(path.read_text())
    info = {}
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as exc:
        log.warning("profile lookup failed for %s: %s", ticker, exc)
    profile = {
        "sector": info.get("sector") or "",
        "industry": info.get("industry") or "",
        "summary": shorten(info.get("longBusinessSummary") or ""),
    }
    if not profile["industry"] and ciks is not None:
        profile["industry"] = _sic_description(ticker, data_dir / "sec", ciks)
    if not any(profile.values()) and path.exists():
        return json.loads(path.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile))
    return profile
