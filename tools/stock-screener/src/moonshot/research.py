"""Ask Claude, with web search, why a stock has lagged since its fundamentals crossing."""

from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
from pathlib import Path

import pandas as pd

from .fundamentals import cik_map, load_fundamentals
from .report import _weekly_series, rise_pct
from .sector import etf_for_sic, load_etf_series, sic_for_ticker
from .valuation import load_valuation

CATEGORIES = ["ai_disruption", "sector_derating", "growth_slowdown", "guidance_cut", "company_specific",
              "macro_rates", "dilution_or_selling", "other"]
VERDICTS = ["likely_deserved", "possibly_undervalued", "unclear"]
TIMEOUT_SECONDS = 600
ALLOWED_TOOLS = ["WebSearch", "WebFetch"]

SYSTEM_PROMPT = (
    "You are an equity research analyst. Search the web for evidence about why a stock has lagged since "
    "its fundamentals turned profitable. Base every claim on what you find, name the sources, and say "
    "'unclear' where the evidence does not settle it. Do not speculate about the future price. "
    "Reply with a single JSON object and nothing else."
)


def research_path(ticker: str, data_dir: Path) -> Path:
    return data_dir / "research" / f"{ticker.upper()}.json"


def load_research(ticker: str, data_dir: Path) -> dict | None:
    path = research_path(ticker, data_dir)
    return json.loads(path.read_text()) if path.exists() else None


def _latest_row(ticker: str, results_dir: Path, prefix: str) -> pd.Series | None:
    files = [f for f in results_dir.glob(f"{prefix}_*.csv") if re.fullmatch(rf"{prefix}_\d{{8}}", f.stem)]
    if not files:
        return None
    df = pd.read_csv(max(files, key=lambda f: f.stem))
    match = df[df["ticker"] == ticker.upper()]
    return match.iloc[0] if len(match) else None


def _crossing_row(ticker: str, results_dir: Path) -> pd.Series | None:
    return _latest_row(ticker, results_dir, "inflections")


def _depressed_row(ticker: str, results_dir: Path) -> pd.Series | None:
    return _latest_row(ticker, results_dir, "depressed")


def _number(row: pd.Series, key: str) -> float | None:
    value = row.get(key)
    return None if value is None or pd.isna(value) else float(value)


def build_depressed_facts(ticker: str, row: pd.Series, data_dir: Path) -> dict:
    etf = row.get("sector_etf") or "SPY"
    year_ago = (pd.Timestamp.today().normalize() - pd.Timedelta(days=365) - pd.Timestamp("1970-01-01")).days
    return {
        "kind": "depressed", "ticker": ticker.upper(), "name": row.get("name"),
        "price": _number(row, "price"),
        "drawdown_52w_pct": _number(row, "dd_52w_pct"), "drawdown_3y_pct": _number(row, "dd_3y_pct"),
        "revenue_growth_ttm_pct": _number(row, "revenue_growth_pct"),
        "ttm_revenue": _number(row, "ttm_revenue"), "ttm_net_income": _number(row, "ttm_net_income"),
        "ttm_free_cash_flow": _number(row, "ttm_free_cash_flow"),
        "net_margin_pct": _number(row, "net_margin_pct"), "fcf_margin_pct": _number(row, "fcf_margin_pct"),
        "price_to_sales_now": _number(row, "ps_now"), "ps_vs_3y_median_pct": _number(row, "ps_vs_3y_median_pct"),
        "sector_etf": etf,
        "sector_etf_change_1y_pct": rise_pct(load_etf_series(etf, data_dir / "sector"), year_ago),
    }


def build_facts(ticker: str, row: pd.Series, data_dir: Path) -> dict:
    sec_dir = data_dir / "sec"
    ciks = cik_map(sec_dir)
    fnd = load_fundamentals(ticker, sec_dir, ciks=ciks)
    period_end = pd.Timestamp(row["crossing_period_end"])
    announced = fnd.loc[fnd["period_end"] == period_end, "announced"]
    known = (announced.iloc[0] if len(announced) else period_end) - pd.Timestamp("1970-01-01")
    etf = etf_for_sic(sic_for_ticker(ticker, sec_dir, ciks))
    facts = {
        "ticker": ticker.upper(), "name": row.get("name"),
        "crossing_quarter_end": f"{period_end:%Y-%m-%d}",
        "crossing_announced": f"{pd.Timestamp('1970-01-01') + known:%Y-%m-%d}",
        "stock_change_since_announcement_pct": rise_pct(_weekly_series(ticker, data_dir / "prices"), known.days),
        "sector_etf": etf,
        "sector_etf_change_same_period_pct": rise_pct(load_etf_series(etf, data_dir / "sector"), known.days),
    }
    valuation = load_valuation(ticker, data_dir)
    if valuation:
        rows = valuation["rows"]
        at = next((r for r in rows if r["period_end"] >= facts["crossing_quarter_end"]), None)
        facts["price_to_sales_at_crossing"] = at and at["ps"]
        facts["price_to_sales_now"] = valuation.get("ps_current") or rows[-1]["ps"]
        facts["market_cap_now"] = rows[-1]["mcap"]
    recent = fnd.sort_values("period_end").tail(4)
    facts["ttm_revenue"] = float(recent["revenue"].sum())
    facts["ttm_net_income"] = float(recent["net_income"].sum())
    facts["latest_revenue_yoy_pct"] = float(fnd.sort_values("period_end")["revenue_yoy_pct"].iloc[-1])
    return facts


DEPRESSED_QUESTION = (
    "Question: this stock is far below its highs, yet its trailing financials show GAAP profit, positive "
    "free cash flow and growing revenue. Why is the stock down, and is the business actually intact? "
    "Check, with web search: analyst commentary, guidance cuts or warnings, slowing growth, AI-disruption "
    "concerns, sector-wide derating, rates and macro, insider or holder selling, dilution, regulatory or "
    "legal events, and company-specific events. Verdict meanings: likely_deserved = the business is "
    "deteriorating, or the trailing numbers flatter it, or the earlier price was stretched; "
    "possibly_undervalued = the business looks intact while the price fell for reasons not tied to the "
    "company; unclear = evidence mixed.\n\n"
)


def _prompt(facts: dict) -> str:
    schema = {
        "summary": "2-4 sentences, plain text, on why the stock is where it is against its sector or its highs",
        "drivers": [{"category": f"one of {CATEGORIES}", "detail": "one sentence of evidence"}],
        "verdict": f"one of {VERDICTS}",
        "verdict_reason": "one sentence",
        "sources": ["url", "..."],
    }
    question = DEPRESSED_QUESTION if facts.get("kind") == "depressed" else CROSSING_QUESTION
    return (
        "Facts computed from SEC filings and price history (trust these):\n"
        f"{json.dumps(facts, indent=2)}\n\n"
        f"{question}"
        f"Answer with exactly this JSON shape:\n{json.dumps(schema, indent=2)}"
    )


CROSSING_QUESTION = (
        "Question: since the results for the crossing quarter were announced, how has this stock performed "
        "against its sector ETF, and why? If it has outperformed, say so and explain what drove it instead "
        "of looking for reasons it lagged. Check, with web search: analyst commentary, guidance changes, slowing growth, "
        "AI-disruption concerns, sector-wide derating, rates and macro, insider or holder selling, "
        "dilution, and company-specific events. Verdict meanings: likely_deserved = the business "
        "deteriorated or the price was stretched; possibly_undervalued = business intact or improving "
        "while the multiple compressed for reasons not tied to the company; unclear = evidence mixed.\n\n"
)


def _parse_json(text: str) -> dict:
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def run_research(ticker: str, data_dir: Path, results_dir: Path, model: str = "sonnet") -> Path:
    path = research_path(ticker, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = _crossing_row(ticker, results_dir)
    if row is not None:
        facts = build_facts(ticker, row, data_dir)
    else:
        drow = _depressed_row(ticker, results_dir)
        if drow is None:
            path.write_text(json.dumps({"skipped": "not in the latest crossings or depressed lists"}))
            return path
        facts = build_depressed_facts(ticker, drow, data_dir)
    cmd = ["claude", "-p", _prompt(facts), "--output-format", "json", "--model", model,
           "--system-prompt", SYSTEM_PROMPT, "--restricted",
           "--disallowedTools", "Bash", "Edit", "Write", "NotebookEdit",
           "--allowedTools", *ALLOWED_TOOLS, "--permission-prompts", "none"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "claude CLI failed")
    payload = json.loads(result.stdout)
    if payload.get("is_error"):
        raise RuntimeError(payload.get("result") or "claude CLI reported an error")
    answer = _parse_json(payload["result"])
    answer["drivers"] = [d for d in answer.get("drivers", []) if d.get("category") in CATEGORIES]
    if answer.get("verdict") not in VERDICTS:
        answer["verdict"] = "unclear"
    path.write_text(json.dumps({"ticker": ticker.upper(), "generated": f"{dt.date.today():%Y-%m-%d}",
                                "model": model, "facts": facts, **answer}))
    return path
