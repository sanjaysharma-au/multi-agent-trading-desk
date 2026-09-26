import hashlib
import json
import re
from pathlib import Path

SLOTS = {
    "revenue": ("money", "level", "Total company revenue for the period."),
    "revenue_growth_yoy": ("pct", "rate", "Total revenue growth versus the same period a year earlier, in percent."),
    "revenue_growth_qoq": ("pct", "rate", "Total revenue growth versus the immediately preceding quarter, in percent."),
    "gross_margin": ("pct", "rate", "Gross margin in percent (GAAP or adjusted; say which in basis)."),
    "gaap_operating_income": ("money", "profit", "GAAP operating income; negative for an operating loss. Only if the call calls it GAAP or states it without an adjusted/non-GAAP qualifier."),
    "gaap_net_income": ("money", "profit", "GAAP net income; negative for a net loss. Only if the call calls it GAAP or states it without an adjusted/non-GAAP qualifier."),
    "adjusted_operating_income": ("money", "profit", "Adjusted / non-GAAP operating income, or adjusted EBITDA if that is the only adjusted profit measure (name it in basis). Negative for a loss."),
    "operating_cash_flow": ("money", "profit", "Cash from operations; negative if cash was consumed."),
    "free_cash_flow": ("money", "profit", "Free cash flow (or adjusted free cash flow; say which in basis); negative if cash was consumed."),
    "cash_and_investments": ("money", "level", "Cash, cash equivalents and marketable securities at period end."),
    "customer_count": ("count", "level", "Total number of customers, clients or accounts for the whole company (not one segment or region)."),
    "large_customer_count": ("count", "level", "Number of customers above a stated size threshold (state the threshold in basis)."),
    "units_delivered": ("count", "level", "Units of the core product delivered, shipped, deployed or installed in the period."),
    "net_dollar_retention": ("pct", "rate", "Net dollar / net revenue retention in percent."),
    "recurring_revenue_share": ("pct", "rate", "Share of revenue that is recurring or from existing customers, in percent."),
    "backlog": ("money", "level", "Whole-company remaining performance obligations (RPO) or reported backlog at period end. Not deal value that includes unexercised options or unfunded amounts, and not one segment's figure; put those in other_hard_figures."),
    "bookings": ("money", "level", "Whole-company contract value signed during the period (bookings or total contract value). Not billings."),
}

FLOW_PERIODS = {"quarter": 1, "half_year": 2, "nine_months": 3, "full_year": 4, "trailing_12m": 4}
PERIODS = [*FLOW_PERIODS, "point_in_time"]
UNIT_NAMES = {"money": "USD millions", "pct": "percent", "count": "count"}
SOFT_KINDS = ["pilot", "pipeline", "assertion", "demand_signal", "partnership_without_figures", "plan", "other"]

STAGES = ["pre_proof", "early_proof", "scaling", "mature"]

PROXIMITY_TIERS = {
    "T0_no_revenue": (0, 15, "no revenue from the business is reported on the call"),
    "T1_revenue_no_profit_or_cash": (5, 35, "revenue is reported but no profit or cash-flow measure is positive"),
    "T2_adjusted_or_cash_positive_gaap_loss": (20, 55, "an adjusted profit or cash-flow measure is positive, but GAAP profitability is not shown on the call"),
    "T3_gaap_profitable": (35, 75, "GAAP operating or net income is positive, but not yet together with positive free cash flow and a repeatable revenue base"),
    "T4_gaap_and_fcf_positive_repeatable": (50, 100, "GAAP profitable, free-cash-flow positive, with a repeatable, growing revenue base"),
}
NO_REPEATABLE_BASE_CAP = 45
MOMENTUM_BAND_HALF_WIDTH = 20
MOMENTUM_BAND_NO_BASIS = (-30, 30)
PCT_UNCHANGED_POINTS = 1.0
LEVEL_UNCHANGED_FRACTION = 0.02
WEIGHTS = {"level": 1, "rate": 3, "profit": 3}
SIGN_FLIP_WEIGHT = 6
RATE_SCALE = 0.10
PROFIT_SCALE = 2.0
LEVEL_SCALE = 4.0
SHRINK_PRIOR_WEIGHT = 6
QUOTE_SHINGLE = 5
QUOTE_MATCH_FRACTION = 0.6

ANCHOR_RULES_VERSION = "momentum-v2"
LEDGER_FILE = "ledger.json"
PRESS_LEDGER_FILE = "press_ledger.json"
PRESS_SLOTS = [
    "revenue", "revenue_growth_yoy", "gross_margin", "gaap_operating_income", "gaap_net_income",
    "adjusted_operating_income", "operating_cash_flow", "free_cash_flow", "cash_and_investments",
]
ANCHORS_FILE = "anchors.json"


def slot_catalogue() -> str:
    return "\n".join(
        f'- "{name}" ({UNIT_NAMES[unit]}): {desc}' for name, (unit, _, desc) in SLOTS.items()
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_json_object(text: str) -> dict:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = cleaned.replace("```json", "```")
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        tail = cleaned[match.start():]
        for attempt in (tail, re.sub(r",\s*([}\]])", r"\1", tail)):
            try:
                obj, _ = decoder.raw_decode(attempt)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    raise ValueError(f"no JSON object found in model output: {text[:300]!r}")


def _to_number(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip().replace(",", "").replace("$", "").replace("%", "")
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class QuoteChecker:
    def __init__(self, source_text: str):
        tokens = _tokens(source_text)
        self.joined = " ".join(tokens)
        self.shingles = {tuple(tokens[i:i + QUOTE_SHINGLE]) for i in range(len(tokens) - QUOTE_SHINGLE + 1)}

    def verified(self, quote: str) -> bool:
        tokens = _tokens(quote or "")
        if not tokens:
            return False
        if len(tokens) < QUOTE_SHINGLE:
            return len(tokens) >= 2 and " ".join(tokens) in self.joined
        grams = [tuple(tokens[i:i + QUOTE_SHINGLE]) for i in range(len(tokens) - QUOTE_SHINGLE + 1)]
        return sum(g in self.shingles for g in grams) / len(grams) >= QUOTE_MATCH_FRACTION


def _clean_text(value, limit: int = 400) -> str:
    return str(value).strip()[:limit] if value is not None else ""


def normalize_ledger(raw: dict, transcript_text: str) -> dict:
    checker = QuoteChecker(transcript_text)
    dropped = []
    figures = {}
    raw_figures = raw.get("figures") if isinstance(raw.get("figures"), dict) else {}
    for name in SLOTS:
        entry = raw_figures.get(name)
        if not isinstance(entry, dict):
            figures[name] = None
            continue
        if entry.get("scope", "whole_company") != "whole_company":
            dropped.append({"slot": name, "reason": "not a whole-company figure", "entry": entry})
            figures[name] = None
            continue
        value = _to_number(entry.get("value"))
        quote = _clean_text(entry.get("quote"), 600)
        period = entry.get("period") if entry.get("period") in PERIODS else None
        if value is None or not quote or period is None:
            dropped.append({"slot": name, "reason": "missing value, quote or valid period", "entry": entry})
            figures[name] = None
            continue
        figures[name] = {
            "value": value,
            "period": period,
            "prior_value_stated": _to_number(entry.get("prior_value_stated")),
            "basis": _clean_text(entry.get("basis")),
            "quote": quote,
            "verified": checker.verified(quote),
        }

    def items(key: str, fields: list[str]) -> list[dict]:
        out = []
        for entry in raw.get(key) or []:
            if not isinstance(entry, dict):
                continue
            quote = _clean_text(entry.get("quote"), 600)
            if not quote:
                dropped.append({"list": key, "reason": "missing quote", "entry": entry})
                continue
            item = {f: _clean_text(entry.get(f)) for f in fields}
            item["quote"] = quote
            item["verified"] = checker.verified(quote)
            out.append(item)
        return out

    other = items("other_hard_figures", ["metric", "value", "unit", "period", "basis"])
    soft = items("soft_evidence", ["claim", "kind"])
    for s in soft:
        if s["kind"] not in SOFT_KINDS:
            s["kind"] = "other"
    return {
        "figures": figures,
        "first_time_achievements": items("first_time_achievements", ["what"]),
        "other_hard_figures": other,
        "soft_evidence": soft,
        "receding_evidence": items("receding_evidence", ["what"]),
        "dropped": dropped,
    }


def load_ledger(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        ledger = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return ledger if isinstance(ledger, dict) and isinstance(ledger.get("figures"), dict) else None


def press_slot_catalogue() -> str:
    return "\n".join(
        f'- "{name}" ({UNIT_NAMES[SLOTS[name][0]]}): {SLOTS[name][2]}' for name in PRESS_SLOTS
    )


def normalize_press_ledger(raw: dict, press_text: str) -> dict:
    raw_figures = raw.get("figures") if isinstance(raw.get("figures"), dict) else {}
    allowed = {k: v for k, v in raw_figures.items() if k in PRESS_SLOTS}
    ledger = normalize_ledger({"figures": allowed}, press_text)
    return {"figures": ledger["figures"], "dropped": ledger["dropped"]}


def load_merged_ledger(path: Path) -> dict | None:
    ledger = load_ledger(path)
    if ledger is None:
        return None
    press = load_ledger(path.parent / PRESS_LEDGER_FILE)
    if press:
        for slot, entry in press["figures"].items():
            have = ledger["figures"].get(slot)
            if entry and entry["verified"] and not (have and have["verified"]):
                ledger["figures"][slot] = {**entry, "source": "press_release"}
    return ledger


def _verified(ledger: dict | None, slot: str) -> dict | None:
    if not ledger:
        return None
    entry = ledger["figures"].get(slot)
    return entry if entry and entry.get("verified") else None


def _value(ledger: dict | None, slot: str) -> float | None:
    entry = _verified(ledger, slot)
    return entry["value"] if entry else None


def _clip(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _row_score(kind: str, current: float, prior: float) -> tuple[float, int]:
    if kind == "rate":
        return _clip((current - prior) * RATE_SCALE), WEIGHTS[kind]
    if kind == "profit" and (prior <= 0 < current or current <= 0 < prior):
        return (1.0 if current > 0 else -1.0), SIGN_FLIP_WEIGHT
    relative = (current - prior) / max(abs(prior), 1e-9)
    return _clip(relative * (PROFIT_SCALE if kind == "profit" else LEVEL_SCALE)), WEIGHTS[kind]


def compare_ledgers(current: dict, prior: dict | None) -> list[dict]:
    rows = []
    for slot, (unit, kind, _) in SLOTS.items():
        cur, pri = _verified(current, slot), _verified(prior, slot)
        row = {"slot": slot, "unit": UNIT_NAMES[unit], "current": cur and cur["value"], "current_period": cur and cur["period"],
               "prior": pri and pri["value"], "prior_period": pri and pri["period"], "direction": None, "weight": 0, "score": 0.0}
        if cur is None and pri is None:
            continue
        if cur is None or pri is None:
            row["direction"] = "not_reported_now" if cur is None else "not_reported_before"
        elif cur["period"] != pri["period"]:
            row["direction"] = "not_comparable_period"
        else:
            c, p = cur["value"], pri["value"]
            score, weight = _row_score(kind, c, p)
            row["weight"] = weight
            if kind == "profit" and (p <= 0 < c or c <= 0 < p):
                row["direction"] = "turned_positive" if c > 0 else "turned_negative"
                row["score"] = score
            else:
                if unit == "pct":
                    unchanged = abs(c - p) < PCT_UNCHANGED_POINTS
                else:
                    unchanged = abs(c - p) <= LEVEL_UNCHANGED_FRACTION * max(abs(p), 1e-9)
                row["direction"] = "unchanged" if unchanged else ("improved" if c > p else "worsened")
                row["score"] = 0.0 if unchanged else score
        rows.append(row)
    return rows


def momentum_basis(rows: list[dict]) -> tuple[int | None, int]:
    compared = [r for r in rows if r["weight"]]
    total = sum(r["weight"] for r in compared)
    if not total:
        return None, 0
    raw = sum(r["weight"] * r["score"] for r in compared) / total
    return round(100 * raw * total / (total + SHRINK_PRIOR_WEIGHT)), len(compared)


def _grew(ledger: dict, prior: dict | None, slot: str) -> bool:
    cur = _verified(ledger, slot)
    if not cur:
        return False
    if cur["prior_value_stated"] is not None and cur["value"] > cur["prior_value_stated"]:
        return True
    pri = _verified(prior, slot)
    return bool(pri and pri["period"] == cur["period"] and cur["value"] > pri["value"])


def _annualized(entry: dict | None) -> float | None:
    if not entry or entry["period"] not in FLOW_PERIODS:
        return None
    return entry["value"] * 4 / FLOW_PERIODS[entry["period"]]


def evidence_hardness(ledger: dict) -> dict:
    hard = sum(1 for e in ledger["figures"].values() if e and e["verified"])
    hard += sum(1 for e in ledger["other_hard_figures"] + ledger["first_time_achievements"] if e["verified"])
    soft = len(ledger["soft_evidence"])
    unverified = sum(1 for e in ledger["figures"].values() if e and not e["verified"])
    unverified += sum(1 for e in ledger["other_hard_figures"] + ledger["first_time_achievements"] if not e["verified"])
    score = round(100 * hard / (hard + soft)) if hard + soft else 0
    return {"score": score, "hard": hard, "soft": soft, "unverified_hard_excluded": unverified}


def _runway_band(ledger: dict) -> tuple[int, int, str]:
    flow = _verified(ledger, "free_cash_flow") or _verified(ledger, "operating_cash_flow")
    cash = _value(ledger, "cash_and_investments")
    if flow and flow["period"] in FLOW_PERIODS:
        if flow["value"] > 0:
            return 40, 100, "operations generated cash in the period"
        burn = -flow["value"] / FLOW_PERIODS[flow["period"]]
        if burn == 0:
            return 40, 100, "cash flow break-even in the period"
        if cash is None:
            return 0, 40, "burning cash and no cash balance reported"
        quarters = cash / burn
        for limit, band in [(4, (0, 25)), (8, (10, 45)), (16, (25, 65))]:
            if quarters < limit:
                return *band, f"burning cash; about {quarters:.1f} quarters of cash at the current burn"
        return 40, 80, f"burning cash but about {quarters:.1f} quarters of cash at the current burn"
    if cash is not None:
        return 10, 60, "cash balance reported but no cash-flow figure"
    return 0, 50, "neither cash flow nor cash balance reported"


def compute_anchors(ledger: dict, prior: dict | None, chain: dict) -> dict:
    rows = compare_ledgers(ledger, prior)
    basis, compared = momentum_basis(rows)
    hardness = evidence_hardness(ledger)

    revenue = _verified(ledger, "revenue")
    growth = _value(ledger, "revenue_growth_yoy")
    revenue_growing = (growth is not None and growth > 0) or _grew(ledger, prior, "revenue")

    repeatable = []
    ndr, recurring = _value(ledger, "net_dollar_retention"), _value(ledger, "recurring_revenue_share")
    if ndr is not None and ndr >= 100:
        repeatable.append(f"net dollar retention {ndr:g}% >= 100%")
    if recurring is not None and recurring >= 50:
        repeatable.append(f"recurring/existing-customer revenue share {recurring:g}% >= 50%")
    for slot in ["customer_count", "large_customer_count", "units_delivered", "backlog"]:
        if _grew(ledger, prior, slot):
            repeatable.append(f"{slot} grew")
    cur_growth, prior_growth = _verified(ledger, "revenue_growth_yoy"), _verified(prior, "revenue_growth_yoy")
    if cur_growth and prior_growth and cur_growth["period"] == prior_growth["period"] and cur_growth["value"] > 0 and prior_growth["value"] > 0:
        repeatable.append(f"year-over-year revenue growth positive on two consecutive calls ({prior_growth['value']:g}% then {cur_growth['value']:g}%)")
    backlog, annual_revenue = _value(ledger, "backlog"), _annualized(revenue)
    if backlog is not None and annual_revenue and backlog >= annual_revenue:
        repeatable.append(f"backlog {backlog:g} >= annualized revenue {annual_revenue:g}")
    repeatable_base = bool(revenue and revenue_growing and repeatable)

    positive = [s for s in ["adjusted_operating_income", "operating_cash_flow", "free_cash_flow",
                            "gaap_operating_income", "gaap_net_income"] if (_value(ledger, s) or 0) > 0]
    gaap_profitable = any(s.startswith("gaap_") for s in positive)
    fcf_positive = (_value(ledger, "free_cash_flow") or 0) > 0

    if cur_growth and prior_growth and cur_growth["period"] == prior_growth["period"]:
        growth_sustained = cur_growth["value"] >= prior_growth["value"] - PCT_UNCHANGED_POINTS
    else:
        growth_sustained = None

    if not revenue:
        tier = "T0_no_revenue"
    elif not positive:
        tier = "T1_revenue_no_profit_or_cash"
    elif not gaap_profitable:
        tier = "T2_adjusted_or_cash_positive_gaap_loss"
    elif not (fcf_positive and repeatable_base):
        tier = "T3_gaap_profitable"
    else:
        tier = "T4_gaap_and_fcf_positive_repeatable"
    floor, cap, tier_reason = PROXIMITY_TIERS[tier]
    caps_applied = []
    if not repeatable_base and cap > NO_REPEATABLE_BASE_CAP:
        cap = NO_REPEATABLE_BASE_CAP
        caps_applied.append(f"no repeatable, growing revenue base shown: cap {NO_REPEATABLE_BASE_CAP}")
    floor = min(floor, cap)
    if hardness["score"] < 50:
        cap = max(floor, (floor + cap) // 2)
        caps_applied.append(f"evidence mostly soft (hardness {hardness['score']}): cap lowered to band midpoint {cap}")

    allowed = ["pre_proof"]
    if revenue:
        allowed.append("early_proof")
    if repeatable_base and positive and growth_sustained:
        allowed.append("scaling")
    if gaap_profitable and fcf_positive and repeatable_base:
        allowed.append("mature")

    if chain["status"] != "linked" or basis is None:
        m_low, m_high = MOMENTUM_BAND_NO_BASIS
        m_reason = ("no prior-quarter ledger (" + chain["status"] + ")" if chain["status"] != "linked"
                    else f"only {compared} figures comparable with the prior quarter")
    else:
        m_low, m_high = max(-100, basis - MOMENTUM_BAND_HALF_WIDTH), min(100, basis + MOMENTUM_BAND_HALF_WIDTH)
        m_reason = f"weighted balance of {compared} figure-by-figure changes versus the preceding call"

    r_low, r_high, r_reason = _runway_band(ledger)

    return {
        "chain": chain,
        "comparison": rows,
        "momentum_basis": basis,
        "comparisons_counted": compared,
        "momentum_band": [m_low, m_high],
        "momentum_band_reason": m_reason,
        "evidence_hardness": hardness,
        "proximity_tier": tier,
        "proximity_tier_reason": tier_reason,
        "proximity_band": [floor, cap],
        "proximity_caps_applied": caps_applied,
        "repeatable_base": repeatable_base,
        "repeatable_base_evidence": repeatable,
        "revenue_growing": revenue_growing,
        "positive_profit_or_cash_measures": positive,
        "gaap_profitable": gaap_profitable,
        "fcf_positive": fcf_positive,
        "growth_sustained_vs_prior": growth_sustained,
        "allowed_stages": allowed,
        "runway_band": [r_low, r_high],
        "runway_band_reason": r_reason,
    }


def _fmt(value, unit: str) -> str:
    if value is None:
        return "-"
    return f"{value:g}%" if unit == "percent" else f"{value:g}"


def render_ledger(ledger: dict) -> str:
    lines = ["Standard figures (money in USD millions; [UNVERIFIED] = quote not found in transcript, not used for gates):"]
    for slot, entry in ledger["figures"].items():
        if not entry:
            continue
        unit = UNIT_NAMES[SLOTS[slot][0]]
        flag = "" if entry["verified"] else " [UNVERIFIED]"
        if entry.get("source") == "press_release":
            flag += " [from the earnings press release, not the call]"
        prior = f", prior stated on call {_fmt(entry['prior_value_stated'], unit)}" if entry["prior_value_stated"] is not None else ""
        lines.append(f"- {slot} = {_fmt(entry['value'], unit)} ({entry['period']}{prior}){flag}; basis: {entry['basis']}; quote: \"{entry['quote']}\"")
    for key, field in [("other_hard_figures", "metric"), ("first_time_achievements", "what"),
                       ("soft_evidence", "claim"), ("receding_evidence", "what")]:
        if ledger[key]:
            lines.append(f"{key}:")
            for e in ledger[key]:
                extra = f" = {e['value']} {e['unit']} ({e['period']})" if key == "other_hard_figures" else ""
                kind = f" [{e['kind']}]" if key == "soft_evidence" else ""
                flag = "" if e["verified"] else " [UNVERIFIED]"
                lines.append(f"- {e[field]}{extra}{kind}{flag}: \"{e['quote']}\"")
    return "\n".join(lines)


def render_comparison(anchors: dict) -> str:
    chain = anchors["chain"]
    if chain["status"] == "first_in_chain":
        header = "PRIOR QUARTER: none. This is the first call on record for this company, so there is no prior-quarter ledger."
    elif chain["status"] == "chain_break":
        header = ("PRIOR QUARTER: the ledger for the preceding call is unavailable, "
                  "so no quarter-on-quarter comparison can be made for this call.")
    else:
        gap = chain.get("gap_quarters") or 0
        gap_note = f" NOTE: {gap} quarter(s) between the two calls have no transcript, so this spans a longer interval." if gap else ""
        header = f"PRIOR QUARTER: the preceding call on record.{gap_note}"
    lines = [header, "Figure-by-figure comparison of verified standard figures (computed mechanically; money in USD millions):"]
    for r in anchors["comparison"]:
        lines.append(f"- {r['slot']}: prior {_fmt(r['prior'], r['unit'])} ({r['prior_period'] or '-'}) -> "
                     f"current {_fmt(r['current'], r['unit'])} ({r['current_period'] or '-'}): {r['direction']}")
    if not anchors["comparison"]:
        lines.append("- (no standard figures to compare)")
    return "\n".join(lines)


def render_gates(anchors: dict) -> str:
    p_low, p_high = anchors["proximity_band"]
    m_low, m_high = anchors["momentum_band"]
    r_low, r_high = anchors["runway_band"]
    h = anchors["evidence_hardness"]
    return "\n".join([
        f"- Proximity tier: {anchors['proximity_tier']} ({anchors['proximity_tier_reason']}); allowed proximity band {p_low}-{p_high}.",
        *[f"  - cap applied: {c}" for c in anchors["proximity_caps_applied"]],
        f"- Repeatable, growing revenue base shown: {anchors['repeatable_base']} ({'; '.join(anchors['repeatable_base_evidence']) or 'no qualifying figure'}).",
        f"- Positive profit or cash measures: {', '.join(anchors['positive_profit_or_cash_measures']) or 'none'}; GAAP profitable: {anchors['gaap_profitable']}; free cash flow positive: {anchors['fcf_positive']}.",
        f"- Revenue growth rate sustained or rising versus prior quarter: {anchors['growth_sustained_vs_prior']} (None = cannot be determined).",
        f"- Allowed stages: {', '.join(anchors['allowed_stages'])}.",
        f"- Momentum basis: {anchors['momentum_basis']} ({anchors['momentum_band_reason']}); allowed momentum band {m_low} to {m_high}.",
        f"- Evidence hardness (verified hard items / all items): {h['score']} ({h['hard']} hard, {h['soft']} soft, {h['unverified_hard_excluded']} unverified hard items excluded).",
        f"- Runway band: {r_low}-{r_high} ({anchors['runway_band_reason']}).",
    ])
