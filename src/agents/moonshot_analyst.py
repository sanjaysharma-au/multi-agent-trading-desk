import json
import re
from datetime import datetime
from pathlib import Path

from agents.earnings_call_analyst import (
    DEFAULT_BACKEND,
    DEFAULT_MODEL,
    _call_llm,
    _nim_base_url,
)
from agents.moonshot_ledger import (
    ANCHORS_FILE,
    LEDGER_FILE,
    PERIODS,
    SOFT_KINDS,
    compute_anchors,
    file_sha256,
    load_ledger,
    normalize_ledger,
    parse_json_object,
    render_comparison,
    render_gates,
    render_ledger,
    slot_catalogue,
)

OUTPUT_DIR = Path("earnings_call_analysis_v3")
REPORT_ORDER = ["ambition", "milestones", "runway"]
PROVENANCE_FILE = "provenance.json"
LEDGER_ATTEMPTS = 3

BLINDERS = """You do not have access to the stock price or any information beyond the transcript text itself. Do not use any knowledge of this company's price, its later results, or how its story turned out, even if you recall it from training data -- deliberately set that aside. The transcript may be anonymized (Company A, Program 1, Person 2, Y+1 for a year); treat those labels as the real names and do not try to work out who the company is. Reason only from what is said on this call."""

LEDGER_SYSTEM_PROMPT = f"""You are a figures clerk. You extract a ledger of the figures actually REPORTED on one earnings call transcript. {BLINDERS}

You do not judge, interpret or summarize. You copy figures that management or the call states as fact for the period being reported.

Rules:
- Only figures stated on this call. Never compute, estimate, annualize or infer a figure that was not said. Never use outside knowledge.
- Guidance, targets, forecasts and hypotheticals are NOT reported figures; leave them out of "figures" (a forecast may appear in soft_evidence as kind "plan").
- Money in USD millions ("$1.8 billion" -> 1800, "$289 million" -> 289, "$450,000" -> 0.45). Losses and cash outflows are negative numbers.
- Percentages as plain numbers (52% -> 52).
- "period" must be one of {", ".join(PERIODS)}. Use "quarter" for the reported quarter, the year-to-date span otherwise; balances (cash, backlog, counts at period end) are "point_in_time". Prefer the quarter figure when both a quarter and a longer-period figure are given.
- "prior_value_stated": the comparison value the call itself gives for the same measure (e.g. the year-ago revenue, the prior customer count), in the same units; null if the call gives none.
- "basis": a few words saying exactly what the figure is (e.g. "adjusted operating income excluding one-off costs", "customers above $1M trailing revenue", "up 52% year over year").
- "quote": the exact words from the transcript that state the figure, copied verbatim (a sentence or clause, not paraphrased). Figures whose quote cannot be found in the transcript are discarded.
- GAAP versus adjusted: put a profit figure in a gaap_ slot only if the call calls it GAAP, or states net income/net loss/operating loss without an adjusted or non-GAAP qualifier. Adjusted, non-GAAP, "excluding" or contribution figures go in adjusted_operating_income or other_hard_figures.
- "scope": "whole_company" or "segment". Standard figures are whole-company figures only; a figure for one segment, region, product line or customer group goes in other_hard_figures instead, even if it is the only figure of its kind on the call.
- Keep each standard figure to the exact measure its description names. Do not substitute a related measure (billings for bookings, deal value for backlog, a segment customer count for the total); put the related measure in other_hard_figures.
- Use null for any standard figure not reported on the call. Absence is information; do not fill gaps.

Standard figures (the "figures" object must contain every one of these keys):
{slot_catalogue()}

Also list:
- "first_time_achievements": things the call says happened for the first time (first profitable quarter, first production deployment, first customer of a kind), each with "what" and "quote".
- "other_hard_figures": other reported, checkable figures that bear on whether the business is working (segment revenue and growth, contract counts and sizes, deployments, margins not covered above, contribution margins, cohort figures), each with "metric", "value", "unit", "period", "basis", "quote".
- "soft_evidence": progress claims with nothing yet delivered or measured: pilots, pipeline, "conversations", demand signals, partnerships without figures, plans and targets. Each with "claim", "kind" (one of {", ".join(SOFT_KINDS)}) and "quote".
- "receding_evidence": slipped timelines, shrunk scope, lost or shrinking customers, declines, concentration warnings, each with "what" and "quote".

Output ONLY one JSON object, no markdown fences and no commentary, of the form:
{{"figures": {{"revenue": {{"value": 0, "period": "quarter", "scope": "whole_company", "prior_value_stated": null, "basis": "", "quote": ""}}, "gaap_net_income": null, ...}}, "first_time_achievements": [], "other_hard_figures": [], "soft_evidence": [], "receding_evidence": []}}
"""

AMBITION_SYSTEM_PROMPT = f"""You are an ambition analyst reviewing an earnings call transcript. {BLINDERS}

Your only job is to establish how big the dream is. Whether it is being achieved is somebody else's job; do not assess progress, tone, or whether management deserves to be believed.

Analyze ONLY:
- The long-term goal, in management's own words. Quote it. If the call states no long-term goal beyond the next few quarters, say so plainly -- do not invent one.
- Scale: if this works, what does the company become? Look for stated addressable market, category creation, displacement of an incumbent, or a claim to become the standard in something large. Separate ambitions that would change an industry from ambitions that would merely grow this business.
- Concreteness: is the dream expressed as falsifiable milestones (a capability, a volume, a customer type, a date) or only as vision language?
- Whether the dream is presented as something already under way or as something still to be started.

Then give a section headed QUANTIFIED TARGETS listing, verbatim, every quote in which management attaches a number or a date to the long-term goal (a market size, a volume, a revenue or margin target, a customer or deployment target, a year). If there are none, write "QUANTIFIED TARGETS: none". Vision language without a number or date does not count.

Ground every claim in a specific quote. End with a single-paragraph summary: how large is the stated dream, and how concrete is it.
"""

MILESTONES_SYSTEM_PROMPT = f"""You are a milestone analyst reviewing an earnings call transcript. {BLINDERS}

The other analysts on this team establish how big the company's dream is and whether it can afford to keep going. Your only job is to establish how close the dream is to being real, judged strictly by REPORTED FIGURES. Management's tone, confidence and the number of upbeat claims are not evidence; ignore them.

You are given, besides the transcript:
1. The LEDGER of figures reported on this call, extracted and checked against the transcript.
2. A mechanical FIGURE-BY-FIGURE COMPARISON of this call's standard figures against the preceding call's ledger (or a statement that no preceding ledger exists).
3. The GATES: mechanical consequences of the ledger (proximity tier, allowed stages, momentum basis). You may not argue past a gate. If you think a gate is wrong because the ledger missed or mis-read a figure, say so and quote the transcript, but still conclude within the gate.

Write these sections:

CHANGE VERSUS THE PRECEDING CALL
Go through the comparison figure by figure: improved, worsened, unchanged, or not comparable, with both values. Then compare any other_hard_figures that plainly measure the same thing on both calls. If there is no preceding ledger, say so and state that the direction of movement cannot be judged quarter on quarter; you may note year-over-year comparisons the call itself gives, labelled as such. Do not describe movement the figures do not show.

HARD VERSUS SOFT
Which ledger figures carry the case that the dream is getting closer, and which of the call's claims are only SOFT (pilots, pipeline, "conversations", demand signals, plans). Soft evidence does not move a company closer to its dream; say explicitly how much of the call's case rests on it.

PILOT VERSUS PRODUCTION
Are customers trying the thing or running their business on it? Quote the deciding language and any figure that supports it.

RECEDING
Slipped timelines, shrunk scope, customers leaving, concentration, figures that worsened, growth that needs heavy discounting or services.

STAGE
Pick ONE of the allowed stages listed in the gates, and name the ledger figures that decide it:
  PRE_PROOF: no evidence yet that the core thing works commercially.
  EARLY_PROOF: it works commercially, but repeatability or economics are not yet shown in the figures.
  SCALING: repeatable, with positive economics, and the growth rate is sustained or rising versus the preceding call.
  MATURE: GAAP profitable with positive free cash flow; the dream is largely realized and growth is now ordinary.

SUMMARY
One paragraph: how far along the dream is on the figures, whether the figures moved closer or further since the preceding call, and the single most important figure.

Ground every claim in a ledger figure or a quote.
"""

RUNWAY_SYSTEM_PROMPT = f"""You are a runway analyst reviewing an earnings call transcript. {BLINDERS}

Your only job is to establish whether the company can afford to keep pursuing its goal long enough to reach it.

Analyze ONLY:
- Cash and liquidity, and how many quarters of operation they support at the current burn. State the cash figure and the cash-flow figure you used and the period each covers.
- Profitability and cash flow: are operations self-funding, close to it, or dependent on outside capital? Distinguish GAAP from adjusted figures.
- Dependence on capital markets: recent or planned equity raises, convertible debt, at-the-market programs, and any language about needing to raise.
- Concentration: reliance on a small number of customers, contracts, or funding sources.
- Whether stated spending plans fit inside the stated resources.

Assurances that the company is "well capitalized" are not evidence; only figures are. If the call gives no cash or cash-flow figure, say so. Ground every claim in a specific quote or number. End with a single-paragraph summary: can this company fund itself to its goal, and what is the single most important number.
"""

SPECIALISTS = {
    "ambition": AMBITION_SYSTEM_PROMPT,
    "milestones": MILESTONES_SYSTEM_PROMPT,
    "runway": RUNWAY_SYSTEM_PROMPT,
}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def extract_ledger(
    transcript_path: Path,
    label: str,
    model: str = DEFAULT_MODEL,
    backend: str = DEFAULT_BACKEND,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    transcript_text = transcript_path.read_text()
    instructions = f"Here is the earnings call transcript. Output only the JSON ledger.\n\n{transcript_text}"
    last_error = None
    for attempt in range(LEDGER_ATTEMPTS):
        text = _call_llm(LEDGER_SYSTEM_PROMPT, instructions, model=model, backend=backend, label=f"{label}:ledger")
        try:
            raw = parse_json_object(text)
            break
        except ValueError as e:
            last_error = e
            print(f"[{label}:ledger] unparseable output (attempt {attempt + 1}/{LEDGER_ATTEMPTS})")
    else:
        raise last_error
    ledger = normalize_ledger(raw, transcript_text)
    ledger["_meta"] = {
        "backend": backend,
        "model": model,
        "transcript": str(transcript_path),
        "transcript_sha256": file_sha256(transcript_path),
        "generated_at": _now(),
    }
    out_dir = output_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / LEDGER_FILE
    _write_json(path, ledger)
    return path


def make_chain(prior_label: str | None, prior_ledger_path: Path | None, gap_quarters: int = 0) -> dict:
    if prior_label is None:
        return {"status": "first_in_chain", "prior_quarter": None, "prior_ledger_sha256": None, "gap_quarters": 0}
    if prior_ledger_path is None or load_ledger(prior_ledger_path) is None:
        return {"status": "chain_break", "prior_quarter": prior_label, "prior_ledger_sha256": None, "gap_quarters": gap_quarters}
    return {
        "status": "linked",
        "prior_quarter": prior_label,
        "prior_ledger_sha256": file_sha256(prior_ledger_path),
        "gap_quarters": gap_quarters,
    }


NOVEL_TERM_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*\b")
NOVEL_TERM_STOPLIST = {
    "yoy", "qoq", "fcf", "ndr", "nrr", "ttm", "ltm", "ytd", "atm", "cro", "sbc", "gaap", "rpo", "arr", "mrr", "tcv", "acv",
    "tam", "usd", "eps", "ceo", "cfo", "coo", "cto", "sec", "ebit", "ebitda", "capex", "opex", "kpi", "json", "llm", "llms",
    "nan", "pre_proof", "early_proof", "t12m", "covid", "ipo", "smb", "dna", "sdk", "mom", "arpu", "dpo", "kyc", "evs", "spacs",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday",
}
TRANSCRIPT_ROOT = Path("data/earnings_calls")
_vocabulary_cache: dict[str, set[str]] = {}


def _lower_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z][a-z'-]*", text.lower()))


def _prompt_vocabulary() -> set[str]:
    words = _lower_words(LEDGER_SYSTEM_PROMPT)
    for prompt in SPECIALISTS.values():
        words |= _lower_words(prompt)
    return words


def _corpus_vocabulary(ticker: str) -> set[str]:
    if ticker not in _vocabulary_cache:
        words: set[str] = set()
        for f in TRANSCRIPT_ROOT.glob("*_*Q*.txt"):
            if not f.name.startswith(f"{ticker}_"):
                words.update(re.findall(r"\b[a-z][a-z'-]*\b", f.read_text()))
        _vocabulary_cache[ticker] = words
    return _vocabulary_cache[ticker]


def identity_terms_found(texts: list[str], terms: list[str], transcript_text: str | None = None) -> list[str]:
    joined = "\n".join(texts)
    found = set()
    for t in terms:
        if not re.search(rf"\b{re.escape(t)}\b", joined, flags=re.IGNORECASE):
            continue
        if transcript_text and all(
            re.search(rf"\b{re.escape(word)}\b", transcript_text, flags=re.IGNORECASE) for word in t.split()
        ):
            continue
        found.add(t)
    return sorted(found)


def novel_proper_nouns(texts: list[str], transcript_text: str, ticker: str = "") -> list[str]:
    joined = "\n".join(texts)
    known = _lower_words(transcript_text) | _prompt_vocabulary() | _corpus_vocabulary(ticker) | NOVEL_TERM_STOPLIST
    lowercase_in_output = set(re.findall(r"\b[a-z][a-z'-]*\b", joined))
    novel = set()
    for term in NOVEL_TERM_RE.findall(joined):
        parts = [p for p in term.split("-") if len(p) >= 3 and not p.isdigit()]
        if any(p.lower() not in known and p.lower() not in lowercase_in_output for p in parts):
            novel.add(term)
    return sorted(novel)


def analyze_call(
    transcript_path: Path,
    label: str,
    chain: dict,
    prior_ledger_path: Path | None,
    model: str = DEFAULT_MODEL,
    backend: str = DEFAULT_BACKEND,
    output_dir: Path = OUTPUT_DIR,
    identity_terms: list[str] | None = None,
) -> Path:
    out_dir = output_dir / label
    ledger_path = out_dir / LEDGER_FILE
    ledger = load_ledger(ledger_path)
    if ledger is None:
        raise RuntimeError(f"{label}: no valid {LEDGER_FILE}; extract the ledger first")
    prior_ledger = load_ledger(prior_ledger_path) if chain["status"] == "linked" else None
    if chain["status"] == "linked" and (prior_ledger is None or file_sha256(prior_ledger_path) != chain["prior_ledger_sha256"]):
        raise RuntimeError(f"{label}: prior ledger changed or vanished while analyzing")

    anchors = compute_anchors(ledger, prior_ledger, chain)
    transcript_text = transcript_path.read_text()
    plain = f"Here is the earnings call transcript to analyze:\n\n{transcript_text}"
    milestones_input = (
        f"LEDGER OF FIGURES REPORTED ON THIS CALL:\n{render_ledger(ledger)}\n\n"
        f"FIGURE-BY-FIGURE COMPARISON:\n{render_comparison(anchors)}\n\n"
        f"GATES:\n{render_gates(anchors)}\n\n"
        f"TRANSCRIPT:\n\n{transcript_text}"
    )
    inputs = {"ambition": plain, "milestones": milestones_input, "runway": plain}
    reports = {
        name: _call_llm(prompt, inputs[name], model=model, backend=backend, label=f"{label}:{name}")
        for name, prompt in SPECIALISTS.items()
    }

    output_texts = [*reports.values(), json.dumps({k: v for k, v in ledger.items() if k != "_meta"})]
    for name, text in reports.items():
        (out_dir / f"{name}.md").write_text(text)
    _write_json(out_dir / ANCHORS_FILE, anchors)
    _write_json(
        out_dir / PROVENANCE_FILE,
        {
            "backend": backend,
            "model": model,
            "endpoint": _nim_base_url() if backend == "nemotron" else None,
            "transcript": str(transcript_path),
            "ledger_sha256": file_sha256(ledger_path),
            "chain": chain,
            "identity_terms_found": None if identity_terms is None else identity_terms_found(
                output_texts, identity_terms, transcript_text
            ),
            "novel_proper_nouns": novel_proper_nouns(output_texts, transcript_text, label.rsplit("_", 1)[0]),
            "generated_at": _now(),
        },
    )
    return out_dir


def call_state(call_dir: Path, prior_call_dir: Path | None, transcript_path: Path | None = None) -> str:
    required = [f"{n}.md" for n in REPORT_ORDER] + [LEDGER_FILE, ANCHORS_FILE, PROVENANCE_FILE]
    if not all((call_dir / f).exists() for f in required):
        return "incomplete"
    try:
        provenance = json.loads((call_dir / PROVENANCE_FILE).read_text())
    except json.JSONDecodeError:
        return "incomplete"
    chain = provenance.get("chain") or {}
    if provenance.get("ledger_sha256") != file_sha256(call_dir / LEDGER_FILE):
        return "stale"
    if transcript_path is not None and not ledger_is_current(call_dir, transcript_path):
        return "stale"
    if prior_call_dir is None:
        return "complete" if chain.get("status") == "first_in_chain" else "stale"
    if chain.get("prior_quarter") != prior_call_dir.name.rsplit("_", 1)[-1]:
        return "stale"
    prior_ledger = prior_call_dir / LEDGER_FILE
    if load_ledger(prior_ledger) is None:
        return "complete" if chain.get("status") == "chain_break" else "stale"
    if chain.get("status") != "linked" or chain.get("prior_ledger_sha256") != file_sha256(prior_ledger):
        return "stale"
    return "complete"


def ledger_is_current(call_dir: Path, transcript_path: Path) -> bool:
    ledger = load_ledger(call_dir / LEDGER_FILE)
    return bool(ledger) and ledger.get("_meta", {}).get("transcript_sha256") == file_sha256(transcript_path)
