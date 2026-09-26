import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.earnings_call_analyst import DEFAULT_MODEL, NIM_MAX_CONCURRENCY, _call_llm
from agents.moonshot_analyst import PROVENANCE_FILE, REPORT_ORDER, call_state
from agents.moonshot_ledger import (
    ANCHOR_RULES_VERSION,
    ANCHORS_FILE,
    LEDGER_FILE,
    STAGES,
    QuoteChecker,
    compute_anchors,
    file_sha256,
    load_ledger,
    parse_json_object,
    render_comparison,
    render_gates,
)

ANALYSIS_DIR = Path("earnings_call_analysis_v3")
SCORES_DIR = Path("data/earnings_call_scores")
CLAUDE_SCORER_MODEL = "haiku"
NEMOTRON_SCORER_MODEL = DEFAULT_MODEL
CLAUDE_CONCURRENCY = 5
SCORER_ATTEMPTS = 3

SCORER_SYSTEM_PROMPT = """You turn three finished analyst reports on one earnings call into numeric scores. You also receive GATES computed mechanically from the ledger of figures reported on the call. You have no other information about the company or the stock, and you must not use any knowledge of its share price or of how its story turned out.

The GATES are binding. Every score must fall inside its allowed band, and the stage must be one of the allowed stages. Inside a band, place the score by the reported figures only. Management's tone, confidence and the number of upbeat claims are not evidence. Soft evidence (pilots, pipeline, conversations, demand signals, plans) never raises a score.

Scores:

"ambition" (0-100, from the AMBITION report): size and concreteness of the stated long-term goal, not whether it is being achieved.
  10 = no long-term goal beyond the coming quarters
  30 = growing the existing business
  50 = a large goal in an existing market with at least one quantified target
  80 = a goal that would change an industry, with a dated or volume milestone and a stated scale figure
  95 = several quantified milestones and scale figures for a category-defining goal
  Before scoring, fill "long_term_goal_stated" and copy into "quantified_targets" the report's verbatim quotes that attach a NUMBER or DATE to the long-term goal (empty list if none). The ambition band is enforced from these.

"proximity" (from the MILESTONES report and GATES): how far the dream is realized, as a LEVEL. The proximity tier fixes the band. Place it low in the band if the figures that qualified for the tier are marginal (barely positive, a single measure, adjusted only) and high if they are large and broad (several measures well clear of zero, sizeable relative to the stated goal).

"momentum" (-100 to 100, from the MILESTONES report and GATES): direction of the figures since the preceding call. Start from the momentum basis. Move away from it only, and within the band, for figure evidence the mechanical comparison could not see: first-time achievements (up), receding evidence and worsening figures (down), other hard figures that plainly moved. If there is no basis, stay near 0 unless year-over-year figures in the report clearly point one way.

"runway" (0-100, from the RUNWAY report and GATES): ability to fund the goal. Place it within the runway band by the cash and cash-flow figures.

"stage": one of the allowed stages; normally the stage the MILESTONES report concluded.

"proximity_reason", "momentum_reason": one sentence each naming the figures that set the score.

"one_line": one sentence under 30 words on how close the dream looks and why.

Output ONLY one JSON object, with no markdown fences, no text before or after it, and exactly these keys:
{"long_term_goal_stated": true, "quantified_targets": ["..."], "ambition": 0, "proximity": 0, "proximity_reason": "...", "momentum": 0, "momentum_reason": "...", "runway": 0, "stage": "early_proof", "one_line": "..."}
All scores are integers.
"""

JSON_REMINDER = "\n\nReply with the JSON object only. The first character of your reply must be { and the last must be }."


def load_reports(call_dir: Path) -> dict[str, str]:
    return {
        name: (call_dir / f"{name}.md").read_text()
        for name in REPORT_ORDER
        if (call_dir / f"{name}.md").exists()
    }


def ambition_band(goal_stated: bool, verified_targets: int) -> tuple[int, int]:
    if not goal_stated:
        return 0, 25
    if verified_targets == 0:
        return 5, 55
    if verified_targets == 1:
        return 15, 80
    return 20, 100


def _int(value, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} is not a number: {value!r}")
    try:
        return round(float(str(value).strip().rstrip("%")))
    except ValueError:
        raise ValueError(f"{name} is not a number: {value!r}")


def _clamp(value: int, band: list[int] | tuple[int, int]) -> int:
    return max(band[0], min(band[1], value))


def _stage_within(stage: str, allowed: list[str]) -> str:
    if stage in allowed:
        return stage
    lower = [s for s in allowed if STAGES.index(s) < STAGES.index(stage)] if stage in STAGES else []
    return lower[-1] if lower else allowed[-1]


def _ask(user_input: str, model: str, backend: str, label: str) -> dict:
    last_error = None
    for attempt in range(SCORER_ATTEMPTS):
        text = _call_llm(
            SCORER_SYSTEM_PROMPT, user_input + (JSON_REMINDER if attempt else ""),
            model=model, backend=backend, label=f"{label}:scorer",
        )
        try:
            raw = parse_json_object(text)
            for key in ["ambition", "proximity", "momentum", "runway", "stage", "one_line"]:
                if key not in raw:
                    raise ValueError(f"missing key {key!r}")
            return raw
        except ValueError as e:
            last_error = e
            print(f"[{label}] scorer output rejected (attempt {attempt + 1}/{SCORER_ATTEMPTS}): {e}")
    raise last_error


def current_anchors(call_dir: Path) -> dict:
    stored = json.loads((call_dir / ANCHORS_FILE).read_text())
    chain = stored["chain"]
    ledger = load_ledger(call_dir / LEDGER_FILE)
    prior = None
    if chain["status"] == "linked":
        prior_dir = call_dir.parent / f"{call_dir.name.rsplit('_', 1)[0]}_{chain['prior_quarter']}"
        prior = load_ledger(prior_dir / LEDGER_FILE)
        if prior is None:
            return stored
    return compute_anchors(ledger, prior, chain)


def score_call(call_dir: Path, model: str, backend: str, label: str = "") -> dict:
    reports = load_reports(call_dir)
    anchors = current_anchors(call_dir)
    sections = "\n\n".join(f"=== {name.upper()} REPORT ===\n{text}" for name, text in reports.items())
    user_input = (
        f"GATES (binding):\n{render_gates(anchors)}\n\n"
        f"{render_comparison(anchors)}\n\n"
        f"Analyst reports to score:\n\n{sections}"
    )
    raw = _ask(user_input, model, backend, label)

    checker = QuoteChecker(reports["ambition"])
    targets = raw.get("quantified_targets") if isinstance(raw.get("quantified_targets"), list) else []
    verified_targets = [t for t in targets if isinstance(t, str) and any(c.isdigit() for c in t) and checker.verified(t)]
    goal_stated = raw.get("long_term_goal_stated") is not False

    bands = {
        "ambition": ambition_band(goal_stated, len(verified_targets)),
        "proximity": tuple(anchors["proximity_band"]),
        "momentum": tuple(anchors["momentum_band"]),
        "runway": tuple(anchors["runway_band"]),
    }
    model_scores = {k: _int(raw[k], k) for k in bands}
    model_scores["stage"] = str(raw["stage"]).strip().lower()
    scores = {k: _clamp(v, bands[k]) for k, v in model_scores.items() if k in bands}
    scores["stage"] = _stage_within(model_scores["stage"], anchors["allowed_stages"])
    clamped = [k for k in model_scores if scores[k] != model_scores[k]]

    return {
        **scores,
        "evidence_hardness": anchors["evidence_hardness"]["score"],
        "one_line": str(raw["one_line"]).strip(),
        "proximity_reason": str(raw.get("proximity_reason", "")).strip(),
        "momentum_reason": str(raw.get("momentum_reason", "")).strip(),
        "model_scores": model_scores,
        "clamped": clamped,
        "bands": {k: list(v) for k, v in bands.items()},
        "allowed_stages": anchors["allowed_stages"],
        "proximity_tier": anchors["proximity_tier"],
        "momentum_basis": anchors["momentum_basis"],
        "quantified_targets_verified": verified_targets,
        "chain_status": anchors["chain"]["status"],
        "prior_quarter": anchors["chain"]["prior_quarter"],
    }


def analysis_fingerprint(call_dir: Path) -> str:
    return f"{file_sha256(call_dir / PROVENANCE_FILE)}:{ANCHOR_RULES_VERSION}"


def scorable(ticker: str, analysis_dir: Path) -> dict[str, Path]:
    out = {}
    for d in sorted(analysis_dir.glob(f"{ticker}_*Q*")):
        if not d.is_dir() or not (d / PROVENANCE_FILE).exists():
            continue
        try:
            prior = (json.loads((d / PROVENANCE_FILE).read_text()).get("chain") or {}).get("prior_quarter")
        except json.JSONDecodeError:
            continue
        if call_state(d, analysis_dir / f"{ticker}_{prior}" if prior else None) == "complete":
            out[d.name.replace(f"{ticker}_", "")] = d
    return out


def _fresh_rows(out_path: Path, ticker: str, analysis_dir: Path) -> list[dict]:
    if not out_path.exists():
        return []
    calls = scorable(ticker, analysis_dir)
    rows = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
    return [
        r for r in rows
        if r.get("quarter") in calls and r.get("analysis_fingerprint") == analysis_fingerprint(calls[r["quarter"]])
    ]


def scored_quarters(ticker: str, analysis_dir: Path, out_path: Path) -> set[str]:
    return {r["quarter"] for r in _fresh_rows(out_path, ticker, analysis_dir)}


def _process_call(
    label: str, call_dir: Path, out_path: Path, write_lock: threading.Lock, model: str, backend: str
) -> None:
    try:
        scores = score_call(call_dir, model, backend, label)
        scores["quarter"] = label
        scores["analysis_fingerprint"] = analysis_fingerprint(call_dir)
        scores["scorer"] = {"backend": backend, "model": model}
        provenance = json.loads((call_dir / PROVENANCE_FILE).read_text())
        scores["identity_terms_found"] = provenance.get("identity_terms_found")
        scores["novel_proper_nouns"] = provenance.get("novel_proper_nouns")
        with write_lock:
            with out_path.open("a") as out:
                out.write(json.dumps(scores) + "\n")
        print(f"[{label}] ambition={scores['ambition']} proximity={scores['proximity']} momentum={scores['momentum']} "
              f"runway={scores['runway']} stage={scores['stage']} clamped={scores['clamped']}")
    except Exception as e:
        print(f"[{label}] FAILED: {type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--model", default=None)
    parser.add_argument("--backend", choices=["nemotron", "claude"], default="claude")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--out-suffix", default="moonshot_v3")
    parser.add_argument("--out-path", type=Path, default=None, help="overrides the path derived from --out-suffix")
    args = parser.parse_args()
    ticker = args.ticker
    model = args.model or (NEMOTRON_SCORER_MODEL if args.backend == "nemotron" else CLAUDE_SCORER_MODEL)
    concurrency = args.concurrency or (NIM_MAX_CONCURRENCY if args.backend == "nemotron" else CLAUDE_CONCURRENCY)

    out_path = args.out_path or SCORES_DIR / f"{ticker}_{args.out_suffix}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fresh = _fresh_rows(out_path, ticker, args.analysis_dir)
    if out_path.exists():
        existing = [line for line in out_path.read_text().splitlines() if line.strip()]
        if len(existing) != len(fresh):
            print(f"dropping {len(existing) - len(fresh)} score rows whose analysis changed or is incomplete")
            out_path.write_text("".join(json.dumps(r) + "\n" for r in fresh))
    done = {r["quarter"] for r in fresh}

    calls = scorable(ticker, args.analysis_dir)
    print(f"{len(calls)} complete calls found, {len(done)} already scored")
    pending = {label: d for label, d in calls.items() if label not in done}

    write_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(_process_call, label, d, out_path, write_lock, model, args.backend)
            for label, d in pending.items()
        ]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
