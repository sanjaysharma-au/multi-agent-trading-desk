import argparse
import json
import subprocess
from pathlib import Path

ANALYSIS_DIR = Path("earnings_call_analysis")
SCORER_MODEL = "haiku"
TIMEOUT_SECONDS = 120

SCORER_SYSTEM_PROMPT = """You extract structured numeric scores from a set of finished analyst reports on a single earnings call. You did not write them and have no other information about the company or the stock.

You are given each specialist's own report plus the lead analyst's synthesis. Score each dimension from THE REPORT THAT OWNS IT, named below, and use the synthesis only for conflict_level and as context. The synthesis is a summary and routinely omits specific evidence that is present in a specialist's own report -- if a specialist report supports a score, use it, even where the synthesis does not repeat it. Only fall back to a mid-range "no basis to judge" score when the OWNING report itself gives you nothing.

These scores exist to DISCRIMINATE between calls. The same company is scored this way across dozens of quarters spanning a decade, and the scores are only useful if a genuinely exceptional call lands far away from a mediocre one. Use the full 0-100 range. A set of scores that all cluster between 40 and 60 carries no information at all, however reasonable each one looks on its own.

Read the reports below and output ONLY a JSON object (no markdown fences, no commentary) with these exact keys:

- "tone_confidence" [from the SENTIMENT report]: 0-100, how confident and assertive management's tone was.
  20 = defensive, evasive under questioning, or visibly rattled
  50 = ordinary corporate delivery: confident in prepared remarks, hedged under pressure
  80 = notably assertive and specific, steady when pressed on hard questions
  95 = reserve for the rare call where confidence is emphatic AND the report says it held up under direct challenge

- "guidance_credibility" [from the GUIDANCE report]: 0-100, how credible the forward guidance was judged to be.
  20 = guidance walked back, contradicted, or quietly abandoned on this call
  50 = a typical mix of hedged aspiration and evidence-backed commitment
  80 = specific, dated, numeric commitments with demonstrated pace behind them
  95 = reserve for guidance both unusually specific AND already substantially evidenced on this same call

- "competitive_strength" [from the COMPETITIVE report]: 0-100, how strong the competitive position was judged to be.
  20 = losing share, no differentiation, claims unsupported by evidence
  50 = defensible but contested, mixed evidence
  80 = clear well-evidenced advantage, pricing power or a structural moat
  95 = reserve for a dominant position the report treats as established by specific numbers

- "financial_health" [from the FINANCIAL report]: 0-100, how strong the financial picture was judged to be.
  20 = solvency or liquidity pressure, capital raise needed on unfavourable terms
  50 = adequate: no distress, no particular strength
  80 = strong cash generation, margins and funding clearly supporting the stated plans
  95 = reserve for unusually strong balance sheet AND cash generation, both in specific figures

- "optionality_strength" [from the OPTIONALITY report]: 0-100, how large and how widening the opportunity ahead of the company appears, per the growth-optionality read.
  20 = opportunity narrowing; programs dead or defunded; capex purely defensive
  50 = stable opportunity; executing the existing business without opening new ones
  80 = clearly widening; offensive capacity investment; new S-curves opening with evidence
  95 = reserve for a large opportunity expanding on several fronts at once with concrete evidence
  Score the SIZE and DIRECTION of the opportunity, not whether management can be trusted. Where the optionality report describes a program as LATE BUT ADVANCING, that is evidence the opportunity is real and being pursued -- do not penalise it as a missed promise. Missed promises belong in guidance_credibility, which is scored separately.

- "growth_acceleration" [from the OPTIONALITY report]: 0-100, the SECOND derivative of growth described on the call, centred at 50.
  20 = growth clearly decelerating
  50 = growth steady, or the optionality report gives no basis to judge direction
  80 = growth clearly accelerating
  95 = reserve for sharp evidenced acceleration across more than one metric
  This is the CHANGE in the growth rate, not its level. A company growing 50% and slowing scores below 50; one growing 15% and accelerating scores above it.

- "conflict_level" [from the SYNTHESIS]: 0-100, how much genuine tension the synthesis found between the specialist reads (0=all reads agreed, 100=reads strongly diverged). Count only tension the synthesis presents as conflict between independent evidence -- several specialists restating one shared concern is agreement, not conflict.

Base every score only on the reports provided.

Uncertainty belongs in "conflict_level", which exists precisely to carry it. Do NOT also pull the other scores toward the middle because a report is hedged: that double-counts the hedging and destroys the discrimination these scores exist to provide. Where the owning report makes a strong claim in one direction, score that dimension strongly even if the overall picture is mixed.
"""


REPORT_ORDER = ["sentiment", "guidance", "competitive", "financial", "optionality", "synthesis"]


def load_reports(quarter_dir: Path) -> dict[str, str]:
    return {
        name: (quarter_dir / f"{name}.md").read_text()
        for name in REPORT_ORDER
        if (quarter_dir / f"{name}.md").exists()
    }


def score_reports(reports: dict[str, str]) -> dict:
    sections = "\n\n".join(
        f"=== {name.upper()} REPORT ===\n{text}" for name, text in reports.items()
    )
    result = subprocess.run(
        [
            "claude", "-p", f"Analyst reports to score:\n\n{sections}",
            "--system-prompt", SCORER_SYSTEM_PROMPT,
            "--model", SCORER_MODEL,
            "--output-format", "text",
            "--restricted",
            "--disallowedTools", "Bash", "Edit", "Write", "NotebookEdit",
            "--permission-prompts", "none",
        ],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI failed (exit {result.returncode}): {result.stderr}")
    text = result.stdout.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def find_quarter_dirs(ticker: str, analysis_dir: Path = ANALYSIS_DIR) -> dict[str, Path]:
    dirs = {}

    all_ticker_dirs = set(analysis_dir.glob(f"{ticker}_*"))
    matched_dirs = set(analysis_dir.glob(f"{ticker}_*Q*_*"))
    skipped_dirs = all_ticker_dirs - matched_dirs
    for d in sorted(skipped_dirs):
        if d.is_dir():
            print(f"Warning: skipping directory {d.name} (does not match quarter pattern)")

    for d in sorted(matched_dirs):
        label = d.name.replace(f"{ticker}_", "").rsplit("_", 1)[0]
        dirs[label] = d
    return dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--out-suffix", default="scores")
    args = parser.parse_args()

    out_path = Path(f"data/earnings_call_scores/{args.ticker}_{args.out_suffix}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            done.add(json.loads(line)["quarter"])

    quarter_dirs = find_quarter_dirs(args.ticker, args.analysis_dir)
    print(f"{len(quarter_dirs)} quarters found, {len(done)} already scored")

    with out_path.open("a") as out:
        for label, d in quarter_dirs.items():
            if label in done:
                continue
            reports = load_reports(d)
            if "synthesis" not in reports:
                print(f"[{label}] no synthesis.md, skipping")
                continue
            try:
                scores = score_reports(reports)
                scores["quarter"] = label
                out.write(json.dumps(scores) + "\n")
                out.flush()
                print(f"[{label}] {scores}")
            except Exception as e:
                print(f"[{label}] FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
