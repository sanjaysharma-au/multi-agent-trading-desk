import argparse
import json
import subprocess
from pathlib import Path

ANALYSIS_DIR = Path("earnings_call_analysis")
SCORER_MODEL = "haiku"
TIMEOUT_SECONDS = 120

SCORER_SYSTEM_PROMPT = """You extract structured numeric scores from a finished four-analyst synthesis report of an earnings call. You did not write the synthesis and have no other information about the company or the stock.

Read the synthesis below and output ONLY a JSON object (no markdown fences, no commentary) with these exact keys:
- "tone_confidence": 0-100, how confident/positive management's tone was overall (0=very hedged/defensive, 100=very confident/assertive)
- "guidance_credibility": 0-100, how credible the forward guidance given on the call was judged to be (0=not credible/walked back, 100=highly credible/evidence-backed)
- "competitive_strength": 0-100, how strong the company's competitive position was judged to be (0=weak/unsupported claims, 100=strong/well-evidenced)
- "financial_health": 0-100, how strong the financial picture was judged to be (0=weak, 100=strong)
- "conflict_level": 0-100, how much genuine tension/disagreement the synthesis found between the four specialist reads (0=all reads agreed, 100=reads strongly diverged)

Base every score only on the synthesis text provided. If the synthesis explicitly says a read is mixed or uncertain, reflect that with a mid-range score rather than guessing toward an extreme.
"""


def score_synthesis(synthesis_text: str) -> dict:
    result = subprocess.run(
        [
            "claude", "-p", f"Synthesis report to score:\n\n{synthesis_text}",
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


def find_quarter_dirs(ticker: str) -> dict[str, Path]:
    dirs = {}
    for d in sorted(ANALYSIS_DIR.glob(f"{ticker}_*Q*_*")):
        label = d.name.replace(f"{ticker}_", "").rsplit("_", 1)[0]
        dirs[label] = d
    return dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    args = parser.parse_args()

    out_path = Path(f"data/earnings_call_scores/{args.ticker}_scores.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            done.add(json.loads(line)["quarter"])

    quarter_dirs = find_quarter_dirs(args.ticker)
    print(f"{len(quarter_dirs)} quarters found, {len(done)} already scored")

    with out_path.open("a") as out:
        for label, d in quarter_dirs.items():
            if label in done:
                continue
            synthesis_path = d / "synthesis.md"
            if not synthesis_path.exists():
                print(f"[{label}] no synthesis.md, skipping")
                continue
            try:
                scores = score_synthesis(synthesis_path.read_text())
                scores["quarter"] = label
                out.write(json.dumps(scores) + "\n")
                out.flush()
                print(f"[{label}] {scores}")
            except Exception as e:
                print(f"[{label}] FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
