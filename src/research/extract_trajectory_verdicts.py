import argparse
import json
import subprocess
from pathlib import Path

SCORER_MODEL = "haiku"
TIMEOUT_SECONDS = 120

EXTRACT_SYSTEM_PROMPT = """You extract a single verdict from a finished quarter-over-quarter execution trajectory report. Output ONLY a JSON object (no markdown fences, no commentary) with exactly these keys:
- "score": an integer from -100 to +100 measuring how strongly the report's own verdict leans deteriorating (negative) vs improving (positive). 0 means genuinely flat/stable. Use the full range -- a report that hedges ("STABLE, tilting DETERIORATING") should land near -20 to -40, not at the same extreme as one describing zero promises delivered and multiple silent misses (-80 to -100). Similarly scale positive scores by how unambiguously good the quarter was.
- "verdict": one of "improving", "stable", "deteriorating" -- the report's own categorical label (for grouping only; "score" carries the real signal)
- "one_line": a single sentence (under 25 words) summarizing why

Base this only on the report's own final verdict section and the language it uses to express confidence in that verdict -- not your own independent judgment of the underlying facts, and not any knowledge of this stock's price or how its story turned out, even from training data.
"""


def extract_verdict(report_text: str) -> dict:
    result = subprocess.run(
        [
            "claude", "-p", f"Trajectory report:\n\n{report_text}",
            "--system-prompt", EXTRACT_SYSTEM_PROMPT,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    args = parser.parse_args()

    in_dir = Path(f"earnings_call_trajectory/{args.ticker}")
    out_path = in_dir / "_verdicts.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            done.add(json.loads(line)["pair"])

    files = sorted(in_dir.glob("*.md"))
    print(f"{len(files)} trajectory reports found, {len(done)} already extracted")

    with out_path.open("a") as out:
        for f in files:
            pair = f.stem
            if pair in done:
                continue
            try:
                verdict = extract_verdict(f.read_text())
                verdict["pair"] = pair
                out.write(json.dumps(verdict) + "\n")
                out.flush()
                print(f"[{pair}] {verdict['verdict']}: {verdict['one_line']}")
            except Exception as e:
                print(f"[{pair}] FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
