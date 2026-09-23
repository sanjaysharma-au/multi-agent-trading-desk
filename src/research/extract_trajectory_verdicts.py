import argparse
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCORER_MODEL = "haiku"
TIMEOUT_SECONDS = 120
# Local `claude` CLI subprocesses, not NVIDIA's shared/rate-limited endpoint --
# the concurrency ceiling found for NIM doesn't apply here. 5 matches the
# unconstrained concurrency the specialist batch already ran at for claude.
DEFAULT_CONCURRENCY = 5

EXTRACT_SYSTEM_PROMPT = """You extract a single verdict from a finished quarter-over-quarter execution trajectory report. Output ONLY a JSON object (no markdown fences, no commentary) with exactly these keys:
- "score": an integer from -100 to +100 measuring how strongly the report's own verdict leans deteriorating (negative) vs improving (positive). 0 means genuinely flat/stable. Use the full range -- a report that hedges ("STABLE, tilting DETERIORATING") should land near -20 to -40, not at the same extreme as one describing zero promises delivered and multiple silent misses (-80 to -100). Similarly scale positive scores by how unambiguously good the quarter was.
- "verdict": one of "improving", "stable", "deteriorating" -- the report's own categorical label (for grouping only; "score" carries the real signal)
- "one_line": a single sentence (under 25 words) summarizing why

Base this only on the report's own final verdict section and the language it uses to express confidence in that verdict -- not your own independent judgment of the underlying facts, and not any knowledge of this stock's price or how its story turned out, even from training data.
"""


def extract_verdict(report_text: str, model: str = SCORER_MODEL) -> dict:
    result = subprocess.run(
        [
            "claude", "-p", f"Trajectory report:\n\n{report_text}",
            "--system-prompt", EXTRACT_SYSTEM_PROMPT,
            "--model", model,
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


def _process_file(f: Path, out_path: Path, write_lock: threading.Lock, model: str) -> None:
    pair = f.stem
    try:
        # Reads only the finished report text on disk -- never sees another
        # scorer's output, so two models scoring the same reports stay blind
        # to each other by construction, not by convention.
        verdict = extract_verdict(f.read_text(), model=model)
        verdict["pair"] = pair
        with write_lock:
            with out_path.open("a") as out:
                out.write(json.dumps(verdict) + "\n")
        print(f"[{pair}] {verdict['verdict']}: {verdict['one_line']}")
    except Exception as e:
        print(f"[{pair}] FAILED: {type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    parser.add_argument("--input-root", type=Path, default=Path("earnings_call_trajectory"))
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--model", default=SCORER_MODEL)
    parser.add_argument("--output-name", default="_verdicts.jsonl")
    args = parser.parse_args()

    in_dir = args.input_root / args.ticker
    out_path = in_dir / args.output_name
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            done.add(json.loads(line)["pair"])

    files = sorted(in_dir.glob("*.md"))
    print(f"{len(files)} trajectory reports found, {len(done)} already extracted")
    pending = [f for f in files if f.stem not in done]
    write_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(_process_file, f, out_path, write_lock, args.model) for f in pending]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
