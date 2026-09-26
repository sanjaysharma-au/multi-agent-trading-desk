import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.earnings_call_analyst import DEFAULT_BACKEND, DEFAULT_MODEL, NIM_MAX_CONCURRENCY, _call_llm
from agents.moonshot_ledger import parse_json_object

from anonymize_transcripts import DEFAULT_OUTPUT_DIR, get_config
from run_moonshot_batch import find_transcripts, quarter_sort_key

PROBE_DIR = Path("data/identity_probe")
DEFAULT_SAMPLE = 8
DEFAULT_EXCERPT_WORDS = 3000
CLAUDE_MODEL = "haiku"

SYSTEM_PROMPT = """You are shown the opening of an earnings call transcript from which the company's name, its products, its executives and calendar years have been replaced by placeholders (Company A, Program N, Person N, Y+k for a year). Decide which public company this most likely is, using only clues in the text.

Output ONLY a JSON object (no markdown fences, no commentary) with these keys:
- "guess": the company's real name, or "unknown" if you cannot tell
- "confidence": an integer from 0 to 100
- "cues": a list of up to 6 short phrases quoting or describing the clues that led you to the guess
"""


def sample_labels(labels: list[str], n: int) -> list[str]:
    if n >= len(labels):
        return labels
    step = (len(labels) - 1) / (n - 1)
    return [labels[round(i * step)] for i in range(n)]


def is_correct(guess: str, aliases: list[str]) -> bool:
    lowered = guess.lower()
    return any(re.search(rf"\b{re.escape(a.lower())}\b", lowered) for a in aliases)


def probe_one(label: str, path: Path, aliases: list[str], words: int, model: str, backend: str) -> dict:
    excerpt = " ".join(path.read_text().split()[:words])
    text = _call_llm(
        SYSTEM_PROMPT, f"Transcript opening:\n\n{excerpt}", model=model, backend=backend, label=f"{label}:probe"
    )
    result = parse_json_object(text)
    guess = str(result.get("guess", "unknown"))
    return {
        "quarter": label,
        "guess": guess,
        "confidence": result.get("confidence"),
        "cues": result.get("cues", []),
        "correct": is_correct(guess, aliases),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Ask a model which company an anonymized transcript is, to measure how well the anonymizer blinds it."
    )
    parser.add_argument("ticker")
    parser.add_argument("--transcript-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default="baseline", help="label for this run, used in the output file name")
    parser.add_argument("--all", action="store_true", help="probe every quarter instead of an evenly spaced sample")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    parser.add_argument("--words", type=int, default=DEFAULT_EXCERPT_WORDS)
    parser.add_argument("--backend", choices=["nemotron", "claude"], default=DEFAULT_BACKEND)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    ticker = args.ticker.upper()

    model = args.model or (DEFAULT_MODEL if args.backend == "nemotron" else CLAUDE_MODEL)
    transcripts = find_transcripts(ticker, args.transcript_dir)
    labels = sorted(transcripts, key=quarter_sort_key)
    if not labels:
        raise SystemExit(f"no transcripts for {ticker} in {args.transcript_dir}")
    chosen = labels if args.all else sample_labels(labels, args.sample)
    aliases = get_config(ticker)["company"]

    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROBE_DIR / f"{ticker}_{args.tag}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(line)["quarter"] for line in out_path.read_text().splitlines() if line.strip()}
    pending = [l for l in chosen if l not in done]
    print(f"{len(chosen)} quarters selected for {ticker}, {len(done)} already probed -> {out_path}")

    lock = threading.Lock()

    def run(label: str) -> None:
        try:
            row = probe_one(label, transcripts[label], aliases, args.words, model, args.backend)
            row["model"] = model
            with lock:
                with out_path.open("a") as out:
                    out.write(json.dumps(row) + "\n")
            print(f"[{label}] guess={row['guess']!r} confidence={row['confidence']} correct={row['correct']}")
        except Exception as e:
            print(f"[{label}] FAILED: {type(e).__name__}: {e}")

    workers = NIM_MAX_CONCURRENCY if args.backend == "nemotron" else 3
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for future in [executor.submit(run, l) for l in pending]:
            future.result()

    rows = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r["quarter"] in chosen]
    print(f"\nrecognized {sum(r['correct'] for r in rows)} of {len(rows)} quarters as {ticker}")
    for r in sorted(rows, key=lambda r: quarter_sort_key(r["quarter"])):
        print(f"  {r['quarter']}: {r['guess']} ({r['confidence']}) {'CORRECT' if r['correct'] else ''}  cues: {'; '.join(map(str, r['cues']))[:160]}")


if __name__ == "__main__":
    main()
