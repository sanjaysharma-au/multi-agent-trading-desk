import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.earnings_trajectory import assess_trajectory

ANALYSIS_DIR = Path("earnings_call_analysis")
TRANSCRIPT_DIR = Path("data/earnings_calls")
OUTPUT_DIR = Path("earnings_call_trajectory")


def find_quarter_dirs(ticker: str) -> dict[str, Path]:
    dirs = {}
    for d in sorted(ANALYSIS_DIR.glob(f"{ticker}_*Q*_*")):
        label = d.name.replace(f"{ticker}_", "").rsplit("_", 1)[0]
        dirs[label] = d
    return dirs


def quarter_sort_key(label: str) -> tuple[int, int]:
    year, q = re.match(r"(\d{4})Q(\d)", label).groups()
    return int(year), int(q)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    parser.add_argument("--model", default="sonnet")
    args = parser.parse_args()
    ticker = args.ticker

    quarter_dirs = find_quarter_dirs(ticker)
    labels = sorted(quarter_dirs.keys(), key=quarter_sort_key)
    print(f"{len(labels)} quarters found for {ticker}, {len(labels) - 1} consecutive pairs to assess")

    out_dir = OUTPUT_DIR / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_log = out_dir / "_progress.txt"
    done = set(progress_log.read_text().splitlines()) if progress_log.exists() else set()

    for prior_label, current_label in zip(labels, labels[1:]):
        pair_key = f"{prior_label}->{current_label}"
        if pair_key in done:
            print(f"[{pair_key}] already done, skipping")
            continue

        prior_guidance = quarter_dirs[prior_label] / "guidance.md"
        current_transcript = TRANSCRIPT_DIR / f"{ticker}_{current_label}.txt"
        if not prior_guidance.exists() or not current_transcript.exists():
            print(f"[{pair_key}] missing input file, skipping")
            continue

        print(f"[{pair_key}] analyzing...")
        try:
            result = assess_trajectory(prior_guidance.read_text(), current_transcript.read_text(), model=args.model)
            out_path = out_dir / f"{pair_key.replace('>', '')}.md"
            out_path.write_text(result)
            with progress_log.open("a") as log:
                log.write(pair_key + "\n")
            print(f"[{pair_key}] -> {out_path}")
        except Exception as e:
            print(f"[{pair_key}] FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
