import argparse
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.earnings_trajectory import assess_trajectory
from agents.earnings_call_analyst import NIM_MAX_CONCURRENCY

ANALYSIS_DIR = Path("earnings_call_analysis")
TRANSCRIPT_DIR = Path("data/earnings_calls")
OUTPUT_DIR = Path("earnings_call_trajectory")


def find_quarter_dirs(ticker: str, analysis_dir: Path) -> dict[str, Path]:
    dirs = {}
    for d in sorted(analysis_dir.glob(f"{ticker}_*Q*_*")):
        label = d.name.replace(f"{ticker}_", "").rsplit("_", 1)[0]
        dirs[label] = d
    return dirs


def quarter_sort_key(label: str) -> tuple[int, int]:
    year, q = re.match(r"(\d{4})Q(\d)", label).groups()
    return int(year), int(q)


def next_quarter(label: str) -> str:
    year, q = quarter_sort_key(label)
    return f"{year + 1}Q1" if q == 4 else f"{year}Q{q + 1}"


def _process_pair(
    pair_key: str,
    prior_label: str,
    current_label: str,
    quarter_dirs: dict[str, Path],
    ticker: str,
    out_dir: Path,
    progress_log: Path,
    progress_lock: threading.Lock,
    model: str | None,
    backend: str,
) -> None:
    prior_guidance = quarter_dirs[prior_label] / "guidance.md"
    current_transcript = TRANSCRIPT_DIR / f"{ticker}_{current_label}.txt"
    if not prior_guidance.exists() or not current_transcript.exists():
        print(f"[{pair_key}] missing input file, skipping")
        return

    print(f"[{pair_key}] analyzing...")
    try:
        result = assess_trajectory(
            prior_guidance.read_text(),
            current_transcript.read_text(),
            model=model,
            backend=backend,
            label=pair_key,
        )
        out_path = out_dir / f"{pair_key.replace('>', '')}.md"
        out_path.write_text(result)
        with progress_lock:
            with progress_log.open("a") as log:
                log.write(pair_key + "\n")
        print(f"[{pair_key}] -> {out_path}")
    except Exception as e:
        print(f"[{pair_key}] FAILED: {type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    parser.add_argument("--model", default=None)
    parser.add_argument("--backend", choices=["claude", "nemotron"], default="claude")
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    ticker = args.ticker

    quarter_dirs = find_quarter_dirs(ticker, args.analysis_dir)
    labels = sorted(quarter_dirs.keys(), key=quarter_sort_key)
    print(f"{len(labels)} quarters found for {ticker} in {args.analysis_dir}")

    out_dir = args.output_dir / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_log = out_dir / "_progress.txt"
    done = set(progress_log.read_text().splitlines()) if progress_log.exists() else set()
    progress_lock = threading.Lock()

    # Trajectory pairs are independent of each other -- unlike the analysis step's
    # guidance.md chain, nothing here depends on another pair's output -- so nemotron
    # runs can reuse the same concurrency cap the analysis step found NIM tolerates
    # (5-way triggered 503s, 2-way was reliable). claude stays sequential as before.
    max_workers = NIM_MAX_CONCURRENCY if args.backend == "nemotron" else 1
    pending = []
    skipped_gaps = 0
    for prior_label, current_label in zip(labels, labels[1:]):
        # A missing quarter (e.g. no transcript source for that period) leaves a
        # hole in `labels`, and zip() would otherwise pair the quarters on either
        # side of it as if they were consecutive -- silently asking the trajectory
        # tool to check "this quarter's promises" against a transcript from a year
        # or more later. Only pair labels that are truly back-to-back quarters.
        if next_quarter(prior_label) != current_label:
            print(f"[{prior_label}->{current_label}] not consecutive quarters (gap), skipping pair")
            skipped_gaps += 1
            continue
        pair_key = f"{prior_label}->{current_label}"
        if pair_key in done:
            print(f"[{pair_key}] already done, skipping")
            continue
        pending.append((pair_key, prior_label, current_label))
    print(f"{len(labels) - 1 - skipped_gaps} consecutive pairs to assess ({skipped_gaps} skipped for non-adjacent gaps)")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _process_pair,
                pair_key, prior_label, current_label,
                quarter_dirs, ticker, out_dir, progress_log, progress_lock,
                args.model, args.backend,
            )
            for pair_key, prior_label, current_label in pending
        ]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
