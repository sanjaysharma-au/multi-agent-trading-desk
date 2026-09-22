import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.earnings_call_analyst import analyze_transcript, DEFAULT_BACKEND

TRANSCRIPT_DIR = Path("data/earnings_calls")
DEFAULT_OUTPUT_DIR = Path("earnings_call_analysis_v2")


def find_transcripts(ticker: str) -> dict[str, Path]:
    transcripts = {}
    for p in TRANSCRIPT_DIR.glob(f"{ticker}_*Q*.txt"):
        transcripts[p.stem.replace(f"{ticker}_", "")] = p
    return transcripts


def quarter_sort_key(label: str) -> tuple[int, int]:
    year, q = re.match(r"(\d{4})Q(\d)", label).groups()
    return int(year), int(q)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    parser.add_argument("--model", default="nemotron-3-ultra")
    parser.add_argument("--backend", choices=["nemotron", "claude"], default=DEFAULT_BACKEND)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--only-quarter", default=None)
    parser.add_argument("--no-prior-guidance", action="store_true")
    args = parser.parse_args()
    ticker = args.ticker

    transcripts = find_transcripts(ticker)
    labels = sorted(transcripts, key=quarter_sort_key)
    if args.only_quarter:
        labels = [l for l in labels if l == args.only_quarter]
        if not labels:
            raise SystemExit(f"quarter {args.only_quarter} not found for {ticker}")

    out_root = args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)
    progress_log = out_root / f"_progress_{ticker}.txt"
    done = set(progress_log.read_text().splitlines()) if progress_log.exists() else set()
    print(f"{len(labels)} quarters for {ticker}, {len(done)} already done -> {out_root}")

    # The guidance specialist's only permitted track record is the prior quarter's
    # guidance.md from this same run, so quarters must be processed in order.
    prior_guidance = None
    for label in labels:
        if label in done:
            existing = sorted(out_root.glob(f"{ticker}_{label}_*"))
            prior_guidance = existing[-1] / "guidance.md" if existing else None
            print(f"[{label}] already done, skipping")
            continue

        print(f"[{label}] analyzing...")
        try:
            out_dir = analyze_transcript(
                transcripts[label],
                f"{ticker}_{label}",
                model=args.model,
                prior_guidance_path=None if args.no_prior_guidance else prior_guidance,
                output_dir=out_root,
                backend=args.backend,
            )
            with progress_log.open("a") as log:
                log.write(label + "\n")
            prior_guidance = out_dir / "guidance.md"
            print(f"[{label}] -> {out_dir}")
        except Exception as e:
            print(f"[{label}] FAILED: {type(e).__name__}: {e}")
            prior_guidance = None


if __name__ == "__main__":
    main()
