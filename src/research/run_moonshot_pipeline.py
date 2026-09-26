import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from anonymize_transcripts import has_config
from run_moonshot_batch import find_transcripts, quarter_sort_key, quarter_states
from score_moonshot import scored_quarters

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
RAW_DIR = Path("data/earnings_calls")
ANON_DIR = Path("data/earnings_calls_anon")
SCORES_DIR = Path("data/earnings_call_scores")

VARIANTS = {
    "identified": {
        "transcript_dir": RAW_DIR,
        "analysis_dir": Path("earnings_call_analysis_v3"),
        "score_suffix": "moonshot_v3",
    },
    "anon": {
        "transcript_dir": ANON_DIR,
        "analysis_dir": Path("earnings_call_analysis_v3_anon"),
        "score_suffix": "moonshot_v3_anon",
    },
}


@dataclass
class Step:
    name: str
    command: list[str]
    missing: Callable[[], list[str]]


def _labels(ticker: str, transcript_dir: Path) -> list[str]:
    return sorted(find_transcripts(ticker, transcript_dir), key=quarter_sort_key)


def _outstanding(labels: list[str], done: set[str]) -> list[str]:
    if not labels:
        return ["no transcripts"]
    return [l for l in labels if l not in done]


def _complete_calls(ticker: str, transcript_dir: Path, analysis_dir: Path) -> set[str]:
    return {label for label, state in quarter_states(ticker, transcript_dir, analysis_dir).items() if state == "complete"}


def _scored(ticker: str, analysis_dir: Path, suffix: str) -> set[str]:
    return scored_quarters(ticker, analysis_dir, SCORES_DIR / f"{ticker}_{suffix}.jsonl")


def _reconcile_progress(ticker: str, transcript_dir: Path, analysis_dir: Path) -> None:
    progress = analysis_dir / f"_progress_{ticker}.txt"
    if not progress.exists():
        return
    complete = _complete_calls(ticker, transcript_dir, analysis_dir)
    logged = progress.read_text().splitlines()
    kept = [label for label in logged if label in complete]
    if kept != logged:
        progress.write_text("".join(label + "\n" for label in kept))
        print(f"  reconciled progress log: dropped {len(logged) - len(kept)} entries that are incomplete or stale")


def build_steps(ticker: str, variants: list[str], args: argparse.Namespace) -> list[Step]:
    py = sys.executable
    steps: list[Step] = []

    raw_missing: Callable[[], list[str]] = lambda: [] if _labels(ticker, RAW_DIR) else ["no transcripts on disk"]
    fetch_cmd = [py, str(SCRIPTS.parent / "data_pipeline" / "fetch_earnings_transcripts.py"), ticker]
    steps.append(Step("fetch", fetch_cmd, raw_missing))

    for variant in variants:
        cfg = VARIANTS[variant]
        analysis_dir, transcript_dir, suffix = cfg["analysis_dir"], cfg["transcript_dir"], cfg["score_suffix"]

        if variant == "anon":
            config_cmd = [
                py, str(SCRIPTS / "generate_anonymize_config.py"), ticker,
                "--backend", args.backend,
            ]
            if args.model:
                config_cmd += ["--model", args.model]
            steps.append(Step(
                "anon-config",
                config_cmd,
                lambda: [] if has_config(ticker) else ["no anonymization config"],
            ))

            anon_cmd = [py, str(SCRIPTS / "anonymize_transcripts.py"), ticker]
            steps.append(Step(
                "anonymize",
                anon_cmd,
                lambda: _outstanding(_labels(ticker, RAW_DIR), set(_labels(ticker, ANON_DIR))),
            ))

        analyze_cmd = [
            py, str(SCRIPTS / "run_moonshot_batch.py"), ticker,
            "--backend", args.backend,
            "--ledger-backend", args.ledger_backend,
            "--transcript-dir", str(transcript_dir),
            "--output-dir", str(analysis_dir),
        ]
        if args.model:
            analyze_cmd += ["--model", args.model]
        if args.ledger_model:
            analyze_cmd += ["--ledger-model", args.ledger_model]
        if args.allow_chain_break:
            analyze_cmd += ["--allow-chain-break"]
        if variant == "anon":
            analyze_cmd += ["--identity-check"]
        steps.append(Step(
            f"analyze[{variant}]",
            analyze_cmd,
            lambda t=transcript_dir, a=analysis_dir: _outstanding(_labels(ticker, t), _complete_calls(ticker, t, a)),
        ))

        score_cmd = [
            py, str(SCRIPTS / "score_moonshot.py"), ticker,
            "--backend", args.score_backend,
            "--analysis-dir", str(analysis_dir),
            "--out-suffix", suffix,
        ]
        if args.score_model:
            score_cmd += ["--model", args.score_model]
        steps.append(Step(
            f"score[{variant}]",
            score_cmd,
            lambda t=transcript_dir, a=analysis_dir, s=suffix: _outstanding(_labels(ticker, t), _scored(ticker, a, s)),
        ))
    return steps


def print_status(steps: list[Step]) -> bool:
    all_done = True
    for step in steps:
        missing = step.missing()
        state = "done" if not missing else f"INCOMPLETE ({len(missing)}: {', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''})"
        print(f"  {step.name:<20} {state}")
        all_done = all_done and not missing
    return all_done


def run_step(step: Step, ticker: str, max_retries: int, retry_delay: float) -> bool:
    for attempt in range(1, max_retries + 2):
        missing = step.missing()
        if not missing:
            return True
        label = f"attempt {attempt}/{max_retries + 1}"
        print(f"\n=== {step.name}: {len(missing)} outstanding ({label}) ===")
        if step.name.startswith("analyze"):
            analysis_dir = Path(step.command[step.command.index("--output-dir") + 1])
            transcript_dir = Path(step.command[step.command.index("--transcript-dir") + 1])
            _reconcile_progress(ticker, transcript_dir, analysis_dir)
        result = subprocess.run(step.command, cwd=REPO_ROOT)
        if result.returncode != 0:
            print(f"  {step.name} exited with code {result.returncode}")
        if not step.missing():
            return True
        if attempt <= max_retries:
            print(f"  still incomplete, retrying in {retry_delay:.0f}s")
            time.sleep(retry_delay)
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Fetch -> (anonymize) -> analyze -> score for one ticker. Safe to re-run: "
        "completed work is detected from the output files and skipped."
    )
    parser.add_argument("ticker")
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=["identified"],
                        help="the anon variant is not blind (see probe_identity_recognition.py), so it is off by default")
    parser.add_argument("--backend", choices=["nemotron", "claude"], default="nemotron")
    parser.add_argument("--model", default=None, help="analysis model")
    parser.add_argument("--ledger-backend", choices=["nemotron", "claude"], default="nemotron")
    parser.add_argument("--ledger-model", default=None)
    parser.add_argument("--score-backend", choices=["nemotron", "claude"], default="claude")
    parser.add_argument("--score-model", default=None)
    parser.add_argument("--max-retries", type=int, default=2, help="extra passes over a step that finishes incomplete")
    parser.add_argument("--retry-delay", type=float, default=60.0)
    parser.add_argument("--refetch", action="store_true", help="fetch transcripts even if some are on disk")
    parser.add_argument("--status", action="store_true", help="show what is done and outstanding, then exit")
    parser.add_argument("--allow-chain-break", action="store_true",
                        help="analyze a call even when the preceding call's ledger could not be produced (recorded as chain_break)")
    args = parser.parse_args()
    ticker = args.ticker.upper()

    variants = list(args.variants)
    steps = build_steps(ticker, variants, args)
    if args.status:
        print(f"{ticker} pipeline status:")
        print_status(steps)
        return

    print(f"{ticker}: {' -> '.join(s.name for s in steps)}")
    try:
        if args.refetch:
            print("\n=== fetch: refetching listed transcripts ===")
            subprocess.run(steps[0].command, cwd=REPO_ROOT)
        for step in steps:
            if not run_step(step, ticker, args.max_retries, args.retry_delay):
                print(f"\nStopped: {step.name} is still incomplete after {args.max_retries + 1} attempts.")
                print("Status:")
                print_status(steps)
                print(f"Fix the cause, then re-run the same command to resume from {step.name}.")
                raise SystemExit(1)
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run the same command to resume where this left off.")
        raise SystemExit(130)

    print("\nAll steps complete:")
    print_status(steps)


if __name__ == "__main__":
    main()
