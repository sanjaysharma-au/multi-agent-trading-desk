import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.earnings_call_analyst import DEFAULT_BACKEND, DEFAULT_MODEL, NIM_MAX_CONCURRENCY
from agents.moonshot_analyst import (
    OUTPUT_DIR,
    PROVENANCE_FILE,
    PRESS_DIR,
    analyze_call,
    call_state,
    extract_ledger,
    extract_press_ledger,
    ledger_is_current,
    make_chain,
    press_ledger_is_current,
)
from agents.moonshot_ledger import LEDGER_FILE

TRANSCRIPT_DIR = Path("data/earnings_calls")
CLAUDE_LEDGER_MODEL = "haiku"


def find_transcripts(ticker: str, transcript_dir: Path) -> dict[str, Path]:
    return {
        p.stem.replace(f"{ticker}_", ""): p
        for p in transcript_dir.glob(f"{ticker}_*Q*.txt")
    }


def quarter_sort_key(label: str) -> tuple[int, int]:
    year, q = re.match(r"(\d{4})Q(\d)", label).groups()
    return int(year), int(q)


def quarters_between(earlier: str, later: str) -> int:
    (y1, q1), (y2, q2) = quarter_sort_key(earlier), quarter_sort_key(later)
    return (y2 * 4 + q2) - (y1 * 4 + q1) - 1


def predecessors(labels: list[str]) -> dict[str, str | None]:
    ordered = sorted(labels, key=quarter_sort_key)
    return {label: ordered[i - 1] if i else None for i, label in enumerate(ordered)}


def quarter_states(ticker: str, transcript_dir: Path, analysis_dir: Path) -> dict[str, str]:
    transcripts = find_transcripts(ticker, transcript_dir)
    prior = predecessors(list(transcripts))
    return {
        label: call_state(
            analysis_dir / f"{ticker}_{label}",
            analysis_dir / f"{ticker}_{prior[label]}" if prior[label] else None,
            transcripts[label],
        )
        for label in transcripts
    }


def _log_progress(progress_log: Path, lock: threading.Lock, label: str) -> None:
    with lock:
        with progress_log.open("a") as log:
            log.write(label + "\n")


def _extract(ticker: str, label: str, path: Path, model: str, backend: str, output_dir: Path) -> None:
    print(f"[{label}] extracting ledger...")
    try:
        out = extract_ledger(path, f"{ticker}_{label}", model=model, backend=backend, output_dir=output_dir)
        print(f"[{label}] ledger -> {out}")
    except Exception as e:
        print(f"[{label}] ledger FAILED: {type(e).__name__}: {e}")


def _extract_press(ticker: str, label: str, path: Path, model: str, backend: str, output_dir: Path) -> None:
    print(f"[{label}] extracting press-release figures...")
    try:
        out = extract_press_ledger(path, f"{ticker}_{label}", model=model, backend=backend, output_dir=output_dir)
        print(f"[{label}] press ledger -> {out}")
    except Exception as e:
        print(f"[{label}] press ledger FAILED: {type(e).__name__}: {e}")


def identity_terms(ticker: str) -> list[str]:
    from anonymize_transcripts import get_config
    cfg = get_config(ticker)
    return [*cfg["company"], *(a for aliases in cfg["programs"].values() for a in aliases)]


def _analyze(
    ticker: str, label: str, path: Path, prior_label: str | None, allow_chain_break: bool,
    model: str, backend: str, output_dir: Path, progress_log: Path, lock: threading.Lock,
    terms: list[str] | None,
) -> None:
    prior_ledger = output_dir / f"{ticker}_{prior_label}" / LEDGER_FILE if prior_label else None
    gap = quarters_between(prior_label, label) if prior_label else 0
    chain = make_chain(prior_label, prior_ledger, gap)
    if chain["status"] == "chain_break" and not allow_chain_break:
        print(f"[{label}] BLOCKED: preceding call {prior_label} has no ledger yet (use --allow-chain-break to analyze without it)")
        return
    print(f"[{label}] analyzing ({chain['status']}{f', prior {prior_label}' if prior_label else ''})...")
    try:
        out_dir = analyze_call(
            path, f"{ticker}_{label}", chain, prior_ledger,
            model=model, backend=backend, output_dir=output_dir, identity_terms=terms,
        )
        _log_progress(progress_log, lock, label)
        provenance = json.loads((out_dir / PROVENANCE_FILE).read_text())
        leaks, novel = provenance.get("identity_terms_found"), provenance.get("novel_proper_nouns")
        print(f"[{label}] -> {out_dir}"
              f"{f'  WARNING identity terms not in transcript: {leaks}' if leaks else ''}"
              f"{f'  novel proper nouns: {novel}' if novel else ''}")
    except Exception as e:
        print(f"[{label}] FAILED: {type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=["nemotron", "claude"], default=DEFAULT_BACKEND)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--transcript-dir", type=Path, default=TRANSCRIPT_DIR)
    parser.add_argument("--only-quarter", nargs="+", default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--press-dir", type=Path, default=PRESS_DIR)
    parser.add_argument("--press-only", action="store_true",
                        help="only extract figures from the earnings press releases in --press-dir, then stop")
    parser.add_argument("--ledger-backend", choices=["nemotron", "claude"], default="nemotron")
    parser.add_argument("--ledger-model", default=None)
    parser.add_argument("--identity-check", action="store_true",
                        help="record any of the ticker's anonymized names that appear in the outputs (for anonymized runs)")
    parser.add_argument("--allow-chain-break", action="store_true",
                        help="analyze a call even if the preceding call's ledger could not be produced")
    args = parser.parse_args()
    ticker = args.ticker

    transcripts = find_transcripts(ticker, args.transcript_dir)
    labels = sorted(transcripts, key=quarter_sort_key)
    prior = predecessors(labels)
    targets = labels
    if args.only_quarter:
        targets = [l for l in labels if l in args.only_quarter]
        if not targets:
            raise SystemExit(f"none of {args.only_quarter} found for {ticker} in {args.transcript_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_log = args.output_dir / f"_progress_{ticker}.txt"
    states = quarter_states(ticker, args.transcript_dir, args.output_dir)
    pending = [l for l in targets if states[l] != "complete"]
    stale = [l for l in pending if states[l] == "stale"]
    print(f"{len(targets)} quarters for {ticker}, {len(targets) - len(pending)} complete, "
          f"{len(stale)} stale{' (' + ', '.join(stale) + ')' if stale else ''} -> {args.output_dir}")

    ledger_model = args.ledger_model or (DEFAULT_MODEL if args.ledger_backend == "nemotron" else CLAUDE_LEDGER_MODEL)
    max_workers = args.concurrency or (NIM_MAX_CONCURRENCY if args.backend == "nemotron" else 3)
    ledger_workers = args.concurrency or (NIM_MAX_CONCURRENCY if args.ledger_backend == "nemotron" else 3)

    needs_ledger = sorted(set(pending) | {prior[l] for l in pending if prior[l]}, key=quarter_sort_key)
    needs_ledger = [l for l in needs_ledger if not ledger_is_current(args.output_dir / f"{ticker}_{l}", transcripts[l])]
    if not args.press_only:
        with ThreadPoolExecutor(max_workers=ledger_workers) as executor:
            for future in [
                executor.submit(_extract, ticker, l, transcripts[l], ledger_model, args.ledger_backend, args.output_dir)
                for l in needs_ledger
            ]:
                future.result()

    press_targets = sorted(set(targets) | {prior[l] for l in targets if prior[l]}, key=quarter_sort_key)
    press_files = {l: args.press_dir / f"{ticker}_{l}.txt" for l in press_targets}
    needs_press = [
        l for l, p in press_files.items()
        if p.exists() and not press_ledger_is_current(args.output_dir / f"{ticker}_{l}", p)
    ]
    if needs_press:
        print(f"{len(needs_press)} quarters need press-release figures extracted")
    with ThreadPoolExecutor(max_workers=ledger_workers) as executor:
        for future in [
            executor.submit(_extract_press, ticker, l, press_files[l], ledger_model, args.ledger_backend, args.output_dir)
            for l in needs_press
        ]:
            future.result()
    if args.press_only:
        return

    states = quarter_states(ticker, args.transcript_dir, args.output_dir)
    ready = []
    for l in targets:
        if states[l] == "complete":
            continue
        if ledger_is_current(args.output_dir / f"{ticker}_{l}", transcripts[l]):
            ready.append(l)
        else:
            print(f"[{l}] skipped: its own ledger is missing")

    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for future in [
            executor.submit(
                _analyze, ticker, l, transcripts[l], prior[l], args.allow_chain_break,
                args.model, args.backend, args.output_dir, progress_log, lock,
                identity_terms(ticker) if args.identity_check else None,
            )
            for l in ready
        ]:
            future.result()


if __name__ == "__main__":
    main()
