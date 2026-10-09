import argparse
import fcntl
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE = Path(__file__).resolve().with_name("run_moonshot_pipeline.py")
SCREENER_BIN = REPO_ROOT / "tools" / "stock-screener" / ".venv" / "bin" / "moonshot"
LOG_DIR = REPO_ROOT / "logs" / "crossing_queue"
STATE_PATH = LOG_DIR / "state.json"
LOCK_PATH = LOG_DIR / "queue.lock"

PIPELINE_ARGS = [
    "--backend", "nemotron", "--ledger-backend", "nemotron",
    "--score-backend", "claude", "--score-model", "haiku",
]


def queue_tickers() -> list[str]:
    result = subprocess.run([str(SCREENER_BIN), "queue"], capture_output=True, text=True, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise SystemExit(f"moonshot queue failed: {result.stderr.strip()}")
    return result.stdout.split()


def is_complete(ticker: str) -> bool:
    result = subprocess.run([sys.executable, str(PIPELINE), ticker, "--status"],
                            capture_output=True, text=True, cwd=REPO_ROOT)
    return result.returncode == 0 and "INCOMPLETE" not in result.stdout


def load_state() -> dict:
    return json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=1, sort_keys=True))


def pending(tickers: list[str], state: dict, max_attempts: int) -> list[str]:
    return [t for t in tickers
            if not state.get(t, {}).get("complete") and state.get(t, {}).get("attempts", 0) < max_attempts]


def run_ticker(ticker: str, state: dict, timeout_hours: float) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{ticker}_{datetime.now():%Y%m%d_%H%M%S}.log"
    entry = state.setdefault(ticker, {"attempts": 0})
    entry["attempts"] += 1
    entry["last_started"] = datetime.now().isoformat(timespec="seconds")
    entry["last_log"] = str(log_path)
    save_state(state)
    cmd = [sys.executable, str(PIPELINE), ticker, *PIPELINE_ARGS]
    print(f"running {ticker}: {' '.join(cmd)}\nlog: {log_path}", flush=True)
    with log_path.open("w") as log:
        try:
            rc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                timeout=timeout_hours * 3600).returncode
        except subprocess.TimeoutExpired:
            rc = -1
    entry["last_exit"] = rc
    entry["complete"] = rc == 0 and is_complete(ticker)
    save_state(state)
    print(f"{ticker}: {'complete' if entry['complete'] else f'incomplete (exit {rc}, attempt {entry['attempts']})'}",
          flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the moonshot pipeline for the next unfinished crossing, most recent crossing first. "
        "Meant to be run by a timer, one ticker per run."
    )
    parser.add_argument("--max-attempts", type=int, default=2,
                        help="give up on a ticker after this many failed runs (default 2)")
    parser.add_argument("--timeout-hours", type=float, default=6.0, help="kill a single ticker's run after this long")
    parser.add_argument("--ticker", help="run this ticker instead of the next one in the queue")
    parser.add_argument("--list", action="store_true", help="print the pending queue and exit")
    parser.add_argument("--retry-failed", action="store_true", help="clear the attempt counters first")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    if args.retry_failed:
        for entry in state.values():
            entry["attempts"] = 0
        save_state(state)

    if args.list:
        todo = pending(queue_tickers(), state, args.max_attempts)
        print(f"{len(todo)} pending:\n" + "\n".join(todo))
        return

    with LOCK_PATH.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another run is still in progress; exiting")
            return
        if args.ticker:
            run_ticker(args.ticker.upper(), state, args.timeout_hours)
            return
        for ticker in pending(queue_tickers(), state, args.max_attempts):
            if is_complete(ticker):
                state.setdefault(ticker, {"attempts": 0})["complete"] = True
                save_state(state)
                continue
            run_ticker(ticker, state, args.timeout_hours)
            return
        print("queue empty: every crossing is complete or out of attempts")


if __name__ == "__main__":
    main()
