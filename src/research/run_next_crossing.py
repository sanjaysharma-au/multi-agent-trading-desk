import argparse
import fcntl
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE = Path(__file__).resolve().with_name("run_moonshot_pipeline.py")
SCREENER_BIN = REPO_ROOT / "tools" / "stock-screener" / ".venv" / "bin" / "moonshot"
LOG_DIR = REPO_ROOT / "logs" / "crossing_queue"
STATE_PATH = LOG_DIR / "state.json"
PRIORITY_PATH = LOG_DIR / "priority.txt"
SCREENER_DATA = REPO_ROOT / "tools" / "stock-screener" / "data"
NO_TRANSCRIPTS = "no transcripts on disk"
LOCK_PATH = LOG_DIR / "queue.lock"

# Infrastructure noise worth retrying on a cooldown rather than burning a real attempt: a transient
# API hiccup (NVIDIA/Nemotron timeout, a dropped connection, Claude's usage limit) looks identical in
# the log to a real per-ticker problem, so it is told apart only by matching known error text.
TRANSIENT_PATTERNS = ("Read timed out", "ConnectionError", "ChunkedEncodingError",
                      "session limit", "usage limit", "hit your limit", "rate limit", "Service Unavailable", "temporarily unavailable")
MAX_TRANSIENT_RETRIES = 6
TRANSIENT_BACKOFF_MINUTES = 5

PIPELINE_ARGS = [
    "--backend", "nemotron", "--ledger-backend", "nemotron",
    "--score-backend", "claude", "--score-model", "haiku",
]


QUEUES = ["inflections", "depressed"]
META_KEY = "_meta"


def queue_tickers(page: str = "inflections") -> list[str]:
    result = subprocess.run([str(SCREENER_BIN), "queue", "--page", page], capture_output=True, text=True, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"moonshot queue --page {page} failed: {result.stderr.strip()}", flush=True)
        return []
    return result.stdout.split()


def is_complete(ticker: str) -> bool:
    result = subprocess.run([sys.executable, str(PIPELINE), ticker, "--status"],
                            capture_output=True, text=True, cwd=REPO_ROOT)
    return result.returncode == 0 and "INCOMPLETE" not in result.stdout


def load_state() -> dict:
    return json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=1, sort_keys=True))


def on_cooldown(entry: dict) -> bool:
    retry_after = entry.get("retry_after")
    return bool(retry_after and datetime.fromisoformat(retry_after) > datetime.now())


def pending(tickers: list[str], state: dict, max_attempts: int) -> list[str]:
    return [t for t in tickers
            if not state.get(t, {}).get("complete") and state.get(t, {}).get("attempts", 0) < max_attempts
            and not on_cooldown(state.get(t, {}))]


def classify_transient(log_tail: str) -> str | None:
    return next((p for p in TRANSIENT_PATTERNS if p in log_tail), None)


def priority_tickers() -> list[str]:
    return PRIORITY_PATH.read_text().replace(",", " ").upper().split() if PRIORITY_PATH.exists() else []


def next_ticker(queues: dict[str, list[str]], state: dict, turn: str, max_attempts: int) -> tuple[str, str] | None:
    """Priority tickers first, then the first unfinished ticker from the queue whose turn it is, else
    from the other queue. A ticker found complete on disk is marked and skipped."""
    order = [turn] + [q for q in QUEUES if q != turn]
    for page, tickers in [("priority", priority_tickers())] + [(q, queues[q]) for q in order]:
        for ticker in pending(tickers, state, max_attempts):
            if is_complete(ticker):
                state.setdefault(ticker, {"attempts": 0})["complete"] = True
                save_state(state)
                continue
            return page, ticker
    return None


def run_without_transcripts(ticker: str) -> bool:
    """For a company with no earnings-call transcripts: skip the transcript steps but still build its
    price-to-sales history and its web research, which do not need them."""
    for command in (["valuation", ticker], ["why", ticker]):
        subprocess.run([str(SCREENER_BIN), *command], cwd=REPO_ROOT, capture_output=True)
    research = SCREENER_DATA / "research" / f"{ticker}.json"
    return (SCREENER_DATA / "valuation" / f"{ticker}.json").exists() and research.exists() \
        and "skipped" not in json.loads(research.read_text())


def run_ticker(ticker: str, state: dict, timeout_hours: float) -> float:
    """Runs the pipeline for one ticker and returns how many seconds it took."""
    started = time.monotonic()
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
    tail = log_path.read_text()[-3000:] if not entry["complete"] else ""
    if not entry["complete"] and rc > 0 and NO_TRANSCRIPTS in tail:
        entry["note"] = "no transcripts available; valuation and research only"
        entry["complete"] = run_without_transcripts(ticker)
    elif not entry["complete"]:
        reason = classify_transient(tail)
        transient_failures = entry.get("transient_failures", 0)
        if reason and transient_failures < MAX_TRANSIENT_RETRIES:
            entry["attempts"] -= 1
            entry["transient_failures"] = transient_failures + 1
            backoff = TRANSIENT_BACKOFF_MINUTES * (2 ** transient_failures)
            entry["retry_after"] = (datetime.now() + timedelta(minutes=backoff)).isoformat(timespec="seconds")
            entry["note"] = f"transient failure ({reason!r}); retrying in {backoff} min, not counted as an attempt"
        elif reason:
            entry["note"] = f"gave up after {MAX_TRANSIENT_RETRIES} transient failures ({reason!r})"
    if entry["complete"]:
        entry.pop("transient_failures", None)
        entry.pop("retry_after", None)
    save_state(state)
    elapsed = time.monotonic() - started
    print(f"{ticker}: {'complete' if entry['complete'] else f'incomplete (exit {rc}, attempt {entry['attempts']})'} "
          f"in {elapsed / 60:.1f} min", flush=True)
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the moonshot pipeline for the next unfinished crossing, most recent crossing first. "
        "Meant to be run by a timer; keeps going until the queue is empty or a transient failure puts a ticker on cooldown."
    )
    parser.add_argument("--max-attempts", type=int, default=2,
                        help="give up on a ticker after this many failed runs (default 2)")
    parser.add_argument("--timeout-hours", type=float, default=6.0, help="kill a single ticker's run after this long")
    parser.add_argument("--ticker", help="run this ticker instead of the next one in the queue")
    parser.add_argument("--prioritize", help="comma-separated tickers to run before everything else, in this order "
                        "(replaces the current priority list; empty string clears it)")
    parser.add_argument("--list", action="store_true", help="print the pending queue and exit")
    parser.add_argument("--retry-failed", action="store_true", help="clear the attempt counters first")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    if args.retry_failed:
        for key, entry in state.items():
            if key != META_KEY:
                entry["attempts"] = 0
        save_state(state)

    if args.prioritize is not None:
        PRIORITY_PATH.write_text(" ".join(t.strip().upper() for t in args.prioritize.split(",") if t.strip()) + "\n")
        print(f"priority list: {priority_tickers() or 'cleared'}")
        return

    queues = {page: queue_tickers(page) for page in QUEUES}
    if args.list:
        todo = pending(priority_tickers(), state, args.max_attempts)
        print(f"priority: {len(todo)} pending" + (f", next: {', '.join(todo[:5])}" if todo else ""))
        for page in QUEUES:
            todo = pending(queues[page], state, args.max_attempts)
            print(f"{page}: {len(todo)} pending" + (f", next: {', '.join(todo[:5])}" if todo else ""))
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
        while True:
            turn = state.get(META_KEY, {}).get("next_queue", QUEUES[0])
            pick = next_ticker(queues, state, turn, args.max_attempts)
            if pick is None:
                print("queues empty: every ticker is complete or out of attempts")
                return
            page, ticker = pick
            if page != "priority":
                state[META_KEY] = {"next_queue": QUEUES[(QUEUES.index(page) + 1) % len(QUEUES)]}
            print(f"[{page}] ", end="", flush=True)
            run_ticker(ticker, state, args.timeout_hours)
            if not state[ticker]["complete"] and on_cooldown(state[ticker]):
                return


if __name__ == "__main__":
    main()
