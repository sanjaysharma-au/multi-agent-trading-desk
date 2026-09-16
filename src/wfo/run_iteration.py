import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from wfo.schema import load_results

ITERATIONS_DIR = Path("iterations")
DATA_DIR = Path("data/minute_aggs")
DAILY_DATA_DIR = Path("data/daily_aggs")
NEWS_DIR = Path("data/news")
EARNINGS_DIR = Path("data/earnings")
SRC_DIR = Path(__file__).resolve().parents[1]
SUBPROCESS_TIMEOUT_SECONDS = 3600


def make_iteration_id(ticker: str) -> str:
    return f"{datetime.now().strftime('%Y%m%dT%H%M%S')}_{ticker}"


def run_iteration(
    script_path: Path,
    ticker: str,
    train_months: int,
    predict_months: int,
    gap_days: int,
    iteration_id: str | None = None,
    contract_version: str | None = None,
    holdout_months: int = 0,
    data_dir: Path = DATA_DIR,
    news_dir: Path | None = None,
    earnings_dir: Path | None = None,
) -> tuple[bool, str]:
    iteration_id = iteration_id or make_iteration_id(ticker)
    output_dir = ITERATIONS_DIR / iteration_id
    output_dir.mkdir(parents=True)
    shutil.copy(script_path, output_dir / "script.py")
    if contract_version is not None:
        (output_dir / "contract_version.txt").write_text(contract_version)

    cmd = [
        sys.executable,
        str(script_path),
        "--data-dir",
        str(data_dir),
        "--ticker",
        ticker,
        "--train-months",
        str(train_months),
        "--predict-months",
        str(predict_months),
        "--gap-days",
        str(gap_days),
        "--output-dir",
        str(output_dir),
        "--holdout-months",
        str(holdout_months),
    ]
    if news_dir is not None:
        cmd += ["--news-dir", str(news_dir)]
    if earnings_dir is not None:
        cmd += ["--earnings-dir", str(earnings_dir)]

    env = {**os.environ, "PYTHONPATH": str(SRC_DIR)}

    log_path = output_dir / "run.log"
    with log_path.open("w") as log_file:
        try:
            subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=SUBPROCESS_TIMEOUT_SECONDS,
                check=True,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return False, f"{iteration_id}: timed out after {SUBPROCESS_TIMEOUT_SECONDS}s"
        except subprocess.CalledProcessError as e:
            return False, f"{iteration_id}: model script exited {e.returncode}, see {log_path}"

    results_path = output_dir / "results.json"
    if not results_path.exists():
        return False, f"{iteration_id}: script did not write results.json"

    try:
        results = load_results(results_path)
    except ValueError as e:
        return False, f"{iteration_id}: invalid results.json ({e})"

    if not results.windows:
        return False, f"{iteration_id}: script produced 0 usable windows (too-tight lookback/thresholds for this ticker's data coverage?)"

    for w in results.windows:
        if not Path(w.weights_path).exists():
            return False, f"{iteration_id}: window {w.index} weights_path missing: {w.weights_path}"
        if not Path(w.ledger_path).exists():
            return False, f"{iteration_id}: window {w.index} ledger_path missing: {w.ledger_path}"

    return True, iteration_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("script_path", type=Path)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--train-months", type=int, required=True)
    parser.add_argument("--predict-months", type=int, required=True)
    parser.add_argument("--gap-days", type=int, default=1)
    parser.add_argument("--holdout-months", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    ok, message = run_iteration(
        args.script_path, args.ticker, args.train_months, args.predict_months, args.gap_days,
        holdout_months=args.holdout_months, data_dir=args.data_dir,
    )
    print(message)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
