import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.earnings_stance import assess_stance

ANALYSIS_DIR = Path("earnings_call_analysis")
SCORES_DIR = Path("data/earnings_call_scores")
OUTPUT_DIR = Path(SCORES_DIR)


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
    parser.add_argument("--only-quarter", default=None, help="Only process this quarter (e.g., 2019Q3)")
    args = parser.parse_args()
    ticker = args.ticker

    quarter_dirs = find_quarter_dirs(ticker)
    labels = sorted(quarter_dirs.keys(), key=quarter_sort_key)
    print(f"{len(labels)} quarters found for {ticker}")

    out_path = OUTPUT_DIR / f"{ticker}_stance.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        import json
        for line in out_path.read_text().splitlines():
            done.add(json.loads(line)["quarter"])

    returns_path = SCORES_DIR / f"{ticker}_returns.csv"
    if not returns_path.exists():
        print(f"Error: returns file not found at {returns_path}")
        return

    import pandas as pd
    returns_df = pd.read_csv(returns_path)
    returns_by_quarter = dict(zip(returns_df["quarter"], returns_df["pre_close"]))

    for label in labels:
        if args.only_quarter and label != args.only_quarter:
            continue

        if label in done:
            print(f"[{label}] already done, skipping")
            continue

        synthesis_path = quarter_dirs[label] / "synthesis.md"
        if not synthesis_path.exists():
            print(f"[{label}] no synthesis.md, skipping")
            continue

        if label not in returns_by_quarter:
            print(f"[{label}] no price data, skipping")
            continue

        current_price = returns_by_quarter[label]

        print(f"[{label}] analyzing...")
        try:
            response, scores = assess_stance(synthesis_path.read_text(), current_price, model=args.model)
            scores["quarter"] = label
            with out_path.open("a") as out:
                import json
                out.write(json.dumps(scores) + "\n")
                out.flush()
            print(f"[{label}] stance={scores.get('stance')} bull={scores.get('bull_pct')} base={scores.get('base_pct')} bear={scores.get('bear_pct')}")
        except Exception as e:
            print(f"[{label}] FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
