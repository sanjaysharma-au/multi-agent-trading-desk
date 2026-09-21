import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

SCORE_COLS = ["tone_confidence", "guidance_credibility", "competitive_strength",
              "financial_health", "conflict_level"]
RETURN_COLS = ["eps_surprise_pct", "fwd_ret_1d", "fwd_ret_5d", "fwd_ret_21d",
               "fwd_ret_63d", "fwd_ret_126d", "fwd_ret_252d"]


def load_panel(ticker: str) -> pd.DataFrame:
    scores = pd.DataFrame([json.loads(l) for l in open(f"data/earnings_call_scores/{ticker}_scores.jsonl")])
    returns = pd.read_csv(f"data/earnings_call_scores/{ticker}_returns.csv")
    panel = returns.merge(scores, on="quarter", how="inner")
    return panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    args = parser.parse_args()
    ticker = args.ticker

    panel = load_panel(ticker)
    print(f"panel: {len(panel)} quarters, {panel['earnings_date'].min()} to {panel['earnings_date'].max()}\n")

    print("=== Pearson correlation: score vs outcome (n=%d) ===" % len(panel))
    header = f"{'':<22}" + "".join(f"{c:>14}" for c in RETURN_COLS)
    print(header)
    for score_col in SCORE_COLS:
        row = f"{score_col:<22}"
        for ret_col in RETURN_COLS:
            sub = panel[[score_col, ret_col]].dropna()
            if len(sub) < 5:
                row += f"{'n/a':>14}"
                continue
            r, p = stats.pearsonr(sub[score_col], sub[ret_col])
            marker = "*" if p < 0.05 else (" " if p >= 0.10 else "~")
            row += f"{r:>+9.2f}{marker:<5}"
        print(row)

    print("\n(* p<0.05, ~ p<0.10, blank otherwise -- n=%d is a small-sample pilot, treat accordingly)" % len(panel))

    print("\n=== Spearman (rank) correlation, same layout ===")
    print(header)
    for score_col in SCORE_COLS:
        row = f"{score_col:<22}"
        for ret_col in RETURN_COLS:
            sub = panel[[score_col, ret_col]].dropna()
            if len(sub) < 5:
                row += f"{'n/a':>14}"
                continue
            r, p = stats.spearmanr(sub[score_col], sub[ret_col])
            marker = "*" if p < 0.05 else (" " if p >= 0.10 else "~")
            row += f"{r:>+9.2f}{marker:<5}"
        print(row)

    print("\n=== Does the qualitative score add anything beyond EPS surprise alone? ===")
    for ret_col in ["fwd_ret_21d", "fwd_ret_63d", "fwd_ret_252d"]:
        sub = panel[["eps_surprise_pct"] + SCORE_COLS + [ret_col]].dropna()
        if len(sub) < 10:
            continue
        r_eps, _ = stats.pearsonr(sub["eps_surprise_pct"], sub[ret_col])
        best_score, best_r = None, 0
        for s in SCORE_COLS:
            r, _ = stats.pearsonr(sub[s], sub[ret_col])
            if abs(r) > abs(best_r):
                best_score, best_r = s, r
        print(f"  {ret_col}: EPS-surprise-alone r={r_eps:+.2f} | best qualitative score ({best_score}) r={best_r:+.2f}")

    out_path = f"data/earnings_call_scores/{ticker}_panel.csv"
    panel.to_csv(out_path, index=False)
    print(f"\nfull panel saved to {out_path}")


if __name__ == "__main__":
    main()
