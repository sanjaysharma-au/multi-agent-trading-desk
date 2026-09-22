import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCORE_COLS = ["tone_confidence", "guidance_credibility", "competitive_strength",
              "financial_health", "conflict_level", "traj_score", "stance_num", "scenario_spread"]
RETURN_COLS = ["eps_surprise_pct", "fwd_ret_1d", "fwd_ret_5d", "fwd_ret_21d",
               "fwd_ret_63d", "fwd_ret_126d", "fwd_ret_252d"]


def load_panel(ticker: str) -> pd.DataFrame:
    scores = pd.DataFrame([json.loads(l) for l in open(f"data/earnings_call_scores/{ticker}_scores.jsonl")])
    returns = pd.read_csv(f"data/earnings_call_scores/{ticker}_returns.csv")
    panel = returns.merge(scores, on="quarter", how="inner")

    verdicts_path = Path(f"earnings_call_trajectory/{ticker}/_verdicts.jsonl")
    if verdicts_path.exists():
        verdicts_data = [json.loads(l) for l in open(verdicts_path)]
        verdicts_df = pd.DataFrame(verdicts_data)
        verdicts_df['quarter'] = verdicts_df['pair'].str.split('-').str[1]
        verdicts_df = verdicts_df[['quarter', 'score']].rename(columns={'score': 'traj_score'})
        panel = panel.merge(verdicts_df, on="quarter", how="left")
    else:
        print(f"Note: trajectory verdicts file not found at {verdicts_path}, skipping traj_score merge")
        panel['traj_score'] = np.nan

    stance_path = Path(f"data/earnings_call_scores/{ticker}_stance.jsonl")
    if stance_path.exists():
        stance_data = [json.loads(l) for l in open(stance_path)]
        stance_df = pd.DataFrame(stance_data)
        stance_mapping = {"bullish": 1, "neutral": 0, "bearish": -1}
        stance_df['stance_num'] = stance_df['stance'].map(stance_mapping)
        stance_df['scenario_spread'] = stance_df['bull_pct'] - stance_df['bear_pct']
        stance_df = stance_df[['quarter', 'stance_num', 'scenario_spread']]
        panel = panel.merge(stance_df, on="quarter", how="left")
    else:
        print(f"Note: stance file not found at {stance_path}, skipping stance merge")
        panel['stance_num'] = np.nan
        panel['scenario_spread'] = np.nan

    return panel


def robustness_report(panel: pd.DataFrame):
    print("\n=== Robustness report: outlier sensitivity ===")
    print(f"{'Score':<22} {'Return':<14} {'Pearson r':>10} {'p':>8} {'Spearman r':>10} {'p':>8} {'Drop-top-4 r':>12} {'Flag':>15}")
    print("-" * 120)

    for score_col in SCORE_COLS:
        for ret_col in ["fwd_ret_63d", "fwd_ret_126d", "fwd_ret_252d"]:
            sub = panel[[score_col, ret_col]].dropna()
            if len(sub) < 10:
                continue

            pearson_r, pearson_p = stats.pearsonr(sub[score_col], sub[ret_col])
            spearman_r, spearman_p = stats.spearmanr(sub[score_col], sub[ret_col])

            sub_dropped = sub.nlargest(4, ret_col)
            sub_without_top4 = sub.drop(sub_dropped.index)
            if len(sub_without_top4) >= 2:
                drop4_r, _ = stats.pearsonr(sub_without_top4[score_col], sub_without_top4[ret_col])
            else:
                drop4_r = np.nan

            # A near-zero correlation has no signal to be outlier-driven; flagging it
            # would drown the cases where a real-looking result rests on a few points.
            if pearson_p >= 0.10 and abs(pearson_r) < 0.25:
                flag = "no signal"
            elif pearson_p < 0.05 and spearman_p >= 0.05:
                flag = "OUTLIER-DRIVEN"
            elif not np.isnan(drop4_r) and (
                np.sign(drop4_r) != np.sign(pearson_r) or abs(drop4_r) < 0.5 * abs(pearson_r)
            ):
                flag = "OUTLIER-DRIVEN"
            else:
                flag = ""

            print(f"{score_col:<22} {ret_col:<14} {pearson_r:>+10.2f} {pearson_p:>8.3f} {spearman_r:>+10.2f} {spearman_p:>8.3f} {drop4_r:>+12.2f} {flag:>15}")


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

    robustness_report(panel)

    print("\n=== Does the qualitative score add anything beyond EPS surprise alone? ===")
    for ret_col in ["fwd_ret_21d", "fwd_ret_63d", "fwd_ret_252d"]:
        eps_sub = panel[["eps_surprise_pct", ret_col]].dropna()
        if len(eps_sub) < 10:
            continue
        r_eps, _ = stats.pearsonr(eps_sub["eps_surprise_pct"], eps_sub[ret_col])
        best_score, best_r, best_n = None, 0, 0
        for s in SCORE_COLS:
            sub = panel[[s, ret_col]].dropna()
            if len(sub) < 10:
                continue
            r, _ = stats.pearsonr(sub[s], sub[ret_col])
            if abs(r) > abs(best_r):
                best_score, best_r, best_n = s, r, len(sub)
        if best_score is None:
            continue
        print(f"  {ret_col}: EPS-surprise-alone r={r_eps:+.2f} (n={len(eps_sub)}) | "
              f"best qualitative score ({best_score}) r={best_r:+.2f} (n={best_n})")

    out_path = f"data/earnings_call_scores/{ticker}_panel.csv"
    panel.to_csv(out_path, index=False)
    print(f"\nfull panel saved to {out_path}")


if __name__ == "__main__":
    main()
