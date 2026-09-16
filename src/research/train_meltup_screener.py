"""Train and honestly evaluate a melt-up screener on the unbiased panel."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

PANEL = "/tmp/meltup_panel.csv"
TARGET_MULTIPLE = 2.0
CUTOFF = pd.Timestamp("2023-01-01")
EMBARGO_DAYS = 400  # calendar days; keeps train label windows from touching the test period

FEATURES = [
    "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "drawdown_from_252d_high", "pct_above_252d_low", "drawdown_from_3y_high",
    "days_since_252d_high",
    "volatility_21d", "volatility_252d", "vol_ratio_21_252", "range_ratio_21_252",
    "volume_ratio_21_252", "volume_ratio_63_252", "log_dollar_volume", "log_price",
    "up_day_ratio_63d", "max_1d_gain_63d", "rsi_14", "listing_age_days",
    "spy_ret_126d", "spy_vol_63d",
]


def main():
    df = pd.read_csv(PANEL, parse_dates=["date"])
    df = df.dropna(subset=["fwd_max_multiple"]).copy()
    df["label"] = (df["fwd_max_multiple"] >= TARGET_MULTIPLE).astype(int)

    train = df[df["date"] < CUTOFF - pd.Timedelta(days=EMBARGO_DAYS)]
    test = df[df["date"] >= CUTOFF]
    print(f"train: {len(train)} rows ({train['label'].mean():.2%} positive)")
    print(f"test:  {len(test)} rows ({test['label'].mean():.2%} positive)")
    print(f"(embargo gap of {EMBARGO_DAYS} days keeps train label windows out of the test period)\n")

    model = HistGradientBoostingClassifier(
        max_depth=4, max_iter=300, learning_rate=0.05,
        min_samples_leaf=50, l2_regularization=1.0, random_state=42,
    )
    model.fit(train[FEATURES], train["label"])
    probs = model.predict_proba(test[FEATURES])[:, 1]

    auc = roc_auc_score(test["label"], probs)
    base = test["label"].mean()
    print(f"Out-of-time AUC: {auc:.3f}")
    print(f"Base rate of reaching {TARGET_MULTIPLE}x in the test period: {base:.2%}\n")

    res = test.copy()
    res["prob"] = probs

    print("=== Screener view: take the top-N% ranked names ===")
    print(f"{'top %':>7} {'n':>6} {'precision':>10} {'lift':>6} {'med fwd ret':>12} {'mean fwd ret':>13} {'hit 5x':>8} {'hit 10x':>8}")
    for pct in (0.1, 0.5, 1, 2, 5, 10):
        k = max(1, int(len(res) * pct / 100))
        top = res.nlargest(k, "prob")
        precision = top["label"].mean()
        lift = precision / base if base else np.nan
        print(f"{pct:>6}% {k:>6} {precision:>9.1%} {lift:>6.2f} "
              f"{top['fwd_252d_return'].median():>11.1%} {top['fwd_252d_return'].mean():>12.1%} "
              f"{(top['fwd_max_multiple'] >= 5).mean():>7.1%} {(top['fwd_max_multiple'] >= 10).mean():>7.1%}")

    print(f"\nFor reference, the whole test population: "
          f"median fwd ret {res['fwd_252d_return'].median():.1%}, "
          f"mean {res['fwd_252d_return'].mean():.1%}, "
          f"hit 5x {(res['fwd_max_multiple'] >= 5).mean():.1%}, "
          f"hit 10x {(res['fwd_max_multiple'] >= 10).mean():.1%}")

    # permutation importance on the test set (what actually drives the ranking)
    print("\n=== Permutation importance (AUC drop when a feature is shuffled) ===")
    rng = np.random.default_rng(42)
    importances = {}
    for feat in FEATURES:
        X = test[FEATURES].copy()
        X[feat] = rng.permutation(X[feat].values)
        importances[feat] = auc - roc_auc_score(test["label"], model.predict_proba(X)[:, 1])
    for feat, drop in sorted(importances.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {feat:<28} {drop:+.4f}")

    res.to_csv("/tmp/meltup_panel_scored.csv", index=False)


if __name__ == "__main__":
    main()
