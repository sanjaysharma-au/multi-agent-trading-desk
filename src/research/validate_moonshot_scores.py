import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
SCORES_DIR = REPO_ROOT / "data" / "earnings_call_scores"
PRESS_DIR = REPO_ROOT / "data" / "press_releases"
PRICES_DIR = REPO_ROOT / "tools" / "stock-screener" / "data" / "prices"
SECTOR_DIR = REPO_ROOT / "tools" / "stock-screener" / "data" / "sector"
SEC_DIR = REPO_ROOT / "tools" / "stock-screener" / "data" / "sec"
HORIZONS = [63, 126, 252]
SCORE_COLS = ["proximity", "momentum", "ambition", "runway", "evidence_hardness", "tier_num", "delta_proximity"]
FILED_RE = re.compile(r"filed (\d{4}-\d{2}-\d{2})")
SECTOR_ETF = {"IGV": "IGV", "SOXX": "SOXX", "XLK": "XLK", "XLV": "XLV", "XLF": "XLF", "XLE": "XLE", "XLI": "XLI",
              "XLC": "XLC", "XLY": "XLY", "XLP": "XLP", "XLB": "XLB", "XLU": "XLU", "XLRE": "XLRE", "GDX": "GDX"}


def load_closes(path: Path) -> pd.Series:
    return pd.read_parquet(path)["Adj Close"].dropna()


def announcement_date(ticker: str, quarter: str) -> pd.Timestamp | None:
    path = PRESS_DIR / f"{ticker}_{quarter}.txt"
    if not path.exists():
        return None
    match = FILED_RE.search(path.read_text()[:200])
    return pd.Timestamp(match.group(1)) if match else None


def forward_return(closes: pd.Series, announced: pd.Timestamp, horizon: int, start: str) -> float | None:
    """`pre` starts at the last close before the announcement, so it includes the earnings-day jump.
    `post` starts at the close of the trading day after the announcement date, when the news is priced
    in (covers both before-open and after-close releases): it measures only what could still be bought."""
    before = closes[closes.index < announced]
    after = closes[closes.index >= announced]
    if before.empty or after.empty:
        return None
    base, window = (before.iloc[-1], after) if start == "pre" else (after.iloc[1] if len(after) > 1 else None, after.iloc[1:])
    if base is None or len(window) < horizon:
        return None
    return float(window.iloc[horizon - 1] / base - 1)


def sector_closes(ticker: str) -> pd.Series | None:
    import sys
    sys.path.insert(0, str(REPO_ROOT / "tools" / "stock-screener" / "src"))
    from moonshot.fundamentals import cik_map
    from moonshot.sector import etf_for_sic, sic_for_ticker
    etf = etf_for_sic(sic_for_ticker(ticker, SEC_DIR, cik_map(SEC_DIR, 9999)))
    path = SECTOR_DIR / f"{etf}.parquet"
    return load_closes(path) if path.exists() else None


def tier_number(tier: str) -> int:
    match = re.match(r"T(\d)", tier or "")
    return int(match.group(1)) if match else np.nan


def build_panel(start: str) -> pd.DataFrame:
    rows = []
    for path in sorted(SCORES_DIR.glob("*_moonshot_v3.jsonl")):
        ticker = path.name.split("_")[0]
        price_paths = sorted(PRICES_DIR.glob(f"*/{ticker}.parquet"), key=lambda p: p.stat().st_mtime)
        if not price_paths:
            continue
        closes, sector = load_closes(price_paths[-1]), sector_closes(ticker)
        previous = None
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            announced = announcement_date(ticker, r["quarter"])
            if announced is None:
                continue
            row = {"ticker": ticker, "quarter": r["quarter"], "announced": announced,
                   "proximity": r["proximity"], "momentum": r["momentum"], "ambition": r["ambition"],
                   "runway": r["runway"], "evidence_hardness": r.get("evidence_hardness"),
                   "tier_num": tier_number(r.get("proximity_tier")),
                   "delta_proximity": r["proximity"] - previous if previous is not None else np.nan}
            previous = r["proximity"]
            for h in HORIZONS:
                raw = forward_return(closes, announced, h, start)
                bench = forward_return(sector, announced, h, start) if sector is not None else None
                row[f"ret_{h}d"] = raw
                row[f"excess_{h}d"] = raw - bench if raw is not None and bench is not None else None
            rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_ci(sub: pd.DataFrame, score: str, ret: str, draws: int = 500, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    tickers = sub["ticker"].unique()
    groups = {t: g for t, g in sub.groupby("ticker")}
    rhos = []
    for _ in range(draws):
        sample = pd.concat([groups[t] for t in rng.choice(tickers, len(tickers))])
        if sample[score].nunique() > 1:
            rhos.append(stats.spearmanr(sample[score], sample[ret])[0])
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def report(panel: pd.DataFrame, min_rows: int) -> None:
    print(f"panel: {len(panel)} scored quarters across {panel['ticker'].nunique()} tickers, "
          f"{panel['announced'].min():%Y-%m} to {panel['announced'].max():%Y-%m}\n")
    print(f"{'score':<18}{'return':<14}{'n':>5}{'spearman':>10}{'p':>8}{'95% CI (by ticker)':>22}{'drop top 4':>12}  flag")
    for ret_col in [f"{kind}_{h}d" for kind in ("excess", "ret") for h in HORIZONS]:
        for score in SCORE_COLS:
            sub = panel[["ticker", score, ret_col]].dropna()
            if len(sub) < min_rows:
                continue
            rho, p = stats.spearmanr(sub[score], sub[ret_col])
            lo, hi = bootstrap_ci(sub, score, ret_col)
            trimmed = sub.drop(sub.nlargest(4, ret_col).index)
            rho4 = stats.spearmanr(trimmed[score], trimmed[ret_col])[0]
            flag = "no signal" if p >= 0.10 else ("survives" if lo * hi > 0 and np.sign(rho4) == np.sign(rho) else "fragile")
            print(f"{score:<18}{ret_col:<14}{len(sub):>5}{rho:>+10.2f}{p:>8.3f}{f'[{lo:+.2f}, {hi:+.2f}]':>22}{rho4:>+12.2f}  {flag}")
        print()
    print("mean excess return vs sector by proximity tier at the time of the call:")
    grouped = panel.assign(tier=panel["tier_num"].map(lambda t: f"T{int(t)}" if pd.notna(t) else "?"))
    table = grouped.groupby("tier").agg(n=("ticker", "size"), **{f"{h}d": (f"excess_{h}d", "median") for h in HORIZONS})
    print((table.round(3)).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Do the moonshot pipeline's per-quarter scores predict forward returns? Pooled across every "
        "scored ticker, measured from the earnings-release date, with outlier and ticker-resampling checks."
    )
    parser.add_argument("--min-rows", type=int, default=40)
    parser.add_argument("--start", choices=["pre", "post"], default="post",
                        help="post (default): returns start the day after the announcement, so the earnings-day "
                        "reaction is excluded; pre: start before the announcement, including the reaction")
    args = parser.parse_args()
    panel = build_panel(args.start)
    print(f"returns start: {'the close before the announcement (includes the reaction)' if args.start == 'pre' else 'the close of the day after the announcement (reaction excluded)'}")
    if panel.empty:
        raise SystemExit("no scored quarters with a press-release date and cached prices")
    report(panel, args.min_rows)


if __name__ == "__main__":
    main()
