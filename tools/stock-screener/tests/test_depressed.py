import numpy as np
import pandas as pd

from moonshot.depressed import fundamentals_snapshot, price_metrics


def prices(values, volume=1_000_000):
    idx = pd.bdate_range("2024-01-01", periods=len(values))
    s = pd.Series(values, index=idx, dtype=float)
    return pd.DataFrame({"Close": s, "Adj Close": s, "Volume": float(volume)})


def quarters(revenue, net_income, fcf, n=8):
    ends = pd.date_range(end="2026-06-30", periods=n, freq="QE")
    return pd.DataFrame({"period_end": ends, "revenue": revenue, "net_income": net_income, "free_cash_flow": fcf})


def test_drawdown_is_measured_from_the_52_week_high():
    values = list(np.linspace(50, 100, 300)) + list(np.linspace(100, 60, 100))
    m = price_metrics(prices(values))
    assert m["dd_52w_pct"] == 40.0
    assert m["price"] == 60.0


def test_short_price_history_is_rejected():
    assert price_metrics(prices([10.0] * 100)) is None


def test_three_year_high_can_sit_below_the_old_peak_only_when_it_is_older():
    values = [200.0] * 50 + list(np.linspace(100, 100, 400))
    m = price_metrics(prices(values))
    assert m["dd_52w_pct"] == 0.0
    assert m["dd_3y_pct"] == 50.0


def test_snapshot_sums_last_four_quarters_against_the_four_before():
    q = quarters([100] * 4 + [120] * 4, [5] * 8, [10] * 8)
    snap = fundamentals_snapshot(q, pd.Timestamp("2026-09-30"))
    assert snap["revenue_growth_pct"] == 20.0
    assert snap["ttm_revenue"] == 480.0
    assert snap["fcf_margin_pct"] == round(40 / 480 * 100, 1)


def test_snapshot_needs_recent_quarters_and_cash_flow():
    q = quarters([100] * 8, [5] * 8, [10] * 8)
    assert fundamentals_snapshot(q, pd.Timestamp("2027-06-30")) is None
    q.loc[7, "free_cash_flow"] = np.nan
    assert fundamentals_snapshot(q, pd.Timestamp("2026-09-30")) is None
    assert fundamentals_snapshot(quarters([100] * 4, [5] * 4, [10] * 4, n=4), pd.Timestamp("2026-09-30")) is None


def test_zero_price_history_is_rejected_instead_of_dividing_by_zero():
    assert price_metrics(prices([0.0] * 400)) is None
