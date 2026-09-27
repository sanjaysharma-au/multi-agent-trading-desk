import pandas as pd

from moonshot.fundamentals import _quarterize_cumulative
from moonshot.inflection import GAAP_PROFIT_AND_FCF, classify_stages, find_crossing, merge_financials


def fact(start, end, val, filed="2024-01-01"):
    return {"start": start, "end": end, "val": val, "filed": filed}


def facts_for(concept, rows, unit="USD"):
    return {"facts": {"us-gaap": {concept: {"units": {unit: rows}}}}}


def q(period_end):
    return pd.Timestamp(period_end)


def make_financials(revenues, net_incomes, fcfs, start="2020-03-31"):
    idx = pd.date_range(start, periods=len(revenues), freq="QE")
    return pd.DataFrame({"period_end": idx, "revenue": revenues, "net_income": net_incomes,
                         "free_cash_flow": fcfs})


# ---------- cumulative (year-to-date) XBRL differencing ----------

def test_cumulative_ytd_facts_are_differenced_into_quarters():
    rows = [
        fact("2023-01-01", "2023-03-31", 100),   # Q1 = 100
        fact("2023-01-01", "2023-06-30", 250),   # H1 = 250 -> Q2 = 150
        fact("2023-01-01", "2023-09-30", 420),   # 9mo = 420 -> Q3 = 170
        fact("2023-01-01", "2023-12-31", 620),   # FY = 620 -> Q4 = 200
    ]
    out = _quarterize_cumulative(facts_for("NetCashProvidedByUsedInOperatingActivities", rows),
                                 ["NetCashProvidedByUsedInOperatingActivities"], "USD")
    got = dict(zip(out["period_end"], out["value"]))
    assert got[q("2023-03-31")] == 100
    assert got[q("2023-06-30")] == 150
    assert got[q("2023-09-30")] == 170
    assert got[q("2023-12-31")] == 200


def test_discrete_quarterly_facts_pass_through_unchanged():
    # Each fact has its own distinct start (a genuinely discrete quarter), not a shared FY start.
    rows = [
        fact("2023-01-01", "2023-03-31", 100),
        fact("2023-04-01", "2023-06-30", 150),
        fact("2023-07-01", "2023-09-30", 170),
    ]
    out = _quarterize_cumulative(facts_for("NetCashProvidedByUsedInOperatingActivities", rows),
                                 ["NetCashProvidedByUsedInOperatingActivities"], "USD")
    got = dict(zip(out["period_end"], out["value"]))
    assert got[q("2023-06-30")] == 150
    assert got[q("2023-09-30")] == 170


def test_earliest_filing_wins_for_restated_cumulative_checkpoint():
    rows = [
        fact("2023-01-01", "2023-03-31", 100, filed="2023-05-01"),
        fact("2023-01-01", "2023-06-30", 250, filed="2023-08-01"),
        fact("2023-01-01", "2023-06-30", 260, filed="2024-08-01"),  # later restatement, ignored
    ]
    out = _quarterize_cumulative(facts_for("NetCashProvidedByUsedInOperatingActivities", rows),
                                 ["NetCashProvidedByUsedInOperatingActivities"], "USD")
    got = dict(zip(out["period_end"], out["value"]))
    assert got[q("2023-06-30")] == 150  # 250 - 100, not 260 - 100


# ---------- stage classification & crossing detection ----------

def test_classify_stages_ladder():
    df = make_financials(
        revenues=[0, 10, 10, 10],
        net_incomes=[None, -5, 5, 5],
        fcfs=[None, None, -1, 1],
    )
    assert classify_stages(df).tolist() == [0, 1, 2, 3]


def test_palantir_shape_is_detected():
    # 8 quarters of revenue-stage losses, then a clean, sustained crossing.
    df = make_financials(
        revenues=[50] * 8 + [60] * 6,
        net_incomes=[-10] * 8 + [5] * 6,
        fcfs=[-5] * 8 + [3] * 6,
    )
    hit = find_crossing(df, min_pre_quarters=4, min_post_quarters=2)
    assert hit is not None
    assert hit["crossing_period_end"] == df.iloc[8]["period_end"]
    assert hit["quarters_since_crossing"] == 5
    assert hit["quarters_of_prior_history"] == 8


def test_always_profitable_company_is_not_a_crossing():
    # Every quarter of available history is already stage 3: no "before" to cross from.
    df = make_financials(revenues=[100] * 10, net_incomes=[20] * 10, fcfs=[15] * 10)
    assert find_crossing(df) is None


def test_one_off_profit_blip_that_reverts_is_not_a_crossing():
    df = make_financials(
        revenues=[50] * 8 + [50] + [50] * 4,
        net_incomes=[-10] * 8 + [5] + [-8] * 4,
        fcfs=[-5] * 8 + [3] + [-6] * 4,
    )
    assert find_crossing(df) is None


def test_recent_crossing_without_enough_track_record_is_not_yet_confirmed():
    df = make_financials(
        revenues=[50] * 8 + [60],
        net_incomes=[-10] * 8 + [5],
        fcfs=[-5] * 8 + [3],
    )
    assert find_crossing(df, min_pre_quarters=4, min_post_quarters=2) is None


def test_blip_then_relapse_then_real_crossing_finds_the_real_one():
    # A one-quarter blip at index 3 that reverts, then the genuine sustained crossing at index 8.
    df = make_financials(
        revenues=[50] * 10,
        net_incomes=[-10, -10, -10, 5, -10, -10, -10, 5, 5, 5],
        fcfs=[-5, -5, -5, 3, -5, -5, -5, 3, 3, 3],
    )
    hit = find_crossing(df, min_pre_quarters=4, min_post_quarters=1)
    assert hit is not None
    assert hit["crossing_period_end"] == df.iloc[7]["period_end"]


def test_single_wobble_quarter_is_tolerated():
    revenues = [50] * 8 + [60] * 6
    net_incomes = [-10] * 8 + [5] * 6
    # One quarter dips to stage 2 (profit but FCF negative) partway through the tail.
    fcfs = [-5] * 8 + [3, 3, -1, 3, 3, 3]
    df = make_financials(revenues, net_incomes, fcfs)
    hit = find_crossing(df, min_pre_quarters=4, min_post_quarters=2, max_wobble_quarters=1)
    assert hit is not None
    assert hit["wobble_quarters"] == 1


def test_wobble_right_up_to_the_present_is_not_a_confirmed_crossing():
    # The two most recent quarters dip to stage 2: not "still there today", and too soon after the
    # dip to find a later clean run either, so this must not be reported as a crossing.
    revenues = [50] * 8 + [60] * 6
    net_incomes = [-10] * 8 + [5] * 6
    fcfs = [-5] * 8 + [3, 3, 3, 3, -1, -1]
    df = make_financials(revenues, net_incomes, fcfs)
    assert find_crossing(df, min_pre_quarters=4, min_post_quarters=2, max_wobble_quarters=1) is None


def test_merge_financials_without_cashflow_data_caps_at_gaap_profit():
    fnd = pd.DataFrame({"period_end": pd.date_range("2020-03-31", periods=3, freq="QE"),
                        "revenue": [10, 10, 10], "net_income": [-1, 1, 1]})
    merged = merge_financials(fnd, None)
    stages = classify_stages(merged)
    assert stages.tolist()[-1] != GAAP_PROFIT_AND_FCF  # can't confirm FCF, so never reaches stage 3
