import pandas as pd

from moonshot.fundamentals import _quarterly, _yoy


def fact(start, end, val, filed):
    return {"start": start, "end": end, "val": val, "filed": filed}


def facts(rows, concept="Revenues", unit="USD"):
    return {"facts": {"us-gaap": {concept: {"units": {unit: rows}}}}}


def test_q4_derived_from_annual_minus_three_quarters():
    rows = [
        fact("2024-01-01", "2024-03-31", 10, "2024-05-01"),
        fact("2024-04-01", "2024-06-30", 20, "2024-08-01"),
        fact("2024-07-01", "2024-09-30", 30, "2024-11-01"),
        fact("2024-01-01", "2024-12-31", 100, "2025-02-15"),
    ]
    q = _quarterly(facts(rows), ["Revenues"], "USD")
    assert q["value"].tolist() == [10, 20, 30, 40]
    assert q["period_end"].iloc[-1] == pd.Timestamp("2024-12-31")


def test_earliest_filing_wins_over_restatement():
    rows = [
        fact("2024-01-01", "2024-03-31", 10, "2024-05-01"),
        fact("2024-01-01", "2024-03-31", 12, "2025-05-01"),  # restated a year later
    ]
    q = _quarterly(facts(rows), ["Revenues"], "USD")
    assert q["value"].tolist() == [10]


def test_concepts_merge_by_priority():
    data = {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [fact("2024-01-01", "2024-03-31", 10, "2024-05-01")]}},
        "SalesRevenueNet": {"units": {"USD": [fact("2024-01-01", "2024-03-31", 99, "2024-05-01"),
                                              fact("2023-01-01", "2023-03-31", 5, "2023-05-01")]}},
    }}}
    q = _quarterly(data, ["Revenues", "SalesRevenueNet"], "USD")
    assert q["value"].tolist() == [5, 10]


def test_yoy():
    df = pd.DataFrame({"period_end": pd.to_datetime(["2023-03-31", "2023-06-30", "2024-03-31"]),
                       "revenue": [10.0, 12.0, 15.0]})
    assert _yoy(df, "revenue").tolist()[2] == 50.0
