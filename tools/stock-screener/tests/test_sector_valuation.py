import pandas as pd

from moonshot.report import rise_pct
from moonshot.report_inflections import _relative_pct
from moonshot.sector import DEFAULT_ETF, etf_for_sic
from moonshot.valuation import _diluted_shares


def test_longest_sic_prefix_wins():
    assert etf_for_sic("7372") == "IGV"
    assert etf_for_sic("3674") == "SOXX"
    assert etf_for_sic("3841") == "XLV"
    assert etf_for_sic("1040") == "GDX"
    assert etf_for_sic("1311") == "XLE"


def test_unknown_or_missing_sic_falls_back_to_market():
    assert etf_for_sic("9999") == DEFAULT_ETF
    assert etf_for_sic("") == DEFAULT_ETF
    assert etf_for_sic(None) == DEFAULT_ETF


def test_rise_starts_from_last_close_before_known_day():
    series = {"d": [10, 20, 30, 40], "p": [100.0, 50.0, 80.0, 150.0]}
    assert rise_pct(series, 30) == 200.0
    assert rise_pct(series, 5) is None
    assert rise_pct(None, 30) is None


def test_relative_rise_compares_growth_factors():
    assert _relative_pct(-50.0, 100.0) == -75.0
    assert _relative_pct(None, 10.0) is None
    assert _relative_pct(10.0, None) is None


def test_diluted_shares_use_annual_only_where_no_quarter_exists():
    def row(start, end, val, filed):
        return {"start": start, "end": end, "val": val, "filed": filed}

    rows = [
        row("2024-01-01", "2024-03-31", 100, "2024-05-01"),
        row("2024-04-01", "2024-06-30", 110, "2024-08-01"),
        row("2024-07-01", "2024-09-30", 120, "2024-11-01"),
        row("2024-01-01", "2024-12-31", 115, "2025-02-15"),
    ]
    facts = {"facts": {"us-gaap": {"WeightedAverageNumberOfDilutedSharesOutstanding": {"units": {"shares": rows}}}}}
    shares = _diluted_shares(facts)
    assert shares[pd.Timestamp("2024-03-31")] == 100
    assert shares[pd.Timestamp("2024-12-31")] == 115
