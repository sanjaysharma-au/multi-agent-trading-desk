import numpy as np
import pandas as pd
import pytest

from moonshot.multibagger import find_best_run, parse_period


def make_df(prices, volume=1_000_000):
    idx = pd.bdate_range("2015-01-01", periods=len(prices))
    prices = np.asarray(prices, dtype=float)
    return pd.DataFrame({"Close": prices, "Adj Close": prices, "Volume": volume}, index=idx)


def ramp(start, end, n):
    return list(np.geomspace(start, end, n))


def test_parse_period():
    assert parse_period("3y") == 756
    assert parse_period("18m") == 378
    assert parse_period("6w") == 30
    assert parse_period("100d") == 100
    with pytest.raises(ValueError):
        parse_period("3x")


def test_detects_clean_5x_run():
    prices = [10] * 50 + ramp(10, 60, 200) + [60] * 50
    run = find_best_run(make_df(prices), multiple=5, window_days=300)
    assert run is not None
    assert run["multiple"] == pytest.approx(6.0, rel=0.02)
    assert run["start_price"] == pytest.approx(10)
    assert run["threshold_crossed_date"] is not None
    assert run["current_multiple"] == pytest.approx(6.0, rel=0.02)


def test_below_threshold_not_detected():
    prices = [10] * 50 + ramp(10, 49, 200) + [49] * 50
    assert find_best_run(make_df(prices), multiple=5, window_days=300) is None


def test_single_day_spike_is_smoothed_out():
    # An 8x spike stays under the jump-segmentation threshold, so this isolates smoothing.
    prices = [10.0] * 300
    prices[150] = 80.0
    assert find_best_run(make_df(prices), multiple=5, window_days=300) is None
    # Without smoothing the spike would count.
    assert find_best_run(make_df(prices), multiple=5, window_days=300, smooth_days=1) is not None


def test_start_below_min_price_ignored():
    prices = [0.5] * 50 + ramp(0.5, 1.8, 200)
    assert find_best_run(make_df(prices), multiple=3, window_days=300, min_price=1.0) is None


def test_illiquid_start_ignored():
    prices = [10] * 50 + ramp(10, 60, 200)
    assert find_best_run(make_df(prices, volume=100), multiple=5, window_days=300) is None


def test_rise_slower_than_window_not_detected():
    prices = ramp(10, 60, 1000)
    assert find_best_run(make_df(prices), multiple=5, window_days=100) is None


def test_short_series():
    assert find_best_run(make_df([10]), multiple=5, window_days=300) is None
    run = find_best_run(make_df([10] * 10 + [60] * 10), multiple=5, window_days=300)
    assert run is not None


def test_bankruptcy_splice_does_not_span_the_jump():
    # Old, wiped-out stock at $0.12 (a 258x overnight jump), then a new issue at $31.
    prices = [0.12] * 60 + [31.0] * 300
    df = make_df(prices)
    # Without the segmentation guard, one contiguous series makes it look like a fake 258x run.
    fake = find_best_run(df, multiple=5, window_days=300, max_jump=1e9)
    assert fake is not None and fake["multiple"] > 200
    # With it, the pre- and post-jump prices are different segments; flat at $31 means no real run.
    assert find_best_run(df, multiple=5, window_days=300) is None


def test_splice_followed_by_real_run_starts_after_the_jump():
    prices = [0.12] * 60 + ramp(31, 250, 250)
    run = find_best_run(make_df(prices), multiple=5, window_days=300)
    assert run is not None
    assert run["start_price"] > 20          # anchored on the new stock, not the $0.12 remnants
    assert run["multiple"] < 12


def test_run_cannot_cross_a_mid_run_jump():
    # A real run to 5x, but a bankruptcy-style splice happens before the peak: the segment
    # containing the splice cannot claim credit for the post-jump price.
    prices = [10] * 20 + ramp(10, 60, 200) + [0.05] * 40 + [900.0] * 100
    run = find_best_run(make_df(prices), multiple=5, window_days=400)
    assert run is not None
    assert run["peak_price"] < 100          # peak stays within the pre-jump segment (~60), not 900
