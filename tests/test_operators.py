import numpy as np
import pandas as pd
import pytest
from equitybt.strategy import operators as ops

X = pd.Series([1.0, 2.0, 4.0, 8.0, 7.0, 3.0])


def test_rank_spans_zero_to_one_and_averages_ties():
    assert ops.rank(pd.Series([1.0, 3.0, 2.0])).tolist() == [0.0, 1.0, 0.5]
    assert ops.rank(pd.Series([2.0, 2.0, 1.0])).tolist() == [0.75, 0.75, 0.0]
    assert ops.rank(pd.Series([5.0])).tolist() == [0.5]
    assert ops.rank(pd.Series([np.nan, 1.0, 3.0])).tolist()[1:] == [0.0, 1.0]


def test_ts_operators_against_hand_computed_windows():
    assert ops.ts_delay(X, 2).tolist()[2:] == [1.0, 2.0, 4.0, 8.0]
    assert ops.ts_delta(X, 1).tolist()[1:] == [1.0, 2.0, 4.0, -1.0, -4.0]
    assert ops.ts_mean(X, 3).iloc[2] == pytest.approx((1 + 2 + 4) / 3)
    assert ops.ts_sum(X, 3).iloc[3] == pytest.approx(14.0)
    assert ops.ts_std(X, 3).iloc[2] == pytest.approx(np.std([1, 2, 4], ddof=1))
    assert ops.ts_min(X, 3).iloc[4] == 4.0
    assert ops.ts_max(X, 3).iloc[4] == 8.0
    assert ops.product(X, 3).iloc[2] == pytest.approx(8.0)


def test_ts_zscore_is_the_deviation_in_trailing_standard_deviations():
    window = X.iloc[1:4]
    expected = (X.iloc[3] - window.mean()) / window.std(ddof=1)
    assert ops.ts_zscore(X, 3).iloc[3] == pytest.approx(expected)


def test_ts_rank_puts_the_window_high_at_one_and_the_low_at_zero():
    ranked = ops.ts_rank(X, 3)

    assert ranked.iloc[3] == pytest.approx(1.0)
    assert ranked.iloc[4] == pytest.approx(0.5)
    assert ranked.iloc[5] == pytest.approx(0.0)
    assert ranked.iloc[:2].isna().all()


def test_ts_correlation_is_one_for_a_series_against_itself():
    assert ops.ts_correlation(X, X, 4).iloc[3] == pytest.approx(1.0)
    assert ops.ts_correlation(X, -X, 4).iloc[3] == pytest.approx(-1.0)


def test_linear_decay_weights_the_latest_bar_most():
    decayed = ops.linear_decay(X, 3)

    assert decayed.iloc[2] == pytest.approx((3 * 4.0 + 2 * 2.0 + 1 * 1.0) / 6)
    assert decayed.iloc[:2].tolist() == [0.0, 0.0]


def test_decay_leaves_an_unchanging_series_alone():
    flat = pd.Series([0.5] * 8)
    assert ops.linear_decay(flat, 4).iloc[3:].round(12).tolist() == [0.5] * 5


def test_zscore_and_group_zscore_centre_their_cross_section():
    values = pd.Series([1.0, 3.0, 10.0, 20.0])
    groups = pd.Series(["a", "a", "b", "b"])

    assert ops.zscore(values).mean() == pytest.approx(0.0)
    assert ops.group_zscore(values, groups).tolist() == pytest.approx(
        [-0.70710678, 0.70710678] * 2
    )


def test_divide_reports_a_zero_denominator_as_missing():
    assert np.isnan(ops.divide(pd.Series([1.0]), pd.Series([0.0]))).all()
    assert ops.divide(pd.Series([3.0]), pd.Series([2.0])).iloc[0] == pytest.approx(1.5)


def test_signed_power_keeps_the_sign():
    assert ops.signed_power(pd.Series([-4.0, 9.0]), 0.5).tolist() == [-2.0, 3.0]


def test_if_else_treats_a_missing_condition_as_false():
    condition = pd.Series([1.0, 0.0, np.nan])
    assert ops.if_else(condition, 10.0, 20.0).tolist() == [10.0, 20.0, 20.0]


def test_trade_when_holds_the_last_triggered_value():
    trigger = pd.Series([False, True, False, False, True, False])
    held = ops.trade_when(trigger, X)

    assert pd.isna(held.iloc[0])
    assert held.iloc[1:4].tolist() == [2.0, 2.0, 2.0]
    assert held.iloc[4:].tolist() == [7.0, 7.0]


def test_an_exit_flattens_the_position_and_blocks_the_stale_entry():
    trigger = pd.Series([True, False, False, False, False, False])
    exits = pd.Series([False, False, True, False, False, False])
    held = ops.trade_when(trigger, X, exits)

    assert held.iloc[:2].tolist() == [1.0, 1.0]
    assert held.iloc[2:].isna().all()


def test_hump_limits_how_far_a_value_moves_in_one_bar():
    assert ops.hump(pd.Series([1.0, 1.5]), hump=0.1).iloc[1] == pytest.approx(1.1)


def test_days_from_last_change_counts_repeats():
    assert ops.days_from_last_change(pd.Series([1, 1, 2, 2, 2])).tolist() == [1, 2, 1, 2, 3]


def test_grouping_sets_name_operators_that_exist():
    grouped = ops.BY_TICKER_OPERATORS | ops.PAIRWISE_BY_TICKER_OPERATORS | ops.BY_DATE_OPERATORS

    assert grouped <= set(ops.__all__)
    assert {"ts_correlation", "ts_covariance"} <= ops.PAIRWISE_BY_TICKER_OPERATORS
    assert "linear_decay" in ops.BY_TICKER_OPERATORS
