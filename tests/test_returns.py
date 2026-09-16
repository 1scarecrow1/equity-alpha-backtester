import equitybt.backtester.returns as br
import numpy as np
import pandas as pd
import pytest
from equitybt.trade_configs import Delay

BOOK_SIZE = 20e6


def book(df, signal_col):
    """Neutral, gross-one weights"""
    centred = df.groupby("date")[signal_col].transform(lambda values: values - values.mean())
    df = df.assign(weights=centred / centred.abs().groupby(df["date"]).transform("sum"))

    return df.dropna(subset=["weights"])


def simulate(df, signal_col, delay):
    allocated = br.get_portfolio_allocations(book(df, signal_col), delay=delay, book_size=BOOK_SIZE)
    return br.asset_returns(allocated).groupby("date")["pnl"].sum()


@pytest.fixture(scope="module")
def foresight(panel):
    df = panel.copy()
    df["tomorrow"] = df.groupby("ticker")["returns"].shift(-1)
    df["today"] = df["returns"]
    df["zero"] = 0.0

    return df


def t_statistic(pnl):
    return pnl.mean() / pnl.std() * np.sqrt(len(pnl))


def test_a_signal_that_knows_tomorrow_pays_only_at_delay_zero(foresight):
    assert t_statistic(simulate(foresight, "tomorrow", Delay.ZERO)) > 20
    assert abs(t_statistic(simulate(foresight, "tomorrow", Delay.ONE))) < 3


def test_a_signal_that_knows_today_pays_at_neither_delay(foresight):
    assert abs(t_statistic(simulate(foresight, "today", Delay.ZERO))) < 3
    assert abs(t_statistic(simulate(foresight, "today", Delay.ONE))) < 3


def test_delay_changes_which_book_is_held(foresight):
    allocated = {
        delay: br.get_portfolio_allocations(
            book(foresight, "today"), delay=delay, book_size=BOOK_SIZE
        ).sort_values(["ticker", "date"])
        for delay in (Delay.ZERO, Delay.ONE)
    }
    zero, one = allocated[Delay.ZERO], allocated[Delay.ONE]
    held_a_day_earlier = zero.groupby("ticker")["exposure"].shift().fillna(0.0)

    assert np.allclose(one["exposure"].to_numpy(), held_a_day_earlier.to_numpy())
    assert not np.allclose(one["exposure"].to_numpy(), zero["exposure"].to_numpy())


def test_a_zero_signal_trades_nothing_and_earns_nothing(foresight):
    df = foresight.assign(weights=0.0)
    allocated = br.get_portfolio_allocations(df, delay=Delay.ONE, book_size=BOOK_SIZE)
    priced = br.asset_returns(allocated)

    assert priced["pnl"].sum() == 0.0
    assert priced["trade_volume"].sum() == 0.0


def test_no_position_is_opened_before_the_delay_has_passed(panel):
    allocated = br.get_portfolio_allocations(
        book(panel, "returns"), delay=Delay.ONE, book_size=BOOK_SIZE
    ).sort_values(["ticker", "date"])
    first_two = allocated.groupby("ticker").head(2)

    assert (first_two["exposure"] == 0.0).all()


def test_daily_returns_are_pnl_over_the_book_size(panel):
    allocated = br.get_portfolio_allocations(
        book(panel, "returns"), delay=Delay.ONE, book_size=BOOK_SIZE
    )
    portfolio = br.portfolio_daily_returns(br.asset_returns(allocated), book_size=BOOK_SIZE)

    assert np.allclose(portfolio["ret"], portfolio["pnl"] / BOOK_SIZE)
    assert portfolio["cum_pnl"].iloc[-1] == pytest.approx(portfolio["pnl"].sum())
    assert (portfolio["drawdown"] <= 1e-12).all()


def test_turnover_is_traded_notional_over_the_book_size(panel):
    allocated = br.get_portfolio_allocations(
        book(panel, "returns"), delay=Delay.ONE, book_size=BOOK_SIZE
    )
    portfolio = br.portfolio_daily_returns(br.asset_returns(allocated), book_size=BOOK_SIZE)

    assert np.allclose(portfolio["turnover"], portfolio["trade_volume"] / BOOK_SIZE)


def test_shifting_the_whole_sample_shifts_the_pnl_with_it(foresight):
    calendar = pd.DatetimeIndex(sorted(foresight["date"].unique()))
    moved = foresight.copy()
    moved["date"] = moved["date"].map(dict(zip(calendar, calendar.shift(1, freq="B"), strict=True)))

    original = simulate(foresight, "today", Delay.ONE)
    shifted = simulate(moved, "today", Delay.ONE)

    assert np.allclose(original.to_numpy(), shifted.to_numpy())
