import numpy as np
import pandas as pd
import pytest
from equitybt.portfolio.weights import (
    WEIGHT_COL,
    apply_decay,
    compute_weights,
    group_neutralize,
    normalize_weights,
    solve_capped_weights,
)
from equitybt.strategy.alpha import Alpha
from equitybt.strategy.build_alpha import compute_signal
from equitybt.strategy.operators import rank, ts_zscore
from equitybt.strategy.strategy import Strategy
from equitybt.trade_configs import Decay, Neutralization
from equitybt.validation import are_weights_valid
from equitybt.variables.equities import returns


def strategy_for(tickers, neutralization, truncation=0.0, decay=Decay.NONE):
    return Strategy(
        alpha=Alpha(-rank(ts_zscore(returns, 20))),
        universe=tickers,
        neutralization=neutralization,
        truncation=truncation,
        decay=decay,
    )


@pytest.mark.parametrize(
    "neutralization",
    [Neutralization.NONE, Neutralization.MARKET, Neutralization.SECTOR, Neutralization.INDUSTRY],
)
@pytest.mark.parametrize("truncation", [0.0, 0.15])
def test_the_book_satisfies_its_invariants_however_it_is_built(
    panel, tickers, neutralization, truncation
):
    strategy = strategy_for(tickers, neutralization, truncation, Decay.WEEK)
    weights = compute_weights(compute_signal(panel.copy(), strategy), strategy)
    report = are_weights_valid(
        weights,
        neutralization=neutralization,
        truncation=truncation,
        delay=strategy.delay,
    )

    assert report, str(report)


def test_normalize_spends_exactly_the_whole_book():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"] * 3 + ["2024-01-03"] * 3),
            "weights": [2.0, -1.0, 1.0, 0.0, 0.0, 0.0],
        }
    )
    normalized = normalize_weights(df)["weights"]

    assert normalized.iloc[:3].tolist() == [0.5, -0.25, 0.25]
    assert normalized.iloc[3:].tolist() == [0.0, 0.0, 0.0]


def test_neutralize_removes_each_group_mean(tickers):
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"] * 4),
            "ticker": ["A", "B", "C", "D"],
            "sector": ["x", "x", "y", "y"],
            "signal": [1.0, 3.0, 10.0, 20.0],
        }
    )
    strategy = strategy_for(tickers, Neutralization.SECTOR)
    out = group_neutralize(df, strategy, apply_on="signal")

    assert out[WEIGHT_COL].tolist() == [-1.0, 1.0, -5.0, 5.0]


def test_truncation_pins_the_cap_and_redistributes_the_rest():
    """
    Tickers under truncation cap should keep their relative sizes
    and absorb what the capped tickers' quantities, 
    so gross exposure is unchanged.
    """
    capped = solve_capped_weights(pd.Series([0.6, 0.25, 0.15]), cap=0.4)

    assert capped.tolist() == pytest.approx([0.4, 0.375, 0.225])
    assert capped.sum() == pytest.approx(1.0)
    assert capped.iloc[1] / capped.iloc[2] == pytest.approx(0.25 / 0.15)


def test_a_binding_cap_leaves_gross_exposure_on_the_table(panel, tickers):
    def book(truncation):
        strategy = strategy_for(tickers, Neutralization.SECTOR, truncation)
        return compute_weights(compute_signal(panel.copy(), strategy), strategy)

    loose, tight = book(0.5), book(0.02)

    def gross(df):
        return df[WEIGHT_COL].abs().groupby(df["date"]).sum().iloc[-1]

    assert gross(loose) == pytest.approx(1.0)
    assert gross(tight) < 1.0
    assert tight[WEIGHT_COL].abs().max() <= 0.02 + 1e-9


def test_decay_cuts_turnover(panel, tickers):
    def turnover(decay):
        strategy = strategy_for(tickers, Neutralization.MARKET, decay=decay)
        weights = compute_weights(compute_signal(panel.copy(), strategy), strategy)
        weights = weights.sort_values(["ticker", "date"])
        return weights.groupby("ticker")[WEIGHT_COL].diff().abs().sum()

    assert turnover(Decay.TWO_WEEKS) < turnover(Decay.NONE)


def test_decay_of_one_bar_changes_nothing():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "ticker": ["A", "A", "A"],
            "weights": [0.2, 0.5, 0.3],
        }
    )
    assert apply_decay(df.copy(), n=1)["weights"].tolist() == pytest.approx([0.2, 0.5, 0.3])


def test_a_name_outside_the_universe_takes_no_position(panel, tickers):
    masked = panel.copy()
    excluded = masked["ticker"] == tickers[0]
    masked.loc[excluded, "in_universe"] = False
    masked.loc[excluded, "returns"] = 100.0

    strategy = strategy_for(tickers, Neutralization.SECTOR)
    weights = compute_weights(compute_signal(masked, strategy), strategy)

    assert np.allclose(weights.loc[weights["ticker"] == tickers[0], WEIGHT_COL], 0.0)
