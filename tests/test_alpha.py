import pandas as pd
import pytest
from equitybt.dataexpression.dataexpression import DataExpression
from equitybt.dataexpression.variable import Variable
from equitybt.strategy.alpha import Alpha
from equitybt.strategy.build_alpha import apply_constraints, compute_signal, evaluate_expression
from equitybt.strategy.strategy import Strategy
from equitybt.variables.equities import adv20, close, open, volume


def test_data_expression_smoke():
    expression = -close / open
    df = pd.DataFrame({"close": [10.0, 12.0], "open": [5.0, 6.0]})

    assert expression.variable_names == {"close", "open"}
    assert evaluate_expression(df, expression).tolist() == [-2.0, -2.0]
    assert Variable("close").description


def test_strategy_accepts_custom_signal_and_constraint():
    strategy = Strategy(signals=-close / open, constraints=volume > 200000)

    assert strategy.signals["signal"].variable_names == {"close", "open"}
    assert strategy.constraints["constraint"].variable_names == {"volume"}
    assert [field.name for field in strategy.fields] == ["close", "open", "volume"]


def test_alpha_accepts_singular_signal_and_constraint():
    alpha = Alpha(signal=-close / open, constraint=volume > 200000)

    assert alpha.signals["signal"].variable_names == {"close", "open"}
    assert alpha.constraints["constraint"].variable_names == {"volume"}
    assert [field.name for field in alpha.fields] == ["close", "open", "volume"]
    assert not alpha.has_exit_condition


def test_strategy_accepts_multiple_signals_and_constraints():
    strategy = Strategy(
        signals={
            "signal": close / open,
            "volume_signal": volume / adv20,
        },
        constraints={
            "liquid": DataExpression("adv20 > cs_mean(adv20)", {"adv20"}),
            "positive_open": open > 0,
        },
    )
    df = pd.DataFrame(
        {
            "ticker": ["A", "B", "A", "B"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]),
            "close": [12.0, 6.0, 10.0, 8.0],
            "open": [10.0, 3.0, 5.0, 8.0],
            "volume": [100.0, 200.0, 120.0, 160.0],
            "adv20": [80.0, 120.0, 150.0, 100.0],
        }
    )

    assert [field.name for field in strategy.fields] == ["adv20", "close", "open", "volume"]
    assert evaluate_expression(df, strategy.signals["signal"]).tolist() == [1.2, 2.0, 2.0, 1.0]
    assert evaluate_expression(df, strategy.signals["volume_signal"]).tolist() == [1.25, 5 / 3, 0.8, 1.6]
    assert str(strategy.constraints["liquid"]) == "adv20 > cs_mean(adv20)"
    assert strategy.constraints["liquid"].variable_names == {"adv20"}
    assert evaluate_expression(df, strategy.constraints["positive_open"]).tolist() == [True] * 4


def test_alpha_constraints_can_reference_generated_signal():
    alpha = Alpha(
        signals=close / open,
        constraints=(Alpha.signal > 1.5) | (Alpha.signal < 0.75),
    )
    df = pd.DataFrame(
        {
            "close": [12.0, 10.0, 6.0],
            "open": [10.0, 5.0, 10.0],
            "signal": [1.2, 2.0, 0.6],
        }
    )

    assert [field.name for field in alpha.fields] == ["close", "open"]

    result = apply_constraints(df, alpha)

    assert pd.isna(result["signal"].iloc[0])
    assert result["signal"].iloc[1:].tolist() == [2.0, 0.6]


def test_alpha_exit_condition_closes_trade_when_true():
    alpha = Alpha(
        signal=close / open,
        constraint=volume > 100,
        exit_condition=volume < 25,
    )
    df = pd.DataFrame(
        {
            "ticker": ["A", "A", "A", "A"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
            "close": [10.0, 12.0, 20.0, 30.0],
            "open": [10.0, 10.0, 10.0, 10.0],
            "volume": [200.0, 50.0, 10.0, 50.0],
        }
    )

    result = compute_signal(df, alpha)

    assert result["signal"].iloc[:2].tolist() == [1.0, 1.0]
    assert pd.isna(result["signal"].iloc[2])
    assert pd.isna(result["signal"].iloc[3])


def test_strategy_delegates_alpha_exit_condition():
    alpha = Alpha(signal=close / open, constraint=volume > 100, exit_condition=volume < 25)
    strategy = Strategy(alpha=alpha)

    assert strategy.exit_condition is alpha.exit_condition
    assert strategy.has_exit_condition


def test_alpha_rejects_non_default_scalar_exit_condition():
    with pytest.raises(TypeError):
        Alpha(signal=close / open, exit_condition=1)


def test_alpha_exit_condition_can_close_without_constraints():
    alpha = Alpha(signal=close / open, exit_condition=volume < 25)
    df = pd.DataFrame(
        {
            "ticker": ["A", "A", "A", "A"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
            "close": [10.0, 12.0, 20.0, 30.0],
            "open": [10.0, 10.0, 10.0, 10.0],
            "volume": [200.0, 50.0, 10.0, 50.0],
        }
    )

    result = compute_signal(df, alpha)

    assert result["signal"].iloc[:2].tolist() == [1.0, 1.2]
    assert pd.isna(result["signal"].iloc[2])
    assert result["signal"].iloc[3] == 3.0


def test_point_in_time_mask_drops_the_signal_outside_the_universe():
    alpha = Alpha(signal=close / open)
    df = pd.DataFrame(
        {
            "ticker": ["A", "A", "B", "B"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"] * 2),
            "close": [12.0, 10.0, 6.0, 8.0],
            "open": [10.0, 5.0, 3.0, 8.0],
            "in_universe": [True, False, True, True],
        }
    )

    result = compute_signal(df, alpha)

    assert result.loc[~result["in_universe"], "signal"].isna().all()
    assert result.loc[result["in_universe"], "signal"].tolist() == [1.2, 2.0, 1.0]
