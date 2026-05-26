import numpy as np
import pandas as pd
from equitybt.data.yfinance_loader import get_tickers, load_variables
from equitybt.strategy import operators
from equitybt.strategy.alpha import Alpha
from equitybt.strategy.strategy import Strategy
from equitybt.trade_configs import Neutralization, TradingPeriod
from functools import partial
from typing import List

DATE_COL = "date"
TICKER_COL = "ticker"
SIGNAL_COL = "signal"

def compute_signal(
    df: pd.DataFrame,
    strategy: Strategy | Alpha, 
    ) -> pd.DataFrame:

    for name, signal in strategy.signals.items():
        df[name] = evaluate_expression(df, signal)
    df = apply_constraints(df, strategy)
    df = apply_pit_mask(df, list(strategy.signals))
    return df


def apply_constraints(
    df: pd.DataFrame,
    strategy: Strategy | Alpha,
    *,
    fill_value: float | None = None,
) -> pd.DataFrame:

    if not strategy.constraints and not strategy.has_exit_condition:
        return df
    
    mask = build_constraint_mask(df, strategy) if strategy.constraints else pd.Series(True, index=df.index)
    exit_condition = evaluate_expression(df, strategy.exit_condition) if strategy.has_exit_condition else -1
    for name in strategy.signals:
        constrained = grouped_operator(df, TICKER_COL, operators.trade_when, mask, df[name], exit_condition)
        df[name] = constrained if fill_value is None else constrained.fillna(fill_value)  
        
    return df

def build_signals_from_data(
    universe: List[str],
    variables: str | List[str] = None,
    start: str = None,
    end: str = None,
    neutralization_groups = True
) -> pd.DataFrame:

    """
    Params
    @universe: a list of tickers
    @variables: any variables in data_fields.py
    @start: start date
    @end: end date
    Returns a dataframe with all fields
    """

    tickers = get_tickers(universe, start=start, end=end)
    print("Loading data...")
    variables = [] if variables is None else ([variables] if isinstance(variables, str) else list(variables))
    
    if neutralization_groups:
        variables += neutralization_group_fields(Neutralization.SECTOR)

    data = load_variables(variables, tickers=tickers, start=start, end=end)

    return data

def build_data_from_signals(
    strategy: Alpha | Strategy,
    trading_period: TradingPeriod,
    ) -> pd.DataFrame:

    """
    Params
    @universe: a list of tickers
    @alpha: alpha object or strategy with signals and constraints
    @trading_period: contains start and end data. To specify custom period, use TradingPeriod(start, end)
    Returns a dataframe with data for the signals
    """
    if isinstance(strategy, Alpha):
        strategy = Strategy(alpha=strategy)

    universe = strategy.universe

    tickers = get_tickers(universe, trading_period=trading_period)
    print("Loading data...")
    fields = list(strategy.fields)
    group_fields = neutralization_group_fields(strategy.neutralization)

    data = load_variables(
        fields+group_fields,
        tickers=tickers,
        start=trading_period.start,
        end=trading_period.end,
    )

    return data

def neutralization_group_fields(neutralization: Neutralization) -> List[str]:
    group = Neutralization(neutralization)
    if group in (Neutralization.NONE, Neutralization.MARKET):
        return []

    return [group.value.lower()]

def evaluate_expression(df: pd.DataFrame, expression) -> pd.Series:
    namespace = {column: df[column] for column in df.columns}
    for name in operators.__all__:
        operator = getattr(operators, name)
        if name in operators.PAIRWISE_BY_TICKER_OPERATORS or name in operators.BY_TICKER_OPERATORS:
            namespace[name] = partial(grouped_operator, df, TICKER_COL, operator)
        elif name in operators.BY_DATE_OPERATORS:
            namespace[name] = partial(grouped_operator, df, DATE_COL, operator)
        else:
            namespace[name] = operator
    result = eval(str(expression), {"__builtins__": {}}, namespace)
    if isinstance(result, pd.Series):
        return result.reindex(df.index)
    return pd.Series(result, index=df.index)


def grouped_operator(df: pd.DataFrame, group_col: str, operator, *args, **kwargs):
    function = operator.function
    first_series_idx = next((i for i, arg in enumerate(args) if isinstance(arg, pd.Series)), None)
    if first_series_idx is None or group_col not in df.columns:
        return function(*args, **kwargs)    

    series = args[first_series_idx]
    pieces = []
    for _, idx in df.loc[series.index].groupby(group_col, sort=False).groups.items():
        group_args = [
            arg.loc[idx] if isinstance(arg, pd.Series) else arg
            for arg in args
        ]
        group_kwargs = {
            key: value.loc[idx] if isinstance(value, pd.Series) else value
            for key, value in kwargs.items()
        }
        result = function(*group_args, **group_kwargs)
        pieces.append(result.reindex(idx) if isinstance(result, pd.Series) else pd.Series(result, index=idx))

    if not pieces:
        return pd.Series(index=series.index, dtype=float)
    
    return pd.concat(pieces).reindex(series.index)

def build_constraint_mask(df: pd.DataFrame, strategy: Strategy | Alpha) -> pd.Series:
    masks = [evaluate_expression(df, constraint).fillna(False).astype(bool) \
             for constraint in strategy.constraints.values()]
    if not masks:
        return pd.Series(True, index=df.index)
    return pd.concat(masks, axis=1).all(axis=1)

def apply_pit_mask(df: pd.DataFrame, signal_names: list[str]) -> pd.DataFrame:
    """
    Drop signals where (date, ticker) is not in the point-in-time universe
    """
    if "in_universe" not in df.columns:
        return df
    mask = df["in_universe"].to_numpy()
    if mask.all():
        return df
    for name in signal_names:
        if name in df.columns:
            df[name] = df[name].where(mask, np.nan)
    return df
