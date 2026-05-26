import numbers
import numpy as np
import pandas as pd
from equitybt.strategy.alpha import Alpha
from equitybt.strategy.build_alpha import DATE_COL, SIGNAL_COL, TICKER_COL
from equitybt.strategy.operators import linear_decay
from equitybt.strategy.strategy import Strategy
from equitybt.trade_configs import Neutralization, Truncation
from typing import List

WEIGHT_COL = "weights"

def compute_weights(
    df: pd.DataFrame | pd.Series, 
    strategy: Strategy = None,
    signal_col: str = SIGNAL_COL,
    weight_col: str = WEIGHT_COL,
    ticker_col: str = TICKER_COL,
    date_col: str = DATE_COL,
    neutralize_on: Neutralization = Neutralization.NONE,
    truncation: float = 0.0,
    decay: int = 0,
    tolerance: float = 1e-9,
    ) -> pd.DataFrame | pd.Series:

    neutralization_group = Neutralization(strategy.neutralization) \
                        if hasattr(strategy, "neutralization") else neutralize_on

    truncation_val = truncation_value(strategy) \
                        if hasattr(strategy, "truncation") else truncation

    decay = strategy.decay if hasattr(strategy, "decay") else int(decay)

    df = df.copy()

    df = group_neutralize(df, strategy, neutralize_on, apply_on=signal_col,
                          weight_col=weight_col, date_col=date_col)

    df = normalize_weights(df, weight_col, date_col)
    
    if decay > 0:
        df = apply_decay(df, n=decay, weight_col=weight_col, ticker_col=ticker_col, date_col=date_col)
        df = group_neutralize(df, strategy, neutralize_on, apply_on=weight_col,
                            weight_col=weight_col, date_col=date_col)
        df = normalize_weights(df, weight_col, date_col)

    if truncation_val > 0:
        if neutralization_group != Neutralization.NONE:
            df = apply_truncation(df, strategy, truncation_val, weight_col, neutralize_on)
        else:
            df = stabilize_weights(df, truncation_val, weight_col, date_col, tolerance)

    return df

def normalize_weights(
    df: pd.DataFrame,
    weight_col=WEIGHT_COL,
    date_col=DATE_COL,
):
    daily_exposure = df[weight_col].abs().groupby(df[date_col]).transform("sum")
    df[weight_col] = np.divide(
                    df[weight_col],
                    daily_exposure,
                    out=np.zeros_like(df[weight_col], dtype=float),
                    where=daily_exposure > 0
                    )     

    return df


def group_neutralize(
    df: pd.DataFrame,
    strategy: Strategy | Alpha = None,
    neutralize_on: Neutralization | str = Neutralization.NONE,
    apply_on: str | List[str] = SIGNAL_COL,
    weight_col: str | List[str] = WEIGHT_COL,
    ticker_col: str = TICKER_COL,
    date_col: str = DATE_COL,
    ) -> pd.DataFrame:

    if hasattr(strategy, "neutralization"):
        group = strategy.neutralization

    else:
        group = neutralize_on

    group = Neutralization(group)

    key_cols = [date_col, ticker_col]
    if set(key_cols).issubset(df.columns) and df.duplicated(key_cols).any():
        raise ValueError("group_neutralize requires unique (date, ticker) rows.")

    if group == Neutralization.NONE:
        df[weight_col] = df[apply_on].fillna(0)
        return df

    if apply_on not in df.columns:
        raise ValueError("Column to neutralise must be present in the dataframe.")
    
    group_cols = neutralization_group_columns(strategy, group, date_col)
    missing_group_cols = [column for column in group_cols if column not in df.columns]
    if missing_group_cols:
        raise ValueError(f"Neutralization columns missing: {missing_group_cols}")

    group_source = df[group_cols]

    group_keys = pd.MultiIndex.from_frame(group_source)
    group_codes, _ = pd.factorize(group_keys, sort=False)
    signal_values = df[apply_on].to_numpy(dtype=float)
    group_valid = (group_codes >= 0) & group_source.notna().all(axis=1).to_numpy()
    n_groups = group_codes.max() + 1

    if n_groups == 0:
        df[weight_col] = 0.0
        return df

    valid = group_valid & np.isfinite(signal_values)
    group_count = np.bincount(group_codes[valid], minlength=n_groups)
    group_sum = np.bincount(group_codes[valid], weights=signal_values[valid], minlength=len(group_count))

    group_means = np.full(len(group_count), np.nan)
    np.divide(group_sum, group_count, out=group_means, where=group_count > 0)

    neutralized = np.zeros(len(df), dtype=float)
    neutralized[group_valid] = signal_values[group_valid] - group_means[group_codes[group_valid]]
    neutralized[~np.isfinite(neutralized)] = 0.0

    df[weight_col] = neutralized

    return df

def apply_truncation(
    df: pd.DataFrame, 
    strategy: Strategy,
    max_weight: int | None = None,
    weight_col: str = WEIGHT_COL,
    neutralize_on: Neutralization = Neutralization.NONE
    ) -> pd.DataFrame:

    truncation = truncation_value(strategy) if hasattr(strategy, "truncation") else max_weight 

    if not truncation:
        return df
    
    if truncation <= 0:
        raise ValueError("Truncation must be between 0 and 1.")

    group_cols = neutralization_group_columns(strategy, neutralize_on) 

    if group_cols:
        missing_group_cols = [column for column in group_cols if column not in df.columns]
        if missing_group_cols:
            raise ValueError(f"Neutralization columns missing: {missing_group_cols}")
        
        df[weight_col] = df.groupby(group_cols, group_keys=False)[weight_col].transform(
                        neutralized_truncated_weights, truncation=truncation
                        )

        return df

    weights = df[weight_col].to_numpy(dtype=float)
    df[weight_col] = np.clip(weights, -truncation, truncation)

    return df

def apply_decay(
    df: pd.DataFrame, 
    n: int = 0,
    weight_col: str = WEIGHT_COL,
    ticker_col: str = TICKER_COL,
    date_col: str = DATE_COL,
    ) -> pd.DataFrame:
    """
    Sets final weights to a linearly decreasing weighted average 
    of raw weights over the past n days 
    """

    if n == 0:
        return df
    
    if n < 0 or not isinstance(n, numbers.Integral):
        raise ValueError("Number of days should be a positive integer")
    
    df = df.sort_values([ticker_col, date_col])
    df[weight_col] = df.sort_values([ticker_col, date_col]).groupby(ticker_col, sort=False)[weight_col].transform(
        linear_decay,
        n=n
    ).to_numpy()

    return df.sort_values([date_col, ticker_col])

def neutralized_truncated_weights(weights: pd.Series, truncation: float) -> pd.Series:
    long_values = weights.clip(lower=0).to_numpy(dtype=float)
    short_values = (-weights.clip(upper=0)).to_numpy(dtype=float)
    target = min(
        long_values.sum(),
        short_values.sum(),
        (long_values > 0).sum() * truncation,
        (short_values > 0).sum() * truncation,
    )
    long_weights = capped_side_weights(long_values, target, truncation)
    short_weights = capped_side_weights(short_values, target, truncation)

    return pd.Series(long_weights - short_weights, index=weights.index)

def stabilize_weights(df, truncation=0, weight_col=WEIGHT_COL, date_col=DATE_COL, tolerance=1e-9):
    if not truncation or truncation <= 0:
        return df
    if not needs_exposure_adjustment(df, truncation, weight_col, date_col, tolerance):
        return df                                              
    df[weight_col] = df.groupby(date_col, group_keys=False)[weight_col].transform(
        solve_capped_weights, cap=truncation, tolerance=tolerance
    )
    return df

def solve_capped_weights(weights: pd.Series, cap: float, tolerance: float = 1e-9) -> pd.Series:
    v = weights.to_numpy(dtype=float)
    sign, magnitude = np.sign(v), np.abs(v)
    n_active = int((magnitude > tolerance).sum())

    if n_active == 0:
        return weights * 0.0

    if magnitude.max() <= cap * (1 + tolerance):
        return weights.copy()                                 

    gross_capacity = n_active * cap                            
    target = 1.0 if gross_capacity >= 1.0 else gross_capacity  

    return pd.Series(sign * capped_side_weights(magnitude, target, cap), index=weights.index)


def capped_side_weights(values: np.ndarray, target: float, cap: float) -> np.ndarray:
    output = np.zeros(len(values), dtype=float)
    active = values > 0
    remaining = target

    while active.any() and remaining > 0:
        scaled = values[active] / values[active].sum() * remaining
        capped = scaled > cap
        active_idx = np.flatnonzero(active)
        if not capped.any():
            output[active_idx] = scaled
            break
        output[active_idx[capped]] = cap
        remaining -= cap * capped.sum()
        active[active_idx[capped]] = False

    return output

def truncation_value(strategy):
    truncation = strategy.truncation.value if isinstance(strategy.truncation, Truncation) else strategy.truncation
    return truncation or 0.0

def needs_exposure_adjustment(df, truncation, weight_col=WEIGHT_COL, date_col=DATE_COL, tolerance=1e-9):
    active = df[weight_col].abs() > tolerance
    if not active.any():
        return False

    w = df.loc[active, weight_col].abs()
    dates = df.loc[active, date_col]
    gross = w.groupby(dates).transform("sum")
    n_active = w.groupby(dates).transform("size")

    target = np.minimum(1.0, n_active * truncation) if truncation > 0 else 1.0
    cap_violated = truncation > 0 and (w > truncation * (1 + tolerance)).any()
    off_target   = (np.abs(gross - target) > tolerance).any()
    return cap_violated or off_target


def neutralization_group_columns(
        strategy: Strategy = None, 
        neutralize_on: Neutralization | str = Neutralization.NONE,
        date_col: str = DATE_COL
        ) -> list[str]:

    group = (
        strategy.neutralization
        if hasattr(strategy, "neutralization")
        else neutralize_on
    )
    group = Neutralization(group)
    if group == Neutralization.NONE:
        return []
    if group == Neutralization.MARKET:
        return [date_col]
    return [date_col, group.value.lower()]