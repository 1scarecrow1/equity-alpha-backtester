import inspect
import numpy as np
import pandas as pd
from equitybt.dataexpression.operator import Operator
from numpy.lib.stride_tricks import sliding_window_view
from typing import Any


def cs_mean(value: Any, *, row: int | None = None, start: int | None = None, end: int | None = None, axis: str = "row") -> Any:
    """Cross-sectional mean of an array-like input."""
    if isinstance(value, pd.Series):
        return pd.Series(value.mean(), index=value.index)

    if axis not in {"row", "column"}:
        raise ValueError("axis must be 'row' or 'column'")

    array = np.asarray(value.T if axis == "column" and hasattr(value, "T") else value)
    if array.ndim == 0:
        return array.item()
    if array.ndim == 1:
        return array.mean()

    selected = array[row] if row is not None else array[start:end]
    return selected.mean() if selected.ndim == 1 else selected.mean(axis=1)


def rank(x, rate=0):
    """
    Cross-sectional rank mapped to [0.0, 1.0].
    rate controls value rounding before ranking. Use rate=0 for exact sorting.
    """
    
    if isinstance(x, pd.Series):
        values = round_rank(x, rate)
        valid = values.notna()
        out = pd.Series(np.nan, index=values.index, dtype=float)
        count = int(valid.sum())
        if count == 0:
            return out
        if count == 1:
            out.loc[valid] = 0.5
            return out
        out.loc[valid] = (values.loc[valid].rank(ascending=True, method="average") - 1) / (count - 1)
        return out

    array = round_rank(np.asarray(x, dtype=float), rate)
    out = np.full(array.shape, np.nan, dtype=float)
    valid = ~np.isnan(array)
    count = valid.sum()
    if count == 0:
        return out
    if count == 1:
        out[valid] = 0.5
        return out
    order = pd.Series(array[valid]).rank(ascending=True, method="average").to_numpy()
    out[valid] = (order - 1) / (count - 1)
    return out


def ts_delay(x, d):
    """x[t-d]."""
    return x.shift(d)


def ts_rank(x, d):
    """Trailing percent rank over d rows, mapped to [0.0, 1.0]."""
    return (x.rolling(d).rank() - 1) / (d - 1)


def ts_delta(x, d):
    """x[t] - x[t-d]."""
    return x.diff(d)


def zscore(x):
    """Z-score of a Series."""
    return (x - x.mean()) / x.std()


def ts_mean(x, window):
    """Rolling mean over window rows."""
    return x.rolling(window).mean()


def ts_std(x, window):
    """Rolling standard deviation over window rows."""
    return x.rolling(window).std()


def ts_zscore(x, window):
    """Rolling z-score over window rows."""
    return (x - ts_mean(x, window)) / ts_std(x, window)


def group_zscore(x, group):
    """Z-score within a group."""
    grouped = x.groupby(group, sort=False)
    return (x - grouped.transform("mean")) / grouped.transform("std")


def sma(x, window):
    """Simple moving average over window rows."""
    return ts_mean(x, window)


def ts_sum(x, window):
    """Rolling sum over window rows."""
    return x.rolling(window).sum()


def product(x, window):
    """Rolling product over window rows."""
    return x.rolling(window).apply(np.prod, raw=True)


def ts_min(x, window):
    """Rolling minimum over window rows."""
    return x.rolling(window).min()


def ts_max(x, window):
    """Rolling maximum over window rows."""
    return x.rolling(window).max()


def ts_argmax(x, window):
    """1-based row where rolling max occurs."""
    return x.rolling(window).apply(np.argmax, raw=True) + 1


def ts_argmin(x, window):
    """1-based row where rolling min occurs."""
    return x.rolling(window).apply(np.argmin, raw=True) + 1


def linear_decay(x, n):
    arr = np.nan_to_num(x.to_numpy(dtype=float))
    out = np.zeros(len(arr))
    weights = np.arange(1, n+1, dtype=float)
    total = weights.sum()
    if len(arr) >= n:
        out[n-1:] = np.convolve(arr, weights[::-1], mode="valid") / total 
    return pd.Series(out, index=x.index)

def ts_correlation(x, y, window):
    """Rolling Pearson correlation."""
    return x.rolling(window, min_periods=window).corr(y)


def ts_covariance(x, y, window):
    """Rolling covariance."""
    return x.rolling(window, min_periods=window).cov(y)


def days_from_last_change(x):
    change = x.ne(x.shift())
    groups = change.cumsum()
    return x.groupby(groups).cumcount() + 1

def hump(x, hump=0.01):
    delta = x.diff()
    return pd.Series(np.where(delta.abs() > hump, x.shift() + np.sign(delta) * hump, x), index=x.index)

def kth_element(x, d, k, ignore=np.nan):
    if k > d:
        raise ValueError(f"{k} cannot be larger than {d}")

    arr = x.to_numpy()
    padded = np.r_[np.full(d - 1, ignore), arr]
    windows = sliding_window_view(padded, d)
    valid = ~np.isnan(windows) if np.isnan(ignore) else windows != ignore
    rank_from_right = np.cumsum(valid[:, ::-1], axis=1)[:, ::-1]
    pick = valid & (rank_from_right == k)
    out = np.nanmax(np.where(pick, windows, np.nan), axis=1)
    out[~pick.any(axis=1)] = np.nan
    
    return pd.Series(out, index=x.index)


def last_diff_value(x, d):
    return x.where(x.ne(x.shift())).shift().ffill(limit=d)


def trade_when(trigger, signal, exit_cond=-1):
    """If trigger is true, change alpha value, else hold previous value, close position to NaN when exit condition true."""
    if isinstance(signal, pd.Series):
        trigger = _as_bool(trigger, signal.index)
        exit_mask = _as_bool(exit_cond, signal.index)
        candidate = signal.where(trigger, np.nan).mask(exit_mask, np.nan)
        return candidate.groupby(exit_mask.cumsum(), sort=False).ffill().mask(exit_mask, np.nan)

    return np.nan if _as_bool(exit_cond).item() else signal if _as_bool(trigger).item() else np.nan

def if_else(event, expression1, expression2):
    index = next((value.index for value in (event, expression1, expression2) if isinstance(value, pd.Series)), None)
    out = np.where(_as_bool(event, index), expression1, expression2)
    return pd.Series(out, index=index) if index is not None else out

def logical_and(cond1, cond2):
    return cond1 & cond2

def logical_or(cond1, cond2):
    return cond1 | cond2

def add(x, y):
    return x + y

def subtract(x, y, filter=True):
    if filter and isinstance(x, pd.Series) and isinstance(y, pd.Series):
        x, y = x.fillna(0), y.fillna(0)
    return x - y

def divide(x, y):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    out = np.divide(x_arr, y_arr, out=np.full(np.broadcast(x_arr, y_arr).shape, np.nan), where=y_arr != 0)
    if isinstance(x, pd.Series):
        return pd.Series(out, index=x.index)
    if isinstance(y, pd.Series):
        return pd.Series(out, index=y.index)
    return out

def inverse(x):
    out = np.divide(1.0, x, out=np.full_like(x, np.nan), where=x!=0)
    return out

def multiply(*args, filter=True):
    if not args:
        return np.nan  
    if filter:
        def clean(arg):
            if isinstance(arg, pd.Series):
                return arg.fillna(1.0)
            if np.isscalar(arg):
                return 1.0 if pd.isna(arg) else arg
            return np.nan_to_num(arg, nan=1.0)

        prod = clean(args[0])
        for arg in args[1:]:
            prod = prod * clean(arg)
        return prod

    prod = args[0]
    for arg in args[1:]:
        prod = prod * arg
    return prod

def power(x, y):
    return x ** y

def reverse(x):
    return -x

def signed_power(x, y):
    return np.sign(x) * power(np.abs(x), y)

def round_rank(x, rate):
    if not rate or rate <= 0:
        return x
    if isinstance(x, pd.Series):
        return x.round(int(rate))
    return np.round(np.asarray(x, dtype=float), int(rate))  
    
def _as_bool(value, index=None):
    if isinstance(value, pd.Series):
        output = value.astype(float).gt(0).fillna(False)
        return output.reindex(index) if index is not None else output
    return pd.Series(bool(value) if isinstance(value, bool) else float(value) > 0, index=index)

ts_arg_max = ts_argmax
ts_arg_min = ts_argmin

def _install_operators() -> None:
    for name in __all__:
        value = globals()[name]
        if inspect.isfunction(value):
            globals()[name] = Operator(value)


__all__ = [
    name
    for name, value in list(globals().items())
    if not name.startswith("_")
    and inspect.isfunction(value)
    and value.__module__ == __name__
]

_install_operators()

BY_TICKER_OPERATORS = {
    "ts_delay",
    "ts_rank",
    "ts_delta",
    "ts_mean",
    "ts_std",
    "ts_zscore",
    "sma",
    "ts_sum",
    "product",
    "ts_min",
    "ts_max",
    "ts_argmax",
    "ts_argmin",
    "ts_arg_max",
    "ts_arg_min",
    "days_from_last_change",
    "hump",
    "kth_element",
    "last_diff_value",
    "linear_decay",
    "trade_when",
}

PAIRWISE_BY_TICKER_OPERATORS = {"ts_correlation", "ts_covariance"}
BY_DATE_OPERATORS = {"rank", "zscore", "cs_mean", "group_zscore"}
