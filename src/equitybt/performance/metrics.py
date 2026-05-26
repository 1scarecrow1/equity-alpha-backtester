import inspect
import numpy as np
from equitybt.utils import with_array_inputs

ANNUALISATION_FACTOR = 252

ANNUALISATION_FACTORS = {'daily': 252, 'monthly': 12, 'yearly': 1}
RESAMPLE_HORIZON = {'daily': 'D', 'monthly': 'ME', 'yearly': 'YE', 'full': None}
RETURN_HORIZON = {'daily': 1, 'weekly': 7, 'monthly': 21, 'quarterly': 63, 'biannual': 126, 'yearly': 252}

def get_mean(x):
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    return np.mean(x)

def get_volatility(x, dof=1):
    x = x[~np.isnan(x)]
    if x.size <= 1:
        return np.nan
    return np.nanstd(x, ddof=dof)

def get_variance(x, dof=1):
    x = x[~np.isnan(x)]
    if x.size <= 1:
        return np.nan
    return np.nanvar(x, ddof=dof)

def annualised_return(pnl, book_size, ann_factor=ANNUALISATION_FACTOR):
    return ann_factor * get_mean(pnl) / book_size


def information_ratio(returns):
    mean_ret = get_mean(returns)
    sd_ret = get_volatility(returns)
    if np.isnan(sd_ret) or sd_ret <= 0:
        return np.nan
    return mean_ret / sd_ret


def sharpe(returns, ann_factor=ANNUALISATION_FACTOR):     
    return information_ratio(returns) * np.sqrt(ann_factor)
 
def turnover(trade_volume, book_size):
    return trade_volume / book_size


def capacity_score(returns, sharpe, turnover, turnover_floor=0.125):
    """
    Return per unit of turnover scaled by the Sharpe
    """
    return sharpe * np.sqrt(np.abs(returns) / np.maximum(turnover, turnover_floor))


def drawdown(pnl, cumulative=True):
    if not cumulative:
        pnl = np.nancumsum(pnl)
    if pnl.size == 0:
        return np.nan    
    return pnl - np.fmax.accumulate(pnl)

def max_drawdown(pnl, cumulative=True):
    dd = drawdown(pnl, cumulative=cumulative)
    return np.nanmin(dd)

def margin(pnl, trade_volume):
    return np.divide(
        pnl,
        trade_volume,
        out=np.zeros_like(pnl, dtype=float),
        where=trade_volume > 0,
    )

def perc_profitable_days(returns):
    returns = returns[~np.isnan(returns)]
    if returns.size == 0:
        return np.nan
    return np.mean(returns > 0)

def count_long(sign):
    long = np.count_nonzero(sign == 1)
    return long

def count_short(sign):
    short = np.count_nonzero(sign == -1)
    return short

def trend_profit_and_loss(pnl):
    delta = pnl.diff()
    avg_delta = np.nanmean(delta)
    return avg_delta


def _install_array_inputs() -> None:
    for name, obj in list(globals().items()):
        if (
            inspect.isfunction(obj)
            and obj.__module__ == __name__
            and not name.startswith("_")
        ):
            globals()[name] = with_array_inputs(obj)

_install_array_inputs()
