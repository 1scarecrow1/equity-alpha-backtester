import equitybt.performance.metrics as pm
import numpy as np
import pandas as pd
from equitybt.portfolio.weights import DATE_COL, TICKER_COL, WEIGHT_COL
from equitybt.trade_configs import Delay

SHARPE_WARMUP = 48
SHARPE_WINDOW = pm.RETURN_HORIZON['quarterly']

def get_portfolio_allocations(
    df: pd.DataFrame,
    weight_col: str = WEIGHT_COL,
    delay: Delay = Delay.ONE,
    ticker_col: str = TICKER_COL,
    date_col: str = DATE_COL,
    book_size: float = 1.0
    ) -> pd.DataFrame:

    df = df.copy().sort_values([ticker_col, date_col])
    g = df.groupby([ticker_col], as_index=False, sort=False)

    df[weight_col] = g[weight_col].shift(delay.value + 1).fillna(0)
    df['exposure'] = df[weight_col] * book_size

    df['previous_exposure'] = g["exposure"].shift().fillna(0) 
    df['trade_size'] = df['exposure'] - df['previous_exposure']
    df['trade_volume'] = df['trade_size'].fillna(0).abs()
    df['sign'] = np.where(df[weight_col] > 0, 1, np.where(df[weight_col] != 0, -1, 0))
    df['rebalanced'] = (~np.isclose(df['trade_volume'], 0.0, atol=1e-8)).astype(float)
    df.loc[g.cumcount() < delay.value + 1, 'rebalanced'] = np.nan

    return df

def asset_returns(
        df: pd.DataFrame, 
        ticker_col: str = TICKER_COL, 
        date_col: str = DATE_COL,
        returns: str = "ret",
        pnl_col: str = "pnl"
        ) -> pd.DataFrame:
    
    df = df.sort_values([ticker_col, date_col])
    df = df.rename(columns={"returns": returns})
    g = df.groupby([ticker_col], as_index=False)

    exposure = df['exposure'].to_numpy()
    df[pnl_col] = np.where(exposure == 0, 0.0, exposure * df[returns].to_numpy())

    df['cum_pnl'] = g[pnl_col].cumsum()
    df["cum_ret"] = (1 + df[returns]).groupby(df[ticker_col]).cumprod() - 1  

    df["log_ret"] = (np.log1p(df[returns].where(df[returns] > -1))).fillna(0)
    df['cum_log_ret'] = df['log_ret'].groupby(df[ticker_col]).cumsum()  

    return df

def portfolio_daily_returns(
        asset_returns: pd.DataFrame, 
        weight_col: str = WEIGHT_COL, 
        date_col: str = DATE_COL,
        returns: str = "ret", 
        pnl_col: str = "pnl",
        trade_volume_col: str = "trade_volume",
        sharpe_warmup: int = SHARPE_WARMUP,
        sharpe_window: int = SHARPE_WINDOW,
        book_size: float = 1.0,
        ann_factor = pm.ANNUALISATION_FACTOR

     ) -> pd.DataFrame:

    """
    Portfolio characteristics:

    Definitions:
    - daily PnL: sum of asset-level PnL per day
    - daily return: daily PnL / book size
    """

    portfolio = (
        asset_returns.groupby(date_col)
        .agg(
            pnl=(pnl_col, 'sum'),
            trade_volume=(trade_volume_col, 'sum'),
            gross_exposure=(weight_col, lambda x: x.abs().sum()),
            net_exposure=(weight_col, 'sum'),
            rebalanced=('rebalanced', 'sum')
        )
        .sort_index()
    )

    portfolio[returns] = portfolio.pnl / book_size
    portfolio['ann_ret'] = portfolio[returns] * ann_factor
    portfolio["log_ret"] = np.log1p(portfolio[returns].where(portfolio[returns] > -1))
    portfolio['cum_pnl'] = portfolio.pnl.cumsum()

    portfolio['cum_ret'] = (1 + portfolio[returns]).cumprod() - 1      
    portfolio["cum_log_ret"] = portfolio.log_ret.cumsum()
    portfolio['turnover'] = pm.turnover(portfolio[trade_volume_col], book_size)

    warmup = max(2, min(sharpe_warmup, len(portfolio)))

    sharpe_col = pnl_col

    # expanding sharpe
    exp_mean_ret = portfolio[sharpe_col].expanding(min_periods=warmup).mean()
    exp_vol = portfolio[sharpe_col].expanding(min_periods=warmup).std(ddof=1)
    exp_valid = exp_vol.to_numpy() > 0

    portfolio['expanding_sharpe'] = np.divide(
        exp_mean_ret.to_numpy(),
        exp_vol.to_numpy(),
        out=np.full(len(portfolio), np.nan),
        where=exp_valid,
    ) * np.sqrt(ann_factor)

    # rolling sharpe 
    roll_mean_ret = portfolio[sharpe_col].rolling(window=sharpe_window, min_periods=warmup).mean()
    roll_vol = portfolio[sharpe_col].rolling(window=sharpe_window, min_periods=warmup).std(ddof=1)
    roll_valid = roll_vol.to_numpy() > 0

    portfolio['rolling_sharpe'] = np.divide(
        roll_mean_ret.to_numpy(),
        roll_vol.to_numpy(),
        out=np.full(len(portfolio), np.nan),
        where=roll_valid,
    )  * np.sqrt(ann_factor)

    exp_ann_ret = exp_mean_ret * ann_factor / book_size
    exp_turnover = portfolio['turnover'].expanding(min_periods=warmup).mean()
    portfolio['expanding_capacity_score'] = np.where(
        exp_valid,
        pm.capacity_score(exp_ann_ret, portfolio.expanding_sharpe, exp_turnover),
        np.nan
    )

    roll_ann_ret = roll_mean_ret * ann_factor / book_size
    roll_turnover = portfolio['turnover'].rolling(window=sharpe_window, min_periods=warmup).mean()
    portfolio['rolling_capacity_score'] = np.where(
        roll_valid,
        pm.capacity_score(roll_ann_ret, portfolio.rolling_sharpe, roll_turnover),
        np.nan
    )

    portfolio['drawdown'] = pm.drawdown(portfolio.cum_pnl) / book_size
    portfolio['margin'] = pm.margin(portfolio[pnl_col], portfolio[trade_volume_col])

    count_df = asset_returns[[date_col, weight_col]].copy()
    portfolio['long_count'] = count_df.groupby(date_col)[weight_col].apply(lambda values: (values > 0).sum())
    portfolio['short_count'] = count_df.groupby(date_col)[weight_col].apply(lambda values: (values < 0).sum())
    portfolio['number_of_trades'] = portfolio['long_count'] + portfolio['short_count']     

    return portfolio

def portfolio_aggregate_returns(
        daily: pd.DataFrame,
        horizon: str = "daily", 
        returns: str = "ret", 
        pnl_col: str = "pnl",
        trade_volume_col: str = "trade_volume",
        sharpe_warmup: int = SHARPE_WARMUP,
        warmup: bool = True,
        delay: Delay = Delay.ONE,
        date_col: str = DATE_COL,
        book_size: float = 1.0,
        ann_factor = pm.ANNUALISATION_FACTOR
) -> pd.DataFrame:

    if horizon not in pm.RESAMPLE_HORIZON:
        raise ValueError(f"horizon must be one of {list(pm.RESAMPLE_HORIZON.keys())}")    
   
    if date_col in daily.columns:
        daily = daily.set_index(date_col)

    metrics = ['ret', 'pnl', 'log_ret', 'turnover', "trade_volume", "long_count", "short_count", 
               "gross_exposure", "net_exposure", 'rebalanced']

    available = [column for column in metrics if column in daily.columns]
    if pnl_col not in available and pnl_col in daily.columns:
        available.append(pnl_col)
    daily = daily[available].copy()

    def _col(frame, name):
        if name in frame.columns:
            return frame[name]
        return pd.Series(np.nan, index=frame.index)

    warmup_cutoff = daily.index[sharpe_warmup] if warmup and len(daily) > sharpe_warmup else None
    build_cutoff = daily.index[delay.value + 1] if len(daily) > delay.value + 1 else None

    if horizon == "full":
        grouped = [(daily.index[-1] if not daily.empty else pd.NaT, daily)]
    else:
        grouped = daily.resample(pm.RESAMPLE_HORIZON[horizon], convention="end")

    rows = []
    for period_end, period in grouped:
        if period.empty:
            continue

        pnl = _col(period, pnl_col)
        ret = _col(period, returns)
        period_ret = pm.annualised_return(pnl, book_size, ann_factor=ann_factor)

        long_count = pm.get_mean(_col(period, "long_count"))
        short_count = pm.get_mean(_col(period, "short_count"))

        period_cutoff = period.loc[period.index >= build_cutoff]
        turnover = pm.get_mean(period_cutoff.turnover if build_cutoff is not None else _col(period, "turnover"))
        gross_exposure = pm.get_mean(period_cutoff.gross_exposure if build_cutoff is not None else _col(period, "gross_exposure"))
        net_exposure = pm.get_mean(period_cutoff.net_exposure if build_cutoff is not None else _col(period, "net_exposure"))
        rebalanced = pm.get_mean(period_cutoff.rebalanced if build_cutoff is not None else _col(period, "rebalanced"))

        trade_volume = _col(period, trade_volume_col).sum()
        total_pnl = pnl.sum()
        drawdown = pm.max_drawdown(pnl, cumulative=False) / book_size

        # post-warmup slice of this period; warmup return/turnover feed fitness only
        post = period.loc[period.index >= warmup_cutoff] if warmup_cutoff is not None else period.iloc[0:0]
        ret_post = pm.annualised_return(_col(post, pnl_col), book_size, ann_factor=ann_factor)
        turnover_post = pm.get_mean(_col(post, "turnover"))
        sharpe_full = pm.sharpe(pnl) if len(period) > 1 else np.nan
        sharpe_post = pm.sharpe(_col(post, pnl_col)) if len(post) > 1 else np.nan
        capacity_full = pm.capacity_score(period_ret, sharpe_full, turnover)
        capacity_post = pm.capacity_score(ret_post, sharpe_post, turnover_post)
        sharpe = sharpe_post if warmup else sharpe_full
        capacity = capacity_post if warmup else capacity_full

        rows.append({
            date_col: period_end,
            "ret": period_ret,
            "vol": pm.get_volatility(ret) * np.sqrt(ann_factor),
            "var": pm.get_variance(ret) * ann_factor,
            "pnl": total_pnl,
            "mean_pnl": pm.get_mean(pnl),
            "log_ret": _col(period, "log_ret").sum(),
            "turnover": turnover,
            "trade_volume": trade_volume,
            "drawdown": drawdown,
            "sharpe": sharpe,
            "capacity_score": capacity,
            "sharpe_full": sharpe_full,
            "sharpe_warmup": sharpe_post,
            "capacity_score_full": capacity_full,
            "capacity_score_warmup": capacity_post,
            "ret_warmup": ret_post,
            "turnover_warmup": turnover_post,
            "margin": pm.margin(np.array([total_pnl]), np.array([trade_volume]))[0] if trade_volume else np.nan,
            "long_count": long_count,
            "short_count": short_count,
            "number_of_trades": long_count + short_count,
            "perc_profitable_days": pm.perc_profitable_days(ret),
            "avg_gross_exposure": gross_exposure,
            "avg_net_exposure": net_exposure,
            "avg_rebalanced": rebalanced     
        })

    portfolio = pd.DataFrame(rows).set_index(date_col) if rows else pd.DataFrame()
    if portfolio.empty:
        return portfolio

    portfolio['cum_pnl'] = np.cumsum(portfolio[pnl_col].to_numpy())
    portfolio["cum_log_ret"] = portfolio.log_ret.cumsum()
    
    return portfolio

def portfolio_summary(
        portfolio,
        returns='ret',
        pnl_col='pnl',
        trade_volume_col='trade_volume',
        sharpe_warmup: int = SHARPE_WARMUP,
        warmup: bool = True,
        delay: Delay = Delay.ONE,
        date_col=DATE_COL,
        book_size: float = 1.0,
        ann_factor=pm.ANNUALISATION_FACTOR
):
    """
    End-of-period total aggregate metrics
    Based on daily portfolio returns
    """
    portfolio = portfolio.copy()
    if date_col in portfolio.columns:
        portfolio = portfolio.set_index(date_col)

    if "number_of_trades" not in portfolio.columns and {"long_count", "short_count"}.issubset(portfolio.columns):
        portfolio["number_of_trades"] = portfolio["long_count"] + portfolio["short_count"]

    pnl = portfolio[pnl_col]
    ret = portfolio[returns]

    total_pnl = pnl.sum() if not portfolio.empty else 0.0
    avg_ret = pm.get_mean(ret)
    annualised_ret = pm.annualised_return(pnl, book_size, ann_factor=ann_factor)
    return_vol = pm.get_volatility(ret) * np.sqrt(ann_factor)
    turnover = pm.get_mean(portfolio.turnover.iloc[delay.value + 1:])
    total_traded_dollars = np.sum(portfolio[trade_volume_col].to_numpy())
    number_of_trades_per_day = pm.get_mean(portfolio.number_of_trades)
    long_trades = pm.get_mean(portfolio.long_count)
    short_trades = pm.get_mean(portfolio.short_count)
    gross_exposure = pm.get_mean(portfolio.gross_exposure.iloc[delay.value + 1:])
    net_exposure = pm.get_mean(portfolio.net_exposure.iloc[delay.value + 1:])
    rebalanced = pm.get_mean(portfolio.rebalanced.iloc[delay.value + 1:])

    sharpe_col = pnl_col
    post_warmup = portfolio.iloc[sharpe_warmup:] if warmup else portfolio
    sharpe = pm.sharpe(post_warmup[sharpe_col]) if len(post_warmup) > 1 else np.nan
    capacity = pm.capacity_score(pm.annualised_return(post_warmup.pnl, book_size, ann_factor=ann_factor), sharpe, pm.get_mean(post_warmup.turnover))
    max_drawdown = pm.max_drawdown(pnl.cumsum()) / book_size
    margin = pm.margin(np.array([total_pnl]), np.array([total_traded_dollars]))[0] if total_traded_dollars else np.nan
    perc_profitable_days = pm.perc_profitable_days(ret)

    metrics = {
        'Total PnL': round(total_pnl, 2), 
        'Annualised return': _format_percent_or_na(annualised_ret),
        'Avg Return': _format_percent_or_na(avg_ret),
        'Volatility': _format_percent_or_na(return_vol),
        'Sharpe': round(sharpe, 2), 
        'Turnover':  _format_percent_or_na(turnover),
        'Capacity score': round(capacity, 2), 
        'Drawdown': _format_percent_or_na(-max_drawdown),
        'Margin': _format_bps_or_na(margin), 
        'Long': f"{round(long_trades, 2)}",
        'Short': f"{round(short_trades, 2)}",
        'Trades per day': f"{round(number_of_trades_per_day, 2)}",
        '% Profitable Days': _format_percent_or_na(perc_profitable_days),
        'Avg Gross Exposure': _format_percent_or_na(gross_exposure),
        'Avg Net Exposure': _format_percent_or_na(net_exposure),
        'Avg Rebalances': round(rebalanced, 2),
    }

    return pd.DataFrame(metrics, index=['value'])

def rolling_beta_alpha(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int,
    annualisation_factor: int = pm.ANNUALISATION_FACTOR,
) -> pd.DataFrame:
    
    joined = pd.concat([portfolio_returns.rename("p"), benchmark_returns.rename("b")], axis=1).dropna()
    cov = joined["p"].rolling(window).cov(joined["b"])
    var = joined["b"].rolling(window).var(ddof=1)
    beta = cov / var.where(var > 0)
    mean_p = joined["p"].rolling(window).mean()
    mean_b = joined["b"].rolling(window).mean()
    daily_alpha = mean_p - beta * mean_b

    return pd.DataFrame({"beta": beta, "alpha_annualised": daily_alpha * annualisation_factor})

def _format_percent_or_na(value, decimals=2):
    if pd.isna(value):
        return ""
    return f"{round(value * 100, decimals)}%"

def _format_bps_or_na(value, decimals=2):
    if pd.isna(value):
        return ""
    return f"{round(value * 10000, decimals)}bps"
