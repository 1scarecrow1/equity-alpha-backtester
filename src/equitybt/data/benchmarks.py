import pandas as pd
from equitybt.data.yfinance_loader import load_bars

BENCHMARK_TICKERS = ("SPY", "QQQ")


def load_benchmark_closes(
    tickers: tuple[str, ...] = BENCHMARK_TICKERS,
    start: str = None,
    end: str | None = None,
    *,
    offline: bool = False,
) -> pd.DataFrame:
    end = end or pd.Timestamp.today().normalize().isoformat()
    bars = load_bars(list(tickers), start, end, offline=offline)

    return bars.pivot(index="date", columns="ticker", values="close").sort_index()


def benchmark_cumulative_pnl(
    portfolio_dates: pd.Series,
    book_size: float,
    tickers: tuple[str, ...] = BENCHMARK_TICKERS,
    *,
    offline: bool = False,
) -> pd.DataFrame:

    portfolio_dates = pd.to_datetime(pd.Series(portfolio_dates)).sort_values().unique()
    if len(portfolio_dates) == 0:
        return pd.DataFrame()

    start = pd.Timestamp(portfolio_dates[0]).date().isoformat()
    end = (pd.Timestamp(portfolio_dates[-1]) + pd.Timedelta(days=1)).date().isoformat()
    closes = load_benchmark_closes(tickers=tickers, start=start, end=end, offline=offline)
    closes = closes.reindex(pd.DatetimeIndex(portfolio_dates)).ffill()

    base = closes.iloc[0]
    cumulative_pnl = (closes.div(base) - 1.0) * book_size
    cumulative_pnl.index.name = "date"

    return cumulative_pnl
