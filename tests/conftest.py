import numpy as np
import pandas as pd
import pytest

TICKERS = [f"T{index:02d}" for index in range(20)]


def build_panel(tickers=TICKERS, bars=400, seed=0, daily_vol=0.015):
    """Dataframe with the columns the pipeline expects"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=bars)
    df = pd.DataFrame([(date, ticker) for date in dates for ticker in tickers],
                      columns=["date", "ticker"])
    df["sector"] = [f"sector_{tickers.index(ticker) % 4}" for ticker in df["ticker"]]
    df["industry"] = [
        f"industry_{tickers.index(ticker) % 4}_{tickers.index(ticker) // 4 % 2}"
        for ticker in df["ticker"]
    ]
    walks = np.cumsum(rng.normal(0, daily_vol, (bars, len(tickers))), axis=0)
    df["close"] = (100 * np.exp(walks)).ravel()
    df["open"] = df["close"] * (1 + rng.normal(0, 0.002, len(df)))
    df["high"] = df[["open", "close"]].max(axis=1) * 1.005
    df["low"] = df[["open", "close"]].min(axis=1) * 0.995
    df["volume"] = rng.uniform(1e6, 5e6, len(df))
    df["dividends"] = 0.0
    df["splits"] = 0.0
    df["dollar_volume"] = df["close"] * df["volume"]
    df["adv20"] = df.groupby("ticker")["dollar_volume"].transform(
        lambda values: values.rolling(20, min_periods=1).mean()
    )
    df["returns"] = df.groupby("ticker")["close"].pct_change()
    df["in_universe"] = True

    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


@pytest.fixture(scope="session")
def panel():
    return build_panel()


@pytest.fixture(scope="session")
def tickers():
    return list(TICKERS)
