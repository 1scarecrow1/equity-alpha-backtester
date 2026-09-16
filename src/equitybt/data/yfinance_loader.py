import json
import pandas as pd
from equitybt.data_fields import PYTHON_DERIVED_FIELDS
from equitybt.dataexpression.dataexpression import DataExpression
from equitybt.dataexpression.variable import Variable
from equitybt.trade_configs import TradingPeriod, Universe, UniverseSize
from equitybt.variables.equities import (
    close,
    date,
    dividends,
    high,
    low,
    open,
    returns,
    splits,
    ticker,
    volume,
)
from pathlib import Path
from typing import Any

DEFAULT_VARS = [date, ticker, open, high, low, close, volume, dividends, splits, returns]
DEFAULT_COLUMN_ORDER = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "returns",
    "volume",
    "dividends",
    "splits",
    "sector",
    "industry",
]

DOWNLOAD_COLUMNS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    "Dividends": "dividends",
    "Stock Splits": "splits",
}

LOCAL_DERIVED_FIELDS = {"returns"}
STATIC_FIELDS = {"sector", "industry", "shares_outstanding"}
CACHE_DIR = Path(".cache") / "bars"
RANGES_PATH = CACHE_DIR / "ranges.json"
STATIC_PATH = Path(".cache") / "static_fields.parquet"
DOWNLOAD_CHUNK = 100

_VARIABLE_CACHE: list[dict[str, Any]] = []
_TICKER_CACHE: dict[Any, list[str]] = {}

VariableInput = str | Variable | DataExpression


def get_tickers(
    universe: Universe | list[str] = None,
    start: str | None = None,
    end: str | None = None,
    trading_period: TradingPeriod | None = None,
):
    """
    Params
    @universe: a list of tickers to trade
    """
    if isinstance(universe, Universe):
        raise ValueError(
            f"{universe} does not name its members. Pass a list of tickers, and use "
            f"{universe} with add_universe_mask to trade the most liquid of them."
        )

    cache_key = tuple(universe)
    if cache_key not in _TICKER_CACHE:
        _TICKER_CACHE[cache_key] = list(universe)

    return list(_TICKER_CACHE[cache_key])


def _as_variable_list(items: VariableInput | list[VariableInput]) -> list[Variable]:
    if items is None:
        return []

    if isinstance(items, str | Variable | DataExpression):
        items = [items]

    variables: dict[str, Variable] = {}
    for item in items:
        if isinstance(item, str):
            variables[item] = Variable(item)
            continue

        if isinstance(item, Variable):
            variables[item.name] = item
            continue

        if isinstance(item, DataExpression):
            for name in item.variable_names:
                variables[name] = Variable(name)
            continue

        raise TypeError(f"Cannot load {type(item).__name__} as a variable")

    return list(variables.values())


def _ordered_unique_variables(variables: list[Variable]) -> list[Variable]:
    seen = set()
    ordered = []

    for variable in variables:
        if variable.name in seen:
            continue
        seen.add(variable.name)
        ordered.append(variable)

    return ordered


def _order_columns(df: pd.DataFrame, extra_variables: list[Variable]) -> pd.DataFrame:
    ordered_columns = [column for column in DEFAULT_COLUMN_ORDER if column in df.columns]
    extra_columns = [
        variable.name
        for variable in extra_variables
        if variable.name in df.columns and variable.name not in ordered_columns
    ]
    remaining_columns = [
        column
        for column in df.columns
        if column not in ordered_columns and column not in extra_columns
    ]

    return df[ordered_columns + extra_columns + remaining_columns]


def clear_variable_cache() -> None:
    _VARIABLE_CACHE.clear()


def _cache_fields(extra_variables: list[Variable]) -> set[str]:
    requested = {variable.name for variable in extra_variables}
    return {variable.name for variable in DEFAULT_VARS} | set(LOCAL_DERIVED_FIELDS) | requested


def _cache_lookup(
    *,
    tickers: list[str],
    start: str,
    end: str,
    fields: set[str],
) -> pd.DataFrame | None:
    requested_tickers = set(tickers)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    for entry in reversed(_VARIABLE_CACHE):
        if not requested_tickers.issubset(entry["tickers"]):
            continue
        if start_ts < entry["start"] or end_ts > entry["end"]:
            continue
        if not fields.issubset(entry["fields"]):
            continue

        df = entry["df"]
        mask = (
            df["ticker"].isin(requested_tickers)
            & df["date"].ge(start_ts)
            & df["date"].le(end_ts)
        )
        columns = [column for column in df.columns if column in fields]
        return df.loc[mask, columns].copy()

    return None


def _cache_store(
    df: pd.DataFrame,
    *,
    tickers: list[str],
    start: str,
    end: str,
    fields: set[str],
) -> None:
    _VARIABLE_CACHE.append(
        {
            "tickers": set(tickers),
            "start": pd.Timestamp(start),
            "end": pd.Timestamp(end),
            "fields": set(fields),
            "df": df.copy(),
        }
    )


def _to_long(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    frames = []
    for name in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            if name not in raw.columns.get_level_values(0):
                continue
            bars = raw[name]
        else:
            bars = raw

        bars = bars.rename(columns=DOWNLOAD_COLUMNS)
        bars = bars[[column for column in DOWNLOAD_COLUMNS.values() if column in bars.columns]]
        bars = bars.dropna(subset=["close"])
        if bars.empty:
            continue

        bars = bars.reset_index(names="date")
        bars["date"] = pd.to_datetime(bars["date"]).dt.tz_localize(None)
        bars.insert(1, "ticker", name)
        frames.append(bars)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _as_date(value: str) -> str:
    return pd.Timestamp(value).date().isoformat()


def _download(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    frames = []
    for offset in range(0, len(tickers), DOWNLOAD_CHUNK):
        chunk = tickers[offset : offset + DOWNLOAD_CHUNK]
        raw = yf.download(
            chunk,
            start=start,
            end=end,
            auto_adjust=False,
            actions=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )
        if raw is None or raw.empty:
            raise RuntimeError(f"No data returned for {chunk} between {start} and {end}")
        frames.append(_to_long(raw, chunk))

    return pd.concat(frames, ignore_index=True)


def load_bars(
    tickers: list[str],
    start: str,
    end: str,
    offline: bool = False,
) -> pd.DataFrame:

    start, end = _as_date(start), _as_date(end)
    ranges = json.loads(RANGES_PATH.read_text()) if RANGES_PATH.exists() else {}

    frames = []
    missing = []
    for name in tickers:
        path = CACHE_DIR / f"{name}.parquet"
        cached = ranges.get(name)
        if path.exists() and cached and cached[0] <= start and cached[1] >= end:
            frames.append(pd.read_parquet(path))
        else:
            missing.append(name)

    if missing and offline:
        raise RuntimeError(f"Not in the cache and offline: {missing}")

    if missing:
        downloaded = _download(missing, start, end)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for name, bars in downloaded.groupby("ticker"):
            bars.to_parquet(CACHE_DIR / f"{name}.parquet", index=False)
            ranges[name] = [start, end]
        RANGES_PATH.write_text(json.dumps(ranges, indent=2, sort_keys=True))
        frames.append(downloaded)

    df = pd.concat(frames, ignore_index=True)
    df = df.loc[df["date"].between(pd.Timestamp(start), pd.Timestamp(end))]

    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def load_static_fields(tickers: list[str], refresh: bool = False) -> pd.DataFrame:
    """
    Sector, industry and share count per ticker
    """
    if STATIC_PATH.exists() and not refresh:
        cached = pd.read_parquet(STATIC_PATH)
        if set(tickers).issubset(cached["ticker"]):
            return cached.loc[cached["ticker"].isin(tickers)].reset_index(drop=True)
    else:
        cached = pd.DataFrame(columns=["ticker", *sorted(STATIC_FIELDS)])

    import yfinance as yf

    rows = []
    for name in tickers:
        info = yf.Ticker(name).info or {}
        rows.append(
            {
                "ticker": name,
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "shares_outstanding": info.get("sharesOutstanding"),
            }
        )

    df = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
    df = df.drop_duplicates("ticker", keep="last").reset_index(drop=True)
    STATIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(STATIC_PATH, index=False)

    return df.loc[df["ticker"].isin(tickers)].reset_index(drop=True)


def add_returns(df: pd.DataFrame) -> pd.Series:
    """
    Compute dividend and split adjusted returns
    """
    close_values = df["close"].astype(float)
    ratio = df["splits"].astype(float).replace(0.0, 1.0)
    grouped = df.groupby("ticker", sort=False)
    return (close_values * ratio + df["dividends"].astype(float)) / grouped["close"].shift() - 1


def _python_derived_names(variables: list[Variable]) -> set[str]:
    return {variable.name for variable in variables if variable.name in PYTHON_DERIVED_FIELDS}


def _expand_python_derived_variables(variables: list[Variable]) -> list[Variable]:
    expanded = {variable.name: variable for variable in variables}
    pending = list(expanded)

    while pending:
        definition = PYTHON_DERIVED_FIELDS.get(pending.pop(), {})
        for dependency in definition.get("dependencies", ()):
            if dependency not in expanded:
                expanded[dependency] = Variable(dependency)
                pending.append(dependency)

    return list(expanded.values())


def _add_python_derived_fields(df: pd.DataFrame, field_names: set[str]) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    for field_name, definition in PYTHON_DERIVED_FIELDS.items():
        if field_name not in field_names:
            continue
        function = definition.get("function")

        if function == "total_return":
            df[field_name] = add_returns(df)
            continue

        if function == "typical_price":
            df[field_name] = df[["high", "low", "close"]].mean(axis=1)
            continue

        if function == "dollar_volume":
            df[field_name] = df["close"].astype(float) * df["volume"].astype(float)
            continue

        if function == "market_cap":
            df[field_name] = df["close"].astype(float) * df["shares_outstanding"].astype(float)
            continue

        if function in ("rolling_mean", "rolling_std"):
            source_column = definition["source_column"]
            window = int(definition["window"])
            min_periods = int(definition.get("min_periods", window))
            rolling = (
                df.groupby("ticker", sort=False)[source_column]
                .rolling(window, min_periods=min_periods)
            )
            values = rolling.mean() if function == "rolling_mean" else rolling.std()
            df[field_name] = values.reset_index(level=0, drop=True)
            continue

        raise ValueError(f"Unsupported python-derived field function: {function}")

    return df


def add_universe_mask(
    df: pd.DataFrame,
    universe: Universe | int = Universe.TOP500_PIT,
) -> pd.DataFrame:
    """
    Rank the most liquid names on each date
    """
    size = UniverseSize[universe.name] if isinstance(universe, Universe) else int(universe)
    rank_within_date = df.groupby("date", sort=False)["adv20"].rank(
        ascending=False, method="first"
    )
    df["in_universe"] = rank_within_date.le(size) & df["adv20"].notna()
    return df


def load_variables(
    variables: VariableInput | list[VariableInput] | None = None,
    *,
    tickers: list[str],
    start: str,
    end: str,
    offline: bool = False,
) -> pd.DataFrame:
    extra_variables = _as_variable_list(variables)
    output_fields = _cache_fields(extra_variables)
    cached = _cache_lookup(tickers=tickers, start=start, end=end, fields=output_fields)

    if cached is not None:
        return (
            _order_columns(cached, extra_variables)
            .sort_values(["date", "ticker"])
            .reset_index(drop=True)
        )

    variable_list = _expand_python_derived_variables(
        _ordered_unique_variables(DEFAULT_VARS + extra_variables)
    )
    requested_names = {variable.name for variable in variable_list}

    df = load_bars(tickers, start, end, offline=offline)
    if requested_names & STATIC_FIELDS:
        df = df.merge(load_static_fields(tickers), on="ticker", how="left")

    df = _add_python_derived_fields(df, _python_derived_names(variable_list))

    df = df[[column for column in df.columns if column in output_fields]]
    df = _order_columns(df, extra_variables)
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    _cache_store(df, tickers=tickers, start=start, end=end, fields=set(df.columns))
    return df
