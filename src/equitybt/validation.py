import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from equitybt.portfolio.weights import DATE_COL, SIGNAL_COL, TICKER_COL, WEIGHT_COL
from equitybt.trade_configs import Delay, Neutralization


@dataclass
class ValidityReport:
    """
    Checks to ascertain if calculations are valid/correct and check for market data anomalies
    """
    is_valid: bool
    checks: dict
    label: str = "Weights"
    violations: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.is_valid

    def __str__(self) -> str:
        passed = [name for name, ok in self.checks.items() if ok]
        failed = [name for name, ok in self.checks.items() if not ok]
        lines = [
            f"{self.label} valid: {self.is_valid}, "
            f"Passed: {', '.join(passed) or 'none'}, "
            f"Failed: {', '.join(failed) or 'none'}"
        ]
        for name in failed:
            if name in self.violations:
                lines.append(f"    - {name}: {self.violations[name]}")
        return "\n".join(lines)


def are_weights_valid(
    df: pd.DataFrame,
    neutralization: Neutralization = Neutralization.NONE,
    truncation: int = 0,
    weight_col=WEIGHT_COL,
    date_col=DATE_COL,
    ticker_col=TICKER_COL,
    delay: Delay = Delay.ONE,
    tolerance=1e-9,
) -> ValidityReport:

    if df.empty:
        raise ValueError("Empty dataframe")
    if weight_col not in df.columns:
        raise KeyError(f"{weight_col} not found")
    if date_col not in df.columns:
        raise KeyError(f"{date_col} not found")

    group = Neutralization(neutralization)
    if group not in (Neutralization.NONE, Neutralization.MARKET) and group.value.lower() not in df.columns:
        raise KeyError(f"neutralization column {group.value.lower()!r} not found")

    active = df[weight_col].abs() > tolerance
    if not active.any():
        return ValidityReport(False, {"has_active_positions": False})

    active_df = df.loc[active]
    weights = active_df[weight_col]
    dates = active_df[date_col]
    has_ticker = ticker_col in df.columns
    gross_exposure = weights.abs().groupby(dates).sum()
    capacity = pd.Series(1.0, index=gross_exposure.index)

    if group == Neutralization.NONE:
        group_neutral = True
        net_exposure = pd.Series(dtype=float)
        if truncation > 0:
            capacity = np.minimum(1.0, weights.groupby(dates).size() * truncation)
    else:
        keys = [dates] if group == Neutralization.MARKET else [dates, active_df[group.value.lower()]]
        net_exposure = weights.groupby(keys).sum()
        group_neutral = np.allclose(net_exposure, 0.0, atol=8 * len(weights) * np.finfo(float).eps)
        if truncation > 0:
            n_long = (weights > 0).groupby(keys).sum()
            n_short = (weights < 0).groupby(keys).sum()
            side_capacity = 2 * np.minimum(n_long, n_short) * truncation
            per_day = side_capacity if group == Neutralization.MARKET else side_capacity.groupby(level=0).sum()
            capacity = np.minimum(1.0, per_day)

    over_weight = active_df[weights.abs() > truncation * (1 + tolerance)] if truncation > 0 else active_df.iloc[:0]
    over_gross = gross_exposure[gross_exposure > capacity + tolerance]
    over_levered = gross_exposure[gross_exposure > 1.0 + tolerance]
    not_finite = df[~np.isfinite(df[weight_col].to_numpy())]
    net_off = net_exposure[net_exposure.abs() > 8 * len(weights) * np.finfo(float).eps] if len(net_exposure) else net_exposure

    checks = {
        "has_active_positions": bool(active.any()),
        "not_nan": bool(np.isfinite(df[weight_col].to_numpy()).all()),
        "under_max_weight": bool(truncation <= 0 or weights.abs().max() <= truncation * (1 + tolerance)),
        "within_gross_capacity": bool((gross_exposure <= capacity + tolerance).all()),
        "not_over_levered": bool((gross_exposure <= 1.0 + tolerance).all()),
        "group_neutral": bool(group_neutral),
    }

    violations = {}
    cols = [date_col] + ([ticker_col] if has_ticker else []) + [weight_col]
    if not checks["not_nan"]:
        violations["not_nan"] = f"{len(not_finite)} non-finite weights, e.g. " + ", ".join(
            f"{r[date_col]}/{r.get(ticker_col, '?')}" for _, r in not_finite.head(5).iterrows())
    if not checks["under_max_weight"]:
        violations["under_max_weight"] = f"{len(over_weight)} over cap {truncation}, e.g. " + ", ".join(
            f"{r[date_col]}/{r.get(ticker_col, '?')}={r[weight_col]:.4f}" for _, r in over_weight.head(5)[cols].iterrows())
    if not checks["within_gross_capacity"]:
        violations["within_gross_capacity"] = f"{len(over_gross)} days over capacity, e.g. " + ", ".join(
            f"{d}: gross={g:.4f}>cap={capacity[d]:.4f}" for d, g in over_gross.head(5).items())
    if not checks["not_over_levered"]:
        violations["not_over_levered"] = f"{len(over_levered)} days gross>1, e.g. " + ", ".join(
            f"{d}={g:.4f}" for d, g in over_levered.head(5).items())
    if not checks["group_neutral"]:
        violations["group_neutral"] = (
            f"delay={getattr(delay, 'value', delay)}: {len(net_off)} groups net!=0, e.g. " + ", ".join(
                f"{k}: net={v:.2e}" for k, v in net_off.head(5).items()))

    return ValidityReport(all(checks.values()), checks, violations=violations)


ALLOCATION_COLS = {"exposure", "previous_exposure", "trade_size", "trade_volume", "sign"}
ASSET_RETURN_COLS = {"ret", "pnl", "cum_pnl", "cum_ret", "log_ret"}
PORTFOLIO_COLS = {"pnl", "ret", "turnover", "cum_pnl", "drawdown"}


def is_source_data_valid(df, strategy, date_col=DATE_COL, ticker_col=TICKER_COL) -> ValidityReport:
    missing_fields = [field.name for field in strategy.fields if field.name not in df.columns]
    checks = {
        "non_empty": not df.empty,
        "has_date_and_ticker": {date_col, ticker_col}.issubset(df.columns),
        "alpha_fields_loaded": not missing_fields,
    }
    return ValidityReport(all(checks.values()), checks, "Source data")


def is_signal_valid(df, strategy, signal_col=SIGNAL_COL) -> ValidityReport:
    has_signal = signal_col in df.columns
    checks = {
        "signal_computed": has_signal,
        "signal_numeric": has_signal and bool(pd.api.types.is_numeric_dtype(df[signal_col])),
        "signal_active": has_signal and bool(df[signal_col].notna().any()),
    }
    return ValidityReport(all(checks.values()), checks, "Signal")


def are_allocations_valid(df) -> ValidityReport:
    has_cols = ALLOCATION_COLS.issubset(df.columns)
    checks = {
        "allocation_columns": has_cols,
        "trade_volume_non_negative": has_cols and bool((df["trade_volume"] >= 0).all()),
    }
    return ValidityReport(all(checks.values()), checks, "Allocations")


def are_asset_returns_valid(df) -> ValidityReport:
    has_cols = ASSET_RETURN_COLS.issubset(df.columns)
    checks = {
        "return_columns": has_cols,
        "pnl_finite": has_cols and bool(np.isfinite(df["pnl"]).all()),
    }
    return ValidityReport(all(checks.values()), checks, "Asset returns")


def is_portfolio_valid(portfolio, yearly, summary) -> ValidityReport:
    checks = {
        "daily_non_empty": not portfolio.empty,
        "daily_metrics": PORTFOLIO_COLS.issubset(portfolio.columns),
        "yearly_non_empty": not yearly.empty,
        "summary_non_empty": not summary.empty,
    }
    return ValidityReport(all(checks.values()), checks, "Portfolio")


def validate_backtest(strategy, weight_col=WEIGHT_COL) -> dict:
    df = strategy.df
    return {
        "source_data": is_source_data_valid(strategy.source_data, strategy),
        "signal": is_signal_valid(df, strategy),
        "weights": are_weights_valid(
            df,
            neutralization=strategy.neutralization,
            truncation=strategy.truncation,
            weight_col=weight_col,
            delay=strategy.delay,
        ),
        "allocations": are_allocations_valid(df),
        "asset_returns": are_asset_returns_valid(df),
        "portfolio": is_portfolio_valid(strategy.portfolio, strategy.yearly, strategy.summary),
    }


def trading_dates(df: pd.DataFrame, date_col=DATE_COL) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(pd.to_datetime(df[date_col]).unique()))


def ticker_changes(df: pd.DataFrame, date_col=DATE_COL, ticker_col=TICKER_COL) -> pd.DataFrame:
    by_date = df.groupby(pd.to_datetime(df[date_col]))[ticker_col].agg(set)
    rows = []
    prev = None
    for date, current in by_date.items():
        if prev is not None:
            entered, exited = current - prev, prev - current
            rows.append({
                date_col: date,
                "entered": len(entered),
                "exited": len(exited),
                "entered_tickers": sorted(entered),
                "exited_tickers": sorted(exited),
            })
        prev = current
    return pd.DataFrame(rows)


def delistings(df: pd.DataFrame, date_col=DATE_COL, ticker_col=TICKER_COL) -> pd.Series:
    last_seen = df.groupby(ticker_col)[date_col].max().map(pd.Timestamp)
    sample_end = pd.to_datetime(df[date_col]).max()
    return last_seen[last_seen < sample_end].sort_values()


def late_listings(df: pd.DataFrame, date_col=DATE_COL, ticker_col=TICKER_COL) -> pd.Series:
    first_seen = df.groupby(ticker_col)[date_col].min().map(pd.Timestamp)
    sample_start = pd.to_datetime(df[date_col]).min()
    return first_seen[first_seen > sample_start].sort_values()


def internal_gaps(df: pd.DataFrame, date_col=DATE_COL, ticker_col=TICKER_COL) -> pd.Series:
    """
    Count of trading dates missing within each ticker's span, excluding point-in-time membership changes
    """
    dates = trading_dates(df, date_col)
    span = df.assign(**{date_col: pd.to_datetime(df[date_col])}).groupby(ticker_col)[date_col].agg(["min", "max", "count"])
    expected = span.apply(lambda r: int(dates.slice_indexer(r["min"], r["max"]).stop - dates.slice_indexer(r["min"], r["max"]).start), axis=1)
    missing = (expected - span["count"]).astype(int)
    return missing[missing > 0].sort_values(ascending=False)


def are_market_changes_valid(df: pd.DataFrame, date_col=DATE_COL, ticker_col=TICKER_COL) -> ValidityReport:
    """
    Check for universe changes, delistings and gaps in ticker data
    """
    if df.empty:
        raise ValueError("Empty dataframe")
    for column in (date_col, ticker_col):
        if column not in df.columns:
            raise KeyError(f"{column} not found")

    de = delistings(df, date_col, ticker_col)
    late = late_listings(df, date_col, ticker_col)
    gaps = internal_gaps(df, date_col, ticker_col)
    changes = ticker_changes(df, date_col, ticker_col)
    churn_days = changes[(changes["entered"] > 0) | (changes["exited"] > 0)] if not changes.empty else changes

    checks = {
        "no_mid_sample_delistings": de.empty,
        "no_late_listings": late.empty,
        "no_internal_gaps": gaps.empty,
    }

    violations = {}
    if not checks["no_mid_sample_delistings"]:
        violations["no_mid_sample_delistings"] = f"{len(de)} delisted, e.g. " + ", ".join(
            f"{t} last {pd.Timestamp(d).date()}" for t, d in de.head(5).items())
    if not checks["no_late_listings"]:
        violations["no_late_listings"] = f"{len(late)} added mid-sample, e.g. " + ", ".join(
            f"{t} from {pd.Timestamp(d).date()}" for t, d in late.head(5).items())
    if not checks["no_internal_gaps"]:
        violations["no_internal_gaps"] = f"{len(gaps)} with gaps, e.g. " + ", ".join(
            f"{t}({n} missing)" for t, n in gaps.head(5).items())

    label = f"Market changes ({len(churn_days)} days with turnover)"
    return ValidityReport(all(checks.values()), checks, label=label, violations=violations)
