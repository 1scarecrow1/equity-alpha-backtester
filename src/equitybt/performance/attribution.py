import equitybt.performance.metrics as pm
import numpy as np
import pandas as pd
from equitybt.backtester.returns import DATE_COL, TICKER_COL, WEIGHT_COL


def prepare_attribution(
    asset_df,
    *,
    horizon="daily",
    group_cols=(TICKER_COL,),
    weight_col=WEIGHT_COL,
    book_size,
):
    """
    Aggregate instrument-level PnL attribution by daily/monthly/yearly horizon.

    contribution_return is contribution_pnl / book size. contribution_pct is the
    group's share of total period PnL after aggregation, so period percentages
    reconcile to 100% when total PnL is non-zero.
    """
    if horizon not in pm.RESAMPLE_HORIZON:
        raise ValueError(f"horizon must be one of {list(pm.RESAMPLE_HORIZON.keys())}")

    asset = prepare_asset_attribution(asset_df, weight_col=weight_col, book_size=book_size)
    group_cols = [group_cols] if isinstance(group_cols, str) else list(group_cols)
    missing = [column for column in group_cols if column not in asset.columns]
    if missing:
        raise ValueError(f"Missing attribution group columns: {missing}")

    grouper = [pd.Grouper(key=DATE_COL, freq=pm.RESAMPLE_HORIZON[horizon]), *group_cols]
    aggregations = {
        "contribution_pnl": ("contribution_pnl", "sum"),
        "contribution_return": ("contribution_return", "sum"),
        "observations": ("contribution_pnl", "size"),
    }
    if weight_col in asset.columns:
        aggregations["avg_abs_weight"] = (weight_col, lambda values: values.abs().mean())
        aggregations["avg_weight"] = (weight_col, "mean")

    attribution = (
        asset.groupby(grouper, dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values([DATE_COL, "contribution_pnl"], ascending=[True, False])
    )

    totals = attribution.groupby(DATE_COL)["contribution_pnl"].transform("sum")
    attribution["period_pnl"] = totals
    attribution["contribution_pct"] = np.where(totals != 0, attribution["contribution_pnl"] / totals, np.nan)
    if "avg_weight" in attribution.columns:
        attribution["side"] = np.where(attribution["avg_weight"] > 0, "long", np.where(attribution["avg_weight"] < 0, "short", "flat"))
    return attribution


def attribution_snapshot(
    asset_df,
    *,
    horizon="daily",
    period=None,
    group_cols=(TICKER_COL,),
    weight_col=WEIGHT_COL,
    limit=10,
    book_size,
):
    attribution = prepare_attribution(
        asset_df,
        horizon=horizon,
        group_cols=group_cols,
        weight_col=weight_col,
        book_size=book_size,
    )
    if attribution.empty:
        return attribution

    selected = nearest_period(attribution[DATE_COL], period) if period is not None else attribution[DATE_COL].max()
    snapshot = attribution.loc[pd.to_datetime(attribution[DATE_COL]).eq(pd.Timestamp(selected))].copy()
    return (
        snapshot.reindex(snapshot["contribution_pnl"].abs().sort_values(ascending=False).index)
        .head(limit)
        .reset_index(drop=True)
    )


def side_attribution(
    asset_df,
    *,
    horizon="daily",
    weight_col=WEIGHT_COL,
    book_size,
):
    asset = prepare_asset_attribution(asset_df, weight_col=weight_col, book_size=book_size)
    side = np.where(asset[weight_col] > 0, "long", np.where(asset[weight_col] < 0, "short", "flat"))
    asset = asset.assign(side=side)
    return prepare_attribution(
        asset,
        horizon=horizon,
        group_cols=("side",),
        weight_col=weight_col,
        book_size=book_size,
    )


def group_return_matrix(
    asset_df: pd.DataFrame,
    group_col: str,
    *,
    book_size: float,
) -> pd.DataFrame:
    if group_col not in asset_df.columns or "pnl" not in asset_df.columns:
        return pd.DataFrame()

    frame = asset_df[[DATE_COL, group_col, "pnl"]].copy()
    frame[DATE_COL] = pd.to_datetime(frame[DATE_COL])
    frame[group_col] = frame[group_col].astype(str)
    wide = (
        frame.groupby([DATE_COL, group_col], dropna=False)["pnl"]
        .sum()
        .unstack(group_col, fill_value=0.0)
        .sort_index()
    )
    return wide / book_size


def group_labels(df, group_cols):
    return df[list(group_cols)].astype(str).agg(" / ".join, axis=1)


def prepare_asset_attribution(asset_df, *, book_size, weight_col=WEIGHT_COL):
    asset = asset_df.copy()
    if DATE_COL not in asset.columns:
        raise ValueError(f"Expected '{DATE_COL}' column in asset dataframe.")
    asset[DATE_COL] = pd.to_datetime(asset[DATE_COL])
    if TICKER_COL not in asset.columns:
        asset[TICKER_COL] = asset.index.astype(str)

    if "contribution_pnl" not in asset.columns:
        if "pnl" in asset.columns:
            asset["contribution_pnl"] = asset["pnl"].fillna(0)
        elif "weighted_ret" in asset.columns:
            asset["contribution_pnl"] = asset["weighted_ret"].fillna(0) * book_size
        elif {"ret", weight_col}.issubset(asset.columns):
            asset["contribution_pnl"] = asset["ret"].fillna(0) * asset[weight_col].fillna(0) * book_size
        else:
            raise ValueError("Need 'pnl', 'weighted_ret', or both 'ret' and weights for attribution.")

    asset["contribution_return"] = asset["contribution_pnl"] / book_size
    return asset


def nearest_period(dates, value):
    value = pd.Timestamp(value)
    dates = pd.Series(pd.to_datetime(dates).sort_values().unique())
    deltas = (dates - value).abs()
    return dates.iloc[deltas.argmin()]
