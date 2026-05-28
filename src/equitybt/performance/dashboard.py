import equitybt.backtester.returns as br
import equitybt.performance.attribution as attr
import equitybt.performance.metrics as pm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sys
from equitybt.backtester.backtester import default_run_path, load_backtest, save_backtest
from equitybt.data.benchmarks import (
    BENCHMARK_TICKERS,
    benchmark_cumulative_pnl,
    load_benchmark_closes,
)
from equitybt.trade_configs import Delay, TradingPeriod
from pathlib import Path

st.set_page_config(page_title="Strategy Dashboard", layout="wide")


PERIOD_COLORS = [("IS_TRAIN","#1f77b4"), ("VALIDATION", "#ff7f0e"), ("OOS", "#2ca02c")]
PERIODS = []
PERCENT_PLOT_COLUMNS = {"ret", "turnover", "drawdown"}
PERCENT_DISPLAY_COLUMNS = {
    "ret",
    "returns",
    "turnover",
    "drawdown",
    "contribution_return",
    "contribution_pct",
    "avg_abs_weight",
    "weights",
    "weighted_ret",
    "cum_ret",
}
BPS_DISPLAY_COLUMNS = {"margin"}
INTEGER_DISPLAY_COLUMNS = {"long_count", "short_count", "number_of_trades", "observations"}
SERIES_COLORS = {
    "Strategy": "#1f77b4",
    "SPY": "#ff7f0e",
    "QQQ": "#2ca02c",
}

def _streamlit_version_at_least(major: int, minor: int) -> bool:
    parts = []
    for part in st.__version__.split(".")[:2]:
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(0)
    while len(parts) < 2:
        parts.append(0)
    return tuple(parts) >= (major, minor)


def _stretch_width_kwargs() -> dict:
    if _streamlit_version_at_least(1, 50):
        return {"width": "stretch"}
    return {"use_container_width": True}


def _split_periods(first, last, train_end, validation_end) -> list:
    bounds = [(first, train_end), (train_end, validation_end), (validation_end, last)]
    return [
        (name, TradingPeriod(start, end, name), color)
        for (name, color), (start, end) in zip(PERIOD_COLORS, bounds, strict=True)
    ]


def _yearly(portfolio: pd.DataFrame, book_size: float, delay: Delay = Delay.ONE) -> pd.DataFrame:
    return br.portfolio_aggregate_returns(
        portfolio, horizon="yearly", delay=delay, book_size=book_size
    ).reset_index()


def _monthly(portfolio: pd.DataFrame, book_size: float, delay: Delay = Delay.ONE) -> pd.DataFrame:
    return br.portfolio_aggregate_returns(
        portfolio, horizon="monthly", delay=delay, book_size=book_size
    ).reset_index()


def _summary(portfolio: pd.DataFrame, book_size: float, delay: Delay = Delay.ONE) -> pd.DataFrame:
    return br.portfolio_summary(portfolio, delay=delay, book_size=book_size)


@st.cache_data(show_spinner=False)
def _attribution(
    asset_df: pd.DataFrame, horizon: str, group_col: str, book_size: float
) -> pd.DataFrame:
    if group_col == "side":
        attribution = attr.side_attribution(asset_df, horizon=horizon, book_size=book_size)
    else:
        attribution = attr.prepare_attribution(
            asset_df, horizon=horizon, group_cols=(group_col,), book_size=book_size
        )
    return _add_group_labels(attribution, group_col)


def _path_from_args() -> str | None:
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    return args[0] if args else None


def _display_name(value):
    return getattr(value, "value", value)


def _add_group_labels(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    output = df.copy()
    if group_col in output.columns:
        output["group_name"] = output[group_col].astype(str)
    return output


def _format_metric_table(df: pd.DataFrame, percent_cols=()):
    df = _arrow_safe(df)
    existing_pct = [
        col for col in percent_cols
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]
    fmt = {
        col: _format_percent_2
        for col in existing_pct
    }
    for col in PERCENT_DISPLAY_COLUMNS:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            fmt[col] = _format_percent_2
    for col in BPS_DISPLAY_COLUMNS:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            fmt[col] = _format_bps_2
    for col in INTEGER_DISPLAY_COLUMNS:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            fmt[col] = _format_integer
    for col in df.select_dtypes(include="number").columns:
        fmt.setdefault(col, _format_number_2)
    labels = {c: {"ret": "Returns", "pnl": "PnL"}.get(c, str(c).replace("_", " ").title().replace("Pnl", "PnL")) for c in df.columns}
    df = df.rename(columns=labels)
    fmt = {labels[col]: func for col, func in fmt.items()}
    return df.style.format(fmt).apply(_style_period_row, axis=1)


def _format_number(value):
    if pd.isna(value):
        return ""
    return f"{value:,.3f}".rstrip("0").rstrip(".")


def _format_number_2(value):
    if pd.isna(value):
        return ""
    return f"{value:,.2f}"


def _format_integer(value):
    if pd.isna(value):
        return ""
    return f"{value:,.0f}"


def _format_percent(value):
    if pd.isna(value):
        return ""
    return f"{value * 100:.3f}".rstrip("0").rstrip(".") + "%"


def _format_percent_2(value):
    if pd.isna(value):
        return ""
    return f"{value * 100:.2f}%"


def _format_bps_2(value):
    if pd.isna(value):
        return ""
    return f"{value * 10000:.2f}bps"


def _period_name(value) -> str:
    if pd.isna(value):
        return ""
    date = pd.Timestamp(value).date()
    for name, period, _ in PERIODS:
        if period.start <= date <= period.end:
            return name
    return ""


def _style_period_row(row):
    period = row.get("Period") or _period_name(row.get("date", row.get("Date")))
    colors = {
        "IS_TRAIN": "#eef5ff",
        "VALIDATION": "#fff4e6",
        "OOS": "#ebf8ef",
    }
    return [f"background-color: {colors[period]}; color: #111111" if period in colors else "" for _ in row]


def _display_start(*frames: pd.DataFrame) -> pd.Timestamp:
    starts = []
    for frame in frames:
        if frame is None or frame.empty:
            continue
        if "date" in frame.columns:
            dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
        else:
            dates = pd.to_datetime(frame.index, errors="coerce").dropna()
        if len(dates):
            starts.append(pd.Timestamp(dates.min()))
    return max(starts) if starts else pd.Timestamp.min


def _filter_start(df: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    output = df.copy()
    if "date" not in output.columns:
        output = output.reset_index()
    output["date"] = pd.to_datetime(output["date"], errors="coerce")
    return output.loc[output["date"].notna() & output["date"].ge(start)].sort_values("date")


def _coerce_period_dates(df: pd.DataFrame, *, horizon: str) -> pd.DataFrame:
    output = df.copy()
    if "date" not in output.columns:
        output = output.reset_index()
    date_values = output["date"]
    if pd.api.types.is_integer_dtype(date_values) or pd.api.types.is_float_dtype(date_values):
        if horizon == "yearly":
            output["date"] = pd.to_datetime(date_values.astype("Int64").astype(str) + "-12-31", errors="coerce")
        elif horizon == "monthly":
            output["date"] = pd.to_datetime(date_values.astype("Int64").astype(str) + "-01", errors="coerce")
        else:
            output["date"] = pd.to_datetime(date_values, errors="coerce")
    else:
        output["date"] = pd.to_datetime(date_values, errors="coerce")
    return output


def _add_period_column(df: pd.DataFrame, *, source_col: str = "date") -> pd.DataFrame:
    output = df.copy()
    if source_col in output.columns:
        output["Period"] = output[source_col].map(_period_name)
    return output


def _period_line_chart(df: pd.DataFrame, columns, *, title=None, ylabel=None, percent_y=False, annotate_max_drawdown=False):
    chart_df = df.copy()
    if "date" not in chart_df.columns:
        chart_df = chart_df.reset_index()
    chart_df["date"] = pd.to_datetime(chart_df["date"], errors="coerce")
    chart_df = chart_df.dropna(subset=["date"]).sort_values("date")

    fig = go.Figure()
    legend_used = set()
    line_styles = ["solid", "dash", "dot"]
    for col_idx, column in enumerate(columns):
        if column not in chart_df.columns:
            continue
        for period_name, period, color in PERIODS:
            mask = chart_df["date"].dt.date.between(period.start, period.end)
            period_df = chart_df.loc[mask, ["date", column]].dropna()
            if period_df.empty:
                continue
            label = period_name if len(columns) == 1 else f"{period_name} {column}"
            hover_y = "%{y:.3%}" if percent_y else "%{y:,.3f}"
            fig.add_trace(
                go.Scatter(
                    x=period_df["date"],
                    y=period_df[column],
                    mode="lines",
                    name=label,
                    legendgroup=period_name,
                    showlegend=label not in legend_used,
                    line={
                        "color": color,
                        "width": 2,
                        "dash": line_styles[col_idx % len(line_styles)],
                    },
                    hovertemplate="%{x|%Y-%m-%d}<br>" + hover_y + "<extra>" + label + "</extra>",
                )
            )
            legend_used.add(label)
    if annotate_max_drawdown and "drawdown" in chart_df.columns:
        drawdown_df = chart_df[["date", "drawdown"]].dropna()
        if not drawdown_df.empty:
            min_idx = drawdown_df["drawdown"].idxmin()
            max_dd = drawdown_df.loc[min_idx, "drawdown"]
            max_dd_date = drawdown_df.loc[min_idx, "date"]
            fig.add_hline(
                y=max_dd,
                line_dash="dash",
                line_color="#d62728",
                annotation_text=f"Max drawdown {_format_percent_2(max_dd)}",
                annotation_position="bottom right",
            )
            fig.add_trace(
                go.Scatter(
                    x=[max_dd_date],
                    y=[max_dd],
                    mode="markers",
                    name="Max drawdown",
                    marker={"color": "#d62728", "size": 8},
                    hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3%}<extra>Max drawdown</extra>",
                    showlegend=False,
                )
            )
    fig.update_layout(
        title={"text": title or "", "x": 0, "xanchor": "left"},
        height=360,
        margin={"l": 48, "r": 24, "t": 40, "b": 40},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        xaxis_title="Date",
        yaxis_title=ylabel or "",
    )
    fig.update_xaxes(showgrid=True, rangeslider_visible=False)
    yaxis_kwargs = {"showgrid": True}
    if percent_y:
        yaxis_kwargs["tickformat"] = ".2%"
    fig.update_yaxes(**yaxis_kwargs)
    st.plotly_chart(fig, **_stretch_width_kwargs())


def _multi_line_chart(
    df: pd.DataFrame,
    *,
    title: str,
    ylabel: str,
    percent_y: bool = False,
    colors: dict[str, str] | None = None,
):
    chart_df = df.copy()
    if "date" not in chart_df.columns:
        chart_df = chart_df.reset_index()
    chart_df["date"] = pd.to_datetime(chart_df["date"], errors="coerce")
    chart_df = chart_df.dropna(subset=["date"]).sort_values("date")

    fig = go.Figure()
    for column in [col for col in chart_df.columns if col != "date"]:
        series = chart_df[["date", column]].dropna()
        if series.empty:
            continue
        hover_y = "%{y:.2%}" if percent_y else "%{y:,.2f}"
        fig.add_trace(
            go.Scatter(
                x=series["date"],
                y=series[column],
                mode="lines",
                name=str(column),
                line={"color": (colors or {}).get(str(column)), "width": 2},
                hovertemplate="%{x|%Y-%m-%d}<br>" + hover_y + "<extra>" + str(column) + "</extra>",
            )
        )
    fig.update_layout(
        title={"text": title, "x": 0, "xanchor": "left"},
        height=360,
        margin={"l": 48, "r": 24, "t": 40, "b": 40},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        xaxis_title="Date",
        yaxis_title=ylabel,
    )
    fig.update_xaxes(showgrid=True, rangeslider_visible=False)
    yaxis_kwargs = {"showgrid": True}
    if percent_y:
        yaxis_kwargs["tickformat"] = ".2%"
    fig.update_yaxes(**yaxis_kwargs)
    st.plotly_chart(fig, **_stretch_width_kwargs())


def _arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    for column in output.select_dtypes(include=["object", "str"]).columns:
        output[column] = output[column].map(_object_cell_to_string)
    return output


def _object_cell_to_string(value):
    if isinstance(value, (list, tuple, set, dict, np.ndarray, pd.Series)):
        return str(value)
    if pd.isna(value):
        return ""
    return str(value)


def _dataframe(df: pd.DataFrame, **kwargs):
    if kwargs.pop("width", None) == "stretch":
        kwargs.update(_stretch_width_kwargs())
    df = _arrow_safe(df)
    df.columns = [{"ret": "Returns", "pnl": "PnL"}.get(c, str(c).replace("_", " ").title()) for c in df.columns]
    st.dataframe(df, **kwargs)


def _period_attribution_charts(attribution: pd.DataFrame, *, limit=12):
    if attribution.empty or "Period" not in attribution.columns:
        return
    plotted = 0
    fig, axes = plt.subplots(1, len(PERIODS), figsize=(15, 4.5), squeeze=False)
    for ax, (period_name, _, color) in zip(axes[0], PERIODS):
        period_attr = attribution.loc[attribution["Period"].eq(period_name)]
        if period_attr.empty:
            ax.set_axis_off()
            continue
        values = (
            period_attr.groupby("group_name", dropna=False)["contribution_pnl"]
            .sum()
            .sort_values(key=lambda data: data.abs(), ascending=False)
            .head(limit)
            .sort_values()
        )
        colors = np.where(values.to_numpy() < 0, "#d62728", color)
        ax.barh(values.index.astype(str), values.to_numpy(), color=colors)
        ax.axvline(0, color="#666666", linewidth=0.8)
        ax.set_title(period_name, loc="left", fontsize=11, fontweight="semibold")
        ax.grid(True, axis="x", alpha=0.25)
        ax.tick_params(axis="y", labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        plotted += 1
    if plotted:
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)


@st.cache_data(show_spinner=False)
def _benchmark_cum_pnl(portfolio_dates: tuple, book_size: float) -> pd.DataFrame:
    return benchmark_cumulative_pnl(pd.Series(portfolio_dates), book_size=book_size)


@st.cache_data(show_spinner=False)
def _benchmark_returns(portfolio_dates: tuple, tickers: tuple[str, ...]) -> pd.DataFrame:
    dates = pd.DatetimeIndex(portfolio_dates)
    start = dates.min().date().isoformat()
    end = (dates.max() + pd.Timedelta(days=1)).date().isoformat()
    closes = load_benchmark_closes(tickers=tickers, start=start, end=end)
    closes = closes.reindex(dates).ffill()
    return closes.pct_change()


def _correlation_heatmap(matrix: pd.DataFrame):
    correlations = matrix.corr()
    if correlations.empty:
        return
    fig = go.Figure(
        data=go.Heatmap(
            z=correlations.to_numpy(),
            x=correlations.columns.astype(str),
            y=correlations.index.astype(str),
            colorscale="RdBu",
            reversescale=True,
            zmin=-1,
            zmax=1,
            colorbar={"title": "corr"},
            hovertemplate="%{y}<br>%{x}<br>%{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=520,
        margin={"l": 140, "r": 24, "t": 16, "b": 120},
        xaxis={"tickangle": -40},
    )
    st.plotly_chart(fig, **_stretch_width_kwargs())


def _ensure_daily_display_columns(portfolio: pd.DataFrame) -> pd.DataFrame:
    output = portfolio.copy()
    if "date" not in output.columns:
        output = output.reset_index()
    output["date"] = pd.to_datetime(output["date"], errors="coerce")
    output = output.sort_values("date")
    if "cum_pnl" not in output.columns and "pnl" in output.columns:
        output["cum_pnl"] = output["pnl"].fillna(0).cumsum()
    if "cum_pnl" in output.columns and "drawdown" not in output.columns:
        output["drawdown"] = pm.drawdown(output["cum_pnl"])
    if "ret" in output.columns and "ann_ret" not in output.columns:
        output["ann_ret"] = output["ret"] * pm.ANNUALISATION_FACTOR
    if "log_ret" not in output.columns and "ret" in output.columns:
        output["log_ret"] = np.log1p(output["ret"].where(output["ret"] > -1).to_numpy())
    if "rolling_sharpe" not in output.columns and "expanding_sharpe" in output.columns:
        output["rolling_sharpe"] = output["expanding_sharpe"]
    # backfill expanding capacity score for older portfolios that predate the
    # rolling/expanding capacity score columns, so the capacity score chart still renders
    if not ({"rolling_capacity_score", "expanding_capacity_score", "capacity_score"} & set(output.columns)):
        sharpe_col = next((col for col in ("expanding_sharpe", "rolling_sharpe", "sharpe") if col in output.columns), None)
        if sharpe_col and {"ret", "turnover"}.issubset(output.columns):
            ann_ret = output["ret"].expanding(min_periods=2).mean() * pm.ANNUALISATION_FACTOR
            turnover = output["turnover"].expanding(min_periods=2).mean()
            output["expanding_capacity_score"] = pm.capacity_score(ann_ret, output[sharpe_col], turnover)
    return output


def _period_aggregate_metrics(
    portfolio: pd.DataFrame, book_size: float, delay: Delay = Delay.ONE
) -> pd.DataFrame:
    rows = []
    in_sample = TradingPeriod(PERIODS[0][1].start, PERIODS[1][1].end, "IS")
    for period_name, period, _ in [*PERIODS[:2], ("IS", in_sample, None), *PERIODS[2:]]:
        mask = portfolio["date"].dt.date.between(period.start, period.end)
        period_df = portfolio.loc[mask].copy()
        if period_df.empty:
            continue
        aggregate = br.portfolio_aggregate_returns(
            period_df, horizon="full", delay=delay, book_size=book_size
        ).reset_index()
        if aggregate.empty:
            continue
        row = aggregate.iloc[0].to_dict()
        row["Period"] = period_name
        if "pnl" in row:
            row["total_pnl"] = row["pnl"]
        if "mean_pnl" in row:
            row["annualised_mean_pnl"] = row["mean_pnl"] * pm.ANNUALISATION_FACTOR
        rows.append(row)
    return pd.DataFrame(rows)


arg_path = _path_from_args()
runs_dir = Path("runs")
run_folders = [p for p in runs_dir.iterdir() if (p / "config.json").exists()] if runs_dir.exists() else []
run_options = [str(p) for p in sorted(run_folders, key=lambda p: p.stat().st_mtime, reverse=True)]
arg_norm = str(Path(arg_path)) if arg_path else None

with st.sidebar:
    st.header("Run")
    if run_options:
        default = arg_norm if arg_norm in run_options else run_options[0]
        selected = st.selectbox(
            "Run folder",
            run_options,
            index=run_options.index(default),
            format_func=lambda p: Path(p).name,
            key="run_folder",
        )
    else:
        selected = ""
    manual = st.text_input("…or a path", value="" if selected else (arg_path or "")).strip()
    path = manual or selected
    horizon = st.selectbox("Attribution horizon", ["monthly", "daily", "yearly"], index=0)

if not path:
    st.info("Pass a run folder on the command line, or write one into runs/.")
    st.stop()

strategy = load_backtest(path)
book_size = strategy.results["book_size"]
asset_df = strategy.df.copy()
portfolio = strategy.portfolio.copy() if hasattr(strategy, "portfolio") else strategy.results["portfolio"].copy()
first_bar = pd.Timestamp(portfolio["date"].min()).date()
last_bar = pd.Timestamp(portfolio["date"].max()).date()
span = (last_bar - first_bar).days
with st.sidebar:
    st.header("Sample split")
    train_end = st.date_input(
        "Train ends", value=first_bar + pd.Timedelta(days=int(span * 0.6)),
        min_value=first_bar, max_value=last_bar,
    )
    validation_end = st.date_input(
        "Validation ends", value=first_bar + pd.Timedelta(days=int(span * 0.8)),
        min_value=train_end, max_value=last_bar,
    )
PERIODS[:] = _split_periods(first_bar, last_bar, train_end, validation_end)

display_start = _display_start(asset_df, portfolio)
portfolio = _filter_start(portfolio, display_start)
asset_df = _filter_start(asset_df, display_start)
portfolio = _ensure_daily_display_columns(portfolio)

# a saved run already carries yearly/summary from run_backtest; only recompute when there is none.
recompute = getattr(strategy, "yearly", None) is None
yearly = _yearly(portfolio, book_size, strategy.delay) if recompute else strategy.yearly
yearly = _filter_start(_coerce_period_dates(yearly, horizon="yearly"), display_start)
yearly_variants = yearly 
monthly = _filter_start(
    _coerce_period_dates(_monthly(portfolio, book_size, strategy.delay), horizon="monthly"),
    display_start,
)

strategy.run_path = str(path)
with st.sidebar:
    st.header("Save")
    default_save_path = getattr(strategy, "run_path", str(default_run_path(strategy)))
    save_path = st.text_input("Save run as", value=default_save_path)
    if st.button("Save run"):
        save_backtest(strategy, save_path)
        st.success(f"Saved {save_path}")

st.title("Strategy Dashboard")

signals = {name: str(expr) for name, expr in strategy.signals.items()}
constraints = {name: str(expr) for name, expr in strategy.constraints.items()}
exit_conditions = (
    {"exit_condition": str(strategy.exit_condition)}
    if strategy.has_exit_condition
    else {}
)
attributes = {
    "instrument_type": _display_name(strategy.instrument_type),
    "region": _display_name(strategy.region),
    "universe": _display_name(strategy.universe),
    "delay": _display_name(strategy.delay),
    "decay": strategy.decay,
    "neutralization": _display_name(strategy.neutralization),
    "truncation": strategy.truncation,
    "book_size": book_size,
}

left, middle = st.columns([1.25, 1.0])
with left:
    st.subheader("Strategy")
    st.write("Alpha signals")
    _dataframe(pd.DataFrame(signals.items(), columns=["name", "expression"]), width="stretch", hide_index=True)
    st.write("Constraints")
    _dataframe(pd.DataFrame(constraints.items(), columns=["name", "expression"]), width="stretch", hide_index=True)
    if exit_conditions:
        st.write("Exit condition")
        _dataframe(pd.DataFrame(exit_conditions.items(), columns=["name", "expression"]), width="stretch", hide_index=True)
with middle:
    st.subheader("Attributes")
    _dataframe(pd.DataFrame(attributes.items(), columns=["attribute", "value"]), width="stretch", hide_index=True)

aggregate_display = _period_aggregate_metrics(portfolio, book_size, strategy.delay)
is_metrics = aggregate_display.loc[aggregate_display["Period"] == "IS"].iloc[0]
is_summary = [
    ("Sharpe", _format_number(is_metrics.get("sharpe", 0))),
    ("Turnover", _format_percent(is_metrics.get("turnover", 0))),
    ("Capacity score", _format_number(is_metrics.get("capacity_score", 0))),
    ("Returns", _format_percent(is_metrics.get("ret", 0))),
    ("Drawdown", _format_percent(is_metrics.get("drawdown", 0))),
    ("Margin", _format_bps_2(is_metrics.get("margin", 0))),
    ("PnL", f"{is_metrics.get('total_pnl', 0):,.0f}"),
]
st.subheader("IS Summary")
for col, (label, value) in zip(st.columns(len(is_summary)), is_summary):
    col.markdown(f"<small>{label}</small><br><b style='font-size:1.5rem'>{value}</b>", unsafe_allow_html=True)

st.subheader("Aggregate Metrics")
aggregate_columns = [
    "Period", "sharpe", "turnover", "capacity_score", "ret",
    "drawdown", "margin", "total_pnl", "annualised_pnl",
]
aggregate_display = aggregate_display[[col for col in aggregate_columns if col in aggregate_display.columns]]
st.dataframe(
    _format_metric_table(
        aggregate_display,
        percent_cols=("ret", "turnover", "drawdown"),
    ),
    **_stretch_width_kwargs(),
    hide_index=True,
)

st.subheader("Portfolio Summary")
# Split by IS_TRAIN/VALIDATION; keep only metrics not already in Aggregate Metrics above.
summary_rows = []
for period_name, period, _ in PERIODS[:2]:
    period_summary = _summary(
        portfolio.loc[portfolio["date"].dt.date.between(period.start, period.end)],
        book_size,
        strategy.delay,
    )
    period_summary.insert(0, "Period", period_name)
    summary_rows.append(period_summary)

keep = ["Period", "Volatility", "Long", "Short", "% Profitable Days", "Avg Gross Exposure", "Avg Net Exposure", "Avg Rebalances"]
summary_display = pd.concat(summary_rows, ignore_index=True)
summary_display = summary_display[[col for col in keep if col in summary_display.columns]]
st.dataframe(
    _format_metric_table(summary_display),
    **_stretch_width_kwargs(),
    hide_index=True,
)

st.subheader("Yearly Aggregate Metrics")
yearly_view = yearly.copy()
yearly_view["Year"] = yearly_view["date"].dt.year.astype(str)
yearly_view = _add_period_column(yearly_view)
columns = ["Year", "Period", "sharpe", "turnover", "capacity_score", "ret", "drawdown", "margin", "long_count", "short_count", "pnl"]
yearly_display = yearly_view[[col for col in columns if col in yearly_view.columns]]
st.dataframe(
    _format_metric_table(yearly_display, percent_cols=("ret", "turnover", "drawdown")),
    **_stretch_width_kwargs(),
    hide_index=True,
)

st.subheader("Yearly Aggregate Metrics — Pre/Post Warmup")
st.caption(
    f"Each metric computed with the first {br.SHARPE_WARMUP}-day Sharpe warmup included (pre) vs excluded (post), "
    "so the effect of the warmup is visible. Only the first year differs from the Yearly table above; later years are identical."
)
warmup_view = yearly_variants.copy()
warmup_view["Year"] = warmup_view["date"].dt.year.astype(str)
warmup_view = _add_period_column(warmup_view)
warmup_pairs = {
    "sharpe_full": "Sharpe (pre)", "sharpe_warmup": "Sharpe (post)",
    "turnover": "Turnover (pre)", "turnover_warmup": "Turnover (post)",
    "capacity_score_full": "Capacity (pre)", "capacity_score_warmup": "Capacity (post)",
    "ret": "Returns (pre)", "ret_warmup": "Returns (post)",
}
warmup_columns = ["Year", "Period"] + [col for col in warmup_pairs if col in warmup_view.columns]
warmup_display = warmup_view.sort_values("date")[warmup_columns].rename(columns=warmup_pairs).head(1)
st.dataframe(
    _format_metric_table(
        warmup_display,
        percent_cols=("Returns (pre)", "Returns (post)", "Turnover (pre)", "Turnover (post)"),
    ),
    **_stretch_width_kwargs(),
    hide_index=True,
)

st.subheader("Cumulative PnL vs Benchmarks")
cum_pnl_df = portfolio.set_index("date")[["cum_pnl"]].rename(columns={"cum_pnl": "Strategy"})
try:
    benchmarks = _benchmark_cum_pnl(tuple(pd.to_datetime(portfolio["date"])), float(book_size))
    cum_pnl_df = cum_pnl_df.join(benchmarks, how="left")
except Exception as exc:
    st.warning(f"Benchmark load failed: {exc}")
_multi_line_chart(
    cum_pnl_df,
    title="Cumulative PnL vs Benchmarks",
    ylabel="Cumulative PnL ($)",
    colors=SERIES_COLORS,
)

st.subheader("Daily Metrics")

DAILY_METRIC_DEFINITIONS = [
    {"label": "Cumulative PnL", "columns": ["cum_pnl"]},
    {"label": "Drawdown", "columns": ["drawdown"], "annotate_max_drawdown": True},
    {"label": "PnL", "columns": ["pnl"]},
    {"label": "Return", "columns": ["ret"]},
    {"label": "Sharpe", "columns": ["rolling_sharpe", "expanding_sharpe"], "ylabel": "Sharpe"},
    {"label": "Turnover", "columns": ["turnover"], "start": display_start},
    {"label": "Capacity score", "columns": ["rolling_capacity_score", "expanding_capacity_score", "capacity_score"], "ylabel": "Capacity score"},
]
for definition in DAILY_METRIC_DEFINITIONS:
    columns = [column for column in definition["columns"] if column in portfolio.columns]
    if not columns:
        continue
    plot_df = portfolio
    start = definition.get("start")
    if start is not None:
        plot_df = portfolio.loc[portfolio["date"].ge(start)]
    label = definition["label"]
    _period_line_chart(
        plot_df,
        columns,
        title=label,
        ylabel=definition.get("ylabel", columns[0]),
        percent_y=any(column in PERCENT_PLOT_COLUMNS for column in columns),
        annotate_max_drawdown=definition.get("annotate_max_drawdown", False),
    )

st.subheader("Portfolio Beta and Alpha vs SPY")
beta_window = st.slider("Rolling beta/alpha window", min_value=30, max_value=252, value=63, step=1)
try:
    benchmark_returns = _benchmark_returns(tuple(pd.to_datetime(portfolio["date"])), BENCHMARK_TICKERS)
    spy_returns = benchmark_returns["SPY"].reindex(pd.to_datetime(portfolio["date"]))
    strategy_returns = portfolio.set_index("date")["ret"]
    beta_alpha = br.rolling_beta_alpha(strategy_returns, spy_returns, window=beta_window)
    beta_col, alpha_col = st.columns(2)
    with beta_col:
        _multi_line_chart(beta_alpha[["beta"]], title="Rolling Beta vs SPY", ylabel="Beta")
    with alpha_col:
        _multi_line_chart(
            beta_alpha[["alpha_annualised"]],
            title="Rolling Annualised Alpha vs SPY",
            ylabel="Alpha",
            percent_y=True,
        )
except Exception as exc:
    st.warning(f"Beta/alpha computation failed: {exc}")

st.subheader("Performance Breakdown")
portfolio_breakdown = portfolio.copy()
perf_options = [col for col in ["cum_pnl", "drawdown", "ret", "pnl", "turnover", "rolling_sharpe", "expanding_sharpe", "rolling_capacity_score", "expanding_capacity_score", "capacity_score"] if col in portfolio_breakdown.columns]
perf_metric = st.selectbox("Breakdown metric", perf_options, index=0)
breakdown_df = portfolio_breakdown
if perf_metric == "turnover":
    breakdown_df = portfolio_breakdown.loc[portfolio_breakdown["date"].ge(display_start)]
_period_line_chart(
    breakdown_df,
    [perf_metric],
    title=perf_metric,
    ylabel=perf_metric,
    percent_y=perf_metric in PERCENT_PLOT_COLUMNS,
    annotate_max_drawdown=perf_metric == "drawdown",
)
monthly_display = monthly.copy()
monthly_display["Month"] = monthly_display["date"].dt.strftime("%Y-%m")
monthly_display = _add_period_column(monthly_display)
monthly_cols = ["Month", "Period", "ret", "pnl", "sharpe", "turnover", "capacity_score", "drawdown", "margin", "long_count", "short_count"]
st.dataframe(
    _format_metric_table(monthly_display[[col for col in monthly_cols if col in monthly_display.columns]], percent_cols=("ret", "turnover", "drawdown")),
    **_stretch_width_kwargs(),
    hide_index=True,
)

st.subheader("Portfolio Attribution")
group_options = [col for col in ["industry", "sector", "subindustry", "ticker", "side"] if col in asset_df.columns or col == "side"]
group_col = st.selectbox("Attribution group", group_options, index=0 if group_options else None)
attribution = _attribution(asset_df, horizon, group_col, book_size)
attribution = _filter_start(_coerce_period_dates(attribution, horizon=horizon), display_start)
attr_cols = [col for col in ["date", "group_name", "contribution_return", "contribution_pnl", "contribution_pct", "avg_abs_weight", "observations"] if col in attribution.columns]
period_attr = _add_period_column(attribution)
_period_attribution_charts(period_attr)
attr_display_cols = [col for col in ["date", "Period", "group_name", "contribution_return", "contribution_pnl", "contribution_pct", "avg_abs_weight", "observations"] if col in period_attr.columns]
st.dataframe(
    _format_metric_table(
        period_attr[attr_display_cols].sort_values("contribution_pnl", key=lambda values: values.abs(), ascending=False).head(25),
        percent_cols=("contribution_return", "contribution_pct", "avg_abs_weight"),
    ),
    **_stretch_width_kwargs(),
    hide_index=True,
)

st.subheader("Strategy Correlation by Grouping")
corr_group_options = [col for col in ["industry", "sector", "subindustry"] if col in asset_df.columns]
if corr_group_options:
    corr_group = st.selectbox("Correlation grouping", corr_group_options, index=0, key="corr_group")
    top_n = st.slider("Top groups by |contribution|", min_value=5, max_value=30, value=15, step=1, key="corr_top_n")
    group_matrix = attr.group_return_matrix(asset_df, corr_group, book_size=book_size)
    if group_matrix.empty:
        st.info(f"No PnL data to correlate by {corr_group}.")
    else:
        ranked = group_matrix.abs().sum().sort_values(ascending=False)
        kept = ranked.head(top_n).index
        _correlation_heatmap(group_matrix[kept])
else:
    st.info("No grouping columns available in the asset frame for correlation.")

st.subheader("Biggest Monthly Winners and Losers")
monthly_ticker_attr = _attribution(asset_df, "monthly", "ticker", book_size)
monthly_ticker_attr = _filter_start(_coerce_period_dates(monthly_ticker_attr, horizon="monthly"), display_start)
month_options = pd.to_datetime(monthly_ticker_attr["date"]).sort_values().unique()
selected_month = st.selectbox("Month", month_options, format_func=lambda value: pd.Timestamp(value).strftime("%Y-%m"))
month_attr = monthly_ticker_attr[pd.to_datetime(monthly_ticker_attr["date"]).eq(pd.Timestamp(selected_month))]
win_col, loss_col = st.columns(2)
with win_col:
    st.write("Winners")
    winners = month_attr.sort_values("contribution_pnl", ascending=False).head(10)
    winners = _add_period_column(winners)
    st.dataframe(_format_metric_table(winners[attr_display_cols], percent_cols=("contribution_return", "contribution_pct", "avg_abs_weight")), **_stretch_width_kwargs(), hide_index=True)
with loss_col:
    st.write("Losers")
    losers = month_attr.sort_values("contribution_pnl", ascending=True).head(10)
    losers = _add_period_column(losers)
    st.dataframe(_format_metric_table(losers[attr_display_cols], percent_cols=("contribution_return", "contribution_pct", "avg_abs_weight")), **_stretch_width_kwargs(), hide_index=True)

st.subheader("Portfolio Specifics")
date_values = asset_df["date"].sort_values().unique()
selected_date = st.selectbox("Date", date_values, index=len(date_values) - 1, format_func=lambda value: pd.Timestamp(value).strftime("%Y-%m-%d"))
snapshot = asset_df[asset_df["date"].eq(pd.Timestamp(selected_date))].copy()
snapshot["abs_pnl"] = snapshot["pnl"].abs() if "pnl" in snapshot.columns else 0
asset_cols = [
    "date", "ticker", "close", "ret", "signal", "weights", 
    "trade_size", "rebalanced",
    "pnl", "cum_pnl"
]
asset_cols = [col for col in asset_cols if col in snapshot.columns]
st.dataframe(
    _format_metric_table(snapshot.sort_values("abs_pnl", ascending=False)[asset_cols], percent_cols=("ret", "weights")),
    **_stretch_width_kwargs(),
    hide_index=True,
)

st.write("Selected-day attribution")
day_attr = _attribution(snapshot, "daily", group_col, book_size)
day_attr_cols = [col for col in attr_cols if col in day_attr.columns]
day_attr = _add_period_column(day_attr)
day_attr_cols = [col for col in ["date", "Period", "group_name", "contribution_return", "contribution_pnl", "contribution_pct", "avg_abs_weight", "observations"] if col in day_attr.columns]
st.dataframe(
    _format_metric_table(
        day_attr[day_attr_cols].sort_values("contribution_pnl", key=lambda values: values.abs(), ascending=False),
        percent_cols=("contribution_return", "contribution_pct", "avg_abs_weight"),
    ),
    **_stretch_width_kwargs(),
    hide_index=True,
)


