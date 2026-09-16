import equitybt.performance.attribution as attr
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

LINE_COLOR = "#1f77b4"
NEGATIVE_COLOR = "#d62728"
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

STYLE = """
body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; margin: 0 auto;
       max-width: 1100px; padding: 32px 24px 64px; color: #111; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 15px; font-weight: 600; margin: 32px 0 8px; }
p.sub { color: #555; margin: 0 0 24px; font-size: 13px; }
table { border-collapse: collapse; font-size: 13px; width: 100%; }
th, td { padding: 6px 10px; text-align: right; border-bottom: 1px solid #eee; }
th:first-child, td:first-child { text-align: left; }
thead th { border-bottom: 1px solid #ccc; font-weight: 600; }
dl { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px 24px; font-size: 13px; margin: 0; }
dt { color: #555; }
dd { margin: 0 0 8px; font-variant-numeric: tabular-nums; font-weight: 600; }
"""


def _line(df: pd.DataFrame, columns, title, percent_y=False) -> go.Figure:
    figure = go.Figure()
    hover = "%{y:.2%}" if percent_y else "%{y:,.2f}"
    for index, column in enumerate(columns):
        if column not in df.columns:
            continue
        figure.add_trace(
            go.Scatter(
                x=df["date"],
                y=df[column],
                mode="lines",
                name=column.replace("_", " "),
                line={"color": PALETTE[index % len(PALETTE)], "width": 2},
                hovertemplate="%{x|%Y-%m-%d}<br>" + hover + "<extra>" + column + "</extra>",
            )
        )
    figure.update_layout(
        title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 14}},
        height=320,
        margin={"l": 56, "r": 24, "t": 44, "b": 40},
        hovermode="x unified",
        showlegend=len(columns) > 1,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        plot_bgcolor="#fff",
    )
    figure.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    figure.update_yaxes(showgrid=True, gridcolor="#f0f0f0", tickformat=".1%" if percent_y else None)
    return figure


def _bars(values: pd.Series, title: str) -> go.Figure:
    colors = [NEGATIVE_COLOR if value < 0 else LINE_COLOR for value in values]
    figure = go.Figure(
        go.Bar(
            x=values.to_numpy(),
            y=values.index.astype(str),
            orientation="h",
            marker={"color": colors},
            hovertemplate="%{y}<br>%{x:,.0f}<extra></extra>",
        )
    )
    figure.update_layout(
        title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 14}},
        height=max(240, 24 * len(values) + 80),
        margin={"l": 140, "r": 24, "t": 44, "b": 40},
        plot_bgcolor="#fff",
    )
    figure.update_xaxes(showgrid=True, gridcolor="#f0f0f0", zerolinecolor="#999")
    return figure


def _table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, border=0, justify="right", float_format=lambda v: f"{v:,.2f}")


def _definition_list(summary: pd.Series) -> str:
    items = "".join(f"<dt>{name}</dt><dd>{value}</dd>" for name, value in summary.items())
    return f"<dl>{items}</dl>"


def write_tearsheet(strategy, path: str | Path, book_size: float) -> Path:
    path = Path(path)
    portfolio = strategy.portfolio.reset_index() if "date" not in strategy.portfolio else strategy.portfolio
    portfolio = portfolio.copy()
    portfolio["date"] = pd.to_datetime(portfolio["date"])

    figures = [
        _line(portfolio, ["cum_pnl"], "Cumulative PnL"),
        _line(portfolio, ["drawdown"], "Drawdown", percent_y=True),
        _line(portfolio, ["rolling_sharpe", "expanding_sharpe"], "Sharpe"),
        _line(portfolio, ["turnover"], "Turnover", percent_y=True),
        _line(portfolio, ["gross_exposure", "net_exposure"], "Exposure", percent_y=True),
    ]

    if "sector" in strategy.df.columns:
        sectors = attr.prepare_attribution(
            strategy.df, horizon="full", group_cols=("sector",), book_size=book_size
        )
        totals = sectors.groupby("sector")["contribution_pnl"].sum().sort_values()
        figures.append(_bars(totals, "PnL by sector"))

    tickers = attr.prepare_attribution(strategy.df, horizon="full", book_size=book_size)
    by_ticker = tickers.groupby("ticker")["contribution_pnl"].sum().sort_values()
    figures.append(_bars(pd.concat([by_ticker.head(10), by_ticker.tail(10)]), "Best and worst names"))

    yearly = strategy.yearly.reset_index() if "date" not in strategy.yearly else strategy.yearly
    yearly = yearly.copy()
    yearly["Year"] = pd.to_datetime(yearly["date"]).dt.year
    columns = [c for c in ("Year", "ret", "sharpe", "turnover", "capacity_score", "drawdown", "pnl") if c in yearly.columns]

    charts = "".join(
        figure.to_html(include_plotlyjs="cdn" if index == 0 else False, full_html=False)
        for index, figure in enumerate(figures)
    )
    signal = next(iter(strategy.signals.values()))
    first, last = portfolio["date"].min().date(), portfolio["date"].max().date()

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{signal}</title>
<style>{STYLE}</style></head><body>
<h1>{signal}</h1>
<p class="sub">{first} to {last} &middot; book {book_size:,.0f} &middot; delay {strategy.delay.value}
 &middot; decay {int(strategy.decay)} &middot; {strategy.neutralization.value.lower()} neutral
 &middot; truncation {strategy.truncation}</p>
<h2>Summary</h2>
{_definition_list(strategy.summary.loc["value"])}
<h2>Performance</h2>
{charts}
<h2>By year</h2>
{_table(yearly[columns])}
</body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(html.encode("utf-8"))

    return path
