import equitybt.backtester.returns as br
import equitybt.portfolio.weights as pw
import equitybt.strategy.build_alpha as ba
import hashlib
import json
import numbers
import pandas as pd
import re
import subprocess
import sys
from equitybt.dataexpression.dataexpression import DataExpression
from equitybt.performance.tearsheet import write_tearsheet
from equitybt.portfolio.weights import SIGNAL_COL, WEIGHT_COL
from equitybt.strategy.alpha import Alpha
from equitybt.strategy.strategy import Strategy
from equitybt.trade_configs import Delay, Neutralization, TradingPeriod
from equitybt.validation import are_market_changes_valid, validate_backtest
from equitybt.variables.equities import extract_known_variables
from pathlib import Path


def run_backtest(
    strategy: Strategy | Alpha,
    data: pd.DataFrame | None = None,
    signal_col: str = SIGNAL_COL,
    weight_col: str = WEIGHT_COL,
    trading_period: TradingPeriod | None = None,
    book_size: float = 1.0,
    display: bool = True,
):
    if isinstance(book_size, bool) or not (isinstance(book_size, numbers.Real) and book_size > 0):
        raise ValueError("Book size must be a positive number")


    if isinstance(strategy, Alpha):
        strategy = Strategy(alpha=strategy)

    source_data = data.copy() if data is not None else ba.build_data_from_signals(strategy, trading_period=trading_period)
    df = source_data.copy()

    df = ba.compute_signal(df, strategy)
    if signal_col not in df.columns:
        first_signal = next(iter(strategy.signals))
        df[signal_col] = df[first_signal]

    df = pw.compute_weights(df, strategy)
    df = br.get_portfolio_allocations(
        df, weight_col=weight_col, delay=strategy.delay, book_size=book_size
    )
    df = br.asset_returns(df)
    portfolio = br.portfolio_daily_returns(df, weight_col=weight_col, book_size=book_size)
    yearly = br.portfolio_aggregate_returns(
        portfolio, horizon="yearly", delay=strategy.delay, book_size=book_size
    )
    summary = br.portfolio_summary(portfolio, delay=strategy.delay, book_size=book_size)

    strategy.source_data = source_data
    strategy.df = df
    strategy.portfolio = portfolio
    strategy.yearly = yearly
    strategy.summary = summary

    strategy.weights = df[weight_col]
    if not summary.empty:
        values = summary.loc["value"]
        strategy.aggregate_metrics = {
            "pnl": values["Total PnL"],
            "ret": values["Annualised return"],
            "sharpe": values["Sharpe"],
            "turnover": values["Turnover"],
            "capacity_score": values["Capacity score"],
            "drawdown": values["Drawdown"],
            "margin": values["Margin"],
        }
    strategy.results["book_size"] = book_size
    strategy.results["source_data"] = source_data
    strategy.results["portfolio"] = portfolio
    strategy.results["yearly"] = yearly
    strategy.results["summary"] = summary

    if display:
        display_backtest(strategy)

    return strategy


def display_backtest(strategy: Strategy) -> bool:
    attributes = {
        "instrument_type": strategy.instrument_type,
        "region": strategy.region,
        "universe": strategy.universe,
        "delay": strategy.delay,
        "decay": strategy.decay,
        "neutralization": strategy.neutralization,
        "truncation": strategy.truncation,
    }
    print("\nStrategy attributes")
    for name, value in attributes.items():
        print(f"  {name:<16}: {getattr(value, 'value', value)}")

    print("\nExpressions")
    for name, expression in strategy.signals.items():
        print(f"  signal[{name}]: {expression}")
    for name, expression in strategy.constraints.items():
        print(f"  constraint[{name}]: {expression}")
    if getattr(strategy, "has_exit_condition", False):
        print(f"  exit_condition: {strategy.exit_condition}")

    print("\nBacktest checks")
    reports = validate_backtest(strategy)
    print(are_market_changes_valid(strategy.df))
    for report in reports.values():
        print(report)
    failed = [stage for stage, report in reports.items() if not report]
    if failed:
        print(f"[warn] checks failed for: {', '.join(failed)}")

    if isinstance(strategy.summary, pd.DataFrame) and not strategy.summary.empty:
        print("\nSummary")
        print(strategy.summary.to_string())

    return not failed

_PARAM_KEYS = (
    "instrument_type",
    "region",
    "universe",
    "delay",
    "decay",
    "neutralization",
    "truncation",
)

def _slug(value) -> str:
    text = str(getattr(value, "value", value)).lower()
    return re.sub(r"[^a-z0-9.-]+", "-", text).strip("-")


def _signature_text(strategy: Strategy) -> str:
    """Canonical, ordered text representation of every parameter that defines a run."""
    parts: list[str] = []
    for key in _PARAM_KEYS:
        parts.append(f"{key}={getattr(strategy, key, None)!s}")
    for name in sorted(strategy.signals):
        parts.append(f"signal::{name}={strategy.signals[name]}")
    for name in sorted(strategy.constraints):
        parts.append(f"constraint::{name}={strategy.constraints[name]}")
    return "|".join(parts)

def strategy_run_name(strategy: Strategy, *, signature_length: int = 8) -> str:
    signals = strategy.signals
    sgn_val = next(iter(signals.values()))
    sgn_key = next(iter(signals.keys()))
    signal_text = str(sgn_val) if str(sgn_key) == "signal" else str(sgn_key)

    readable = [
        _slug(signal_text),
        _slug(strategy.universe),
        _slug(strategy.neutralization),
        f"d{_slug(strategy.delay)}",
        f"decay{_slug(strategy.decay)}",
        f"trunc{_slug(strategy.truncation)}",
    ]
    signature = _signature_text(strategy)
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:signature_length]
    return "_".join(filter(None, readable + [digest]))


def default_run_path(strategy: Strategy, root: str | Path = "runs") -> Path:
    return Path(root) / strategy_run_name(strategy)


RUN_TABLES = ("source_data", "df", "portfolio", "yearly")


def save_backtest(strategy: Strategy, path: str | Path) -> Strategy:
    """Write a run as parquet tables, with the settings and summary as JSON."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    config = {key: str(getattr(strategy, key, None)) for key in _PARAM_KEYS}
    config["universe"] = list(strategy.universe)
    config["universe_size"] = strategy.universe_size
    config["book_size"] = strategy.results["book_size"]
    config["signals"] = {name: str(expression) for name, expression in strategy.signals.items()}
    config["constraints"] = {name: str(expression) for name, expression in strategy.constraints.items()}
    config["exit_condition"] = str(strategy.exit_condition)
    (path / "config.json").write_text(json.dumps(config, indent=2))
    (path / "summary.json").write_text(json.dumps(strategy.summary.loc["value"].to_dict(), indent=2))

    for name in RUN_TABLES:
        table = getattr(strategy, name, None)
        if table is None or table.empty:
            continue
        table.reset_index().to_parquet(path / f"{name}.parquet", index=False)

    write_tearsheet(strategy, path / "tearsheet.html", strategy.results["book_size"])
    strategy.run_path = str(path)
    return strategy


def _expression(text: str) -> DataExpression:
    """A rendered expression read back from a run folder, with its fields recovered."""
    return DataExpression(text, set(extract_known_variables(text)))


def load_backtest(path: str | Path, root: str | Path = "runs") -> Strategy:
    """Rebuild a Strategy from a run folder"""
    path = Path(path)
    if not path.exists():
        path = Path(root) / path

    config = json.loads((path / "config.json").read_text())
    exit_condition = config["exit_condition"]
    strategy = Strategy(
        alpha=Alpha(
            signals={name: _expression(text) for name, text in config["signals"].items()},
            constraints={name: _expression(text) for name, text in config["constraints"].items()},
            exit_condition=_expression(exit_condition) if exit_condition != "-1" else -1,
        ),
        universe=config["universe"],
        delay=Delay[config["delay"].rsplit(".", 1)[-1]],
        decay=int(config["decay"]),
        truncation=float(config["truncation"]),
        neutralization=Neutralization(config["neutralization"].rsplit(".", 1)[-1]),
    )

    for name in RUN_TABLES:
        table_path = path / f"{name}.parquet"
        if table_path.exists():
            setattr(strategy, name, pd.read_parquet(table_path))

    strategy.results["book_size"] = float(config["book_size"])
    summary = json.loads((path / "summary.json").read_text())
    strategy.summary = pd.DataFrame(summary, index=["value"])
    strategy.aggregate_metrics = dict(summary)
    strategy.run_path = str(path)

    return strategy


def launch_dashboard(strategy: Strategy | str | Path, *, path: str | Path | None = None):
    if isinstance(strategy, Strategy):
        save_backtest(strategy, Path(path or getattr(strategy, "run_path", default_run_path(strategy))))
        dashboard_path = strategy.run_path
    else:
        dashboard_path = str(strategy)

    return subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(Path(__file__).parent.parent / "performance" / "dashboard.py"),
            "--",
            dashboard_path,
        ],
        check=False,
    )
