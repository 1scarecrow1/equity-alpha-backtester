import json
import numpy as np
import pytest
from equitybt.backtester.backtester import (
    load_backtest,
    run_backtest,
    save_backtest,
    strategy_run_name,
)
from equitybt.performance import attribution as pa
from equitybt.strategy.alpha import Alpha
from equitybt.strategy.operators import rank, ts_zscore
from equitybt.strategy.strategy import Strategy
from equitybt.trade_configs import Decay, Delay, Neutralization
from equitybt.validation import are_market_changes_valid, validate_backtest
from equitybt.variables.equities import returns

BOOK_SIZE = 20e6


@pytest.fixture(scope="module")
def run(panel, tickers):
    strategy = Strategy(
        alpha=Alpha(-rank(ts_zscore(returns, 20))),
        universe=tickers,
        delay=Delay.ONE,
        decay=Decay.WEEK,
        truncation=0.1,
        neutralization=Neutralization.SECTOR,
    )

    return run_backtest(strategy, data=panel, book_size=BOOK_SIZE, display=False)


def test_every_stage_of_a_completed_run_checks_out(run):
    reports = validate_backtest(run)
    failed = {stage: str(report) for stage, report in reports.items() if not report}

    assert not failed, failed
    assert are_market_changes_valid(run.df)


def test_a_run_produces_the_tables_and_the_summary(run):
    assert not run.portfolio.empty
    assert not run.yearly.empty
    assert list(run.summary.index) == ["value"]
    assert run.aggregate_metrics["sharpe"] == run.summary.loc["value", "Sharpe"]


def test_the_book_is_sized_by_the_book_size(run):
    held = run.df["exposure"].abs().groupby(run.df["date"]).sum()

    assert held.max() <= BOOK_SIZE * (1 + 1e-9)
    assert held.iloc[-1] > 0


def test_the_run_name_changes_with_every_setting(panel, tickers):
    def name(**changes):
        settings = dict(
            alpha=Alpha(-rank(ts_zscore(returns, 20))),
            universe=tickers,
            delay=Delay.ONE,
            decay=Decay.WEEK,
            truncation=0.1,
            neutralization=Neutralization.SECTOR,
        )
        return strategy_run_name(Strategy(**{**settings, **changes}))

    assert name() == name()
    assert name(delay=Delay.ZERO) != name()
    assert name(decay=Decay.MONTH) != name()
    assert name(truncation=0.2) != name()
    assert name(neutralization=Neutralization.MARKET) != name()


def test_a_run_folder_round_trips(run, tmp_path):
    save_backtest(run, tmp_path / "reversal")
    written = sorted(path.name for path in (tmp_path / "reversal").iterdir())

    assert written == [
        "compact_summary.html",
        "config.json",
        "df.parquet",
        "portfolio.parquet",
        "source_data.parquet",
        "summary.json",
        "yearly.parquet",
    ]

    reloaded = load_backtest(tmp_path / "reversal")

    assert reloaded.results["book_size"] == run.results["book_size"]
    assert reloaded.delay is run.delay
    assert reloaded.neutralization is run.neutralization
    assert len(reloaded.portfolio) == len(run.portfolio)
    assert reloaded.summary.loc["value", "Sharpe"] == run.summary.loc["value", "Sharpe"]


def test_the_config_records_the_formula_it_ran(run, tmp_path):
    save_backtest(run, tmp_path / "reversal")
    config = json.loads((tmp_path / "reversal" / "config.json").read_text())

    assert config["signals"]["signal"] == "(-rank(ts_zscore(returns, 20)))"
    assert config["book_size"] == BOOK_SIZE


def test_contributions_reconcile_to_the_period_pnl(run):
    attribution = pa.prepare_attribution(run.df, horizon="yearly", book_size=BOOK_SIZE)
    last = attribution[attribution["date"] == attribution["date"].max()]

    assert last["contribution_pct"].sum() == pytest.approx(1.0)
    assert last["contribution_pnl"].sum() == pytest.approx(last["period_pnl"].iloc[0])


def test_sector_attribution_covers_every_sector(run):
    attribution = pa.prepare_attribution(
        run.df, horizon="yearly", group_cols=("sector",), book_size=BOOK_SIZE
    )

    assert set(attribution["sector"]) == set(run.df["sector"])


def test_the_sector_return_matrix_is_one_column_per_sector(run):
    matrix = pa.group_return_matrix(run.df, "sector", book_size=BOOK_SIZE)

    assert sorted(matrix.columns) == sorted(run.df["sector"].unique())
    assert np.isfinite(matrix.to_numpy()).all()


def test_the_compact_summary_is_one_interactive_page(run, tmp_path):
    save_backtest(run, tmp_path / "reversal")
    html = (tmp_path / "reversal" / "compact_summary.html").read_text(encoding="utf-8")

    assert html.startswith("<!doctype html>")
    assert "plotly" in html
    assert "hovertemplate" in html
    assert html.count("plotly-graph-div") == 7
    assert "Cumulative PnL" in html and "PnL by sector" in html
    assert str(run.summary.loc["value", "Sharpe"]) in html
