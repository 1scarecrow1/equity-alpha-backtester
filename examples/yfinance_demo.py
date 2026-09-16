"""
End-to-end run on public daily data: download bars, compute a signal, build a book,
simulate it, and write the results to a run folder.

    python examples/yfinance_demo.py
    python examples/yfinance_demo.py --offline
"""
import argparse
import equitybt.performance.attribution as pa
from equitybt.backtester.backtester import run_backtest, save_backtest
from equitybt.data.yfinance_loader import add_universe_mask, get_tickers, load_variables
from equitybt.portfolio.weights import SIGNAL_COL
from equitybt.strategy import operators as ops
from equitybt.strategy.alpha import Alpha
from equitybt.strategy.strategy import Strategy
from equitybt.trade_configs import Decay, Delay, Neutralization, TradingPeriod, Universe
from equitybt.utils import timer
from equitybt.variables import equities as v

PERIOD = TradingPeriod("2021-01-01", "2024-12-31", "demo")
BOOK_SIZE = 20e6

# Liquid large caps across several sectors, so sector neutralisation has something to
# remove. Membership is not point in time: these are names that exist today, which
# leaves the results survivorship biased. See the limitations section of the README.
TICKERS = [
    "AAPL", "MSFT", "NVDA", "AVGO", "CSCO", "ORCL", "CRM", "AMD",
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP", "C",
    "JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY", "TMO", "ABT",
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "VLO",
    "PG", "KO", "PEP", "WMT", "COST", "MCD", "NKE", "HD",
]

ALPHAS = {
    "reversal": -ops.rank(ops.ts_zscore(v.returns, 20)),
    "momentum": ops.rank(ops.ts_delay(v.close, 21) / ops.ts_delay(v.close, 252)),
    "low_volatility": -ops.rank(ops.ts_std(v.returns, 60)),
    "liquid_reversal": ops.trade_when(
        v.dollar_volume > v.adv20, -ops.rank(ops.ts_zscore(v.returns, 20)), -1
    ),
}


@timer
def load(alpha: Alpha, offline: bool):
    tickers = get_tickers(TICKERS)
    data = load_variables(
        # sector for neutralisation, adv20 for the liquidity ranking the mask uses
        list(alpha.fields) + ["sector", "adv20"],
        tickers=tickers,
        start=PERIOD.start,
        end=PERIOD.end,
        offline=offline,
    )

    return add_universe_mask(data, Universe.TOP500_PIT)


@timer
def simulate(alpha: Alpha, data, delay: Delay):
    strategy = Strategy(
        alpha=alpha,
        universe=TICKERS,
        delay=delay,
        decay=Decay.WEEK,
        truncation=0.05,
        neutralization=Neutralization.SECTOR,
    )

    return run_backtest(
        strategy, data=data.copy(), trading_period=PERIOD, book_size=BOOK_SIZE, display=False
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", choices=sorted(ALPHAS), default="reversal")
    parser.add_argument("--delay", type=int, choices=(0, 1), default=1)
    parser.add_argument("--offline", action="store_true", help="read the bar cache only")
    args = parser.parse_args()

    alpha = Alpha(ALPHAS[args.alpha])
    print(f"Alpha: {alpha.signals[SIGNAL_COL]}")
    print(f"Fields: {', '.join(field.name for field in alpha.fields)}")

    data = load(alpha, args.offline)
    coverage = data.groupby("ticker")["close"].count()
    print(f"\n{len(coverage)}/{len(TICKERS)} tickers, {data['date'].nunique()} dates")
    if len(coverage) < len(TICKERS):
        print(f"No data for: {', '.join(sorted(set(TICKERS) - set(coverage.index)))}")

    strategy = simulate(alpha, data, Delay(args.delay))
    print("\nSummary")
    print(strategy.summary.to_string())

    sectors = pa.prepare_attribution(
        strategy.df,
        horizon="full",
        group_cols=("sector",),
        book_size=BOOK_SIZE,
    )
    totals = sectors.groupby("sector")["contribution_pnl"].sum().sort_values(ascending=False)
    print("\nPnL by sector")
    print((totals / totals.abs().sum()).map(lambda share: f"{share:+.1%}").to_string())

    # The same alpha at both delays: the gap is what one session of lag costs.
    other = Delay.ZERO if args.delay == 1 else Delay.ONE
    comparison = simulate(alpha, data, other)
    print(f"\nSharpe at delay {args.delay}: {strategy.summary.loc['value', 'Sharpe']}")
    print(f"Sharpe at delay {other.value}: {comparison.summary.loc['value', 'Sharpe']}")

    save_backtest(strategy, f"runs/{args.alpha}_d{args.delay}")
    print(f"\nWritten to {strategy.run_path}")


if __name__ == "__main__":
    main()
