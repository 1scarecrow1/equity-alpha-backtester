# equity-alpha-backtester

End-to-end equity alpha research on daily data. You can write a signal as a formula, turn
it into a cross-sectionally neutral book, simulate it with an explicit trading delay, and
read the results in a dashboard or tearsheet.

```
formula  ->  signal  ->  weights  ->  exposures  ->  PnL  ->  report
  DSL       constraints   neutralize   book size    delay    tearsheet
            exit rules    decay        turnover     mark      dashboard
                          truncate
```

## Python setup

Use Python >=3.11.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,dashboard]"
```

## Quickstart

```powershell
python examples/yfinance_demo.py
```

The backtester is demonstrated with yfinance. The demo loads daily bars for 40 large caps, 
computes a 20-day reversal signal and builds a sector-neutral book. It then runs the strategy with
1-day delay, prints the summary and the sector attribution, runs the same alpha with 0 delay for 
comparison, and writes everything to `runs/reversal_d1/`. You can change the book size and return
calculation methods.

```
With a book size of 20M

        Total PnL  Annualised return  Sharpe  Turnover  Capacity score  Drawdown
value  6731272.82              8.45%    0.67    67.75%            0.24     7.87%

Sharpe at delay 1: 0.67
Sharpe at delay 0: 0.86
```

Other alphas and settings:

```powershell
python examples/yfinance_demo.py --alpha momentum
python examples/yfinance_demo.py --alpha liquid_reversal --delay 0
python examples/yfinance_demo.py --offline
```

Bars are cached under `.cache/bars/`, 1 parquet file per ticker, so a second run does
not download anything. `--offline` reads the cache and refuses to download.

## Writing an alpha

A formula is built from field objects and operators. It records which fields are required,
so the loader knows what to download.

```python
from equitybt.strategy.alpha import Alpha
from equitybt.strategy.operators import rank, trade_when, ts_std, ts_zscore
from equitybt.variables.equities import adv20, dollar_volume, returns

alpha = Alpha(
    signal=-rank(ts_zscore(returns, 20)),
    constraints={"liquid": dollar_volume > adv20},
)

print(alpha.signals["signal"])   # (-rank(ts_zscore(returns, 20)))
print(alpha.fields)              # [adv20, dollar_volume, returns]
```
You can also specify triggers and exit conditions.

```python
alpha = Alpha(signal=trade_when(dollar_volume > adv20, -rank(ts_zscore(returns, 20)), -1))
```

Time series and cross-sectional operators are in `strategy/operators.py`. 

## Running a backtest

```python
from equitybt.backtester.backtester import run_backtest, save_backtest
from equitybt.data.yfinance_loader import add_universe_mask, load_variables
from equitybt.strategy.strategy import Strategy
from equitybt.trade_configs import Decay, Delay, Neutralization, Universe

data = load_variables(
    list(alpha.fields) + ["sector", "adv20"],
    tickers=["AAPL", "MSFT", "JPM", "XOM", "JNJ", "PG"],
    start="2021-01-01",
    end="2024-12-31",
)
data = add_universe_mask(data, Universe.TOP500_PIT)

strategy = Strategy(
    alpha=alpha,
    universe=["AAPL", "MSFT", "JPM", "XOM", "JNJ", "PG"],
    delay=Delay.ONE,
    decay=Decay.WEEK,
    truncation=0.05,
    neutralization=Neutralization.SECTOR,
)

strategy = run_backtest(strategy, data=data, book_size=20e6)
save_backtest(strategy, "runs/my_alpha")
```

A run folder holds `config.json`, `summary.json`, `tearsheet.html`, and parquet tables for
the source data, the per-name panel, the daily portfolio and the yearly aggregates. 

## Dashboard

```powershell
python -m streamlit run src/equitybt/performance/dashboard.py -- runs/my_alpha
```

Or from Python, which saves the run first:

```python
from equitybt.backtester.backtester import launch_dashboard

launch_dashboard(strategy)
```

The dashboard reads a run folder and shows the summary, per-period and yearly aggregates,
the portfolio breakdown, cumulative PnL against benchmarks, attribution by ticker, sector
or side, a group correlation matrix, and a per-day snapshot. 



## Licence

MIT. See `LICENSE`.
