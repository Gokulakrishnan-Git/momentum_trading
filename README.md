# Momentum Strategy Backtest

## Start the app

Double-click **Start Backtest App.bat**. On first launch, it creates a local `.venv`, installs the packages in `requirements.txt`, and opens the app in your browser. Later launches reuse that environment and start the app directly.

The app reads the saved files under `market_data/`, then attempts an incremental Yahoo Finance update through yesterday for NIFTY 500, S&P 500, and TOPIX 500. It only requests the tickers already present in each universe's saved close-price file, and merges new adjusted OHLC bars and benchmark closes into the existing CSVs. Internet access is needed for initial package installation and data updates.

## Backtest options and reports

Choose a universe, date range, regime filter, allowed regimes, rebalance frequency, holding count, stop-loss settings, fee toggle, and starting capital. The end date defaults to yesterday. The report shows strategy metrics and equity curves, and provides CSV downloads for trade log, top five trade frequency, strategy curve, benchmark curve, monthly returns, backtest results, and regime.

The strategy implementation in `strategy_engine.py` is extracted from definition cells in `momentum_updated_regime_exact_tested_updated.ipynb`; notebook download and execution cells are excluded. Existing `backtest_output/` files are retained, and new runs save the notebook's CSV reports there.

## Forward paper tests

The Forward test tab lets you freeze multiple independent tests using S&P 500, NIFTY 500, or TOPIX 500 data. Each test saves its own starting capital, rebalance frequency, holding count, regime filter, stop-loss settings, fee toggle, simulated latency, and slippage in `forward_test.json`. Freezing commits and pushes the configuration to GitHub; a push-triggered Actions run starts it at the next open session. Repeated tests with different parameters keep separate cash, positions, fills, portfolio value, and P&L in `forward_state.json`.

Forward signals use the same `MomentumStrategy`, `ScoreWeightAllocator`, `MarketRegime`, stop managers, and market fee profiles as the backtester. The workflow updates daily OHLC history in memory through yfinance, calculates the selected strategy at the market open, applies sells before buys, then holds until the next configured rebalance date. It monitors enabled stop losses during the session and marks the portfolio to current Yahoo Finance prices every 30 minutes. Simulated fills apply the frozen delay and adverse slippage; no brokerage orders are sent.

GitHub Actions must be enabled, and the workflow must exist on the repository's default branch. The workflow commits forward state between ephemeral runners. GitHub cron times are approximate, and Yahoo Finance can delay or omit prices, so the results are a paper simulation rather than exchange-grade fills. The market calendars account for local exchange sessions, holidays, and the JPX lunch break.

## Desktop shortcut

The **Momentum Strategy Backtest** shortcut opens **Start Backtest App.bat** from this folder.

## Data and limitations

Yahoo Finance can return no bar for weekends, exchange holidays, or a delayed data publication. The app reports the latest saved date and continues with available saved data when an update fails. Only the three universes with existing OHLC and benchmark files are enabled. Prices use Yahoo Finance's `auto_adjust=True` convention, matching the notebook's downloader.
