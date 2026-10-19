# Momentum Strategy Backtest

## Start the app

Double-click **Start Backtest App.bat**. On first launch, it creates a local `.venv`, installs the packages in `requirements.txt`, and opens the app in your browser. Later launches reuse that environment and start the app directly.

The app reads the saved files under `market_data/`, then attempts an incremental Yahoo Finance update through yesterday for NIFTY 500, S&P 500, and TOPIX 500. It only requests the tickers already present in each universe's saved close-price file, and merges new adjusted OHLC bars and benchmark closes into the existing CSVs. Internet access is needed for initial package installation and data updates.

## Backtest options and reports

Choose a universe, date range, regime filter, allowed regimes, rebalance frequency, holding count, stop-loss settings, fee toggle, and starting capital. The end date defaults to yesterday. The report shows strategy metrics and equity curves, and provides CSV downloads for trade log, top five trade frequency, strategy curve, benchmark curve, monthly returns, backtest results, and regime.

The strategy implementation in `strategy_engine.py` is extracted from definition cells in `momentum_updated_regime_exact_tested_updated.ipynb`; notebook download and execution cells are excluded. Existing `backtest_output/` files are retained, and new runs save the notebook's CSV reports there.

## US forward test

`.github/workflows/us-forward-test.yml` schedules `forward_test.py` every 30 minutes across the US session. The script checks the NYSE calendar (including holidays and daylight saving time), ranks the saved S&P 500 symbols using 12-month momentum, downloads current intraday Yahoo Finance prices for the top five and current holdings, and saves paper portfolio snapshots in `forward_state.json`. It simulates a rebalance on the first run of each session, with one second of delay and 5 basis points of adverse slippage per fill. It does not send brokerage orders.

GitHub Actions must be enabled and the workflow must exist on the repository's default branch. The workflow needs permission to push commits so it can persist `forward_state.json` between ephemeral runners. GitHub cron runs are approximate and can be delayed; Yahoo Finance prices may also be delayed or unavailable, so this is an operational forward-test simulation rather than exchange-grade execution data. The app's Forward test tab reports the current NYSE market state and displays the latest saved portfolio state.

## Desktop shortcut

The **Momentum Strategy Backtest** shortcut opens **Start Backtest App.bat** from this folder.

## Data and limitations

Yahoo Finance can return no bar for weekends, exchange holidays, or a delayed data publication. The app reports the latest saved date and continues with available saved data when an update fails. Only the three universes with existing OHLC and benchmark files are enabled. Prices use Yahoo Finance's `auto_adjust=True` convention, matching the notebook's downloader.
