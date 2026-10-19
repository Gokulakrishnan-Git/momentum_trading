"""Shared, paper-only forward testing using the backtest signal engine."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import time
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf

from strategy_engine import (
    ExecutionEngine,
    FixedPercentStopLoss,
    MarketRegime,
    MomentumStrategy,
    REGIME_CONFIG,
    ScoreWeightAllocator,
    TrailingPercentStopLoss,
    get_ibkr_fee_profile,
    get_rebalance_dates,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "market_data"
CONFIG_PATH = ROOT / "forward_test.json"
STATE_PATH = ROOT / "forward_state.json"
UNIVERSES = {
    "SP500": {"calendar": "NYSE", "timezone": "America/New_York", "market": "US", "currency": "USD", "index": "^GSPC"},
    "NIFTY500": {"calendar": "NSE", "timezone": "Asia/Kolkata", "market": "INDIA", "currency": "INR", "index": "^CRSLDX"},
    "TOPIX500": {"calendar": "JPX", "timezone": "Asia/Tokyo", "market": "JAPAN", "currency": "JPY", "index": "1306.T"},
}


def market_state(now: datetime, universe: str = "SP500") -> tuple[bool, str]:
    cfg = UNIVERSES[universe]
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    local = now.astimezone(ZoneInfo(cfg["timezone"]))
    display_tz = ZoneInfo("Asia/Kolkata")
    calendar = mcal.get_calendar(cfg["calendar"])
    session = calendar.schedule(start_date=local.date(), end_date=local.date())
    if session.empty:
        return False, f"{universe}: market closed today (weekend or exchange holiday)."

    row = session.iloc[0]
    opening = pd.Timestamp(row["market_open"]).to_pydatetime().astimezone(local.tzinfo)
    closing = pd.Timestamp(row["market_close"]).to_pydatetime().astimezone(local.tzinfo)
    if local < opening:
        return False, f"{universe}: market closed. Opens at {opening.astimezone(display_tz):%H:%M IST}."
    if local >= closing:
        return False, f"{universe}: market closed. Closed at {closing.astimezone(display_tz):%H:%M IST}."
    break_start, break_end = row.get("break_start"), row.get("break_end")
    if pd.notna(break_start) and pd.notna(break_end):
        pause_start = pd.Timestamp(break_start).to_pydatetime().astimezone(local.tzinfo)
        pause_end = pd.Timestamp(break_end).to_pydatetime().astimezone(local.tzinfo)
        if pause_start <= local < pause_end:
            return False, f"{universe}: market is on its scheduled break until {pause_end.astimezone(display_tz):%H:%M IST}."
    opening_ist = opening.astimezone(display_tz)
    closing_ist = closing.astimezone(display_tz)
    return True, f"{universe}: market open · {opening_ist:%H:%M}–{closing_ist:%H:%M} IST."


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    idx = pd.to_datetime(frame.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    frame.index = idx.normalize()
    frame = frame.sort_index()
    return frame.loc[~frame.index.duplicated(keep="last")]


def _read_frame(path: Path) -> pd.DataFrame:
    return _clean_frame(pd.read_csv(path, index_col=0, parse_dates=True))


def _field_frame(bars: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
    if isinstance(bars.columns, pd.MultiIndex):
        if field in bars.columns.get_level_values(0):
            result = bars[field]
        else:
            result = bars.xs(field, axis=1, level=1)
    else:
        result = bars[[field]].copy()
        if len(tickers) == 1:
            result.columns = tickers
    if isinstance(result, pd.Series):
        result = result.to_frame(name=tickers[0])
    result.columns = result.columns.astype(str)
    return _clean_frame(result)


def _merge(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    columns = old.columns.union(new.columns)
    combined = pd.concat([old.reindex(columns=columns), new.reindex(columns=columns)])
    return combined.sort_index().loc[lambda frame: ~frame.index.duplicated(keep="last")]


def load_market_history(universe: str) -> dict:
    """Load the committed history and append fresh Yahoo daily bars in memory."""
    cfg = UNIVERSES[universe]
    kinds = ("open", "high", "low", "close")
    frames = {kind: _read_frame(DATA_DIR / f"{universe}_{kind}.csv") for kind in kinds}
    tickers = frames["close"].columns.astype(str).tolist()
    if not tickers:
        raise RuntimeError(f"No saved tickers exist for {universe}.")

    latest = min(frame.index.max() for frame in frames.values() if not frame.empty)
    local_today = datetime.now(ZoneInfo(cfg["timezone"])).date()
    start = (latest + pd.Timedelta(days=1)).date()
    end = local_today + timedelta(days=1)
    if start <= local_today:
        bars = yf.download(tickers=tickers, start=start.isoformat(), end=end.isoformat(),
                           interval="1d", auto_adjust=True, progress=False, threads=True,
                           group_by="column")
        if bars is not None and not bars.empty:
            for kind in kinds:
                fresh = _field_frame(bars, kind.title(), tickers)
                frames[kind] = _merge(frames[kind], fresh)

    benchmark_path = DATA_DIR / f"{universe}_index_close.csv"
    benchmark_saved = _read_frame(benchmark_path).iloc[:, 0].astype(float)
    bench_start = (benchmark_saved.index.max() + pd.Timedelta(days=1)).date()
    if bench_start <= local_today:
        index_bars = yf.download(tickers=[cfg["index"]], start=bench_start.isoformat(),
                                 end=end.isoformat(), interval="1d", auto_adjust=True,
                                 progress=False, threads=True)
        if index_bars is not None and not index_bars.empty:
            fresh = _field_frame(index_bars, "Close", [cfg["index"]]).iloc[:, 0]
            benchmark_saved = pd.concat([benchmark_saved, fresh]).sort_index()
            benchmark_saved = benchmark_saved.loc[~benchmark_saved.index.duplicated(keep="last")]

    return {**frames, "benchmark": benchmark_saved, "tickers": tickers, "config": cfg,
            "today": pd.Timestamp(local_today)}


def current_quotes(tickers: list[str], fallback: dict[str, float]) -> dict[str, float]:
    tickers = list(dict.fromkeys(tickers))
    result = dict(fallback)
    if not tickers:
        return result
    bars = yf.download(tickers=tickers, period="1d", interval="1m", auto_adjust=True,
                       prepost=False, progress=False, threads=True, group_by="column")
    if bars is None or bars.empty:
        return result
    close = _field_frame(bars, "Close", tickers)
    for ticker in tickers:
        if ticker in close:
            values = close[ticker].dropna()
            if not values.empty and float(values.iloc[-1]) > 0:
                result[ticker] = float(values.iloc[-1])
    return result


def _new_test_state(test: dict) -> dict:
    params = test["parameters"]
    return {"cash": float(params["initial_capital"]), "initial_capital": float(params["initial_capital"]),
            "positions": {}, "trades": [], "snapshots": [], "last_prices": {},
            "last_processed_rebalance_date": None, "started_at": None, "monthly_volume": []}


def _restore_engine(test: dict, state: dict) -> ExecutionEngine:
    params = test["parameters"]
    cfg = UNIVERSES[test["universe"]]
    engine = ExecutionEngine(
        initial_capital=params["initial_capital"], use_charges=params["use_charges"],
        market=cfg["market"], currency=cfg["currency"],
        fee_model=get_ibkr_fee_profile(cfg["market"]),
        simulated_latency_seconds=params.get("simulated_latency_seconds", 1.0),
        simulated_slippage_bps=params.get("simulated_slippage_bps", 5.0),
    )
    engine.cash = float(state["cash"])
    engine.positions = {
        ticker: {**position, "buy_date": pd.Timestamp(position["buy_date"])}
        for ticker, position in state.get("positions", {}).items()
    }
    engine.monthly_volume = {
        (int(item["year"]), int(item["month"])): float(item["volume"])
        for item in state.get("monthly_volume", [])
    }
    return engine


def _store_engine(test_state: dict, engine: ExecutionEngine) -> None:
    test_state["cash"] = float(engine.cash)
    test_state["positions"] = {
        ticker: {**position, "buy_date": pd.Timestamp(position["buy_date"]).date().isoformat()}
        for ticker, position in engine.positions.items()
    }
    test_state["monthly_volume"] = [
        {"year": key[0], "month": key[1], "volume": value}
        for key, value in engine.monthly_volume.items()
    ]


def _rebalance_dates(prices: pd.DataFrame, frequency) -> pd.DatetimeIndex:
    return get_rebalance_dates(prices, frequency)


def process_test(test: dict, test_state: dict, history: dict, now: datetime) -> dict:
    universe = test["universe"]
    params = test["parameters"]
    date = history["today"]
    open_prices = history["open"].dropna(how="all")
    if date not in open_prices.index:
        raise RuntimeError(f"Yahoo Finance has not published today’s {universe} opening bars yet.")

    # Keep the strategy in one place: this is the same indicator, eligibility,
    # ranking, portfolio score threshold, and score-weight allocator as the backtester.
    strategy = MomentumStrategy(top_n=params["top_n"],
                                rebalance_frequency=params["rebalance_frequency"])
    normal_dates = _rebalance_dates(open_prices, params["rebalance_frequency"])
    first_run = test_state.get("started_at") is None
    scheduled = first_run or (date in normal_dates and
                              test_state.get("last_processed_rebalance_date") != date.date().isoformat())
    selection = None
    if scheduled:
        selection = strategy.generate_positions(open_prices, rebalance_dates=[date]).get(date)

    holdings = list(test_state.get("positions", {}))
    quote_tickers = holdings + (list(selection.index.astype(str)) if selection is not None else [])
    fallback_prices = {ticker: float(value) for ticker, value in test_state.get("last_prices", {}).items()
                       if ticker in quote_tickers and pd.notna(value)}
    fallback_prices.update({ticker: float(open_prices.loc[date, ticker])
                            for ticker in quote_tickers
                            if ticker in open_prices and pd.notna(open_prices.loc[date, ticker])})
    prices = current_quotes(quote_tickers, fallback_prices)
    if not prices and holdings:
        prices = {ticker: float(value) for ticker, value in test_state.get("last_prices", {}).items()}
    if not prices and selection is None:
        # Still establish the run and allow the test to be visible before its
        # next configured rebalance date.
        prices = fallback_prices

    price_row = pd.DataFrame([prices], index=[date])
    engine = _restore_engine(test, test_state)
    test_state.setdefault("trades", [])
    test_state.setdefault("snapshots", [])
    if scheduled:
        test_state["last_processed_rebalance_date"] = date.date().isoformat()
        test_state["started_at"] = test_state.get("started_at") or now.isoformat(timespec="seconds")
        if selection is not None:
            live_regime = "OFF"
            if params["use_regime_filter"]:
                benchmark = history["benchmark"].dropna()
                if not benchmark.empty:
                    # Use the most recent benchmark value available to the runner.
                    if date in benchmark.index:
                        current_regime = MarketRegime(**REGIME_CONFIG).detect(benchmark)
                        live_regime = str(current_regime.loc[date, "Regime"])
                    else:
                        current_regime = MarketRegime(**REGIME_CONFIG).detect(benchmark)
                        live_regime = str(current_regime.iloc[-1]["Regime"])
            if params["use_regime_filter"] and live_regime not in params["allowed_regimes"]:
                sell_prices = {ticker: price * (1 - params.get("simulated_slippage_bps", 5.0) / 10000)
                               for ticker, price in prices.items()}
                engine.sell_positions(list(engine.positions), price_row, date,
                                      execution_prices=sell_prices, sell_reason="REGIME_EXIT")
            else:
                weights = ScoreWeightAllocator().allocate(selection, open_prices, date)
                buys = {ticker: prices[ticker] * (1 + params.get("simulated_slippage_bps", 5.0) / 10000)
                        for ticker in weights if ticker in prices}
                sells = {ticker: prices[ticker] * (1 - params.get("simulated_slippage_bps", 5.0) / 10000)
                         for ticker in prices}
                if len(buys) == len(weights):
                    engine.rebalance(weights, price_row, date,
                                     buy_execution_prices=buys, sell_execution_prices=sells)

    # Daily OHLC stop checks mirror the backtester, using the current session's
    # developing high/low values. The 30-minute run sells after a trigger.
    if params["use_stop_loss"] and engine.positions:
        manager = (FixedPercentStopLoss(params["stop_loss_pct"])
                   if params["stop_loss_type"] == "fixed"
                   else TrailingPercentStopLoss(params["stop_loss_pct"]))
        stop_info = manager.get_stop_loss_tickers(
            engine.positions, open_prices, history["high"], history["low"], date)
        if stop_info:
            stop_prices = {}
            for ticker, info in stop_info.items():
                current = prices.get(ticker, info["execution_price"])
                trigger = float(info["execution_price"])
                stop_prices[ticker] = min(current, trigger) * (
                    1 - params.get("simulated_slippage_bps", 5.0) / 10000)
            engine.sell_positions(list(stop_prices), price_row, date,
                                  execution_prices=stop_prices, sell_reason="STOP_LOSS")

    _store_engine(test_state, engine)
    test_state["realized_pnl_cumulative"] = float(test_state.get("realized_pnl_cumulative", 0.0)) + sum(
        float(trade.get("RealizedPnL", 0.0)) for trade in engine.trade_log
    )
    test_state["trades"].extend(engine.trade_log)
    test_state["trades"] = test_state["trades"][-2000:]
    test_state["last_prices"].update(prices)
    equity = float(engine.cash) + sum(
        float(position["shares"]) * float(prices.get(ticker, position["entry_price"]))
        for ticker, position in engine.positions.items())
    unrealized = sum(
        (float(prices.get(ticker, position["entry_price"])) - float(position["entry_price"]))
        * float(position["shares"])
        for ticker, position in engine.positions.items())
    realized = float(test_state["realized_pnl_cumulative"])
    stamp = now.astimezone(ZoneInfo(history["config"]["timezone"])).isoformat(timespec="seconds")
    test_state["snapshots"].append({
        "timestamp": stamp, "portfolio_value": equity,
        "pnl": equity - float(test_state["initial_capital"]),
        "pnl_pct": equity / float(test_state["initial_capital"]) - 1,
        "cash": float(engine.cash), "unrealized_pnl": float(unrealized),
        "realized_pnl": float(realized), "holdings": len(engine.positions),
        "prices": prices,
    })
    test_state["snapshots"] = test_state["snapshots"][-500:]
    test_state["last_updated_at"] = stamp
    return test_state


def main() -> None:
    now = datetime.now(timezone.utc)
    if not CONFIG_PATH.exists():
        print("No frozen forward tests are configured.")
        return
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    tests = [test for test in config.get("tests", []) if test.get("status", "active") == "active"]
    if not tests:
        print("No active forward tests.")
        return
    state = json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else {"schema_version": 2, "tests": {}}
    state.setdefault("tests", {})
    histories = {}
    updated_any = False
    for test in tests:
        is_open, message = market_state(now, test["universe"])
        print(message)
        if not is_open:
            continue
        universe = test["universe"]
        if universe not in histories:
            histories[universe] = load_market_history(universe)
        test_state = state["tests"].setdefault(test["id"], _new_test_state(test))
        process_test(test, test_state, histories[universe], now)
        updated_any = True
    if not updated_any:
        print("No active test's market is open; state was not changed.")
        return
    state["last_updated_at"] = now.isoformat(timespec="seconds")
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    tmp.replace(STATE_PATH)
    print(json.dumps({"updated_tests": list(state["tests"]), "at": state["last_updated_at"]}, indent=2))


if __name__ == "__main__":
    main()
