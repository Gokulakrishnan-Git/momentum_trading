"""GitHub Actions runner for a small, paper-only US momentum forward test."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import time
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "market_data" / "SP500_close.csv"
STATE = ROOT / "forward_state.json"
NY = ZoneInfo("America/New_York")


def market_state(now: datetime) -> tuple[bool, str]:
    local = now.astimezone(NY)
    calendar = mcal.get_calendar("NYSE")
    session = calendar.schedule(start_date=local.date(), end_date=local.date())
    if session.empty:
        return False, "NYSE is closed today (weekend or market holiday)."
    opening = session.iloc[0]["market_open"].to_pydatetime().astimezone(NY)
    closing = session.iloc[0]["market_close"].to_pydatetime().astimezone(NY)
    if local < opening:
        return False, f"Market closed. NYSE opens at {opening:%I:%M %p} ET."
    if local >= closing:
        return False, f"Market closed. NYSE closed at {closing:%I:%M %p} ET."
    return True, f"Market open · NYSE session {opening:%I:%M %p}–{closing:%I:%M %p} ET."


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"schema_version": 1, "cash": 1000000.0, "initial_capital": 1000000.0,
            "positions": {}, "trades": [], "snapshots": [], "last_rebalance_date": None}


def save_state(state: dict) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE)


def main() -> None:
    now = datetime.now(timezone.utc)
    is_open, message = market_state(now)
    print(message)
    if not is_open:
        return

    closes = pd.read_csv(DATA, index_col=0, parse_dates=True)
    closes = closes.apply(pd.to_numeric, errors="coerce")
    closes = closes.loc[:, closes.notna().sum() >= 200]
    momentum = closes.iloc[-1].div(closes.iloc[-min(252, len(closes))]).sub(1).dropna()
    picks = momentum.nlargest(5).index.astype(str).tolist()
    if not picks:
        raise RuntimeError("Could not rank the saved S&P 500 universe.")

    state = load_state()
    quote_tickers = list(dict.fromkeys(picks + list(state.get("positions", {}))))
    bars = yf.download(quote_tickers, period="1d", interval="1m", prepost=False,
                        auto_adjust=True, progress=False, threads=True)
    if bars is None or bars.empty:
        raise RuntimeError("Yahoo Finance returned no current intraday prices.")
    latest = bars["Close"].iloc[-1]
    prices = {ticker: float(latest[ticker]) for ticker in quote_tickers
              if ticker in latest and pd.notna(latest[ticker]) and latest[ticker] > 0}
    if not prices:
        raise RuntimeError("No usable current prices returned for the top five.")

    stamp = now.astimezone(NY).isoformat(timespec="seconds")
    state["snapshots"].append({"timestamp": stamp, "prices": prices, "ranked_top5": picks})
    state["snapshots"] = state["snapshots"][-500:]

    # One rebalance on the first scheduled run of each open session. Later
    # half-hour runs collect price snapshots only.
    session_date = now.astimezone(NY).date().isoformat()
    did_rebalance = state.get("last_rebalance_date") != session_date
    if did_rebalance:
        current = state.get("positions", {})
        target_value = (float(state["cash"]) + sum(
            float(position["shares"]) * prices.get(ticker, float(position["last_price"]))
            for ticker, position in current.items())) / len(picks)
        targets = {ticker: target_value / prices[ticker] for ticker in picks if ticker in prices}
        for ticker in list(current):
            if ticker not in targets:
                position = current.pop(ticker)
                time.sleep(1.0)  # simple simulated order latency
                fill = prices.get(ticker, float(position["last_price"])) * (1 - 0.0005)
                proceeds = float(position["shares"]) * fill
                state["cash"] += proceeds
                state["trades"].append({"timestamp": stamp, "side": "SELL", "ticker": ticker,
                                         "shares": position["shares"], "fill_price": fill,
                                         "slippage_bps": 5, "simulated_latency_seconds": 1})
        for ticker, target_shares in targets.items():
            old_shares = float(current.get(ticker, {}).get("shares", 0))
            delta = target_shares - old_shares
            if delta <= 0:
                continue
            time.sleep(1.0)
            fill = prices[ticker] * (1 + 0.0005)
            affordable = min(delta, float(state["cash"]) / fill)
            if affordable <= 0:
                continue
            state["cash"] -= affordable * fill
            if ticker in current:
                current[ticker]["shares"] = old_shares + affordable
            else:
                current[ticker] = {"shares": affordable, "last_price": prices[ticker]}
            state["trades"].append({"timestamp": stamp, "side": "BUY", "ticker": ticker,
                                     "shares": affordable, "fill_price": fill,
                                     "slippage_bps": 5, "simulated_latency_seconds": 1})
        for ticker, position in current.items():
            if ticker in prices:
                position["last_price"] = prices[ticker]
        state["positions"] = current
        state["last_rebalance_date"] = session_date

    state["last_updated_at"] = stamp
    save_state(state)
    print(json.dumps({"timestamp": stamp, "prices": prices, "top5": picks,
                      "rebalanced_this_run": did_rebalance,
                      "positions": state["positions"]}, indent=2))


if __name__ == "__main__":
    main()
