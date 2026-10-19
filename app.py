from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
import json
import os
import subprocess
import time
import uuid
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from forward_test import STATE_PATH as FORWARD_STATE_PATH, market_state

from strategy_engine import run_strategy_backtest


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "market_data"
OUTPUT_DIR = ROOT / "backtest_output"
FORWARD_CONFIG_PATH = ROOT / "forward_test.json"
STARTUP_CHECK_PATH = ROOT / ".market_data_last_checked.json"
WARMUP_DAYS = 400
UNIVERSES = {
    "NIFTY500": {"label": "NIFTY 500", "market": "INDIA", "currency": "INR", "ticker": "^CRSLDX", "benchmark": "Nifty 500"},
    "SP500": {"label": "S&P 500", "market": "US", "currency": "USD", "ticker": "^GSPC", "benchmark": "S&P 500"},
    "TOPIX500": {"label": "TOPIX 500", "market": "JAPAN", "currency": "JPY", "ticker": "1306.T", "benchmark": "TOPIX"},
}


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    return frame.sort_index()


def _write_merged(path: Path, fresh: pd.DataFrame) -> None:
    old = _read_csv(path) if path.exists() else pd.DataFrame()
    combined = pd.concat([old, fresh]).sort_index()
    combined = combined.loc[~combined.index.duplicated(keep="last")]
    temp = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(temp)
    os.replace(temp, path)


def _latest_saved(universe: str) -> pd.Timestamp | None:
    paths = [DATA_DIR / f"{universe}_{kind}.csv" for kind in ("open", "high", "low", "close", "index_close")]
    if not all(p.exists() for p in paths):
        return None
    latest = []
    for path in paths:
        frame = _read_csv(path)
        if not frame.empty:
            latest.append(frame.index.max())
    return min(latest) if len(latest) == len(paths) else None


def _last_market_check_date() -> date | None:
    try:
        value = json.loads(STARTUP_CHECK_PATH.read_text(encoding="utf-8"))
        return date.fromisoformat(value["checked_date"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _claim_market_data_check(today: date) -> bool:
    """Claim the once-daily update so reloads and parallel sessions don't repeat it."""
    lock_path = ROOT / f".market_data_check_{today.isoformat()}.lock"
    if _last_market_check_date() == today:
        return False
    try:
        with lock_path.open("x", encoding="utf-8") as lock:
            lock.write(today.isoformat())
    except FileExistsError:
        return False
    if _last_market_check_date() == today:
        lock_path.unlink(missing_ok=True)
        return False
    return True


def _finish_market_data_check(today: date) -> None:
    lock_path = ROOT / f".market_data_check_{today.isoformat()}.lock"
    temporary = STARTUP_CHECK_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps({"checked_date": today.isoformat()}), encoding="utf-8")
    os.replace(temporary, STARTUP_CHECK_PATH)
    lock_path.unlink(missing_ok=True)


def update_universe(universe: str, through: date) -> tuple[str, bool]:
    """Incrementally extend saved adjusted OHLC and benchmark CSVs."""
    DATA_DIR.mkdir(exist_ok=True)
    names = ["open", "high", "low", "close", "index_close"]
    paths = {name: DATA_DIR / f"{universe}_{name}.csv" for name in names}
    cfg = UNIVERSES[universe]
    old_latest = _latest_saved(universe)
    if old_latest is None:
        return ("Required OHLC or benchmark CSV is missing; restore the notebook's initial download first.", False)
    through_ts = pd.Timestamp(through)
    if old_latest.normalize() >= through_ts:
        return (f"Already current through {old_latest.date()}.", True)

    tickers = _read_csv(paths["close"]).columns.astype(str).tolist()
    start = (old_latest + pd.Timedelta(days=1)).date().isoformat()
    end = (through_ts + pd.Timedelta(days=1)).date().isoformat()  # yfinance end is exclusive
    try:
        bars = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, group_by="column", threads=True)
        if bars is None or bars.empty:
            return (f"No new stock bars returned ({start} to {through}). Data may already be current for this market calendar.", True)
        for kind in ("Open", "High", "Low", "Close"):
            fresh = bars[kind]
            if isinstance(fresh, pd.Series):
                fresh = fresh.to_frame(name=tickers[0])
            fresh.index = pd.to_datetime(fresh.index).tz_localize(None)
            _write_merged(paths[kind.lower()], fresh)

        index = yf.download(cfg["ticker"], start=start, end=end, auto_adjust=True, progress=False)
        if index is not None and not index.empty:
            close = index["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close.index = pd.to_datetime(close.index).tz_localize(None)
            _write_merged(paths["index_close"], close.to_frame(name="Close"))
        latest = _latest_saved(universe)
        if latest is None:
            return ("Downloaded bars, but could not verify the saved files.", False)
        if latest.normalize() < through_ts:
            return (f"Updated through {latest.date()}; the data source has no later session through {through}.", True)
        return (f"Updated through {latest.date()}.", True)
    except Exception as exc:
        return (f"Market data update failed: {exc}", False)


def load_universe(universe: str, start_date: date, end_date: date):
    requested_start = pd.Timestamp(start_date)
    requested_end = pd.Timestamp(end_date)
    warmup_start = requested_start - pd.Timedelta(days=WARMUP_DAYS)
    frames = {kind: _read_csv(DATA_DIR / f"{universe}_{kind}.csv") for kind in ("open", "high", "low", "close")}
    common_dates = frames["open"].index
    common_tickers = frames["open"].columns
    for frame in list(frames.values())[1:]:
        common_dates = common_dates.intersection(frame.index)
        common_tickers = common_tickers.intersection(frame.columns)
    common_dates = common_dates.sort_values()
    mask = (common_dates >= warmup_start) & (common_dates <= requested_end)
    dates = common_dates[mask]
    frames = {kind: frame.loc[dates, common_tickers] for kind, frame in frames.items()}
    if dates.empty:
        raise ValueError("No OHLC data is available in the requested date range and warm-up period.")
    benchmark_frame = _read_csv(DATA_DIR / f"{universe}_index_close.csv")
    benchmark_frame = benchmark_frame.loc[(benchmark_frame.index >= warmup_start) & (benchmark_frame.index <= requested_end)]
    benchmark = benchmark_frame.iloc[:, 0].dropna()
    return frames["open"], frames["open"], frames["high"], frames["low"], frames["close"], benchmark


def csv_bytes(frame: pd.DataFrame, index: bool = False) -> bytes:
    buffer = StringIO()
    frame.to_csv(buffer, index=index, encoding="utf-8-sig")
    return buffer.getvalue().encode("utf-8-sig")


def output_frames(result: dict, cfg: dict) -> dict[str, pd.DataFrame]:
    trades = result["trade_log"].copy()
    if not trades.empty and "Ticker" in trades:
        frequency = trades.groupby("Ticker").size().sort_values(ascending=False).head(5).rename("TradeFrequency").reset_index()
    else:
        frequency = pd.DataFrame(columns=["Ticker", "TradeFrequency"])
    strategy_curve = result["strategy_curve"].rename("PortfolioValue").to_frame()
    benchmark_curve = result["benchmark_curve"].rename(cfg["benchmark"]).to_frame()
    monthly = result["monthly_returns"].copy()
    # Match the notebook's *_backtest_results.csv (daily engine output).
    summary = result["results"].copy()
    regime = result["regime"]
    if regime is None:
        regime = pd.DataFrame()
    return {
        "Trade log": trades,
        "Top 5 trade frequency": frequency,
        "Strategy curve": strategy_curve,
        "Benchmark curve": benchmark_curve,
        "Monthly returns": monthly,
        "Backtest results": summary,
        "Regime": regime,
    }


def read_forward_config() -> dict | None:
    if not FORWARD_CONFIG_PATH.exists():
        return None
    try:
        return json.loads(FORWARD_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_forward_config(config: dict) -> None:
    temporary = FORWARD_CONFIG_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(config, indent=2), encoding="utf-8")
    os.replace(temporary, FORWARD_CONFIG_PATH)


def publish_forward_config() -> tuple[bool, str]:
    """Commit the newly frozen configuration so Actions can start the test."""
    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=ROOT, text=True,
                              capture_output=True, timeout=120)

    added = git("add", "--", FORWARD_CONFIG_PATH.name)
    if added.returncode:
        return False, added.stderr.strip() or "Could not stage the forward-test configuration."
    committed = git("commit", "-m", "Freeze forward test", "--", FORWARD_CONFIG_PATH.name)
    if committed.returncode:
        return False, committed.stderr.strip() or "Could not commit the forward-test configuration."
    fetched = git("fetch", "origin", "main")
    if fetched.returncode:
        return False, fetched.stderr.strip() or "Could not fetch the latest GitHub state."
    merged = git("merge", "--no-edit", "origin/main")
    if merged.returncode:
        return False, merged.stderr.strip() or "Could not merge the latest GitHub state. Resolve the Git conflict and try again."
    pushed = git("push", "origin", "main")
    if pushed.returncode:
        return False, pushed.stderr.strip() or "Could not push the frozen test to GitHub."
    return True, "Frozen test pushed; GitHub Actions will start it at the next scheduled run."


def read_forward_state() -> dict:
    """Read the latest state Actions committed; fall back to a local copy."""
    try:
        response = requests.get(
            f"https://raw.githubusercontent.com/Gokulakrishnan-Git/momentum_trading/main/forward_state.json?ts={int(time.time())}",
            timeout=4,
        )
        if response.ok:
            return response.json()
    except (requests.RequestException, json.JSONDecodeError):
        pass
    if FORWARD_STATE_PATH.exists():
        try:
            return json.loads(FORWARD_STATE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {"tests": {}}


st.set_page_config(page_title="Momentum Strategy Backtest", page_icon="📈", layout="wide")
st.markdown("<style>.block-container{max-width:1450px;padding-top:2rem}.hero{padding:1.5rem 1.8rem;border-radius:18px;background:linear-gradient(115deg,#10283f,#173f56);color:#f4f8fb;margin-bottom:1.2rem}.hero h1{margin:0 0 .35rem;font-size:2rem}.hero p{margin:0;color:#c6d9e4}[data-testid='stMetricValue'],[data-testid='stMetricValue'] div{white-space:normal!important;overflow:visible!important;text-overflow:clip!important;overflow-wrap:anywhere;font-size:clamp(1.25rem,2.2vw,1.85rem)!important}</style>", unsafe_allow_html=True)
st.markdown('<section class="hero"><h1>Momentum Strategy Backtest</h1><p>Run the notebook strategy against saved market data and export every report table.</p></section>', unsafe_allow_html=True)

today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
yesterday = today - timedelta(days=1)
if st.session_state.get("startup_update_date") != today:
    if _last_market_check_date() == today:
        messages = [(name, f"Daily OHLC check already completed for {today.isoformat()}.", True)
                    for name in UNIVERSES]
    elif _claim_market_data_check(today):
        messages = []
        try:
            with st.spinner(f"Checking market data through {yesterday}…"):
                for name in UNIVERSES:
                    msg, okay = update_universe(name, yesterday)
                    messages.append((name, msg, okay))
        finally:
            _finish_market_data_check(today)
    else:
        messages = [(name, "Today's market data check is already running in another session.", True)
                    for name in UNIVERSES]
    st.session_state.startup_update = messages
    st.session_state.startup_update_date = today
for name, msg, okay in st.session_state.startup_update:
    (st.success if okay else st.warning)(f"{UNIVERSES[name]['label']}: {msg}")

with st.sidebar:
    st.header("Backtest inputs")
    universe = st.selectbox("Universe", list(UNIVERSES), format_func=lambda k: UNIVERSES[k]["label"])
    start = st.date_input("Start date", value=date(2024, 1, 1), min_value=date(2016, 1, 1), max_value=yesterday)
    end = st.date_input("End date", value=yesterday, min_value=date(2016, 1, 1), max_value=yesterday)
    st.divider()
    st.subheader("Strategy")
    use_regime = st.toggle("Use regime filter", value=True)
    allowed = st.multiselect("Allowed market regimes", ["Bull", "Bear"], default=["Bull"], disabled=not use_regime)
    frequency_choice = st.selectbox("Trade frequency", ["Monthly", "Every 2 weeks", "Weekly", "Every N trading days"])
    if frequency_choice == "Every N trading days":
        frequency = st.number_input("Trading days between rebalance", min_value=1, max_value=252, value=3)
    else:
        frequency = {"Monthly": "monthly", "Every 2 weeks": "2weeks", "Weekly": "weekly"}[frequency_choice]
    top_n = st.number_input("Number of holdings", min_value=1, max_value=100, value=5)
    st.subheader("Risk and costs")
    use_stop = st.toggle("Use stop loss", value=False)
    stop_type = st.selectbox("Stop loss type", ["Fixed percent", "Trailing percent"], disabled=not use_stop)
    stop_pct = st.slider("Stop loss (%)", min_value=1, max_value=50, value=8, disabled=not use_stop) / 100
    use_charges = st.toggle("Include notebook fee model charges", value=True)
    capital = st.number_input("Initial capital", min_value=1, value=1_000_000, step=100_000)
    run = st.button("Run backtest", type="primary", use_container_width=True)

backtest_tab, forward_tab = st.tabs(["Backtest", "Forward test"])

with backtest_tab:
    st.caption("OHLC files are adjusted prices from Yahoo Finance. The app checks all saved universes once per day (IST) and appends any newer bars available through yesterday; reloading again that day skips the download check.")
    if run:
        if start > end:
            st.error("Start date must be on or before end date.")
        elif use_regime and not allowed:
            st.error("Choose at least one allowed regime, or turn off the regime filter.")
        else:
            cfg = UNIVERSES[universe]
            try:
                prices, open_prices, high_prices, low_prices, close_prices, benchmark_prices = load_universe(universe, start, end)
                with st.spinner("Running backtest and building reports…"):
                    result = run_strategy_backtest(
                        universe_name=universe, prices=prices, open_prices=open_prices, high_prices=high_prices,
                        low_prices=low_prices, close_prices=close_prices, benchmark_ticker=cfg["ticker"],
                        benchmark_name=cfg["benchmark"], benchmark_prices=benchmark_prices,
                        regime_index_prices=benchmark_prices, backtest_start_date=start, backtest_end_date=end,
                        use_regime_filter=use_regime, allowed_regimes=tuple(allowed or ["Bull"]),
                        use_stop_loss=use_stop, stop_loss_type="fixed" if stop_type == "Fixed percent" else "trailing",
                        stop_loss_pct=stop_pct, rebalance_frequency=int(frequency) if isinstance(frequency, (int, float)) else frequency,
                        top_n=int(top_n), use_charges=use_charges, market=cfg["market"], currency=cfg["currency"],
                        initial_capital=float(capital), show_plots=False, save_csv=True, output_dir=str(OUTPUT_DIR),
                    )
                st.session_state.last_report = (result, cfg, universe, start, end)
            except Exception as exc:
                st.exception(exc)

    if "last_report" in st.session_state:
        result, cfg, report_universe, report_start, report_end = st.session_state.last_report
        frames = output_frames(result, cfg)
        st.subheader(f"{cfg['label']} report · {report_start:%d %b %Y} – {report_end:%d %b %Y}")
        metrics = result["metrics"]
        metric_specs = [
            ("Strategy return", "total_return", "percent"), ("Benchmark return", "benchmark_total_return", "percent"),
            ("Strategy CAGR", "cagr", "percent"), ("Benchmark CAGR", "benchmark_cagr", "percent"),
            ("Volatility", "volatility", "percent"), ("Sharpe", "sharpe", "number"),
            ("Max drawdown", "max_drawdown", "percent"), ("Final value", "final_value", "money"),
        ]
        for offset in (0, 4):
            cols = st.columns(4)
            for col, (label, key, display) in zip(cols, metric_specs[offset:offset + 4]):
                value = metrics.get(key)
                if value is None or pd.isna(value):
                    rendered = "—"
                elif display == "percent":
                    rendered = f"{value:.2%}"
                elif display == "money":
                    rendered = f"{value:,.2f}"
                else:
                    rendered = f"{value:.2f}"
                col.metric(label, rendered)
        st.subheader("Equity curves")
        curve = pd.concat([frames["Strategy curve"], frames["Benchmark curve"]], axis=1).sort_index().dropna(how="all")
        st.line_chart(curve, height=360)
        left, right = st.columns(2)
        with left:
            st.subheader("Monthly return heatmap")
            monthly_strategy = result["strategy_monthly_returns"] * 100
            if not monthly_strategy.empty:
                heatmap_data = pd.DataFrame({"Year": monthly_strategy.index.year, "Month": monthly_strategy.index.strftime("%b"), "Return": monthly_strategy.values})
                month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                heatmap = heatmap_data.pivot(index="Year", columns="Month", values="Return").reindex(columns=month_order)
                st.dataframe(heatmap.style.background_gradient(cmap="RdYlGn", axis=None).format("{:.1f}%", na_rep=""), use_container_width=True)
            else:
                st.info("No monthly returns are available for this period.")
        with right:
            st.subheader("Portfolio drawdown")
            drawdown = result["drawdown"] * 100
            st.area_chart(drawdown.rename("Drawdown (%)"), height=240)
        tab_names = list(frames)
        tabs = st.tabs(tab_names)
        slug = {"Trade log": "trade_log", "Top 5 trade frequency": "top5_trade_frequency", "Strategy curve": "strategy_curve", "Benchmark curve": "benchmark_curve", "Monthly returns": "monthly_returns", "Backtest results": "backtest_results", "Regime": "regime"}
        for tab, name in zip(tabs, tab_names):
            with tab:
                frame = frames[name]
                st.dataframe(frame, use_container_width=True, height=420)
                st.download_button(f"Download {name} CSV", csv_bytes(frame, index=name in ("Strategy curve", "Benchmark curve", "Monthly returns", "Regime")), file_name=f"{report_universe}_{slug[name]}.csv", mime="text/csv", key=f"download_{slug[name]}")


@st.fragment(run_every="60s")
def render_forward_tab():
        st.subheader("Forward paper tests")
        st.write("Each frozen test uses the backtest’s own momentum, MACD, RSI, eligibility, score threshold, score-weighted allocation, regime filter, stop-loss rules, and fee model. The first run after freezing starts at the next open session; it then holds until the selected rebalance schedule. Orders remain simulated.")
        st.caption("GitHub Actions checks open market sessions every five minutes: S&P 500 (NYSE), NIFTY 500 (NSE), and TOPIX 500 (JPX). Fills use Yahoo Finance prices with configurable simulated latency and slippage. P&L is marked to the latest available quote. This tab reloads the latest GitHub snapshot automatically every 60 seconds while open.")

        config = read_forward_config() or {}
        frozen_tests = config.get("tests", []) if isinstance(config.get("tests"), list) else []
        active_count = sum(test.get("status", "active") == "active" for test in frozen_tests)
        st.markdown("### Freeze a new test")
        with st.form("forward_setup"):
            f1, f2, f3 = st.columns(3)
            with f1:
                f_name = st.text_input("Test name", value=f"Forward test {len(frozen_tests) + 1}")
                f_universe = st.selectbox("Market universe", list(UNIVERSES), index=1,
                                          format_func=lambda key: UNIVERSES[key]["label"])
                f_capital = st.number_input("Starting capital", min_value=1, value=1_000_000, step=100_000)
            with f2:
                f_top_n = st.number_input("Number of holdings", min_value=1, max_value=100, value=5)
                f_frequency_label = st.selectbox("Rebalance frequency", ["Monthly", "Every 2 weeks", "Weekly", "Every N trading days"])
                f_frequency_days = st.number_input("Trading days between rebalances", min_value=1, max_value=252, value=3)
                st.caption("Used when rebalance frequency is Every N trading days.")
            with f3:
                f_use_regime = st.toggle("Use market regime filter", value=True)
                f_allowed = st.multiselect("Allowed regimes", ["Bull", "Bear"], default=["Bull"], disabled=not f_use_regime)
                f_use_stop = st.toggle("Use stop loss", value=False)
                f_stop_type = st.selectbox("Stop type", ["Fixed percent", "Trailing percent"], disabled=not f_use_stop)
                f_stop_pct = st.slider("Stop distance (%)", min_value=1, max_value=50, value=8) / 100
                st.caption("Used when stop loss is enabled.")
            f4, f5 = st.columns(2)
            with f4:
                f_charges = st.toggle("Include backtest fee model", value=True)
            with f5:
                f_latency = st.number_input("Simulated order latency (seconds)", min_value=0.0, max_value=10.0, value=1.0, step=0.25)
                f_slippage = st.number_input("Adverse slippage (basis points)", min_value=0.0, max_value=100.0, value=5.0, step=1.0)
            freeze = st.form_submit_button("Freeze and start test", type="primary")

        if freeze:
            if not f_name.strip():
                st.error("Enter a name for this forward test.")
            elif f_use_regime and not f_allowed:
                st.error("Choose at least one allowed regime, or turn off the regime filter.")
            else:
                frequency = int(f_frequency_days) if f_frequency_label == "Every N trading days" else {
                    "Monthly": "monthly", "Every 2 weeks": "2weeks", "Weekly": "weekly"
                }[f_frequency_label]
                new_test = {
                    "id": uuid.uuid4().hex[:12], "name": f_name.strip(), "universe": f_universe,
                    "status": "active", "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "parameters": {
                        "top_n": int(f_top_n), "rebalance_frequency": frequency,
                        "use_regime_filter": f_use_regime, "allowed_regimes": f_allowed or ["Bull"],
                        "use_stop_loss": f_use_stop,
                        "stop_loss_type": "fixed" if f_stop_type == "Fixed percent" else "trailing",
                        "stop_loss_pct": float(f_stop_pct), "use_charges": f_charges,
                        "initial_capital": float(f_capital),
                        "simulated_latency_seconds": float(f_latency),
                        "simulated_slippage_bps": float(f_slippage),
                    },
                }
                updated_config = {"schema_version": 2, "tests": [*frozen_tests, new_test]}
                write_forward_config(updated_config)
                frozen_tests.append(new_test)
                active_count += 1
                pushed, publish_message = publish_forward_config()
                (st.success if pushed else st.error)(publish_message)
                if pushed:
                    st.info("The test is active. If its market is closed, its first portfolio will be created at the next market open.")

        forward_state = read_forward_state()
        states_by_id = forward_state.get("tests", {})
        st.markdown(f"### Frozen tests · {active_count} active")
        st.components.v1.html("""
        <div id="actions-countdown" style="font: 15px sans-serif; color: #c8d1dc; padding: 5px 0"></div>
        <script>
          (() => {
            const output = document.getElementById("actions-countdown");
            const timeFormatter = new Intl.DateTimeFormat("en-IN", {
              timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit", hour12: false
            });
            function nextScheduledRun(after) {
              const candidate = new Date(after);
              candidate.setUTCSeconds(0, 0);
              candidate.setUTCMinutes(candidate.getUTCMinutes() - candidate.getUTCMinutes() % 5);
              if (candidate.getTime() <= after.getTime()) candidate.setUTCMinutes(candidate.getUTCMinutes() + 5);
              for (let i = 0; i < 5000; i++) {
                const day = candidate.getUTCDay();
                const hour = candidate.getUTCHours();
                if (day >= 1 && day <= 5 && ((hour >= 0 && hour <= 10) || (hour >= 13 && hour <= 21))) {
                  return candidate;
                }
                candidate.setUTCMinutes(candidate.getUTCMinutes() + 5);
              }
              return candidate;
            }
            let target = nextScheduledRun(new Date());
            function render() {
              let remaining = target.getTime() - Date.now();
              if (remaining <= 0) {
                target = nextScheduledRun(new Date());
                remaining = target.getTime() - Date.now();
              }
              const totalSeconds = Math.max(0, Math.floor(remaining / 1000));
              const minutes = Math.floor(totalSeconds / 60);
              const seconds = String(totalSeconds % 60).padStart(2, "0");
              output.textContent = `Next GitHub Actions check scheduled in ${minutes}m ${seconds}s · ${timeFormatter.format(target)} IST`;
            }
            render();
            setInterval(render, 1000);
          })();
        </script>
        """, height=38, scrolling=False)
        st.caption("The countdown shows the next five-minute scheduled slot in the workflow's UTC schedule, displayed in IST. GitHub may start a scheduled run late.")
        if not frozen_tests:
            st.caption("No forward tests are frozen yet. Freezing a test publishes it to GitHub Actions; you can create additional tests with different parameters at any time.")
        for test in frozen_tests:
            test_id = test.get("id", "unknown")
            params = test.get("parameters", {})
            test_state = states_by_id.get(test_id, {})
            latest = (test_state.get("snapshots") or [{}])[-1]
            currency = UNIVERSES.get(test.get("universe"), {}).get("currency", "USD")
            status = test.get("status", "active")
            st.markdown(f"#### {test.get('name', test_id)} · {test.get('universe', '')} · {status.title()}")
            if st.button("Pause test" if status == "active" else "Resume test", key=f"toggle_test_{test_id}"):
                test["status"] = "paused" if status == "active" else "active"
                write_forward_config({"schema_version": 2, "tests": frozen_tests})
                saved, message = publish_forward_config()
                (st.success if saved else st.error)(message)
                if saved:
                    st.rerun()
            open_now, market_message = market_state(datetime.now(timezone.utc), test.get("universe", "SP500"))
            (st.success if open_now else st.info)(market_message)
            st.caption(f"{params.get('top_n', 5)} holdings · {params.get('rebalance_frequency', 'monthly')} rebalance · Frozen {test.get('frozen_at', 'unknown')}")
            if latest:
                c1, c2, c3 = st.columns(3)
                c1.metric("Portfolio value", f"{currency} {float(latest.get('portfolio_value', 0)):,.2f}")
                c2.metric("Total P&L", f"{currency} {float(latest.get('pnl', 0)):,.2f}", f"{float(latest.get('pnl_pct', 0)):.2%}")
                c3.metric("Unrealized P&L", f"{currency} {float(latest.get('unrealized_pnl', 0)):,.2f}")
                c4, c5, c6 = st.columns(3)
                c4.metric("Realized P&L", f"{currency} {float(latest.get('realized_pnl', 0)):,.2f}")
                c5.metric("Cash", f"{currency} {float(latest.get('cash', 0)):,.2f}")
                c6.metric("Holdings", int(latest.get("holdings", 0)))
                snapshots = test_state.get("snapshots", [])
                st.metric("Next rebalance date", test_state.get("next_rebalance_date") or "Pending")
                initial_capital = float(params.get("initial_capital", 0))
                curve_rows = [{
                    "Time": pd.to_datetime(row["timestamp"], utc=True).tz_convert("Asia/Kolkata"),
                    "Portfolio value": float(row["portfolio_value"]),
                    "Portfolio P&L": float(row["portfolio_value"]) - initial_capital,
                    "Portfolio return (%)": 100 * (float(row["portfolio_value"]) / initial_capital - 1),
                    "Benchmark value": row.get("benchmark_value"),
                } for row in snapshots if row.get("timestamp") and initial_capital > 0]
                if curve_rows:
                    curve = pd.DataFrame(curve_rows).set_index("Time")
                    benchmark_values = pd.to_numeric(curve["Benchmark value"], errors="coerce")
                    benchmark_values = benchmark_values.dropna()
                    if not benchmark_values.empty and float(benchmark_values.iloc[0]) > 0:
                        curve["S&P 500 return (%)"] = 100 * (pd.to_numeric(curve["Benchmark value"], errors="coerce") / float(benchmark_values.iloc[0]) - 1)
                    curve = curve.drop(columns=["Benchmark value", "Portfolio value"])
                    st.subheader("Portfolio P&L")
                    st.caption(f"Cumulative paper P&L from the frozen starting capital, in {currency}.")
                    st.area_chart(curve[["Portfolio P&L"]], height=220)
                    st.subheader("Portfolio return vs S&P 500")
                    return_series = ["Portfolio return (%)"]
                    if "S&P 500 return (%)" in curve:
                        return_series.append("S&P 500 return (%)")
                    st.line_chart(curve[return_series], height=260)
                    st.caption("Returns are percentages from the starting capital / first S&P 500 reading. The Total P&L metric above shows both the currency amount and percentage.")
                    if "S&P 500 return (%)" in curve:
                        benchmark_returns = curve[["S&P 500 return (%)"]].dropna()
                        if len(benchmark_returns) > 1:
                            benchmark_change = float(benchmark_returns.iloc[-1, 0] - benchmark_returns.iloc[0, 0])
                            st.subheader("S&P 500 intraday movement · zoomed")
                            st.caption(f"S&P 500 changed {benchmark_change:+.4f}% across the recorded checks. This chart uses its own vertical scale to make small moves visible; compare with the shared-scale chart above for relative performance.")
                            st.line_chart(benchmark_returns, height=220)
                selected_tickers = test_state.get("selected_tickers") or list(test_state.get("positions", {}))
                if selected_tickers:
                    st.markdown(f"**Top {len(selected_tickers)} selected** · {test_state.get('selected_at', 'selection date pending')}")
                    positions = test_state.get("positions", {})
                    last_prices = latest.get("prices", {})
                    selected_rows = [{
                        "Rank": rank, "Ticker": ticker,
                        "Shares held": positions.get(ticker, {}).get("shares", 0),
                        "Latest price": last_prices.get(ticker),
                    } for rank, ticker in enumerate(selected_tickers, 1)]
                    st.dataframe(pd.DataFrame(selected_rows), use_container_width=True, hide_index=True)
                st.caption(f"Last update: {test_state.get('last_updated_at', 'pending')}")
                if test_state.get("trades"):
                    with st.expander(f"Recent simulated trades · {len(test_state['trades'])} total"):
                        st.dataframe(pd.DataFrame(test_state["trades"][-20:]), use_container_width=True, hide_index=True)
            else:
                st.info("Waiting for the next market-open Actions run to build the first portfolio.")


with forward_tab:
    render_forward_tab()
