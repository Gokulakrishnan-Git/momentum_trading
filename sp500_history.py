"""Point-in-time S&P 500 constituents and local Yahoo Finance OHLC cache."""

from __future__ import annotations

from datetime import date
import json
import os
from pathlib import Path
import re

import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "market_data"
MEMBERSHIP_PATH = DATA_DIR / "SP500_historical_membership.csv"
CACHE_DIR = ROOT / ".sp500_historical_cache"
MANIFEST_PATH = CACHE_DIR / "manifest.json"
OHLC_FIELDS = {"Open": "open", "High": "high", "Low": "low", "Close": "close"}
BATCH_SIZE = 30


def yahoo_symbol(ticker: str) -> str:
    """Translate class-share dots in the source data to Yahoo's hyphen form."""
    return str(ticker).strip().upper().replace(".", "-")


def load_membership() -> pd.DataFrame:
    if not MEMBERSHIP_PATH.exists():
        raise FileNotFoundError(f"Historical S&P 500 membership file is missing: {MEMBERSHIP_PATH}")
    membership = pd.read_csv(MEMBERSHIP_PATH, dtype={"ticker": str})
    required = {"ticker", "start_date", "end_date"}
    if not required.issubset(membership.columns):
        raise ValueError("Historical membership CSV must contain ticker, start_date, and end_date columns")
    membership["ticker"] = membership["ticker"].map(yahoo_symbol)
    membership["start_date"] = pd.to_datetime(membership["start_date"], errors="coerce")
    membership["end_date"] = pd.to_datetime(membership["end_date"], errors="coerce")
    membership = membership.dropna(subset=["ticker", "start_date"])
    return membership.sort_values(["start_date", "ticker"]).reset_index(drop=True)


def _ticker_cache_path(ticker: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", yahoo_symbol(ticker))
    return CACHE_DIR / f"{safe}.csv"


def _read_cached_ohlc(ticker: str) -> pd.DataFrame:
    path = _ticker_cache_path(ticker)
    if not path.exists():
        return pd.DataFrame(columns=OHLC_FIELDS.values())
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None)
    return frame.reindex(columns=list(OHLC_FIELDS.values())).sort_index()


def _read_manifest() -> dict:
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _manifest_covers(spans: list[dict], start: pd.Timestamp, end: pd.Timestamp) -> bool:
    for span in spans:
        span_start = pd.Timestamp(span.get("start"))
        span_end = pd.Timestamp(span.get("end"))
        if span_start <= start and span_end >= end:
            return True
    return False


def _save_manifest(manifest: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, MANIFEST_PATH)


def _persist_ticker(ticker: str, fresh: pd.DataFrame) -> bool:
    if fresh.empty:
        return False
    path = _ticker_cache_path(ticker)
    old = _read_cached_ohlc(ticker)
    merged = pd.concat([old, fresh]).sort_index()
    merged = merged.loc[~merged.index.duplicated(keep="last")]
    temporary = path.with_suffix(".tmp")
    merged.to_csv(temporary)
    os.replace(temporary, path)
    return True


def _active_symbols(membership: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    overlaps = membership[
        (membership["start_date"] <= end)
        & (membership["end_date"].isna() | (membership["end_date"] >= start))
    ]
    return sorted(overlaps["ticker"].dropna().unique().tolist())


def _coverage_status(manifest: dict, ticker: str, start: pd.Timestamp, end: pd.Timestamp,
                     retry_unavailable: bool) -> str | None:
    for span in manifest.get(ticker, []):
        if pd.Timestamp(span["start"]) <= start and pd.Timestamp(span["end"]) >= end:
            if span.get("status") == "no_data" and retry_unavailable:
                return None
            return span.get("status", "cached")
    return None


def _bars_for_ticker(bars: pd.DataFrame, field: str, ticker: str) -> pd.Series:
    if bars is None or bars.empty:
        return pd.Series(dtype=float)
    if isinstance(bars.columns, pd.MultiIndex):
        level0 = bars.columns.get_level_values(0)
        level1 = bars.columns.get_level_values(1)
        if field in level0:
            block = bars[field]
            if isinstance(block, pd.Series):
                return block
            if ticker in block.columns:
                return block[ticker]
        if field in level1:
            block = bars.xs(field, axis=1, level=1)
            if isinstance(block, pd.Series):
                return block
            if ticker in block.columns:
                return block[ticker]
    if field in bars.columns:
        return bars[field]
    return pd.Series(dtype=float)


def _membership_matrix(membership: pd.DataFrame, dates: pd.Index,
                       tickers: pd.Index) -> pd.DataFrame:
    index = pd.DatetimeIndex(pd.to_datetime(dates))
    if index.tz is not None:
        index = index.tz_localize(None)
    columns = pd.Index([yahoo_symbol(ticker) for ticker in tickers])
    result = pd.DataFrame(False, index=index, columns=columns, dtype=bool)
    present = set(columns)
    for row in membership.itertuples(index=False):
        ticker = yahoo_symbol(row.ticker)
        if ticker not in present:
            continue
        start = pd.Timestamp(row.start_date)
        end = pd.Timestamp(row.end_date) if pd.notna(row.end_date) else (index.max() if len(index) else start)
        if start.tzinfo is not None:
            start = start.tz_localize(None)
        if end.tzinfo is not None:
            end = end.tz_localize(None)
        dates_in_membership = (index >= start) & (index <= end)
        result.loc[dates_in_membership, ticker] = True
    return result


def membership_mask(dates: pd.Index, tickers: pd.Index) -> pd.DataFrame:
    """Build an inclusive point-in-time membership matrix for a price panel."""
    return _membership_matrix(load_membership(), dates, tickers)


def prepare_sp500_history(start_date: date | pd.Timestamp, end_date: date | pd.Timestamp,
                          warmup_days: int = 400, progress_callback=None,
                          retry_unavailable: bool = False,
                          existing_tickers=None) -> tuple[dict[str, pd.DataFrame], dict]:
    """Download/cache missing former-constituent OHLC and build membership mask.

    Only symbols that were constituents during the requested backtest interval
    are considered. The requested price range includes warm-up history before
    the backtest start; saved data is adjusted OHLC to match the existing app.
    """
    membership = load_membership()
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if start.tzinfo is not None:
        start = start.tz_localize(None)
    if end.tzinfo is not None:
        end = end.tz_localize(None)
    start, end = start.normalize(), end.normalize()
    data_start = start - pd.Timedelta(days=int(warmup_days))
    all_symbols = _active_symbols(membership, start, end)
    existing = set()
    if existing_tickers is not None:
        existing.update(yahoo_symbol(ticker) for ticker in list(existing_tickers))
    # Existing universe files already hold today's constituents' adjusted
    # OHLC, so only download histories for former/missing constituents.
    symbols = [ticker for ticker in all_symbols if ticker not in existing]
    manifest = _read_manifest()

    needed = []
    cached = []
    previous_unavailable = []
    for ticker in symbols:
        status = _coverage_status(manifest, ticker, data_start, end, retry_unavailable)
        if status is None:
            needed.append(ticker)
        elif status == "no_data":
            previous_unavailable.append(ticker)
        else:
            cached.append(ticker)

    downloaded = []
    unavailable = list(previous_unavailable)
    failed = []
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    batches = [needed[pos:pos + BATCH_SIZE] for pos in range(0, len(needed), BATCH_SIZE)]
    for batch_number, batch in enumerate(batches, 1):
        if progress_callback:
            progress_callback(batch_number - 1, len(batches), f"Downloading S&P 500 OHLC batch {batch_number}/{len(batches)}")
        try:
            bars = yf.download(
                batch,
                start=data_start.date().isoformat(),
                end=(end + pd.Timedelta(days=1)).date().isoformat(),
                auto_adjust=True,
                progress=False,
                group_by="column",
                threads=True,
            )
        except Exception as exc:
            failed.extend(batch)
            continue

        for ticker in batch:
            parts = {}
            for vendor_field, local_field in OHLC_FIELDS.items():
                series = _bars_for_ticker(bars, vendor_field, ticker)
                if not series.empty:
                    parts[local_field] = pd.to_numeric(series, errors="coerce")
            if len(parts) == len(OHLC_FIELDS):
                fresh = pd.DataFrame(parts).dropna(how="all")
                fresh.index = pd.DatetimeIndex(pd.to_datetime(fresh.index))
                if fresh.index.tz is not None:
                    fresh.index = fresh.index.tz_localize(None)
            else:
                fresh = pd.DataFrame(columns=list(OHLC_FIELDS.values()))

            status = "data" if _persist_ticker(ticker, fresh) else "no_data"
            manifest.setdefault(ticker, []).append({
                "start": data_start.date().isoformat(),
                "end": end.date().isoformat(),
                "status": status,
            })
            if status == "data":
                downloaded.append(ticker)
            else:
                unavailable.append(ticker)
        _save_manifest(manifest)

    if progress_callback:
        progress_callback(len(batches), len(batches), "Merging cached former-constituent OHLC")

    cache_frames = {field: [] for field in OHLC_FIELDS.values()}
    available = []
    for ticker in symbols:
        frame = _read_cached_ohlc(ticker)
        if frame.empty:
            continue
        available.append(ticker)
        for field in cache_frames:
            cache_frames[field].append(frame[field].rename(ticker))

    cached_ohlc = {
        field: (pd.concat(series, axis=1).sort_index() if series else pd.DataFrame())
        for field, series in cache_frames.items()
    }
    report = {
        "requested_symbols": len(symbols),
        "cached_symbols": len(cached),
        "downloaded_symbols": len(downloaded),
        "available_symbols": len(available),
        "unavailable_symbols": sorted(set(unavailable)),
        "failed_symbols": sorted(set(failed)),
        "data_start": data_start.date().isoformat(),
        "data_end": end.date().isoformat(),
    }
    return cached_ohlc, report
