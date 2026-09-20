"""Closed Binance minute candles, validated configuration, and an atomic cache."""

import asyncio
import json
import os
from pathlib import Path
import tempfile

import aiohttp
import numpy as np
import pandas as pd

BINANCE_BASE_URL = "https://api.binance.com"
MAX_RETRIES = 5
CONCURRENT_REQUESTS = 10
MINUTE = pd.Timedelta(minutes=1)
CONFIG_FILE = Path(__file__).with_name("config.json")
DEFAULT_CONFIG = {
    "symbol": "BNBUSDT",
    "bucket_target_bars_per_day": 252,
    "adv_lookback_days": 90,
    "bucket_size_base": None,
    "vpin_window": 10,
    "cdf_lookback_days": 90,
    "start_date": "2020-01-01",
    "zoom_center_date": None,
    "latest_zoom_days_back": 14,
    "fee_bps": 10,
    "slippage_bps": 2,
}
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume", "ignore",
]


def load_config(path=CONFIG_FILE):
    """Merge an optional JSON file with defaults; reject invalid or unknown settings."""
    path = Path(path)
    config = DEFAULT_CONFIG.copy()
    if path.exists():
        supplied = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(supplied, dict) or supplied.keys() - config.keys():
            raise ValueError("Config must contain only supported settings")
        config.update(supplied)

    symbol = config["symbol"]
    if not isinstance(symbol, str) or not symbol.isascii() or not symbol.isalnum():
        raise ValueError("symbol must be an alphanumeric Binance symbol")
    config["symbol"] = symbol.upper()

    for key in ("bucket_target_bars_per_day", "adv_lookback_days", "vpin_window", "cdf_lookback_days"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("fee_bps", "slippage_bps", "bucket_size_base"):
        value = config[key]
        if key == "bucket_size_base" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"{key} must be finite and numeric")
        valid_range = value > 0 if key == "bucket_size_base" else 0 <= value < 10000
        if not valid_range:
            raise ValueError(f"Invalid {key}")

    for key in ("start_date", "zoom_center_date"):
        if key == "zoom_center_date" and config[key] is None:
            continue
        value = config[key]
        if (
            not isinstance(value, str)
            or pd.to_datetime(value, format="%Y-%m-%d", utc=True).strftime("%Y-%m-%d") != value
        ):
            raise ValueError(f"{key} must use YYYY-MM-DD")
    if type(config["latest_zoom_days_back"]) is not int or config["latest_zoom_days_back"] < 0:
        raise ValueError("latest_zoom_days_back must be a nonnegative integer")
    return config


def get_data_file(symbol):
    return f"{symbol.lower()}_1m.feather"


def validate_candles(df, start=None, end=None):
    """Return a UTC copy with finite OHLC/base volumes and contiguous minute rows.

    Naive timestamps mean UTC. Optional bounds require exact [start, end) coverage;
    malformed prices, volumes, timestamps, or missing candles raise ValueError.
    """
    if df.empty:
        raise ValueError("No minute candles available")
    df = df.copy()
    for name in ("open_time", "close_time"):
        df[name] = pd.to_datetime(df[name], utc=True)
    times = df["open_time"]
    if times.isna().any() or not times.is_monotonic_increasing or times.duplicated().any():
        raise ValueError("Candles must have unique, increasing UTC opening times")
    if (times != times.dt.floor("min")).any() or not times.diff().iloc[1:].eq(MINUTE).all():
        raise ValueError("Missing or misaligned minute candles")
    if not df["close_time"].eq(times + MINUTE - pd.Timedelta(milliseconds=1)).all():
        raise ValueError("Invalid minute candle closing times")
    numeric = ["open", "high", "low", "close", "volume", "taker_buy_base_asset_volume"]
    df[numeric] = df[numeric].apply(pd.to_numeric)
    if not np.isfinite(df[numeric].to_numpy()).all():
        raise ValueError("Candle prices and volumes must be finite")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Candle prices must be positive")
    invalid_high = df["high"] < df[["open", "close", "low"]].max(axis=1)
    invalid_low = df["low"] > df[["open", "close", "high"]].min(axis=1)
    if invalid_high.any() or invalid_low.any():
        raise ValueError("Inconsistent candle high/low prices")
    volume, buy_volume = df["volume"], df["taker_buy_base_asset_volume"]
    if ((volume < 0) | (buy_volume < 0) | (buy_volume > volume)).any():
        raise ValueError("Require 0 <= taker-buy volume <= total volume")
    if start is not None and times.iloc[0] != pd.to_datetime(start, utc=True):
        raise ValueError("Requested start is not covered by the data")
    if end is not None and times.iloc[-1] + MINUTE != pd.to_datetime(end, utc=True):
        raise ValueError("Requested end is not covered by the data")
    return df


async def fetch_kline_chunk(session, symbol, interval, start_time, end_time, limit=1000):
    """Binance bounds are inclusive; failures must not look like empty data."""
    params = dict(symbol=symbol, interval=interval, startTime=start_time, endTime=end_time, limit=limit)
    error = "no response"
    for attempt in range(MAX_RETRIES):
        delay = 2 ** attempt
        try:
            await asyncio.sleep(0.05)
            async with session.get(f"{BINANCE_BASE_URL}/api/v3/klines", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if not isinstance(data, list):
                        raise ValueError("Expected a kline array")
                    return data
                error = f"HTTP {response.status}"
                if response.status in (429, 418):
                    delay = max(delay, float(response.headers.get("Retry-After", delay)))
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            error = str(exc)
        if attempt + 1 < MAX_RETRIES:
            await asyncio.sleep(delay)
    raise RuntimeError(f"Kline download failed for {symbol} at {start_time}: {error}")


async def download_historical_data_async(symbol, interval, start_ts, end_ts):
    """Return 1m klines with UTC timestamps for [start_ts, end_ts), in epoch milliseconds.

    The caller sets end_ts to a closed-minute cutoff. Empty exchange responses are
    allowed here; manage_local_data enforces coverage before saving or analysis.
    """
    if interval != "1m":
        raise ValueError("Only 1m candles are supported")
    chunk_ms = 1000 * 60_000
    # Use the working OS resolver rather than an optional aiodns installation.
    connector = aiohttp.TCPConnector(resolver=aiohttp.ThreadedResolver())
    async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=60)) as session:
        semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

        async def fetch(start):
            async with semaphore:
                inclusive_end = min(start + chunk_ms, end_ts) - 1
                return await fetch_kline_chunk(session, symbol, interval, start, inclusive_end)

        chunks = await asyncio.gather(*(fetch(start) for start in range(start_ts, end_ts, chunk_ms)))
    df = pd.DataFrame([row for chunk in chunks for row in chunk], columns=KLINE_COLUMNS)
    for name in ("open_time", "close_time"):
        df[name] = pd.to_datetime(df[name], unit="ms", utc=True)
    numeric = [
        "open", "high", "low", "close", "volume", "taker_buy_base_asset_volume",
        "quote_asset_volume", "taker_buy_quote_asset_volume",
    ]
    df[numeric] = df[numeric].apply(pd.to_numeric)
    return df


def manage_local_data(symbol, interval="1m", start_date_dt=None, *, now=None):
    """Return closed UTC candles from start_date_dt through floor(now, minute), exclusive.

    Repair coverage and refresh two overlapping rows before atomically saving the
    validated cache. Legacy naive-time caches are rebuilt; failed validation leaves
    the old file intact. Omitting now uses the current UTC clock.
    """
    if interval != "1m":
        raise ValueError("Only 1m candles are supported")
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.to_datetime(now, utc=True)
    end = now.floor("min")
    path = Path(get_data_file(symbol))
    cached = pd.read_feather(path) if path.exists() else pd.DataFrame()

    # Old naive-time caches may contain permanently truncated historical candles.
    if not cached.empty and not isinstance(cached["open_time"].dtype, pd.DatetimeTZDtype):
        print("Rebuilding legacy cache; original stays intact until validation succeeds.")
        cached = pd.DataFrame()
    if not cached.empty:
        for name in ("open_time", "close_time"):
            cached[name] = pd.to_datetime(cached[name], utc=True)
        cached = cached[(cached["open_time"] < end) & (cached["close_time"] < end)]
        cached = (
            cached.drop_duplicates("open_time", keep="last")
            .sort_values("open_time").reset_index(drop=True)
        )

    if start_date_dt is not None:
        requested_start = pd.to_datetime(start_date_dt, utc=True)
    elif not cached.empty:
        requested_start = cached["open_time"].min()
    else:
        requested_start = end - pd.Timedelta(days=2)
    if requested_start != requested_start.floor("min") or requested_start >= end:
        raise ValueError("Requested start must be minute-aligned and precede the closed-data cutoff")
    start = min(requested_start, cached["open_time"].min()) if not cached.empty else requested_start

    if cached.empty:
        ranges = [(start, end)]
    else:
        expected = pd.date_range(start, end, freq="min", inclusive="left")
        missing = expected.difference(cached["open_time"])
        missing = missing.union(pd.DatetimeIndex(cached["open_time"].tail(2))).sort_values()
        # Group adjacent missing/overlapping minutes into single download requests.
        breaks = (missing[1:] - missing[:-1]) != MINUTE
        range_starts = missing[np.r_[True, breaks]]
        range_ends = missing[np.r_[breaks, True]] + MINUTE
        ranges = zip(range_starts, range_ends)

    frames = [cached] if not cached.empty else []
    for left, right in ranges:
        fetched = asyncio.run(download_historical_data_async(
            symbol, interval, int(left.timestamp() * 1000), int(right.timestamp() * 1000)
        ))
        if not fetched.empty:
            for name in ("open_time", "close_time"):
                fetched[name] = pd.to_datetime(fetched[name], utc=True)
            in_range = (fetched["open_time"] >= left) & (fetched["open_time"] < right)
            closed = fetched["close_time"] < end
            frames.append(fetched[in_range & closed])
    if not frames:
        raise ValueError("No data downloaded for the requested period")
    combined = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("open_time", keep="last")
        .sort_values("open_time").reset_index(drop=True)
    )
    combined = validate_candles(combined, start, end)

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.resolve().parent, suffix=".feather", delete=False) as handle:
            temporary = Path(handle.name)
        combined.to_feather(temporary)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Validated {len(combined):,} closed candles in {path}")
    return combined[combined["open_time"] >= requested_start].reset_index(drop=True)


if __name__ == "__main__":
    config = load_config()
    manage_local_data(config["symbol"], start_date_dt=config["start_date"])
