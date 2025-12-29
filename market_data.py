import asyncio
import aiohttp
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# --- Constants ---
BINANCE_BASE_URL = "https://api.binance.com"
MAX_RETRIES = 5
CONCURRENT_REQUESTS = 10 
DATA_FILE = "bnbusdt_1m.feather"

async def fetch_kline_chunk(session, symbol, interval, start_time, end_time, limit=1000):
    """
    Fetch a single chunk of klines asynchronously with retry logic.
    """
    url = f"{BINANCE_BASE_URL}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": limit
    }
    
    # "Rate limit of like 50 ms" - adding delay before request
    await asyncio.sleep(0.05)
    
    for attempt in range(MAX_RETRIES):
        try:
            async with session.get(url, params=params) as response:
                if response.status == 429:
                    wait_time = int(response.headers.get("Retry-After", 2 ** attempt))
                    print(f"Rate limited. Waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                
                if response.status == 418:
                    wait_time = int(response.headers.get("Retry-After", 60))
                    print(f"IP Ban imminent! Waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue

                if response.status != 200:
                    print(f"Error fetching chunk {start_time}: Status {response.status}")
                    await asyncio.sleep(2 ** attempt)
                    continue
                    
                data = await response.json()
                return data
                
        except Exception as e:
            print(f"Exception fetching chunk {start_time}: {e}")
            await asyncio.sleep(2 ** attempt)
            
    return []

async def download_historical_data_async(symbol, interval, start_ts, end_ts):
    """
    Download historical data in parallel chunks.
    """
    print(f"Downloading data from {datetime.fromtimestamp(start_ts/1000)} to {datetime.fromtimestamp(end_ts/1000)}...")
    
    chunk_size_ms = 1000 * 60 * 1000 
    
    tasks = []
    chunk_starts = range(start_ts, end_ts, chunk_size_ms)
    
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
        
        async def bound_fetch(s, e):
            async with semaphore:
                actual_end = min(e, end_ts)
                return await fetch_kline_chunk(session, symbol, interval, s, actual_end)
        
        tasks = [bound_fetch(s, s + chunk_size_ms - 1) for s in chunk_starts]
        results = await asyncio.gather(*tasks)
        
    all_data = []
    for res in results:
        if res:
            all_data.extend(res)
            
    if not all_data:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume", 
        "close_time", "quote_asset_volume", "number_of_trades", 
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    
    numeric_cols = ["open", "high", "low", "close", "volume", "taker_buy_base_asset_volume", "quote_asset_volume", "taker_buy_quote_asset_volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    
    return df

def manage_local_data(symbol, interval="1m", start_date_dt=None):
    """
    Load local feather file, update it with new data (forward and backward), and save back.
    Ensures no duplicates and checks for gaps.
    """
    df_existing = pd.DataFrame()
    now_ts = datetime.now()
    
    # 1. Load existing
    if os.path.exists(DATA_FILE):
        try:
            df_existing = pd.read_feather(DATA_FILE)
            print(f"Loaded {len(df_existing)} rows from {DATA_FILE}")
        except Exception as e:
            print(f"Error loading cache: {e}. Starting fresh.")
    
    # 1b. Skip downloads if the cache was updated recently
    if not df_existing.empty:
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(DATA_FILE))
            if now_ts - mtime < timedelta(minutes=15):
                print(f"Last data update was {mtime} (<15 minutes ago). Skipping downloads.")
                return df_existing
        except Exception as e:
            print(f"Could not read cache mtime: {e}")
    
    end_ts = int(now_ts.timestamp() * 1000)
    
    # List of new dataframes to concat
    dfs_to_concat = []
    if not df_existing.empty:
        dfs_to_concat.append(df_existing)

    # 2a. Forward Fill (Newer data)
    if not df_existing.empty:
        last_time = df_existing["open_time"].max()
        forward_start_ts = int(last_time.timestamp() * 1000) + 60000 
    else:
        # If empty, we start from start_date_dt or reasonable default
        if start_date_dt:
            forward_start_ts = int(start_date_dt.timestamp() * 1000)
        else:
            # Default to 2 days ago if no start date provided and no existing data
            forward_start_ts = int((current_now - timedelta(days=2)).timestamp() * 1000)

    if forward_start_ts < end_ts:
        print(f"Checking for new data from {datetime.fromtimestamp(forward_start_ts/1000)}...")
        try:
            df_forward = asyncio.run(download_historical_data_async(symbol, interval, forward_start_ts, end_ts))
            if not df_forward.empty:
                 print(f"Downloaded {len(df_forward)} new rows (forward fill).")
                 dfs_to_concat.append(df_forward)
        except Exception as e:
            print(f"Async download (forward) failed: {e}")

    # 2b. Backward Fill (Older data)
    if not df_existing.empty and start_date_dt:
        first_time = df_existing["open_time"].min()
        # If existing data starts AFTER requested start date, we need to backfill
        if first_time > start_date_dt:
            backfill_end_ts = int(first_time.timestamp() * 1000) - 60000
            backfill_start_ts = int(start_date_dt.timestamp() * 1000)
            
            if backfill_start_ts < backfill_end_ts:
                print(f"Backfilling data from {start_date_dt} to {first_time}...")
                try:
                    df_backward = asyncio.run(download_historical_data_async(symbol, interval, backfill_start_ts, backfill_end_ts))
                    if not df_backward.empty:
                        print(f"Downloaded {len(df_backward)} older rows (backfill).")
                        dfs_to_concat.append(df_backward)
                except Exception as e:
                    print(f"Async download (backward) failed: {e}")

    # 3. Merge and Save
    if not dfs_to_concat:
        print("No data in cache or downloaded.")
        return pd.DataFrame()
        
    df_combined = pd.concat(dfs_to_concat)
        
    # 4. Sanity Checks
    initial_len = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    dedup_len = len(df_combined)
    if initial_len != dedup_len:
        print(f"Removed {initial_len - dedup_len} duplicate rows.")
        
    time_diff = df_combined["open_time"].diff()
    gaps = time_diff[time_diff > timedelta(minutes=1)]
    if not gaps.empty:
        print(f"WARNING: Found {len(gaps)} potential data gaps!")
        print(gaps.head())
    else:
        print("Data continuity check passed (no gaps > 1m).")
        
    df_combined.to_feather(DATA_FILE)
    print(f"Saved updated data to {DATA_FILE}")
    
    return df_combined

if __name__ == "__main__":
    import json
    import asyncio
    
    # Load config if available, else defaults
    config_path = "config.json"
    symbol = "BNBUSDT"
    start_date_str = "2025-01-01"
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                conf = json.load(f)
                symbol = conf.get("symbol", symbol)
                start_date_str = conf.get("start_date", start_date_str)
                print(f"Loaded config: Symbol={symbol}, Start={start_date_str}")
        except Exception as e:
            print(f"Failed to load config: {e}. Using defaults.")
    
    try:
        start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    except:
        start_dt = datetime(2025, 1, 1)
        
    print(f"--- Starting Standalone Data Download for {symbol} ---")
    manage_local_data(symbol, "1m", start_date_dt=start_dt)


async def get_adv_async(symbol, days=90):
    """
    Quick async fetch for daily data to compute ADV.
    """
    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    df = await download_historical_data_async(symbol, "1d", start_ts, end_ts)
    if df.empty:
        raise ValueError("Could not fetch daily data for ADV")
    return df["volume"].mean()
