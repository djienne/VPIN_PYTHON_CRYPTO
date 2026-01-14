import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import asyncio
import json
import os
from datetime import datetime, timedelta
import market_data 

# --- Load Config ---
CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "symbol": "BNBUSDT",
    # Target ~200 buckets/day per book Ch.22 calibration (range: 50-200)
    "bucket_target_bars_per_day": 200,
    # ADV lookback (days) used to size the bucket: V = ADV / bucket_target_bars_per_day
    "adv_lookback_days": 90,
    # Optional manual override for bucket size in base units (e.g., BNB, BTC). If set, skips ADV sizing.
    "bucket_size_base": None,
    # VPIN rolling window: 30 bars per book Ch.22 calibration
    "vpin_window": 30,
    "cdf_lookback_days": 90,
    "start_date": "2025-01-01",
    # Include partial final bucket if >10% filled (default False to match original behavior)
    "include_partial_bucket": False,
    # Center date for zoom plot (YYYY-MM-DD format, or null to use data midpoint)
    "zoom_center_date": None
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}. Using defaults.")
    return DEFAULT_CONFIG

CONFIG = load_config()
SYMBOL = CONFIG.get("symbol", "BNBUSDT")
TARGET_BUCKETS_PER_DAY = CONFIG.get("bucket_target_bars_per_day", 50)
ADV_LOOKBACK_DAYS = CONFIG.get("adv_lookback_days", 90)
BUCKET_SIZE_BASE = CONFIG.get("bucket_size_base")
VPIN_WINDOW = CONFIG.get("vpin_window", 50)
CDF_LOOKBACK_DAYS = CONFIG.get("cdf_lookback_days", 90)
START_DATE_STR = CONFIG.get("start_date", "2025-01-01")
INCLUDE_PARTIAL_BUCKET = CONFIG.get("include_partial_bucket", False)
ZOOM_CENTER_DATE = CONFIG.get("zoom_center_date")

def make_volume_buckets(df_1m, bucket_size, include_partial=False):
    """
    Resample 1-minute klines into Volume Buckets (Volume Clock sampling).

    Per book Ch.22 calibration: prices each volume bar using the volume-weighted
    median price of all trades within the bar, not just the closing price.

    Uses base asset volume (Binance fields: Volume, Taker buy base asset volume).

    Args:
        df_1m: DataFrame with 1-minute klines
        bucket_size: Target volume for each bucket (V)
        include_partial: If True, include the final partially-filled bucket (default False)

    Returns:
        DataFrame with columns: time, buy_volume, sell_volume, total_volume, price
    """
    buckets = []

    current_bucket_buy = 0.0
    current_bucket_sell = 0.0
    # Volume-weighted price accumulator (per book Ch.22: use median price of trades in bar)
    current_bucket_price_x_vol = 0.0
    volume_remaining = bucket_size

    if df_1m.empty:
        return pd.DataFrame()

    current_bucket_start_time = None

    print(f"Processing {len(df_1m)} 1-minute candles into buckets of size {bucket_size:,.2f} (base units)...")

    times = df_1m["open_time"].values
    # Use Median Price (High + Low)/2 per book Ch.22 calibration
    highs = df_1m["high"].values
    lows = df_1m["low"].values
    median_prices = (highs + lows) / 2.0

    total_vols = df_1m["volume"].values.astype(float)
    buy_vols_raw = df_1m["taker_buy_base_asset_volume"].values.astype(float)

    # Clamp taker-buy volume to [0, total] to avoid anomalies where buy > total.
    buy_vols = np.minimum(np.maximum(buy_vols_raw, 0.0), total_vols)
    # Sell vol = Total - Buy (non-negative by construction).
    sell_vols = total_vols - buy_vols

    for i in range(len(df_1m)):
        total_vol = total_vols[i]
        buy_vol = buy_vols[i]
        sell_vol = sell_vols[i]
        current_time = times[i]
        current_price = median_prices[i]

        while total_vol > 1e-12:
            if current_bucket_start_time is None:
                current_bucket_start_time = current_time

            fill_amount = min(total_vol, volume_remaining)
            if total_vol == 0:
                break
            ratio = fill_amount / total_vol

            add_buy = buy_vol * ratio
            add_sell = sell_vol * ratio

            current_bucket_buy += add_buy
            current_bucket_sell += add_sell
            # Accumulate volume-weighted price contribution
            current_bucket_price_x_vol += fill_amount * current_price

            total_vol -= fill_amount
            buy_vol -= add_buy
            sell_vol -= add_sell
            volume_remaining -= fill_amount

            if volume_remaining <= 1e-12:
                # Compute volume-weighted average price for this bucket
                bucket_total_vol = current_bucket_buy + current_bucket_sell
                bucket_price = current_bucket_price_x_vol / bucket_total_vol if bucket_total_vol > 0 else current_price

                buckets.append({
                    "time": current_bucket_start_time,
                    "buy_volume": current_bucket_buy,
                    "sell_volume": current_bucket_sell,
                    "total_volume": bucket_total_vol,
                    "price": bucket_price
                })

                # Reset accumulators for next bucket
                current_bucket_buy = 0.0
                current_bucket_sell = 0.0
                current_bucket_price_x_vol = 0.0
                volume_remaining = bucket_size
                current_bucket_start_time = None

    # Include partial final bucket if requested and there's meaningful volume
    if include_partial and current_bucket_start_time is not None:
        partial_volume = current_bucket_buy + current_bucket_sell
        if partial_volume > bucket_size * 0.1:  # Only include if >10% filled
            partial_price = current_bucket_price_x_vol / partial_volume if partial_volume > 0 else 0.0
            buckets.append({
                "time": current_bucket_start_time,
                "buy_volume": current_bucket_buy,
                "sell_volume": current_bucket_sell,
                "total_volume": partial_volume,
                "price": partial_price,
                "partial": True  # Flag to indicate this is incomplete
            })
            print(f"Included partial final bucket ({partial_volume/bucket_size*100:.1f}% filled)")

    return pd.DataFrame(buckets)

def calculate_vpin_metric(buckets_df, bucket_size, window_n=30, cdf_lookback_days=90):
    """
    Calculate VPIN and rolling CDF per book Section 19.5.2 and Ch.22 calibration.

    VPIN Formula (book p.292):
        VPIN_τ = Σ|V_τ^B - V_τ^S| / (n × V)

    Where:
        - n: number of volume bars in the rolling window (default 30 per Ch.22)
        - V: fixed volume size of each bar (bucket_size)
        - V_τ^B, V_τ^S: buy/sell volumes within bar τ

    CDF normalization (Section 19.7): converts raw VPIN to percentile rank
    within a historical lookback window for threshold-based signaling.

    Args:
        buckets_df: DataFrame from make_volume_buckets()
        bucket_size: The fixed volume V per bucket
        window_n: Rolling window size in buckets (default 30 per Ch.22)
        cdf_lookback_days: Lookback period for CDF calculation

    Returns:
        DataFrame with VPIN and CDF columns added
    """
    df = buckets_df.copy()
    if "time" in df.columns:
        df = df.set_index("time")

    # Order Flow Imbalance: OI_τ = |V_τ^B - V_τ^S|
    df["OI"] = (df["buy_volume"] - df["sell_volume"]).abs()

    # VPIN = Σ OI / (n × V) over rolling window of n buckets
    df["OI_rolling_sum"] = df["OI"].rolling(window=window_n).sum()
    df["VPIN"] = df["OI_rolling_sum"] / (window_n * bucket_size)

    # Calculate Rolling CDF (per Section 19.7)
    print(f"Calculating Rolling CDF ({cdf_lookback_days} days)...")
    df = df.sort_index()

    window_str = f"{cdf_lookback_days}D"

    # Percentile rank: P(X <= VPIN_t) within the lookback window
    # Per book: CDF > 0.99 signals extreme toxicity
    def rank_func(x):
        current = x[-1]
        if np.isnan(current):
            return np.nan
        x = x[~np.isnan(x)]
        if x.size == 0:
            return np.nan
        return (x <= current).mean()

    df["CDF"] = df["VPIN"].rolling(window=window_str).apply(rank_func, raw=True)

    df = df.reset_index()  # Bring time back as column
    return df

def _find_toxic_spans(plot_data, threshold=0.99):
    """
    Find contiguous time spans where CDF exceeds the threshold.
    """
    toxic_mask = plot_data["CDF"] >= threshold
    spans = []
    current_start = None
    last_time = None
    for t, toxic in zip(plot_data["time"], toxic_mask):
        if toxic and current_start is None:
            current_start = t
        if not toxic and current_start is not None:
            spans.append((current_start, last_time))
            current_start = None
        last_time = t
    if current_start is not None and last_time is not None:
        spans.append((current_start, last_time))
    return spans


def export_extreme_events(vpin_df, threshold=0.99, cluster_gap_days=3, output_path="vpin_extreme_events.csv"):
    """
    Export clustered extreme events (CDF >= threshold) to CSV.

    Events within cluster_gap_days of each other are grouped as a single event.

    Args:
        vpin_df: DataFrame with VPIN and CDF columns
        threshold: CDF threshold for extreme events (default 0.99)
        cluster_gap_days: Days gap to consider events as separate clusters
        output_path: Output CSV file path

    Returns:
        DataFrame of clustered events
    """
    df = vpin_df.dropna(subset=["CDF"]).copy()
    extreme = df[df["CDF"] >= threshold].copy()

    if extreme.empty:
        print(f"No extreme events (CDF >= {threshold}) found.")
        return pd.DataFrame()

    extreme = extreme.sort_values("time")
    extreme["time_diff"] = extreme["time"].diff()
    extreme["new_cluster"] = (
        (extreme["time_diff"] > pd.Timedelta(days=cluster_gap_days)) |
        extreme["time_diff"].isna()
    )
    extreme["cluster_id"] = extreme["new_cluster"].cumsum()

    # Aggregate clusters
    clusters = extreme.groupby("cluster_id").agg(
        start_time=("time", "min"),
        end_time=("time", "max"),
        duration_hours=("time", lambda x: (x.max() - x.min()).total_seconds() / 3600),
        peak_vpin=("VPIN", "max"),
        peak_cdf=("CDF", "max"),
        avg_vpin=("VPIN", "mean"),
        bucket_count=("time", "count")
    ).reset_index(drop=True)

    clusters["start_date"] = clusters["start_time"].dt.strftime("%Y-%m-%d")
    clusters["end_date"] = clusters["end_time"].dt.strftime("%Y-%m-%d")

    # Reorder columns for clarity
    clusters = clusters[[
        "start_date", "end_date", "start_time", "end_time",
        "duration_hours", "peak_vpin", "peak_cdf", "avg_vpin", "bucket_count"
    ]]

    clusters.to_csv(output_path, index=False)
    print(f"Exported {len(clusters)} extreme events to {output_path}")

    # Print summary by year
    clusters["year"] = pd.to_datetime(clusters["start_time"]).dt.year
    print(f"\nExtreme events per year (CDF >= {threshold}):")
    print(clusters.groupby("year").size().to_string())

    return clusters

def plot_vpin(vpin_df, symbol, bucket_size, cdf_window_days):
    print("Generating plot with Price, VPIN, and CDF...")
    plot_data = vpin_df.dropna(subset=["VPIN", "CDF"]).sort_values("time")
    
    if plot_data.empty:
        print("No VPIN data to plot.")
        return

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [2, 2, 2]})
    
    # 0. Price
    if "price" in plot_data.columns:
        ax0.plot(plot_data["time"], plot_data["price"], label="Price", color='black', linewidth=1)
        ax0.set_title(f"{symbol} Price")
        ax0.set_ylabel("Price")
        ax0.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax0.legend(loc='upper left')

    # 1. Raw VPIN
    ax1.plot(plot_data["time"], plot_data["VPIN"], label="VPIN", color='blue', linewidth=1)
    ax1.set_title(f"VPIN Metric (Bucket: {bucket_size:,.0f} base units)")
    ax1.set_ylabel("VPIN Value")
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.legend(loc='upper left')
    
    # 2. CDF with Thresholds
    ax2.plot(plot_data["time"], plot_data["CDF"], label=f"CDF ({cdf_window_days}d Rolling)", color='purple', linewidth=1)
    ax2.set_title("VPIN CDF (Toxicity Probability)")
    ax2.set_ylabel("CDF Percentile")
    ax2.set_xlabel("Date")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Thresholds
    ax2.axhline(y=0.90, color='gold', linestyle='--', label="> 0.90 (Elevated)")
    ax2.axhline(y=0.95, color='orange', linestyle='--', label="> 0.95 (High)")
    ax2.axhline(y=0.99, color='red', linestyle='-', label="> 0.99 (Extreme)")
    
    # Shade regions where CDF exceeds 0.99 across all panels (primary highlight on price chart)
    spans = _find_toxic_spans(plot_data, threshold=0.99)
    shaded = False
    for start, end in spans:
        ax0.axvspan(start, end, color="red", alpha=0.1, label="CDF > 0.99" if not shaded else None)
        ax1.axvspan(start, end, color="red", alpha=0.05)
        ax2.axvspan(start, end, color="red", alpha=0.05)
        shaded = True

    ax2.legend(loc='lower right')
    
    output_img = "vpin_plot.png"
    plt.tight_layout()
    plt.savefig(output_img)
    print(f"Plot saved to {output_img}")

def plot_vpin_zoom_range(vpin_df, symbol, bucket_size, cdf_window_days, center_date, days_window=15, output_path="vpin_plot_zoom.png"):
    """
    Plot a zoomed window around center_date +/- days_window.
    """
    try:
        center_dt = pd.to_datetime(center_date)
    except Exception:
        print(f"Invalid center_date {center_date}; skipping zoom plot.")
        return

    start_dt = center_dt - pd.Timedelta(days=days_window)
    end_dt = center_dt + pd.Timedelta(days=days_window)

    subset = vpin_df[(vpin_df["time"] >= start_dt) & (vpin_df["time"] <= end_dt)].dropna(subset=["VPIN", "CDF"]).sort_values("time")
    if subset.empty:
        print(f"No data in zoom window {start_dt} to {end_dt}; skipping zoom plot.")
        return

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [2, 2, 2]})

    ax0.plot(subset["time"], subset["price"], label="Price", color='black', linewidth=1)
    ax0.set_title(f"{symbol} Price (Zoom {center_date} +/- {days_window}d)")
    ax0.set_ylabel("Price")
    ax0.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax0.legend(loc='upper left')

    ax1.plot(subset["time"], subset["VPIN"], label="VPIN", color='blue', linewidth=1)
    ax1.set_title(f"VPIN Metric (Bucket: {bucket_size:,.0f} base units)")
    ax1.set_ylabel("VPIN Value")
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.legend(loc='upper left')

    ax2.plot(subset["time"], subset["CDF"], label=f"CDF ({cdf_window_days}d Rolling)", color='purple', linewidth=1)
    ax2.set_title("VPIN CDF (Toxicity Probability)")
    ax2.set_ylabel("CDF Percentile")
    ax2.set_xlabel("Date")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax2.axhline(y=0.90, color='gold', linestyle='--', label="> 0.90 (Elevated)")
    ax2.axhline(y=0.95, color='orange', linestyle='--', label="> 0.95 (High)")
    ax2.axhline(y=0.99, color='red', linestyle='-', label="> 0.99 (Extreme)")

    spans = _find_toxic_spans(subset, threshold=0.99)
    shaded = False
    for start, end in spans:
        ax0.axvspan(start, end, color="red", alpha=0.1, label="CDF > 0.99" if not shaded else None)
        ax1.axvspan(start, end, color="red", alpha=0.05)
        ax2.axvspan(start, end, color="red", alpha=0.05)
        shaded = True

    ax2.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Zoom plot saved to {output_path}")

def compute_bucket_size_base(df_1m, symbol, target_bars_per_day, adv_lookback_days, override=None, allow_adv_download=True):
    """
    Determine bucket size V in base units (ADV / target_bars_per_day).
    Preference order: override -> ADV fetch -> local data fallback.
    """
    if override is not None:
        print(f"Using bucket size override: {override:,.2f} (base units)")
        return float(override)

    # 1) ADV from Binance daily data
    if allow_adv_download:
        try:
            adv = asyncio.run(market_data.get_adv_async(symbol, days=adv_lookback_days))
            bucket_size = adv / target_bars_per_day
            print(f"Computed bucket size from ADV ({adv_lookback_days}d): ADV={adv:,.2f} -> V={bucket_size:,.2f} (base units)")
            return bucket_size
        except Exception as e:
            print(f"Could not compute bucket size from ADV API: {e}")
    else:
        print("Skipping ADV download (cache updated <15 minutes ago). Using local data to size bucket.")

    # 2) Fallback: use the loaded 1m data to approximate ADV
    try:
        daily_vol = df_1m.set_index("open_time")["volume"].resample("1D").sum()
        if not daily_vol.empty:
            adv_local = daily_vol.mean()
            bucket_size = adv_local / target_bars_per_day
            print(f"Using local ADV fallback over {len(daily_vol)} days: ADV={adv_local:,.2f} -> V={bucket_size:,.2f} (base units)")
            return bucket_size
    except Exception as e:
        print(f"Local ADV fallback failed: {e}")

    return None

def main():
    print(f"--- Starting VPIN Calculation for {SYMBOL} ---")
    print(f"Config: Target ~{TARGET_BUCKETS_PER_DAY} buckets/day, VPIN Window={VPIN_WINDOW}, CDF Window={CDF_LOOKBACK_DAYS}d, Start={START_DATE_STR}")
    
    # 1. Manage Data
    try:
        start_date = datetime.strptime(START_DATE_STR, "%Y-%m-%d")
    except ValueError:
        print(f"Invalid date format for {START_DATE_STR}. Using 2025-01-01")
        start_date = datetime(2025, 1, 1)

    df_1m = market_data.manage_local_data(SYMBOL, "1m", start_date_dt=start_date)
    
    if df_1m.empty:
        print("No data available.")
        return
        
    df_1m = df_1m[df_1m["open_time"] >= start_date]
    print(f"Data available: {len(df_1m)} rows from {df_1m['open_time'].min()} to {df_1m['open_time'].max()}")

    # 2. Determine bucket size (base units) and build volume buckets
    allow_adv_download = True
    try:
        mtime = datetime.fromtimestamp(os.path.getmtime(market_data.get_data_file(SYMBOL)))
        if datetime.now() - mtime < timedelta(minutes=15):
            allow_adv_download = False
    except Exception as e:
        print(f"Could not read cache mtime for ADV decision: {e}")

    bucket_size_base = compute_bucket_size_base(
        df_1m,
        SYMBOL,
        TARGET_BUCKETS_PER_DAY,
        ADV_LOOKBACK_DAYS,
        BUCKET_SIZE_BASE,
        allow_adv_download=allow_adv_download
    )

    if bucket_size_base is None or bucket_size_base <= 0:
        print("Unable to determine bucket size; aborting.")
        return

    buckets_df = make_volume_buckets(df_1m, bucket_size_base, include_partial=INCLUDE_PARTIAL_BUCKET)
    print(f"Created {len(buckets_df)} volume buckets.")
    
    if len(buckets_df) < VPIN_WINDOW:
        print(f"Not enough buckets for VPIN calculation (Need {VPIN_WINDOW}).")
        return

    # 3. Calculate VPIN & CDF
    vpin_df = calculate_vpin_metric(buckets_df, bucket_size_base, window_n=VPIN_WINDOW, cdf_lookback_days=CDF_LOOKBACK_DAYS)
    
    # 4. Output
    print("\n--- Recent VPIN Values ---")
    print(vpin_df[["time", "VPIN", "CDF"]].dropna().tail(10))
    
    plot_vpin(vpin_df, SYMBOL, bucket_size_base, CDF_LOOKBACK_DAYS)

    # Determine zoom center date: use config, or fall back to data midpoint
    if ZOOM_CENTER_DATE:
        zoom_date = ZOOM_CENTER_DATE
    else:
        # Use midpoint of available data
        valid_times = vpin_df.dropna(subset=["VPIN"])["time"]
        if not valid_times.empty:
            mid_idx = len(valid_times) // 2
            zoom_date = valid_times.iloc[mid_idx].strftime("%Y-%m-%d")
        else:
            zoom_date = None

    if zoom_date:
        plot_vpin_zoom_range(vpin_df, SYMBOL, bucket_size_base, CDF_LOOKBACK_DAYS, center_date=zoom_date, days_window=15, output_path="vpin_plot_zoom.png")

    vpin_df.to_csv("vpin_results.csv", index=False)
    print("Saved results to vpin_results.csv")

    # Export extreme events (CDF >= 0.99) to separate CSV
    # cluster_gap_days=1 groups events within the same day
    export_extreme_events(vpin_df, threshold=0.99, cluster_gap_days=1, output_path="vpin_extreme_events.csv")

if __name__ == "__main__":
    main()
