"""Adaptive VPIN approximation from completed one-minute candles."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import market_data
from market_data import load_config

MIN_CDF_HISTORY = 100


def compute_bucket_size_base(df_1m, target_bars_per_day, adv_lookback_days, override=None):
    """Return index-aligned bucket targets in base units from preceding complete UTC days.

    Calibration rows have NaN targets. A positive override supplies a constant Series
    and bypasses ADV calibration; targets take effect when a new bucket starts.
    """
    if target_bars_per_day <= 0 or adv_lookback_days <= 0:
        raise ValueError("Bucket count and ADV lookback must be positive")
    if override is not None:
        if not np.isfinite(override) or override <= 0:
            raise ValueError("Bucket override must be finite and positive")
        return pd.Series(float(override), index=df_1m.index)
    times = pd.to_datetime(df_1m["open_time"], utc=True)
    daily_volume = df_1m.set_index(times)["volume"].resample("1D").agg(["sum", "count"])
    complete_days = daily_volume["sum"].where(daily_volume["count"] == 1440)
    prior_adv = complete_days.rolling(adv_lookback_days, min_periods=adv_lookback_days).mean().shift(1)
    sizes = times.dt.normalize().map(prior_adv / target_bars_per_day)
    if (sizes.dropna() <= 0).any() or np.isinf(sizes).any():
        raise ValueError("Preceding ADV must be finite and positive")
    return sizes


def make_volume_buckets(df_1m, bucket_sizes):
    """Build full buckets from scalar or row-aligned Series targets; omit the partial tail.

    Leading NaNs skip calibration. Each target stays fixed until its bucket fills.
    Buy/sell volume is split proportionally; time is final candle close + 1 ms and
    price is that candle's close, not an execution price. Equal-time order is retained.
    """
    columns = ["time", "buy_volume", "sell_volume", "total_volume", "price"]
    if df_1m.empty:
        return pd.DataFrame(columns=columns)
    df = market_data.validate_candles(df_1m)
    if np.isscalar(bucket_sizes):
        sizes = np.full(len(df), float(bucket_sizes))
    else:
        if not isinstance(bucket_sizes, pd.Series) or not bucket_sizes.index.equals(df.index):
            raise ValueError("Bucket targets must align with candle rows")
        sizes = bucket_sizes.to_numpy(dtype=float)
    valid = np.isfinite(sizes)
    if np.isinf(sizes).any() or (sizes[valid] <= 0).any():
        raise ValueError("Bucket targets must be positive and finite")
    if valid.any() and not valid[np.flatnonzero(valid)[0]:].all():
        raise ValueError("Missing bucket target after calibration")
    buckets = []
    remaining = buy = sell = 0.0
    available_times = df["close_time"] + pd.Timedelta(milliseconds=1)
    for time, price, volume, buy_volume, target in zip(
        available_times, df["close"], df["volume"], df["taker_buy_base_asset_volume"], sizes
    ):
        if not np.isfinite(target) or volume == 0:
            continue
        buy_fraction = buy_volume / volume
        while volume > 0:
            if remaining == 0:
                if volume - target == volume:
                    raise ValueError("Bucket target is too small relative to candle volume")
                remaining = target
            fill = min(volume, remaining)
            buy_part = fill * buy_fraction
            buy += buy_part
            sell += fill - buy_part
            volume -= fill
            remaining -= fill
            if remaining == 0:
                buckets.append((time, buy, sell, buy + sell, price))
                buy = sell = 0.0
    result = pd.DataFrame(buckets, columns=columns)
    result["time"] = pd.to_datetime(result["time"], utc=True)
    return result


def calculate_vpin_metric(buckets_df, window_n=10, cdf_lookback_days=90):
    """Add VPIN using actual rolling volume and CDF using preceding bucket observations.

    CDF excludes the current bucket, counts ties, and preserves equal-time order.
    It is NaN until the full time lookback and minimum history are available, or
    when the reference is constant. It is a historical rank, not an event probability.
    """
    if window_n <= 0 or cdf_lookback_days <= 0:
        raise ValueError("VPIN and CDF windows must be positive")
    df = buckets_df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time", kind="stable").set_index("time")
    df["OI"] = (df["buy_volume"] - df["sell_volume"]).abs()
    df["OI_rolling_sum"] = df["OI"].rolling(window_n).sum()
    df["VPIN"] = df["OI_rolling_sum"] / df["total_volume"].rolling(window_n).sum()
    if df.empty:
        return df.assign(CDF=pd.Series(dtype=float)).reset_index()

    def percentile(values):
        history = values[:-1]
        history = history[np.isfinite(history)]
        if (
            not np.isfinite(values[-1]) or len(history) < MIN_CDF_HISTORY
            or history.min() == history.max()
        ):
            return np.nan
        return np.mean(history <= values[-1])

    reference = df["VPIN"].rolling(f"{cdf_lookback_days}D", min_periods=MIN_CDF_HISTORY + 1)
    df["CDF"] = reference.apply(percentile, raw=True)
    first_valid = df["VPIN"].first_valid_index()
    if first_valid is not None:
        df.loc[df.index < first_valid + pd.Timedelta(days=cdf_lookback_days), "CDF"] = np.nan
    return df.reset_index()


def configured_buckets(df_1m, config, buckets_per_day=None):
    sizes = compute_bucket_size_base(
        df_1m,
        config["bucket_target_bars_per_day"] if buckets_per_day is None else buckets_per_day,
        config["adv_lookback_days"],
        config["bucket_size_base"],
    )
    return make_volume_buckets(df_1m, sizes)


def _find_toxic_spans(plot_data, threshold=0.99):
    spans = []
    start = last = None
    for time, toxic in zip(plot_data["time"], plot_data["CDF"] >= threshold):
        if toxic and start is None:
            start = time
        if not toxic and start is not None:
            spans.append((start, last))
            start = None
        last = time
    if start is not None:
        spans.append((start, last))
    return spans


def export_extreme_events(vpin_df, threshold=0.99, cluster_gap_days=3, output_path="vpin_extreme_events.csv"):
    extreme = vpin_df.loc[vpin_df["CDF"] >= threshold].sort_values("time", kind="stable").copy()
    columns = [
        "start_date", "end_date", "start_time", "end_time", "duration_hours",
        "peak_vpin", "peak_cdf", "avg_vpin", "bucket_count",
    ]
    if extreme.empty:
        clusters = pd.DataFrame(columns=columns)
    else:
        gaps = extreme["time"].diff()
        extreme["cluster"] = (gaps.isna() | (gaps > pd.Timedelta(days=cluster_gap_days))).cumsum()
        clusters = extreme.groupby("cluster").agg(
            start_time=("time", "min"), end_time=("time", "max"),
            peak_vpin=("VPIN", "max"), peak_cdf=("CDF", "max"),
            avg_vpin=("VPIN", "mean"), bucket_count=("time", "size"),
        ).reset_index(drop=True)
        clusters["duration_hours"] = (clusters["end_time"] - clusters["start_time"]).dt.total_seconds() / 3600
        clusters["start_date"] = clusters["start_time"].dt.strftime("%Y-%m-%d")
        clusters["end_date"] = clusters["end_time"].dt.strftime("%Y-%m-%d")
        clusters = clusters[columns]
    clusters.to_csv(output_path, index=False)
    print(f"Exported {len(clusters)} extreme-event clusters to {output_path}")
    return clusters


def plot_vpin(vpin_df, symbol, cdf_window_days, output_path="vpin_plot.png", title_suffix=""):
    data = vpin_df.dropna(subset=["VPIN"]).sort_values("time", kind="stable")
    if data.empty:
        print("No completed VPIN windows to plot.")
        return
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    panels = [
        ("price", "black", f"{symbol} candle close at bucket completion"),
        ("VPIN", "blue", "VPIN (actual rolling volume)"),
        ("CDF", "purple", f"VPIN percentile ({cdf_window_days} preceding days)"),
    ]
    toxic_spans = _find_toxic_spans(data)
    for ax, (column, color, title) in zip(axes, panels):
        ax.plot(data["time"], data[column], color=color, linewidth=1)
        ax.set_title(title + title_suffix)
        ax.grid(True, alpha=0.3)
        for start, end in toxic_spans:
            ax.axvspan(start, end, color="red", alpha=0.1)
    for threshold, color in [(0.90, "gold"), (0.95, "orange"), (0.99, "red")]:
        axes[2].axhline(threshold, color=color, linestyle="--", label=str(threshold))
    axes[1].set_ylim(0, 1.05)
    axes[2].set_ylim(0, 1.05)
    axes[2].legend()
    axes[2].set_xlabel("Signal availability (UTC)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_vpin_zoom_range(vpin_df, symbol, cdf_window_days, center_date, days_window=15, output_path="vpin_plot_zoom.png"):
    center = pd.to_datetime(center_date, utc=True)
    window = pd.Timedelta(days=days_window)
    subset = vpin_df[vpin_df["time"].between(center - window, center + window)]
    plot_vpin(subset, symbol, cdf_window_days, output_path, f" ({center.date()} +/- {days_window}d)")


def plot_vpin_recent_range(vpin_df, symbol, cdf_window_days, days_back=14, output_path="vpin_plot_zoom_latest.png"):
    latest = vpin_df["time"].max()
    subset = vpin_df[vpin_df["time"] >= latest - pd.Timedelta(days=days_back)]
    plot_vpin(subset, symbol, cdf_window_days, output_path, f" (last {days_back} days)")


def main():
    config = load_config()
    candles = market_data.manage_local_data(config["symbol"], start_date_dt=config["start_date"])
    buckets = configured_buckets(candles, config)
    result = calculate_vpin_metric(buckets, config["vpin_window"], config["cdf_lookback_days"])
    result.to_csv("vpin_results.csv", index=False)
    export_extreme_events(result, cluster_gap_days=1)
    if result["VPIN"].notna().sum() == 0:
        print("Insufficient history after ADV calibration for a complete VPIN window.")
        return
    if result["CDF"].notna().sum() == 0:
        print(f"CDF unavailable: need the full lookback, {MIN_CDF_HISTORY} prior VPINs, and a nonconstant reference.")
    print(result[["time", "total_volume", "VPIN", "CDF"]].tail(10).to_string(index=False))
    plot_vpin(result, config["symbol"], config["cdf_lookback_days"])
    center = config["zoom_center_date"] or result["time"].iloc[len(result) // 2]
    plot_vpin_zoom_range(result, config["symbol"], config["cdf_lookback_days"], center)
    if config["latest_zoom_days_back"]:
        plot_vpin_recent_range(result, config["symbol"], config["cdf_lookback_days"], config["latest_zoom_days_back"])


if __name__ == "__main__":
    main()
