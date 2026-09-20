"""One cash/quantity simulator shared by grid search and Optuna."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import market_data
from vpin_calculator import configured_buckets, calculate_vpin_metric

PARAM_GRID = {
    "buckets_per_day": [25, 50, 75, 100, 125, 150, 175, 200, 250, 300],
    "vpin_window": [15, 20, 30, 50, 75, 100, 125, 150, 175, 200, 250, 300],
}
BUCKETS_PER_DAY_RANGE = (20, 350)
VPIN_WINDOW_RANGE = (10, 350)
CDF_THRESHOLD = 0.99
COOLDOWN_DAYS = 2
MIN_EXITS_REQUIRED = 30
METRICS = ("total_pnl", "buy_hold_pnl", "excess_return", "num_exits", "win_rate",
           "max_drawdown", "buy_hold_max_drawdown", "cagr", "calmar")


def calculate_drawdowns(equity):
    """Percentage decline from peak wealth, including initial capital of 1."""
    peaks = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    return (equity / peaks - 1) * 100


def run_single_backtest(vpin_df, df_1m, *, start_time, end_time,
                        cdf_threshold=CDF_THRESHOLD, cooldown_days=COOLDOWN_DAYS,
                        fee_bps=10, slippage_bps=2):
    """Simulate [start_time, end_time) in UTC from unit cash using available CDF signals.

    Fees/slippage are basis points per side; missing CDF leaves the position unchanged.
    Return percentage metrics, unit-capital wealth at minute closes, and trade records.
    The final close liquidates holdings but is excluded from the signal-exit count.
    """
    start = pd.to_datetime(start_time, utc=True)
    end = pd.to_datetime(end_time, utc=True)
    if start >= end or start != start.floor("min") or end != end.floor("min"):
        raise ValueError("Evaluation bounds must be increasing, minute-aligned UTC times")
    if not 0 <= cdf_threshold <= 1 or not np.isfinite(cooldown_days) or cooldown_days < 0:
        raise ValueError("Invalid threshold or cooldown")
    if not all(np.isfinite(x) and 0 <= x < 10000 for x in (fee_bps, slippage_bps)):
        raise ValueError("Costs must be finite basis points in [0, 10000)")

    candles = df_1m.copy()
    candles["open_time"] = pd.to_datetime(candles["open_time"], utc=True)
    candles = candles[(candles["open_time"] >= start) & (candles["open_time"] < end)]
    candles = market_data.validate_candles(candles, start, end).reset_index(drop=True)

    signals = vpin_df[["time", "CDF"]].copy()
    signals["time"] = pd.to_datetime(signals["time"], utc=True)
    signals = signals.sort_values("time", kind="stable").drop_duplicates("time", keep="last")
    if signals.empty:
        known_cdf = np.full(len(candles), np.nan)
    else:
        known_cdf = pd.merge_asof(
            candles[["open_time"]], signals,
            left_on="open_time", right_on="time", direction="backward",
        )["CDF"].to_numpy()

    fee, slip = fee_bps / 10000, slippage_bps / 10000
    cash, quantity = 1.0, 0.0
    cooldown_until = start
    entry_time = entry_price = entry_cash = None
    trades = []
    equity = np.empty(len(candles))
    positions = np.empty(len(candles), dtype=bool)

    def close_position(market_price, time, cdf, reason):
        """Sell the current position and return net cash, recording the same costs for every exit."""
        exit_price = market_price * (1 - slip)
        proceeds = quantity * exit_price * (1 - fee)
        trades.append(dict(
            entry_time=entry_time, exit_time=time,
            entry_price=entry_price, exit_price=exit_price,
            pnl_pct=(proceeds / entry_cash - 1) * 100, exit_cdf=cdf, reason=reason,
        ))
        return proceeds

    for i, (time, open_price, close_price, cdf) in enumerate(zip(
        candles["open_time"], candles["open"], candles["close"], known_cdf
    )):
        if np.isfinite(cdf):
            if cdf >= cdf_threshold:
                if quantity > 0:
                    cash = close_position(open_price, time, cdf, "signal")
                    quantity = 0.0
                    cooldown_until = time + pd.Timedelta(days=cooldown_days)
            elif quantity == 0 and time >= cooldown_until:
                entry_time, entry_price, entry_cash = time, open_price * (1 + slip), cash
                quantity = cash / (entry_price * (1 + fee))
                cash = 0.0
        equity[i] = cash + quantity * close_price
        positions[i] = quantity > 0

    if quantity > 0:
        equity[-1] = close_position(candles["close"].iloc[-1], end, known_cdf[-1], "terminal")
        positions[-1] = False

    buy_hold_quantity = 1 / (candles["open"].iloc[0] * (1 + slip) * (1 + fee))
    buy_hold_equity = buy_hold_quantity * candles["close"].to_numpy(dtype=float)
    buy_hold_equity[-1] *= (1 - slip) * (1 - fee)

    drawdowns = calculate_drawdowns(equity)
    max_drawdown = float(drawdowns.min())
    days = (end - start).total_seconds() / 86400
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        cagr = float(np.expm1(np.log(equity[-1]) * 365.25 / days) * 100)
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 and np.isfinite(cagr) else np.nan
    signal_trades = [trade for trade in trades if trade["reason"] == "signal"]
    wins = sum(trade["pnl_pct"] > 0 for trade in signal_trades)
    win_rate = 100 * wins / len(signal_trades) if signal_trades else 0.0
    total_pnl = (equity[-1] - 1) * 100
    buy_hold_pnl = (buy_hold_equity[-1] - 1) * 100
    curve = pd.DataFrame({
        "time": candles["close_time"] + pd.Timedelta(milliseconds=1),
        "equity": equity, "buy_hold_equity": buy_hold_equity,
        "position": np.where(positions, "long", "flat"),
        "drawdown": drawdowns,
    })
    return dict(
        total_pnl=float(total_pnl), buy_hold_pnl=float(buy_hold_pnl),
        excess_return=float(total_pnl - buy_hold_pnl),
        num_exits=len(signal_trades), win_rate=win_rate,
        max_drawdown=max_drawdown, buy_hold_max_drawdown=float(calculate_drawdowns(buy_hold_equity).min()),
        cagr=cagr, calmar=calmar, trades=trades, equity_curve=curve,
    )


def prepare_evaluation(df_1m, config):
    """Return UTC (training_start, split, end) after common warmup, with an 80/20 time split.

    Raise ValueError for invalid candles or insufficient common history. Both scoring
    intervals are half-open; the end follows the last completed candle.
    """
    candles = market_data.validate_candles(df_1m)
    slowest_count = min(BUCKETS_PER_DAY_RANGE[0], config["bucket_target_bars_per_day"])
    largest_window = max(VPIN_WINDOW_RANGE[1], config["vpin_window"])
    slow = configured_buckets(candles, config, slowest_count)
    slow = calculate_vpin_metric(slow, largest_window, config["cdf_lookback_days"])
    ready = slow.loc[slow["CDF"].notna(), "time"]
    if ready.empty:
        raise ValueError("Insufficient history for common ADV, VPIN, and CDF warmup")
    start = ready.iloc[0]
    end = candles["close_time"].iloc[-1] + pd.Timedelta(milliseconds=1)
    split = (start + (end - start) * 0.8).floor("min")
    if split <= start or split >= end:
        raise ValueError("Insufficient history for both training and held-out evaluation")
    return start, split, end


def evaluate_parameters(df_1m, config, buckets_per_day, vpin_window, start, end, bucket_cache=None):
    """Score one training candidate, or return None if its initial CDF is unavailable.

    Data at/after end is excluded before calculation. Reuse bucket_cache only for
    the same input data, configuration, and end boundary; it holds one bucket set.
    """
    # Slice before indicator construction so held-out observations cannot affect fitness.
    training = df_1m[pd.to_datetime(df_1m["open_time"], utc=True) < end]
    cache = {} if bucket_cache is None else bucket_cache
    if buckets_per_day not in cache:
        # Keep one bucket set; use a bounded LRU only if rebuild time becomes a bottleneck.
        cache.clear()
        cache[buckets_per_day] = configured_buckets(training, config, buckets_per_day)
    vpin = calculate_vpin_metric(cache[buckets_per_day], vpin_window, config["cdf_lookback_days"])
    prior = vpin[vpin["time"] <= start]
    if prior.empty or not np.isfinite(prior["CDF"].iloc[-1]):
        return None
    return run_single_backtest(
        vpin, training, start_time=start, end_time=end,
        fee_bps=config["fee_bps"], slippage_bps=config["slippage_bps"],
    )


def result_record(buckets_per_day, vpin_window, result, start, end, phase="train"):
    """Flatten metrics for CSV; only qualifying training rows are eligible for selection."""
    row = dict(buckets_per_day=int(buckets_per_day), vpin_window=int(vpin_window),
               phase=phase, start_time=start, end_time=end)
    row.update({key: result[key] if result is not None else np.nan for key in METRICS})
    row["eligible"] = bool(
        phase == "train" and result is not None
        and result["num_exits"] >= MIN_EXITS_REQUIRED and np.isfinite(result["calmar"])
    )
    return row


def run_parameter_sweep(df_1m, config, start, end):
    counts = sorted(set(PARAM_GRID["buckets_per_day"] + [config["bucket_target_bars_per_day"]]))
    if config["bucket_size_base"] is not None:
        counts = [config["bucket_target_bars_per_day"]]
    windows = sorted(set(PARAM_GRID["vpin_window"] + [config["vpin_window"]]))
    results, cache = [], {}
    for count in tqdm(counts, desc="Training grid"):
        for window in windows:
            result = evaluate_parameters(df_1m, config, count, window, start, end, cache)
            results.append(result_record(count, window, result, start, end))
    return pd.DataFrame(results)


def plot_equity_curves(best_result, result, output_path="vpin_backtest_equity.png"):
    curve = result["equity_curve"]
    fig, ax = plt.subplots(figsize=(14, 8))
    for column, label in [("equity", "VPIN strategy"), ("buy_hold_equity", "Buy and hold")]:
        ax.plot(curve["time"], (curve[column] - 1) * 100, label=label)
    ax.set_title(
        f"Held-out evaluation: {int(best_result['buckets_per_day'])} buckets/day, "
        f"window {int(best_result['vpin_window'])}"
    )
    ax.set_xlabel("UTC")
    ax.set_ylabel("Net compounded return (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def report_results(results, df_1m, config, start, split, end, output_path):
    """Select from training rows, evaluate the winner on [split, end), and save both phases.

    Return its held-out simulator result, or None if no training row qualifies.
    Later performance never changes the selected parameters.
    """
    if results.empty:
        results = pd.DataFrame(columns=result_record(0, 0, None, start, split).keys())
    results = results.copy()
    valid = results[results["eligible"].eq(True)].sort_values("calmar", ascending=False, kind="stable")
    print(f"Training: [{start}, {split}); held out: [{split}, {end})")
    held_out = None
    if valid.empty:
        print(f"No valid training trials: require full warmup, {MIN_EXITS_REQUIRED} exits, and finite Calmar.")
    else:
        best = valid.iloc[0]
        count, window = int(best["buckets_per_day"]), int(best["vpin_window"])
        buckets = configured_buckets(df_1m, config, count)
        vpin = calculate_vpin_metric(buckets, window, config["cdf_lookback_days"])
        held_out = run_single_backtest(
            vpin, df_1m, start_time=split, end_time=end,
            fee_bps=config["fee_bps"], slippage_bps=config["slippage_bps"],
        )
        test_row = result_record(count, window, held_out, split, end, phase="test")
        results = pd.concat([results, pd.DataFrame([test_row])], ignore_index=True)
        print(f"Training winner: {count} buckets/day, window {window}, Calmar={best['calmar']:.3f}")
        print(
            f"Held out: return={held_out['total_pnl']:.2f}%, "
            f"drawdown={held_out['max_drawdown']:.2f}%, Calmar={held_out['calmar']:.3f}"
        )
        plot_equity_curves(best, held_out)
    for key in ("symbol", "adv_lookback_days", "cdf_lookback_days", "bucket_size_base", "fee_bps", "slippage_bps"):
        results[key] = config[key]
    results.to_csv(output_path, index=False)
    return held_out


def main():
    config = market_data.load_config()
    candles = market_data.manage_local_data(config["symbol"], start_date_dt=config["start_date"])
    try:
        start, split, end = prepare_evaluation(candles, config)
    except ValueError as exc:
        print(exc)
        return
    training = candles[candles["open_time"] < split]
    results = run_parameter_sweep(training, config, start, split)
    report_results(results, candles, config, start, split, end, "vpin_backtest_results.csv")


if __name__ == "__main__":
    main()
