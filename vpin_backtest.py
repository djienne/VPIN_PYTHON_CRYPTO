"""
VPIN Backtest Strategy

Strategy:
- Always long by default (exposed to market)
- Exit when VPIN CDF >= 0.99 (extreme toxicity)
- Stay out of market for 2 days after exit
- Re-enter after cooldown period

Sweeps bucket sizes and VPIN windows to find optimal parameters.
Compares against buy-and-hold baseline.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import functions from vpin_calculator
from vpin_calculator import make_volume_buckets, calculate_vpin_metric
import market_data


# Parameter grid to explore (expanded)
PARAM_GRID = {
    'buckets_per_day': [25, 50, 75, 100, 125, 150, 175, 200, 250, 300],
    'vpin_window': [15, 20, 30, 50, 75, 100, 125, 150, 175, 200, 250, 300],
}

CDF_THRESHOLD = 0.99
COOLDOWN_DAYS = 2
CDF_LOOKBACK_DAYS = 90
MIN_EXITS_REQUIRED = 30


def calculate_buy_hold_pnl(prices):
    """Calculate simple buy-and-hold return."""
    if len(prices) < 2:
        return 0.0
    return (prices.iloc[-1] / prices.iloc[0] - 1) * 100


def run_single_backtest(df, cdf_threshold=0.99, cooldown_days=2):
    """
    Run backtest on pre-computed VPIN/CDF data.

    Strategy:
    - Start long at first valid price
    - Exit when CDF >= threshold
    - Stay flat for cooldown_days
    - Re-enter after cooldown

    Args:
        df: DataFrame with 'time', 'price', 'CDF' columns
        cdf_threshold: CDF level to trigger exit
        cooldown_days: Days to stay out after exit

    Returns:
        dict with backtest results
    """
    df = df.dropna(subset=['CDF', 'price']).copy()
    df = df.sort_values('time').reset_index(drop=True)

    if len(df) < 10:
        return None

    # State tracking
    position = 'long'
    entry_price = df['price'].iloc[0]
    entry_time = df['time'].iloc[0]
    cooldown_until = None

    trades = []
    equity_curve = []

    # Track cumulative PnL
    cumulative_pnl = 0.0

    for idx, row in df.iterrows():
        current_time = row['time']
        current_price = row['price']
        current_cdf = row['CDF']

        # Check for re-entry after cooldown
        if position == 'flat' and cooldown_until is not None:
            if current_time >= cooldown_until:
                # Re-enter long
                position = 'long'
                entry_price = current_price
                entry_time = current_time
                cooldown_until = None

        # Check for exit signal
        if position == 'long' and current_cdf >= cdf_threshold:
            # Exit position
            exit_price = current_price
            trade_pnl = (exit_price / entry_price - 1) * 100  # Percentage return

            trades.append({
                'entry_time': entry_time,
                'exit_time': current_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': trade_pnl,
                'exit_cdf': current_cdf
            })

            cumulative_pnl += trade_pnl
            position = 'flat'
            cooldown_until = current_time + timedelta(days=cooldown_days)

        # Track equity (cumulative PnL + unrealized if in position)
        if position == 'long':
            unrealized = (current_price / entry_price - 1) * 100
            equity_curve.append({
                'time': current_time,
                'equity': cumulative_pnl + unrealized,
                'position': 'long'
            })
        else:
            equity_curve.append({
                'time': current_time,
                'equity': cumulative_pnl,
                'position': 'flat'
            })

    # Close any open position at end
    if position == 'long':
        final_price = df['price'].iloc[-1]
        final_pnl = (final_price / entry_price - 1) * 100
        trades.append({
            'entry_time': entry_time,
            'exit_time': df['time'].iloc[-1],
            'entry_price': entry_price,
            'exit_price': final_price,
            'pnl_pct': final_pnl,
            'exit_cdf': df['CDF'].iloc[-1]
        })
        cumulative_pnl += final_pnl

    # Calculate metrics
    num_exits = len([t for t in trades if t['exit_cdf'] >= cdf_threshold])
    total_pnl = cumulative_pnl
    buy_hold_pnl = calculate_buy_hold_pnl(df['price'])

    # Win rate (profitable exits due to VPIN signal)
    vpin_trades = [t for t in trades if t['exit_cdf'] >= cdf_threshold]
    if vpin_trades:
        wins = sum(1 for t in vpin_trades if t['pnl_pct'] > 0)
        win_rate = wins / len(vpin_trades) * 100
    else:
        win_rate = 0.0

    # Max drawdown from equity curve
    equity_df = pd.DataFrame(equity_curve)
    if not equity_df.empty:
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = equity_df['equity'] - equity_df['peak']
        max_drawdown = equity_df['drawdown'].min()
    else:
        max_drawdown = 0.0

    return {
        'total_pnl': total_pnl,
        'buy_hold_pnl': buy_hold_pnl,
        'excess_return': total_pnl - buy_hold_pnl,
        'num_exits': num_exits,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'trades': trades,
        'equity_curve': equity_df
    }


def run_parameter_sweep(df_1m, adv):
    """
    Sweep over parameter grid and collect results.

    Args:
        df_1m: 1-minute kline data
        adv: Average daily volume for bucket sizing

    Returns:
        DataFrame with results for each parameter combination
    """
    results = []

    # Cache bucket DataFrames for each buckets_per_day
    bucket_cache = {}

    total_combos = len(PARAM_GRID['buckets_per_day']) * len(PARAM_GRID['vpin_window'])
    pbar = tqdm(total=total_combos, desc="Parameter Sweep")

    for buckets_per_day in PARAM_GRID['buckets_per_day']:
        bucket_size = adv / buckets_per_day

        # Create volume buckets (cached per buckets_per_day)
        if buckets_per_day not in bucket_cache:
            print(f"\nCreating buckets for {buckets_per_day} buckets/day (V={bucket_size:,.0f})...")
            buckets_df = make_volume_buckets(df_1m, bucket_size, include_partial=False)
            bucket_cache[buckets_per_day] = buckets_df
        else:
            buckets_df = bucket_cache[buckets_per_day]

        if len(buckets_df) < 100:
            pbar.update(len(PARAM_GRID['vpin_window']))
            continue

        for vpin_window in PARAM_GRID['vpin_window']:
            pbar.set_postfix({'buckets/day': buckets_per_day, 'window': vpin_window})

            if len(buckets_df) < vpin_window:
                pbar.update(1)
                continue

            # Calculate VPIN and CDF
            try:
                vpin_df = calculate_vpin_metric(
                    buckets_df, bucket_size,
                    window_n=vpin_window,
                    cdf_lookback_days=CDF_LOOKBACK_DAYS
                )
            except Exception as e:
                print(f"Error calculating VPIN: {e}")
                pbar.update(1)
                continue

            # Run backtest
            result = run_single_backtest(vpin_df, CDF_THRESHOLD, COOLDOWN_DAYS)

            if result is not None:
                results.append({
                    'buckets_per_day': buckets_per_day,
                    'vpin_window': vpin_window,
                    'bucket_size': bucket_size,
                    'total_pnl': result['total_pnl'],
                    'buy_hold_pnl': result['buy_hold_pnl'],
                    'excess_return': result['excess_return'],
                    'num_exits': result['num_exits'],
                    'win_rate': result['win_rate'],
                    'max_drawdown': result['max_drawdown'],
                    'num_buckets': len(buckets_df)
                })

            pbar.update(1)

    pbar.close()
    return pd.DataFrame(results)


def plot_equity_curves(best_result, vpin_df, output_path="vpin_backtest_equity.png"):
    """Plot equity curve for best strategy vs buy-and-hold."""
    result = run_single_backtest(vpin_df, CDF_THRESHOLD, COOLDOWN_DAYS)

    if result is None or result['equity_curve'].empty:
        print("No equity curve to plot.")
        return

    equity_df = result['equity_curve']

    # Calculate buy-and-hold equity curve
    prices = vpin_df.dropna(subset=['price']).sort_values('time')
    bh_equity = (prices['price'] / prices['price'].iloc[0] - 1) * 100

    fig, ax = plt.subplots(figsize=(14, 8))

    # VPIN strategy equity
    ax.plot(equity_df['time'], equity_df['equity'],
            label=f"VPIN Strategy (PnL: {result['total_pnl']:.1f}%)", color='blue', linewidth=1)

    # Buy-and-hold equity
    ax.plot(prices['time'], bh_equity,
            label=f"Buy & Hold (PnL: {result['buy_hold_pnl']:.1f}%)", color='gray', linewidth=1, alpha=0.7)

    # Mark exits
    trades = result['trades']
    exit_times = [t['exit_time'] for t in trades if t['exit_cdf'] >= CDF_THRESHOLD]
    exit_equities = []
    for et in exit_times:
        match = equity_df[equity_df['time'] == et]
        if not match.empty:
            exit_equities.append(match['equity'].iloc[0])
        else:
            exit_equities.append(np.nan)

    ax.scatter(exit_times, exit_equities, color='red', marker='v', s=50, label='VPIN Exit', zorder=5)

    ax.set_title(f"VPIN Backtest: {best_result['buckets_per_day']} buckets/day, window={best_result['vpin_window']}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Equity curve saved to {output_path}")


def main():
    print("=" * 60)
    print("VPIN BACKTEST - Parameter Optimization")
    print("=" * 60)
    print(f"CDF Threshold: {CDF_THRESHOLD}")
    print(f"Cooldown Days: {COOLDOWN_DAYS}")
    print(f"Min Exits Required: {MIN_EXITS_REQUIRED}")
    print(f"Parameter Grid: {len(PARAM_GRID['buckets_per_day'])} x {len(PARAM_GRID['vpin_window'])} = "
          f"{len(PARAM_GRID['buckets_per_day']) * len(PARAM_GRID['vpin_window'])} combinations")
    print("=" * 60)

    # Load 1-minute data
    print("\nLoading 1-minute kline data...")
    df_1m = market_data.manage_local_data("BNBUSDT", "1m", start_date_dt=pd.Timestamp("2020-01-01"))

    if df_1m.empty:
        print("No data available!")
        return

    print(f"Loaded {len(df_1m):,} 1-minute candles")
    print(f"Date range: {df_1m['open_time'].min()} to {df_1m['open_time'].max()}")

    # Calculate ADV for bucket sizing
    daily_vol = df_1m.set_index("open_time")["volume"].resample("1D").sum()
    adv = daily_vol.mean()
    print(f"Average Daily Volume: {adv:,.0f}")

    # Run parameter sweep
    print("\nRunning parameter sweep...")
    results_df = run_parameter_sweep(df_1m, adv)

    if results_df.empty:
        print("No valid results!")
        return

    # Filter by minimum exits
    valid_results = results_df[results_df['num_exits'] >= MIN_EXITS_REQUIRED].copy()
    print(f"\n{len(valid_results)} / {len(results_df)} combinations have >= {MIN_EXITS_REQUIRED} exits")

    if valid_results.empty:
        print(f"No combinations meet the {MIN_EXITS_REQUIRED} exits requirement!")
        print("\nAll results (sorted by num_exits):")
        print(results_df.sort_values('num_exits', ascending=False).head(10))
        return

    # Sort by total PnL
    valid_results = valid_results.sort_values('total_pnl', ascending=False)

    # Save all results
    results_df.to_csv("vpin_backtest_results.csv", index=False)
    print(f"\nAll results saved to vpin_backtest_results.csv")

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Buy-and-hold baseline
    bh_pnl = valid_results['buy_hold_pnl'].iloc[0]
    print(f"\nBuy-and-Hold Baseline PnL: {bh_pnl:.2f}%")

    # Top 5 results
    print(f"\nTop 5 Parameter Combinations (by Total PnL, >= {MIN_EXITS_REQUIRED} exits):")
    print("-" * 80)
    for i, row in valid_results.head(5).iterrows():
        print(f"  {row['buckets_per_day']:3.0f} buckets/day, window={row['vpin_window']:3.0f}: "
              f"PnL={row['total_pnl']:7.2f}%, Excess={row['excess_return']:+7.2f}%, "
              f"Exits={row['num_exits']:.0f}, WinRate={row['win_rate']:.1f}%, MaxDD={row['max_drawdown']:.1f}%")

    # Best result
    best = valid_results.iloc[0]
    print(f"\n{'=' * 60}")
    print("BEST PARAMETERS")
    print(f"{'=' * 60}")
    print(f"  Buckets per day: {best['buckets_per_day']:.0f}")
    print(f"  VPIN Window: {best['vpin_window']:.0f}")
    print(f"  Bucket Size: {best['bucket_size']:,.0f}")
    print(f"  Total PnL: {best['total_pnl']:.2f}%")
    print(f"  Buy-Hold PnL: {best['buy_hold_pnl']:.2f}%")
    print(f"  Excess Return: {best['excess_return']:+.2f}%")
    print(f"  Number of Exits: {best['num_exits']:.0f}")
    print(f"  Win Rate: {best['win_rate']:.1f}%")
    print(f"  Max Drawdown: {best['max_drawdown']:.1f}%")

    # Generate equity curve plot for best params
    print("\nGenerating equity curve for best parameters...")
    bucket_size = adv / best['buckets_per_day']
    buckets_df = make_volume_buckets(df_1m, bucket_size, include_partial=False)
    vpin_df = calculate_vpin_metric(buckets_df, bucket_size,
                                    window_n=int(best['vpin_window']),
                                    cdf_lookback_days=CDF_LOOKBACK_DAYS)
    plot_equity_curves(best, vpin_df)

    print("\nBacktest complete!")


if __name__ == "__main__":
    main()
