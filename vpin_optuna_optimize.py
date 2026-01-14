"""
VPIN Optuna Optimization

Objective: Maximize cumulative drawdown avoided by VPIN 99% exit signals.

For each exit signal, we measure how much the price dropped during the cooldown period.
This represents the drawdown we avoided by exiting when VPIN was high.

Uses Bayesian optimization (TPE sampler) to efficiently search the parameter space.
"""

import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

from vpin_calculator import make_volume_buckets, calculate_vpin_metric
import market_data

# Constants
CDF_THRESHOLD = 0.99
COOLDOWN_DAYS = 2
CDF_LOOKBACK_DAYS = 90
MIN_EXITS_REQUIRED = 30

# Optimization settings
N_TRIALS = 100  # Number of Optuna trials
N_STARTUP_TRIALS = 20  # Random exploration before TPE kicks in


def run_backtest_for_optuna(df, cdf_threshold=0.99, cooldown_days=2):
    """
    Run backtest and calculate cumulative avoided drawdown.

    For each VPIN exit signal:
    - Record exit price
    - Find minimum price during cooldown period
    - Calculate avoided drawdown = (exit_price - min_price) / exit_price * 100

    Returns metrics including cumulative_avoided_drawdown.
    """
    df = df.dropna(subset=['CDF', 'price']).copy()
    df = df.sort_values('time').reset_index(drop=True)

    if len(df) < 10:
        return None

    position = 'long'
    entry_price = df['price'].iloc[0]
    entry_time = df['time'].iloc[0]
    cooldown_until = None
    exit_price_for_dd = None

    cumulative_pnl = 0.0
    equity_curve = []
    num_exits = 0

    # Track avoided drawdowns
    avoided_drawdowns = []
    min_price_during_cooldown = None

    for idx, row in df.iterrows():
        current_time = row['time']
        current_price = row['price']
        current_cdf = row['CDF']

        # Track min price during cooldown
        if position == 'flat' and cooldown_until is not None:
            if min_price_during_cooldown is None or current_price < min_price_during_cooldown:
                min_price_during_cooldown = current_price

        # Re-entry after cooldown
        if position == 'flat' and cooldown_until is not None:
            if current_time >= cooldown_until:
                # Calculate avoided drawdown for this exit
                if exit_price_for_dd is not None and min_price_during_cooldown is not None:
                    # How much did price drop during cooldown?
                    avoided_dd = (exit_price_for_dd - min_price_during_cooldown) / exit_price_for_dd * 100
                    if avoided_dd > 0:  # Only count if price actually dropped
                        avoided_drawdowns.append(avoided_dd)

                # Re-enter
                position = 'long'
                entry_price = current_price
                entry_time = current_time
                cooldown_until = None
                exit_price_for_dd = None
                min_price_during_cooldown = None

        # Exit on high CDF
        if position == 'long' and current_cdf >= cdf_threshold:
            trade_pnl = (current_price / entry_price - 1) * 100
            cumulative_pnl += trade_pnl
            position = 'flat'
            cooldown_until = current_time + timedelta(days=cooldown_days)
            exit_price_for_dd = current_price
            min_price_during_cooldown = current_price  # Initialize with exit price
            num_exits += 1

        # Track equity
        if position == 'long':
            unrealized = (current_price / entry_price - 1) * 100
            equity_curve.append(cumulative_pnl + unrealized)
        else:
            equity_curve.append(cumulative_pnl)

    # Handle final cooldown if still flat at end
    if position == 'flat' and exit_price_for_dd is not None and min_price_during_cooldown is not None:
        avoided_dd = (exit_price_for_dd - min_price_during_cooldown) / exit_price_for_dd * 100
        if avoided_dd > 0:
            avoided_drawdowns.append(avoided_dd)

    # Close open position at end
    if position == 'long':
        final_pnl = (df['price'].iloc[-1] / entry_price - 1) * 100
        cumulative_pnl += final_pnl

    # Calculate max drawdown of equity curve
    equity_series = pd.Series(equity_curve)
    peak = equity_series.cummax()
    drawdown = equity_series - peak
    max_drawdown = drawdown.min()

    # Cumulative avoided drawdown
    cumulative_avoided_dd = sum(avoided_drawdowns)
    avg_avoided_dd = np.mean(avoided_drawdowns) if avoided_drawdowns else 0

    return {
        'max_drawdown': max_drawdown,
        'num_exits': num_exits,
        'total_pnl': cumulative_pnl,
        'cumulative_avoided_dd': cumulative_avoided_dd,
        'avg_avoided_dd': avg_avoided_dd,
        'num_successful_exits': len(avoided_drawdowns)  # Exits where price actually dropped
    }


class VPINOptimizer:
    """Optuna-based optimizer for VPIN parameters."""

    def __init__(self, df_1m, adv):
        self.df_1m = df_1m
        self.adv = adv
        self.bucket_cache = {}
        self.best_valid_result = None
        self.trial_results = []

    def objective(self, trial):
        """
        Optuna objective function.

        Maximizes cumulative avoided drawdown from VPIN exits.
        Penalizes trials with < 30 exits heavily.
        """
        # Sample parameters
        buckets_per_day = trial.suggest_int('buckets_per_day', 20, 350)
        vpin_window = trial.suggest_int('vpin_window', 10, 350)

        bucket_size = self.adv / buckets_per_day

        # Create volume buckets (cached)
        if buckets_per_day not in self.bucket_cache:
            buckets_df = make_volume_buckets(self.df_1m, bucket_size, include_partial=False)
            self.bucket_cache[buckets_per_day] = buckets_df
        else:
            buckets_df = self.bucket_cache[buckets_per_day]

        if len(buckets_df) < vpin_window:
            return float('-inf')

        # Calculate VPIN
        try:
            vpin_df = calculate_vpin_metric(
                buckets_df, bucket_size,
                window_n=vpin_window,
                cdf_lookback_days=CDF_LOOKBACK_DAYS
            )
        except Exception as e:
            return float('-inf')

        # Run backtest
        result = run_backtest_for_optuna(vpin_df, CDF_THRESHOLD, COOLDOWN_DAYS)

        if result is None:
            return float('-inf')

        num_exits = result['num_exits']
        cumulative_avoided_dd = result['cumulative_avoided_dd']
        max_drawdown = result['max_drawdown']
        total_pnl = result['total_pnl']

        # Store for reporting
        self.trial_results.append({
            'trial': trial.number,
            'buckets_per_day': buckets_per_day,
            'vpin_window': vpin_window,
            'cumulative_avoided_dd': cumulative_avoided_dd,
            'avg_avoided_dd': result['avg_avoided_dd'],
            'num_successful_exits': result['num_successful_exits'],
            'max_drawdown': max_drawdown,
            'num_exits': num_exits,
            'total_pnl': total_pnl
        })

        # Track best valid result
        if num_exits >= MIN_EXITS_REQUIRED:
            if self.best_valid_result is None or cumulative_avoided_dd > self.best_valid_result['cumulative_avoided_dd']:
                self.best_valid_result = {
                    'buckets_per_day': buckets_per_day,
                    'vpin_window': vpin_window,
                    'cumulative_avoided_dd': cumulative_avoided_dd,
                    'max_drawdown': max_drawdown,
                    'num_exits': num_exits,
                    'total_pnl': total_pnl
                }

        # Constraint: penalize if not enough exits
        if num_exits < MIN_EXITS_REQUIRED:
            # Heavy penalty
            penalty = -10000 * (MIN_EXITS_REQUIRED - num_exits)
            return penalty

        # Maximize cumulative avoided drawdown
        return cumulative_avoided_dd

    def run(self, n_trials=N_TRIALS):
        """Run the optimization."""
        sampler = TPESampler(
            n_startup_trials=N_STARTUP_TRIALS,
            seed=42
        )

        study = optuna.create_study(
            direction='maximize',  # Maximize avoided drawdown
            sampler=sampler,
            study_name='vpin_avoided_drawdown_optimization'
        )

        # Suppress Optuna's verbose logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        print(f"Running {n_trials} Optuna trials...")
        print(f"Objective: Maximize cumulative avoided drawdown from VPIN exits")
        print(f"Constraint: >= {MIN_EXITS_REQUIRED} exit signals")
        print("-" * 60)

        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)

        return study


def main():
    print("=" * 60)
    print("VPIN OPTUNA OPTIMIZATION")
    print("Objective: Maximize Cumulative Avoided Drawdown")
    print("=" * 60)
    print("\nFor each VPIN 99% exit, we measure how much the price")
    print("dropped during the 2-day cooldown. This represents the")
    print("drawdown we avoided by exiting on the VPIN signal.")
    print("=" * 60)

    # Load data
    print("\nLoading 1-minute kline data...")
    df_1m = market_data.manage_local_data("BNBUSDT", "1m", start_date_dt=pd.Timestamp("2020-01-01"))

    if df_1m.empty:
        print("No data available!")
        return

    print(f"Loaded {len(df_1m):,} 1-minute candles")
    print(f"Date range: {df_1m['open_time'].min()} to {df_1m['open_time'].max()}")

    # Calculate ADV
    daily_vol = df_1m.set_index("open_time")["volume"].resample("1D").sum()
    adv = daily_vol.mean()
    print(f"Average Daily Volume: {adv:,.0f}")

    # Run optimization
    print("\n" + "=" * 60)
    optimizer = VPINOptimizer(df_1m, adv)
    study = optimizer.run(n_trials=N_TRIALS)

    # Results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)

    # Save all trial results
    results_df = pd.DataFrame(optimizer.trial_results)
    results_df.to_csv("vpin_optuna_results.csv", index=False)
    print(f"\nAll trial results saved to vpin_optuna_results.csv")

    # Filter valid results (>= 30 exits)
    valid_results = results_df[results_df['num_exits'] >= MIN_EXITS_REQUIRED].copy()

    if valid_results.empty:
        print(f"\nNo trials achieved >= {MIN_EXITS_REQUIRED} exits!")
        print("\nTop 5 by num_exits:")
        print(results_df.nlargest(5, 'num_exits')[['buckets_per_day', 'vpin_window', 'num_exits', 'cumulative_avoided_dd']])
        return

    # Sort by cumulative avoided drawdown (higher = better)
    valid_results = valid_results.sort_values('cumulative_avoided_dd', ascending=False)

    print(f"\n{len(valid_results)} / {len(results_df)} trials achieved >= {MIN_EXITS_REQUIRED} exits")

    print("\nTop 10 by Cumulative Avoided Drawdown:")
    print("-" * 90)
    print(f"  {'Buckets':>7} {'Window':>7} {'Avoided DD':>12} {'Avg DD':>8} {'Exits':>6} {'Success':>8} {'MaxDD':>10} {'PnL':>10}")
    print("-" * 90)
    for i, row in valid_results.head(10).iterrows():
        print(f"  {row['buckets_per_day']:7.0f} {row['vpin_window']:7.0f} "
              f"{row['cumulative_avoided_dd']:12.1f}% {row['avg_avoided_dd']:7.1f}% "
              f"{row['num_exits']:6.0f} {row['num_successful_exits']:8.0f} "
              f"{row['max_drawdown']:9.1f}% {row['total_pnl']:9.1f}%")

    # Best result
    best = valid_results.iloc[0]
    print(f"\n{'=' * 60}")
    print("BEST PARAMETERS (Maximum Avoided Drawdown)")
    print(f"{'=' * 60}")
    print(f"  Buckets per day: {best['buckets_per_day']:.0f}")
    print(f"  VPIN Window: {best['vpin_window']:.0f}")
    print(f"  Cumulative Avoided DD: {best['cumulative_avoided_dd']:.1f}%")
    print(f"  Avg Avoided DD per Exit: {best['avg_avoided_dd']:.1f}%")
    print(f"  Number of Exits: {best['num_exits']:.0f}")
    print(f"  Successful Exits (price dropped): {best['num_successful_exits']:.0f}")
    print(f"  Max Drawdown: {best['max_drawdown']:.1f}%")
    print(f"  Total PnL: {best['total_pnl']:.1f}%")

    # Success rate
    if best['num_exits'] > 0:
        success_rate = best['num_successful_exits'] / best['num_exits'] * 100
        print(f"  Signal Success Rate: {success_rate:.1f}%")

    print("\nOptimization complete!")


if __name__ == "__main__":
    main()
