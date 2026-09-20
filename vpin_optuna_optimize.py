"""Optuna parameter selection on training Calmar; the later period stays held out."""

import pandas as pd
import optuna
from optuna.samplers import TPESampler

import market_data
from vpin_backtest import (
    BUCKETS_PER_DAY_RANGE, VPIN_WINDOW_RANGE,
    evaluate_parameters, prepare_evaluation, report_results, result_record,
)

N_TRIALS = 100
N_STARTUP_TRIALS = 20


class VPINOptimizer:
    """TPE selection using only pre-split candles and fixed training scoring dates."""
    def __init__(self, df_1m, config, start, split):
        self.df_1m = df_1m[pd.to_datetime(df_1m["open_time"], utc=True) < split].copy()
        self.config = config
        self.start, self.split = start, split
        self.bucket_cache = {}
        self.trial_results = []

    def objective(self, trial):
        if self.config["bucket_size_base"] is None:
            count = trial.suggest_int("buckets_per_day", *BUCKETS_PER_DAY_RANGE)
        else:
            count = self.config["bucket_target_bars_per_day"]
        window = trial.suggest_int("vpin_window", *VPIN_WINDOW_RANGE)
        result = evaluate_parameters(
            self.df_1m, self.config, count, window,
            self.start, self.split, self.bucket_cache,
        )
        row = result_record(count, window, result, self.start, self.split)
        row["trial"] = trial.number
        self.trial_results.append(row)
        return row["calmar"] if row["eligible"] else float("-inf")

    def run(self, n_trials=N_TRIALS):
        """Return a seeded study; inspect eligible trial_results before selecting a winner."""
        study = optuna.create_study(
            direction="maximize", sampler=TPESampler(n_startup_trials=N_STARTUP_TRIALS, seed=42),
            study_name="vpin_training_calmar",
        )
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        return study


def main():
    config = market_data.load_config()
    candles = market_data.manage_local_data(config["symbol"], start_date_dt=config["start_date"])
    try:
        start, split, end = prepare_evaluation(candles, config)
    except ValueError as exc:
        print(exc)
        return
    optimizer = VPINOptimizer(candles, config, start, split)
    optimizer.run()
    report_results(pd.DataFrame(optimizer.trial_results), candles, config,
                   start, split, end, "vpin_optuna_results.csv")


if __name__ == "__main__":
    main()
