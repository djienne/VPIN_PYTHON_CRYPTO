"""Run offline regression checks with: python -m unittest -v test_vpin."""

import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import pandas as pd
import optuna

import market_data as md
import vpin_calculator as vc
import vpin_backtest as bt
from vpin_optuna_optimize import VPINOptimizer


def candles(volumes, buys=None, prices=None, start="2020-01-01", opens=None):
    volumes = np.asarray(volumes, dtype=float)
    n = len(volumes)
    closes = np.full(n, 100.0) if prices is None else np.asarray(prices, dtype=float)
    opens = closes.copy() if opens is None else np.asarray(opens, dtype=float)
    times = pd.date_range(start, periods=n, freq="min", tz="UTC")
    return pd.DataFrame(dict(
        open_time=times, close_time=times + md.MINUTE - pd.Timedelta(milliseconds=1),
        open=opens, high=np.maximum(opens, closes) * 1.01, low=np.minimum(opens, closes) * 0.99,
        close=closes, volume=volumes,
        taker_buy_base_asset_volume=volumes / 2 if buys is None else buys,
    ))


def synthetic_days(days):
    i = np.arange(days * 1440)
    volumes = 1 + 0.3 * np.sin(i / 1440 * 2 * np.pi / 13)
    fractions = 0.5 + 0.45 * np.sin(i / 1020 * 2 * np.pi)
    prices = 100 * np.exp(i / 1440 * 0.0002 + 0.04 * np.sin(i / 1440 * 2 * np.pi / 7))
    return candles(volumes, volumes * fractions, prices)


def signals(frame, offsets, values):
    return pd.DataFrame({"time": [frame.open_time.iloc[0] + x * md.MINUTE for x in offsets], "CDF": values})


def simulate(frame, sig, **kwargs):
    return bt.run_single_backtest(
        sig, frame, start_time=frame.open_time.iloc[0],
        end_time=frame.open_time.iloc[-1] + md.MINUTE,
        fee_bps=kwargs.pop("fee_bps", 0), slippage_bps=kwargs.pop("slippage_bps", 0), **kwargs,
    )


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = Path(self.temp.name) / "test.feather"
        self.filename = patch.object(md, "get_data_file", return_value=str(self.cache))
        self.filename.start()
        self.addCleanup(self.filename.stop)

    def test_config_validation(self):
        path = Path(self.temp.name) / "config.json"
        self.assertEqual(md.load_config(path)["vpin_window"], 10)
        for invalid in ({"vpin_window": 0}, {"bucket_size_base": float("nan")},
                        {"fee_bps": -1}, {"symbol": "../BTC"}, {"include_partial_bucket": True},
                        {"start_date": "invalid"}, {"adv_lookback_days": True}):
            path.write_text(json.dumps(invalid))
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                md.load_config(path)
        path.write_text('{"symbol":"btcusdt","fee_bps":4}')
        self.assertEqual(md.load_config(path)["symbol"], "BTCUSDT")

    def test_report_fingerprints_survive_git_line_endings(self):
        from docs.build_vpin_guide import sha256
        for suffix in (".py", ".csv"):
            path = Path(self.temp.name) / ("example" + suffix)
            path.write_bytes(b"value=1\n")
            original = sha256(path)
            path.write_bytes(b"value=1\r\n")
            self.assertEqual(sha256(path), original)
            path.write_bytes(b"value=2\r\n")
            self.assertNotEqual(sha256(path), original)
        binary = Path(self.temp.name) / "example.bin"
        binary.write_bytes(b"\x00\n")
        original = sha256(binary)
        binary.write_bytes(b"\x00\r\n")
        self.assertNotEqual(sha256(binary), original)

    def test_repair_overlap_and_exclude_unfinished_candle(self):
        full = candles([10, 20, 30, 40, 50])
        cached = full.iloc[[1, 3, 4]].copy()
        cached.loc[cached.index == 3, ["volume", "taker_buy_base_asset_volume"]] = [1, 0.5]
        cached.reset_index(drop=True).to_feather(self.cache)
        fetch = AsyncMock(return_value=full)
        with patch.object(md, "download_historical_data_async", fetch):
            result = md.manage_local_data("TEST", start_date_dt="2020-01-01", now="2020-01-01 00:04:30Z")
        pd.testing.assert_frame_equal(result, full.iloc[:4])
        pd.testing.assert_frame_equal(pd.read_feather(self.cache), full.iloc[:4])
        self.assertEqual(fetch.call_args.args[3], int(full.open_time.iloc[4].timestamp() * 1000))

    def test_recent_cache_does_not_hide_one_minute_backfill(self):
        full = candles(np.ones(10))
        full.iloc[1:].reset_index(drop=True).to_feather(self.cache)

        async def fetch(symbol, interval, start, end):
            times = full.open_time
            return full[(times >= pd.to_datetime(start, unit="ms", utc=True)) &
                        (times < pd.to_datetime(end, unit="ms", utc=True))].copy()

        with patch.object(md, "download_historical_data_async", side_effect=fetch) as mock:
            result = md.manage_local_data("TEST", start_date_dt=full.open_time.iloc[0], now="2020-01-01 00:10Z")
        pd.testing.assert_frame_equal(result, full)
        self.assertEqual(mock.call_args_list[0].args[3] - mock.call_args_list[0].args[2], 60_000)

    def test_unresolved_gap_keeps_original_cache(self):
        full = candles([1, 1, 1])
        full.iloc[[0, 2]].reset_index(drop=True).to_feather(self.cache)
        original = self.cache.read_bytes()
        with patch.object(md, "download_historical_data_async", AsyncMock(return_value=pd.DataFrame())):
            with self.assertRaisesRegex(ValueError, "Missing"):
                md.manage_local_data("TEST", start_date_dt="2020-01-01", now="2020-01-01 00:03Z")
        self.assertEqual(self.cache.read_bytes(), original)

    def test_missing_start_and_atomic_failure_preserve_cache(self):
        full = candles([1, 1, 1])
        full.to_feather(self.cache)
        original = self.cache.read_bytes()
        with patch.object(md, "download_historical_data_async", AsyncMock(return_value=full)):
            with self.assertRaisesRegex(ValueError, "start"):
                md.manage_local_data("TEST", start_date_dt="2019-12-31 23:59Z", now="2020-01-01 00:03Z")
            with self.assertRaisesRegex(ValueError, "end"):
                md.manage_local_data("TEST", start_date_dt="2020-01-01", now="2020-01-01 00:04Z")
            with patch.object(md.os, "replace", side_effect=OSError("write failed")):
                with self.assertRaises(OSError):
                    md.manage_local_data("TEST", start_date_dt="2020-01-01", now="2020-01-01 00:03Z")
        self.assertEqual(self.cache.read_bytes(), original)
        self.assertEqual(list(Path(self.temp.name).iterdir()), [self.cache])

    def test_legacy_cache_is_rebuilt(self):
        full = candles([5, 6, 7])
        legacy = full.copy()
        legacy["open_time"] = legacy.open_time.dt.tz_localize(None)
        legacy["close_time"] = legacy.close_time.dt.tz_localize(None)
        legacy.loc[0, ["volume", "taker_buy_base_asset_volume"]] = [1, 0.5]
        legacy.to_feather(self.cache)
        with patch.object(md, "download_historical_data_async", AsyncMock(return_value=full)) as fetch:
            result = md.manage_local_data("TEST", start_date_dt="2020-01-01", now="2020-01-01 00:03Z")
        pd.testing.assert_frame_equal(result, full)
        self.assertEqual(fetch.call_args.args[2], int(full.open_time.iloc[0].timestamp() * 1000))

    def test_failed_retries_raise(self):
        response = Mock(status=500, headers={})
        manager = AsyncMock()
        manager.__aenter__.return_value = response
        session = Mock()
        session.get.return_value = manager
        with patch.object(md.asyncio, "sleep", AsyncMock()):
            with self.assertRaisesRegex(RuntimeError, "HTTP 500"):
                asyncio.run(md.fetch_kline_chunk(session, "TEST", "1m", 0, 59999))
        self.assertEqual(session.get.call_count, md.MAX_RETRIES)

    def test_download_inclusive_api_bounds_and_utc(self):
        raw = [0, "100", "101", "99", "100", "2", 59999, "200", 1, "1", "100", "0"]
        with patch.object(md, "fetch_kline_chunk", AsyncMock(return_value=[raw])) as fetch:
            result = asyncio.run(md.download_historical_data_async("TEST", "1m", 0, 1001 * 60000))
        self.assertEqual([(c.args[3], c.args[4]) for c in fetch.call_args_list],
                         [(0, 1000 * 60000 - 1), (1000 * 60000, 1001 * 60000 - 1)])
        self.assertEqual(str(result.open_time.dt.tz), "UTC")
        self.assertEqual(result.volume.iloc[0], 2)

    def test_rate_limit_retry_after_then_success(self):
        for status in (429, 418):
            with self.subTest(status=status):
                limited = Mock(status=status, headers={"Retry-After": "7"})
                success = Mock(status=200, json=AsyncMock(return_value=[["candle"]]))
                responses = []
                for response in (limited, success):
                    manager = AsyncMock()
                    manager.__aenter__.return_value = response
                    responses.append(manager)
                session = Mock()
                session.get.side_effect = responses
                with patch.object(md.asyncio, "sleep", AsyncMock()) as sleep:
                    result = asyncio.run(md.fetch_kline_chunk(session, "TEST", "1m", 0, 59999))
                self.assertEqual(result, [["candle"]])
                self.assertEqual(session.get.call_count, 2)
                sleep.assert_any_await(7)


class IndicatorTests(unittest.TestCase):
    def test_bucket_carries_target_across_midnight(self):
        frame = candles([6, 12, 12, 4], [1.5, 9, 6, 2], [100, 90, 80, 70], start="2020-01-01 23:59")
        buckets = vc.make_volume_buckets(frame, pd.Series([10., 20., 20., 20.]))
        np.testing.assert_allclose(buckets.total_volume, [10, 20])
        np.testing.assert_allclose(buckets.buy_volume, [4.5, 12])
        np.testing.assert_allclose(buckets.price, [90, 80])
        self.assertEqual(list(buckets.time), list(frame.open_time.iloc[[2, 3]]))
        result = vc.calculate_vpin_metric(buckets, window_n=2)
        self.assertAlmostEqual(result.VPIN.iloc[-1], 5 / 30)

    def test_full_buckets_only_and_same_minute_order(self):
        frame = candles([35, 5], [35, 0])
        result = vc.calculate_vpin_metric(vc.make_volume_buckets(frame, 10), window_n=1)
        np.testing.assert_allclose(result.VPIN, [1, 1, 1, 0])
        np.testing.assert_allclose(result.total_volume, [10] * 4)
        self.assertTrue(result.time.iloc[:3].eq(frame.open_time.iloc[1]).all())
        self.assertEqual(result.time.iloc[-1], frame.close_time.iloc[-1] + pd.Timedelta(milliseconds=1))
        self.assertEqual(len(vc.make_volume_buckets(frame.iloc[:1], 10)), 3)

    def test_tiny_carried_remainder_does_not_stall_or_reject_valid_bucket(self):
        frame = candles([np.nextafter(10., 0.), 100.])
        buckets = vc.make_volume_buckets(frame, 10.)
        self.assertEqual(len(buckets), 11)
        np.testing.assert_allclose(buckets.total_volume, 10.)
        self.assertAlmostEqual(buckets.total_volume.sum(), frame.volume.sum())

    def test_adaptive_sizes_and_prefix_are_causal(self):
        frame = synthetic_days(6)
        prefix = frame.iloc[:4 * 1440]
        config = dict(md.DEFAULT_CONFIG, adv_lookback_days=1, cdf_lookback_days=1, bucket_target_bars_per_day=180, vpin_window=3)
        sizes = vc.compute_bucket_size_base(frame, 180, 1)
        self.assertTrue(sizes.iloc[:1440].isna().all())
        self.assertAlmostEqual(sizes.iloc[1440], frame.volume.iloc[:1440].sum() / 180)
        local = frame.copy()
        local["open_time"] = local.open_time.dt.tz_convert("Europe/Berlin")
        pd.testing.assert_series_equal(sizes, vc.compute_bucket_size_base(local, 180, 1))
        def calculate(data):
            return vc.calculate_vpin_metric(vc.configured_buckets(data, config), 3, 1)
        old = calculate(prefix)
        changed = frame.copy()
        changed.loc[changed.index >= len(prefix), ["volume", "taker_buy_base_asset_volume"]] *= 20
        newer = calculate(changed)
        newer = newer[newer.time <= prefix.close_time.iloc[-1] + pd.Timedelta(milliseconds=1)]
        pd.testing.assert_frame_equal(old, newer)
        self.assertTrue(old.CDF.notna().any())
        self.assertTrue(old.VPIN.dropna().between(0, 1).all())

    def test_manual_override_and_all_buy_normalization(self):
        frame = candles([6, 12, 12], [6, 12, 12])
        sizes = vc.compute_bucket_size_base(frame, 252, 90, override=10)
        self.assertTrue(sizes.eq(10).all())
        varying = vc.make_volume_buckets(frame, pd.Series([10., 20., 20.]))
        result = vc.calculate_vpin_metric(varying, window_n=2)
        self.assertEqual(result.VPIN.iloc[-1], 1)
        with self.assertRaises(ValueError):
            vc.make_volume_buckets(frame, 0)
        with self.assertRaisesRegex(ValueError, "too small"):
            vc.make_volume_buckets(frame, 1e-30)

    def test_cdf_warmup_prior_only_and_constant_history(self):
        values = np.linspace(0, 1, 301)
        values[-1] = 0.25
        buckets = pd.DataFrame(dict(time=pd.date_range("2020-01-01", periods=301, freq="5min", tz="UTC"),
                                    buy_volume=(1 + values) / 2, sell_volume=(1 - values) / 2,
                                    total_volume=np.ones(301), price=np.ones(301)))
        result = vc.calculate_vpin_metric(buckets, window_n=1, cdf_lookback_days=1)
        self.assertTrue(result.CDF.iloc[:288].isna().all())
        # 63 of the 287 preceding values are <= 0.25; neither the expired anchor nor self counts.
        self.assertAlmostEqual(result.CDF.iloc[-1], 63 / 287)
        buckets["buy_volume"] = buckets["sell_volume"] = 0.5
        self.assertTrue(vc.calculate_vpin_metric(buckets, 1, 1).CDF.isna().all())

    def test_cdf_requires_100_prior_values_even_after_time_warmup(self):
        values = np.linspace(0, 1, 110)
        buckets = pd.DataFrame(dict(time=pd.date_range("2020-01-01", periods=110, freq="30min", tz="UTC"),
                                    buy_volume=(1 + values) / 2, sell_volume=(1 - values) / 2,
                                    total_volume=np.ones(110), price=np.ones(110)))
        self.assertTrue(vc.calculate_vpin_metric(buckets, 1, 1).CDF.isna().all())

    def test_cdf_exact_history_boundary_includes_ties(self):
        times = pd.date_range("2020-01-01", periods=101, freq="min", tz="UTC")
        times = times.append(pd.DatetimeIndex([pd.Timestamp("2020-01-02", tz="UTC")]))
        values = np.r_[0.9, np.full(50, 0.1), np.full(50, 0.9), 0.1]
        buckets = pd.DataFrame(dict(time=times, buy_volume=(1 + values) / 2,
                                    sell_volume=(1 - values) / 2, total_volume=np.ones(102)))
        self.assertAlmostEqual(vc.calculate_vpin_metric(buckets, 1, 1).CDF.iloc[-1], 0.5)
        self.assertTrue(np.isnan(vc.calculate_vpin_metric(buckets.drop(index=1), 1, 1).CDF.iloc[-1]))

    def test_invalid_volume_and_missing_minutes_rejected(self):
        frame = candles([1, 1, 1])
        with self.assertRaises(ValueError):
            vc.make_volume_buckets(frame.iloc[[0, 2]], 1)
        frame.loc[1, "taker_buy_base_asset_volume"] = 2
        with self.assertRaises(ValueError):
            vc.make_volume_buckets(frame, 1)

    def test_empty_event_export_replaces_old_result(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "events.csv"
            output.write_text("obsolete result")
            empty = pd.DataFrame({"time": pd.to_datetime([], utc=True), "VPIN": [], "CDF": []})
            result = vc.export_extreme_events(empty, output_path=output)
            self.assertTrue(result.empty)
            self.assertEqual(list(pd.read_csv(output).columns), [
                "start_date", "end_date", "start_time", "end_time", "duration_hours",
                "peak_vpin", "peak_cdf", "avg_vpin", "bucket_count",
            ])


class BacktestTests(unittest.TestCase):
    def test_compounding_two_doublings(self):
        frame = candles(np.ones(10), prices=[100, 200, 200, 200, 400, 400, 400, 400, 400, 400])
        result = simulate(frame, signals(frame, [0, 1, 2, 4], [0, 1, 0, 1]), cooldown_days=0)
        self.assertAlmostEqual(result["total_pnl"], 300)
        self.assertEqual(result["num_exits"], 2)
        np.testing.assert_allclose([t["pnl_pct"] for t in result["trades"]], [100, 100])
        self.assertTrue(np.isnan(result["calmar"]))  # Positive return with zero drawdown is ineligible.

    def test_known_annualized_return_and_calmar(self):
        # 1,461 minutes is exactly 1/360 of a 365.25-day year. Independent compound-power oracle.
        prices = np.full(1461, 100.1)
        prices[:2] = [100., 125.125]
        frame = candles(np.ones(1461), prices=prices)
        result = simulate(frame, signals(frame, [0], [0]))
        self.assertAlmostEqual(result["total_pnl"], 0.1)
        self.assertAlmostEqual(result["max_drawdown"], -20.)
        self.assertAlmostEqual(result["cagr"], 43.30716103239724, places=8)
        self.assertAlmostEqual(result["calmar"], 2.165358051619862, places=8)

    def test_percentage_drawdown_and_terminal_exit(self):
        frame = candles(np.ones(10), prices=[100, 200, 100, 100, 100, 100, 100, 100, 100, 100])
        result = simulate(frame, signals(frame, [0, 10], [0, 1]))
        self.assertAlmostEqual(result["max_drawdown"], -50)
        self.assertAlmostEqual(result["buy_hold_max_drawdown"], -50)
        self.assertEqual(result["num_exits"], 0)
        self.assertEqual(result["trades"][-1]["reason"], "terminal")

    def test_signals_execute_at_available_minute_open(self):
        frame = candles(np.ones(4), prices=[100, 100, 70, 60], opens=[100, 100, 80, 70])
        sig = signals(frame, [0, 2], [0, 1])
        sig["price"] = [100, 9999]
        result = simulate(frame, sig)
        trade = result["trades"][0]
        self.assertEqual(trade["exit_time"], frame.open_time.iloc[2])
        self.assertEqual(trade["exit_price"], 80)
        self.assertAlmostEqual(result["total_pnl"], -20)

    def test_transaction_costs_and_identical_benchmark(self):
        frame = candles(np.ones(10))
        result = simulate(frame, signals(frame, [0], [0]), fee_bps=10, slippage_bps=2)
        expected = (1 - .0002) * (1 - .001) / ((1 + .0002) * (1 + .001))
        self.assertAlmostEqual(result["total_pnl"], (expected - 1) * 100)
        np.testing.assert_allclose(result["equity_curve"].equity, result["equity_curve"].buy_hold_equity)
        self.assertEqual(result["equity_curve"].time.iloc[-1], frame.open_time.iloc[-1] + md.MINUTE)

    def test_cooldown_and_exit_precedence(self):
        frame = candles(np.ones(2885))
        result = simulate(frame, signals(frame, [0, 1, 2, 2881, 2882], [0, 1, np.nan, 1, 0]))
        self.assertEqual(result["num_exits"], 1)
        self.assertEqual(len(result["trades"]), 2)
        self.assertEqual(result["trades"][-1]["entry_time"], frame.open_time.iloc[2882])
        self.assertTrue(result["equity_curve"].position.iloc[1:2882].eq("flat").all())

    def test_last_same_minute_signal_and_nan_are_respected(self):
        frame = candles(np.ones(5))
        result = simulate(frame, signals(frame, [0, 0, 1, 2], [0, 1, np.nan, 0]))
        self.assertEqual(result["trades"][0]["entry_time"], frame.open_time.iloc[2])
        empty = simulate(frame, signals(frame, [0], [np.nan]))
        self.assertEqual(empty["total_pnl"], 0)
        self.assertTrue(np.isnan(empty["calmar"]))
        invested = simulate(frame, signals(frame, [0, 1], [0, np.nan]))
        self.assertTrue(invested["equity_curve"].position.iloc[:-1].eq("long").all())
        self.assertEqual(invested["num_exits"], 0)

    def test_invalid_evaluation_coverage_rejected(self):
        frame = candles(np.ones(5))
        with self.assertRaises(ValueError):
            simulate(frame.iloc[[0, 2, 3, 4]], signals(frame, [0], [0]))


class OptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = synthetic_days(35)
        cls.config = dict(md.DEFAULT_CONFIG, adv_lookback_days=1, cdf_lookback_days=6,
                          bucket_target_bars_per_day=100, vpin_window=10)
        cls.start, cls.split, cls.end = bt.prepare_evaluation(cls.frame, cls.config)

    def test_grid_optuna_and_future_independence(self):
        changed = self.frame.copy()
        future = changed.open_time >= self.split
        changed.loc[future, ["open", "high", "low", "close"]] *= 3
        changed.loc[future, ["volume", "taker_buy_base_asset_volume"]] *= 7
        self.assertEqual(bt.prepare_evaluation(changed, self.config), (self.start, self.split, self.end))
        first = VPINOptimizer(self.frame, self.config, self.start, self.split)
        second = VPINOptimizer(changed, self.config, self.start, self.split)
        with patch.object(bt, "MIN_EXITS_REQUIRED", 1):
            for i, params in enumerate([{"buckets_per_day": 100, "vpin_window": 10},
                                        {"buckets_per_day": 150, "vpin_window": 20}]):
                a = first.objective(optuna.trial.FixedTrial(params, number=i))
                b = second.objective(optuna.trial.FixedTrial(params, number=i))
                self.assertAlmostEqual(a, b)
                self.assertTrue(np.isfinite(a))
            with patch.object(bt, "PARAM_GRID", {"buckets_per_day": [100], "vpin_window": [10]}):
                grid = bt.run_parameter_sweep(self.frame, self.config, self.start, self.split)
            for key in bt.METRICS:
                self.assertAlmostEqual(grid.iloc[0][key], first.trial_results[0][key])
        self.assertEqual(max(first.trial_results, key=lambda x: x["calmar"])["buckets_per_day"],
                         max(second.trial_results, key=lambda x: x["calmar"])["buckets_per_day"])
        self.assertLess(first.df_1m.open_time.max(), self.split)

    def test_minimum_exits_and_empty_results(self):
        result = dict.fromkeys(bt.METRICS, 0.0)
        result.update(num_exits=29, calmar=1)
        self.assertFalse(bt.result_record(100, 10, result, self.start, self.split)["eligible"])
        result["num_exits"] = 30
        self.assertTrue(bt.result_record(100, 10, result, self.start, self.split)["eligible"])
        result["calmar"] = np.nan
        self.assertFalse(bt.result_record(100, 10, result, self.start, self.split)["eligible"])
        with tempfile.TemporaryDirectory() as directory, patch.object(bt, "run_single_backtest") as run:
            path = Path(directory) / "results.csv"
            self.assertIsNone(bt.report_results(pd.DataFrame(), self.frame, self.config,
                                               self.start, self.split, self.end, path))
            self.assertTrue(pd.read_csv(path).empty)
            run.assert_not_called()

    def test_small_optuna_study_and_held_out_report(self):
        optimizer = VPINOptimizer(self.frame, self.config, self.start, self.split)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        for window in (10, 20):
            study.enqueue_trial({"buckets_per_day": 100, "vpin_window": window})
        with patch.object(bt, "MIN_EXITS_REQUIRED", 1):
            study.optimize(optimizer.objective, n_trials=2)
        self.assertTrue(np.isfinite(study.best_value))
        with tempfile.TemporaryDirectory() as directory, patch.object(bt, "plot_equity_curves") as plot:
            path = Path(directory) / "results.csv"
            with patch.object(bt, "run_single_backtest", wraps=bt.run_single_backtest) as run:
                held_out = bt.report_results(pd.DataFrame(optimizer.trial_results), self.frame, self.config,
                                             self.start, self.split, self.end, path)
                self.assertEqual(run.call_count, 1)
                self.assertEqual(run.call_args.kwargs["start_time"], self.split)
            saved = pd.read_csv(path)
            self.assertEqual(saved.phase.tolist(), ["train", "train", "test"])
            self.assertEqual(len(held_out["equity_curve"]), int((self.end - self.split) / md.MINUTE))
            plot.assert_called_once()

    def test_report_selects_eligible_training_calmar(self):
        rows = pd.DataFrame([
            dict(buckets_per_day=25, vpin_window=15, calmar=1., total_pnl=50., eligible=True),
            dict(buckets_per_day=50, vpin_window=20, calmar=2., total_pnl=10., eligible=True),
            dict(buckets_per_day=100, vpin_window=30, calmar=9., total_pnl=100., eligible=False),
        ])
        rows["phase"] = "train"
        poor_test_result = dict.fromkeys(bt.METRICS, 0.)
        poor_test_result["calmar"] = -999.
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(bt, "configured_buckets", return_value=pd.DataFrame()) as buckets, \
                patch.object(bt, "calculate_vpin_metric", return_value=pd.DataFrame()) as vpin, \
                patch.object(bt, "run_single_backtest", return_value=poor_test_result) as simulate_test, \
                patch.object(bt, "plot_equity_curves"):
            output = Path(directory) / "results.csv"
            bt.report_results(rows, self.frame, self.config, self.start, self.split, self.end, output)
            self.assertEqual(buckets.call_args.args[2], 50)
            self.assertEqual(vpin.call_args.args[1], 20)
            simulate_test.assert_called_once()
            selected = pd.read_csv(output).iloc[-1]
            self.assertEqual((selected.buckets_per_day, selected.vpin_window), (50, 20))
            self.assertEqual(selected.calmar, -999.)

    def test_insufficient_history(self):
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            bt.prepare_evaluation(self.frame.iloc[:100], self.config)


if __name__ == "__main__":
    unittest.main()
