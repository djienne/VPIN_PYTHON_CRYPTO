# VPIN Calculator (Binance 1m)

Adaptive volume-bucket VPIN and a long-or-cash backtest for Binance spot candles.

The [guide](output/pdf/vpin_guide.pdf) is the canonical explanation of the calculation, strategy, book context, study protocol, and evidence. Its [LaTeX source](docs/vpin_guide.tex) is included.

## Setup

Python 3.10+:

```bash
pip install pandas numpy aiohttp matplotlib pyarrow tqdm optuna
python -m unittest -q test_vpin
```

Tests cover independent numerical examples, candle coverage, causality, execution, costs, and training-only selection. Agreement between grid search and Optuna checks integration; it is not independent proof of the model.

## Basic use

Run from the project root:

```bash
python market_data.py
python vpin_calculator.py
python vpin_backtest.py
python vpin_optuna_optimize.py
```

Edit [config.json](config.json) for the symbol, requested start date, bucket targets, indicator windows, per-side fee/slippage assumptions, and plot dates. All commands use the same validated configuration. The 252/10 settings remain a provisional baseline.

The downloader accepts only contiguous, closed UTC minute candles. It repairs missing ranges and replaces overlapping recent rows before saving atomically. Old naive-time caches are rebuilt. Costs, warmup, trading rules, and output-column meanings are specified in the guide.

## Study evidence

Read the study status and any validated results in the guide and [run manifest](docs/vpin_guide_assets/run_manifest.json). The original history beginning in 2020 failed coverage validation on 2026-09-20; its [directly rechecked gap intervals](docs/vpin_guide_assets/data_gaps.csv) are retained as evidence. A failed study does not acquire substitute performance claims.

The configured start date must cover a valid continuous interval before the analysis commands can finish. Changing the research interval is an explicit study choice.

## Build the guide

MiKTeX or TeX Live with `pdflatex` is needed to compile the PDF.

```bash
python docs/build_vpin_guide.py
python docs/build_vpin_guide.py --run-backtests
```

The first command rebuilds from saved evidence. The second downloads a frozen snapshot, runs both full searches, saves evidence, and builds the guide. Explicit `--start-date YYYY-MM-DD` and `--end-date YYYY-MM-DD` options set that study's interval (end exclusive) without changing `config.json`. Searches use separate processes against the same snapshot; the held-out data never enters parameter selection.

A completed study stores full trial CSVs, selected held-out metrics, hashes and versions, generated tables, and two performance figures in `docs/vpin_guide_assets/`. Original teaching diagrams explain buckets, percentiles, and trading rules independently of market results. Failed coverage is recorded without performance tables or figures.

Raw caches, ordinary analysis outputs, and LaTeX scratch files are ignored by Git. The three copyrighted source books remain local and ignored; the authored guide contains original summaries and citations.
