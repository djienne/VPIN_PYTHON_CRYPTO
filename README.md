# VPIN Calculator (Binance 1m)

Compute the Volume-Synchronized Probability of Informed Trading (VPIN) for Binance symbols using 1-minute klines. The scripts fetch/cach historical data, build volume buckets, calculate VPIN plus a rolling CDF “toxicity” score, and plot the results.

## Repository contents
- `market_data.py` – async downloader for 1m klines with retry/backoff; maintains a feather cache (`bnbusdt_1m.feather`) and can back/forward fill gaps.
- `vpin_calculator.py` – builds volume buckets, computes VPIN and rolling CDF, saves CSV results, and generates plots.
- `config.json` – runtime parameters (symbol, bucket sizing, windows, start date).
- `info_VPIN.md` – notes on the VPIN methodology and formulas.
- `bnbusdt_1m.feather` – cached sample 1m data; `vpin_results.csv`, `vpin_plot.png`, `vpin_plot_oct_zoom.png` are sample outputs.

## Requirements
- Python 3.10+
- Packages: `pandas`, `numpy`, `aiohttp`, `matplotlib`, `pyarrow` (for feather I/O)

Example setup:
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\activate
pip install --upgrade pip
pip install pandas numpy aiohttp matplotlib pyarrow
```

## Configuration
`config.json` controls the run:
- `symbol`: Binance symbol, e.g., `BNBUSDT`.
- `bucket_target_bars_per_day`: desired number of volume buckets per day (50 is common in VPIN literature).
- `adv_lookback_days`: days used to estimate ADV for bucket sizing.
- `bucket_size_base`: manual bucket size override in base units; leave `null` to auto-size from ADV/local data.
- `vpin_window`: rolling bucket window for VPIN numerator (default 50).
- `cdf_lookback_days`: rolling time window (days) for percentile ranking of VPIN.
- `start_date`: earliest candle to load/analyze (YYYY-MM-DD).

## Usage
1) **Download/refresh klines**
```bash
python market_data.py
```
This loads `bnbusdt_1m.feather` if present, fetches missing data from Binance (handles 429/418 responses with backoff), and saves the updated cache. It skips downloads if the cache was modified within the last 15 minutes.

2) **Calculate VPIN and plots**
```bash
python vpin_calculator.py
```
This will:
- Determine bucket size `V` (ADV / `bucket_target_bars_per_day`, or override if provided).
- Build volume buckets from 1m data (buy volume = taker buy base; sell volume = total - buy).
- Compute VPIN and a rolling CDF percentile.
- Save `vpin_results.csv`, `vpin_plot.png`, and `vpin_plot_oct_zoom.png` (center date is set in the script; adjust if desired).

## Interpreting results
- VPIN is between 0 and 1; higher values indicate greater order-flow imbalance.
- The CDF line gives a percentile of current VPIN vs. recent history (`cdf_lookback_days` window). Rule-of-thumb alerts:
  - > 0.90: elevated
  - > 0.95: high
  - > 0.99: extreme (potential liquidity stress/flow toxicity)

## Methodology snapshot
- Volume bucket size: `V = ADV / bucket_target_bars_per_day` (ADV from Binance daily data; falls back to local data).
- Bucketing: candles are split fractionally so each bucket sums to `V` base units.
- Order flow imbalance: `OI = |buy_volume - sell_volume|` per bucket.
- VPIN: rolling sum of `OI` over `vpin_window` buckets divided by `vpin_window * V`.
- CDF: rolling percentile rank of VPIN over `cdf_lookback_days` to contextualize toxicity.

## Notes
- If you point `symbol` to something other than the cached data, run `market_data.py` first to build a fresh feather file.
- The downloader is conservative with request pacing but Binance rate limits still apply; very long histories may need multiple runs.
- See `info_VPIN.md` for a fuller explanation of the algorithm and thresholds.
