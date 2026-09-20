"""Rebuild the guide from saved evidence; --run-backtests refreshes the frozen study."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
import pandas as pd

import market_data as md
import vpin_backtest as bt
import vpin_calculator as vc
from vpin_optuna_optimize import VPINOptimizer, N_TRIALS, N_STARTUP_TRIALS

ASSETS = ROOT / "docs" / "vpin_guide_assets"
MANIFEST = ASSETS / "run_manifest.json"
OUTPUT = ROOT / "output" / "pdf"
ANALYSIS_FILES = (
    "market_data.py", "vpin_calculator.py", "vpin_backtest.py",
    "vpin_optuna_optimize.py", "test_vpin.py", "config.json",
)
EVIDENCE_FILES = (
    "grid_trials.csv", "optuna_trials.csv", "held_out.csv",
    "equity_drawdown.pdf", "indicator_example.pdf",
)


def save_explanation_figures():
    """Original worked examples, independent of market data or study eligibility."""
    blue, orange, gray = "#2878B5", "#E69F00", "#707070"
    fig, ax = plt.subplots(figsize=(7, 2.7))
    left = 0
    for time, volume, buys in [("23:59", 6, 1.5), ("00:00", 12, 9), ("00:01", 12, 6), ("00:02", 4, 2)]:
        fraction = buys / volume
        ax.add_patch(Rectangle((left, 2), volume, .65 * fraction, facecolor=blue, edgecolor="white"))
        ax.add_patch(Rectangle((left, 2 + .65 * fraction), volume, .65 * (1 - fraction),
                               facecolor=orange, edgecolor="white"))
        ax.text(left + volume / 2, 2.78, f"{time}\nv={volume}, B={buys:g}", ha="center", fontsize=9)
        left += volume
    left = 0
    for label, volume, buys in [("Bucket 1", 10, 4.5), ("Bucket 2", 20, 12), ("Tail", 4, 2)]:
        fraction = buys / volume
        ax.add_patch(Rectangle((left, .65), volume, .65 * fraction, facecolor=blue, edgecolor="white"))
        ax.add_patch(Rectangle((left, .65 + .65 * fraction), volume, .65 * (1 - fraction),
                               facecolor=orange, edgecolor="white"))
        caption = f"{label}\nB={buys:g}, S={volume-buys:g}" if label != "Tail" else "Tail\nomitted"
        ax.text(left + volume / 2, .48, caption, ha="center", va="top", fontsize=9)
        ax.annotate("", xy=(left + volume / 2, 1.35), xytext=(left + volume / 2, 1.9),
                    arrowprops=dict(arrowstyle="->", color=gray))
        left += volume
    for boundary in (10, 30):
        ax.plot([boundary, boundary], [.6, 2.67], color=gray, linestyle="--", linewidth=.8)
    ax.text(0, 3.55, "Blue: buys     Orange: sells     Width: base volume", fontsize=10)
    ax.set(xlim=(-.5, 34.5), ylim=(-.1, 3.85))
    ax.axis("off")
    fig.tight_layout(pad=.5)
    fig.savefig(ASSETS / "bucket_example.pdf")
    plt.close(fig)

    values = np.array([.10, .18, .18, .22, .29, .32, .35, .40, .45, .50])
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(np.arange(1, 11), values, color=np.where(values <= .35, blue, "#BBBBBB"))
    ax.axhline(.35, color=orange, linewidth=1.6, linestyle="--", label="Current VPIN = 0.35")
    ax.text(.98, .93, "70 of 100 prior values are at or below 0.35\nPercentile = 0.70", transform=ax.transAxes,
            ha="right", va="top", fontsize=10, bbox=dict(facecolor="white", edgecolor="#DDDDDD"))
    ax.set(xlabel="Sorted groups (10 prior observations per bar)", ylabel="Prior VPIN", ylim=(0, .68),
           xticks=np.arange(1, 11))
    ax.legend(loc="upper left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS / "percentile_example.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.text(3.5, 3.3, r"Completed minute $\rightarrow$ available signal $\rightarrow$ next minute's opening fill",
            ha="center", fontsize=10)
    for x, title, detail, color in [(0.3, "CASH", "Start here; after an exit,\nwait two days", orange),
                                    (4.4, "LONG", "Hold asset quantity;\nmark equity at minute closes", blue)]:
        ax.add_patch(FancyBboxPatch((x, 1), 2.3, 1.15, boxstyle="round,pad=0.05",
                                    facecolor=color, alpha=.16, edgecolor=color))
        ax.text(x + 1.15, 1.8, title, ha="center", weight="bold", fontsize=12)
        ax.text(x + 1.15, 1.3, detail, ha="center", va="center", fontsize=9)
    ax.annotate("", xy=(4.4, 2.1), xytext=(2.6, 2.1),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.4", color=blue))
    ax.text(3.5, 2.65, "CDF < 0.99, available,\nand cooldown complete", ha="center", fontsize=9)
    ax.annotate("", xy=(2.6, 1.1), xytext=(4.4, 1.1),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.35", color=orange))
    ax.text(3.5, .5, r"CDF $\geq$ 0.99: exit", ha="center", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.text(3.5, .1, "Missing CDF: keep the current position.  End of test: liquidate holdings.",
            ha="center", fontsize=9)
    ax.set(xlim=(0, 7), ylim=(-.1, 3.65))
    ax.axis("off")
    fig.tight_layout(pad=.5)
    fig.savefig(ASSETS / "strategy_flow.pdf")
    plt.close(fig)


def sha256(path):
    """Canonical LF fingerprints for text; byte-exact fingerprints for binary assets."""
    path = Path(path)
    if path.suffix.lower() in {".py", ".json", ".csv", ".tex", ".md"}:
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def search(method, config, bounds):
    """Each worker reads the same frozen cache; only pre-split rows reach selection."""
    start, split, _ = map(pd.Timestamp, bounds)
    data = pd.read_feather(ROOT / md.get_data_file(config["symbol"]))
    data = data[(data.open_time >= pd.to_datetime(config["start_date"], utc=True)) & (data.open_time < split)]
    if method == "grid":
        trials = bt.run_parameter_sweep(data, config, start, split)
    else:
        optimizer = VPINOptimizer(data, config, start, split)
        optimizer.run(N_TRIALS)
        trials = pd.DataFrame(optimizer.trial_results)
    trials.to_csv(ASSETS / f"{method}_trials.csv", index=False)
    print(f"{method}: saved {len(trials)} training trials", flush=True)


def save_figures(results, indicator, symbol, end):
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    benchmark = next(iter(results.values()))["equity_curve"]
    for label, result in results.items():
        curve = result["equity_curve"].set_index("time")
        daily = curve.resample("1D", closed="right", label="right")
        axes[0].plot(daily.equity.last(), label=label)
        axes[1].plot(daily.drawdown.min(), label=label)
    benchmark = benchmark.set_index("time")
    wealth = benchmark.buy_hold_equity
    drawdown = pd.Series(bt.calculate_drawdowns(wealth.to_numpy()), index=wealth.index)
    axes[0].plot(wealth.resample("1D", closed="right", label="right").last(),
                 color="black", linestyle="--", label="Buy and hold")
    axes[1].plot(drawdown.resample("1D", closed="right", label="right").min(),
                 color="black", linestyle="--", label="Buy and hold")
    axes[0].set_ylabel("Wealth (initial = 1)")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("UTC")
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ASSETS / "equity_drawdown.pdf")
    plt.close(fig)

    last = indicator[indicator.time >= end - pd.Timedelta(days=14)]
    fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    spans = vc._find_toxic_spans(last)
    for ax, column, label in zip(axes, ["price", "VPIN", "CDF"], [f"{symbol} close", "VPIN", "Percentile"]):
        ax.plot(last.time, last[column], linewidth=0.8)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        for left, right in spans:
            ax.axvspan(left, right, color="red", alpha=0.12)
    axes[1].set_ylim(0, 1.05)
    axes[2].set_ylim(0, 1.05)
    axes[2].axhline(0.99, color="red", linestyle="--", linewidth=0.8)
    axes[2].set_xlabel("UTC")
    fig.tight_layout()
    fig.savefig(ASSETS / "indicator_example.pdf")
    plt.close(fig)


def run_backtests(end, start_date=None):
    checked = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromName("test_vpin"))
    if not checked.wasSuccessful():
        raise RuntimeError("Regression checks failed; study was not started")
    config = md.load_config()
    if start_date is not None:
        config["start_date"] = pd.to_datetime(start_date, format="%Y-%m-%d", utc=True).strftime("%Y-%m-%d")
    ASSETS.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        status="running", created_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        hash_convention="sha256-lf-text-v1",
        requested_start=config["start_date"], data_end_exclusive=end.isoformat(),
        config=config, source_sha256={name: sha256(ROOT / name) for name in ANALYSIS_FILES},
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
        python=sys.version.split()[0],
        packages={name: version(name) for name in ("numpy", "pandas", "aiohttp", "pyarrow", "matplotlib", "optuna")},
        search=dict(grid_combinations=143, optuna_trials=N_TRIALS, seed=42,
                    startup_trials=N_STARTUP_TRIALS, buckets_per_day=list(bt.BUCKETS_PER_DAY_RANGE),
                    vpin_window=list(bt.VPIN_WINDOW_RANGE), minimum_exits=bt.MIN_EXITS_REQUIRED),
        tests=dict(command="python -m unittest -q test_vpin", count=checked.testsRun, passed=True),
    )
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    try:
        data = md.manage_local_data(config["symbol"], start_date_dt=config["start_date"], now=end)
        start, split, stop = bt.prepare_evaluation(data, config)
    except (ValueError, RuntimeError) as error:
        manifest.update(status="data_unavailable", error=str(error))
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Study stopped without numerical results: {error}", flush=True)
        return

    bounds = [value.isoformat() for value in (start, split, stop)]
    manifest.update(bounds=bounds, data_rows=len(data),
                    data_first_open=data.open_time.iloc[0].isoformat(),
                    data_last_close=data.close_time.iloc[-1].isoformat(),
                    data_sha256=sha256(ROOT / md.get_data_file(config["symbol"])))
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen scoring dates: {bounds}", flush=True)
    # Independent searches share the snapshot, not trial state or held-out scores.
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(search, method, config, bounds) for method in ("grid", "optuna")]
        for future in futures:
            future.result()

    trials = {method: pd.read_csv(ASSETS / f"{method}_trials.csv") for method in ("grid", "optuna")}
    if len(trials["grid"]) != 143 or len(trials["optuna"]) != N_TRIALS:
        raise ValueError("Incomplete search output; refusing to publish a completed study")
    selected = {}
    all_valid = []
    for method, rows in trials.items():
        valid = rows[rows.eligible.eq(True)].sort_values("calmar", ascending=False, kind="stable")
        manifest[method + "_eligible"] = len(valid)
        if not valid.empty:
            winner = valid.iloc[0]
            key = (int(winner.buckets_per_day), int(winner.vpin_window))
            selected.setdefault(key, []).append(method.capitalize())
            all_valid.append(valid)
    default = (config["bucket_target_bars_per_day"], config["vpin_window"])
    selected.setdefault(default, []).append("Default")
    champion = pd.concat(all_valid).sort_values("calmar", ascending=False, kind="stable").iloc[0] if all_valid else None
    illustrated = ((int(champion.buckets_per_day), int(champion.vpin_window))
                   if champion is not None else default)
    manifest["best_training"] = (dict(buckets_per_day=illustrated[0], vpin_window=illustrated[1],
                                      calmar=float(champion.calmar)) if champion is not None else None)

    results, held_out = {}, []
    for (count, window), roles in selected.items():
        label = "/".join(roles)
        buckets = vc.configured_buckets(data, config, count)
        indicator = vc.calculate_vpin_metric(buckets, window, config["cdf_lookback_days"])
        result = bt.run_single_backtest(indicator, data, start_time=split, end_time=stop,
                                       fee_bps=config["fee_bps"], slippage_bps=config["slippage_bps"])
        results[f"{label}: {count}/{window}"] = result
        record = bt.result_record(count, window, result, split, stop, phase="test")
        record["selected_by"] = label
        held_out.append(record)
        if (count, window) == illustrated:
            shown_indicator = indicator

    reference = next(iter(results.values()))
    benchmark_wealth = reference["equity_curve"].buy_hold_equity.iloc[-1]
    years = (stop - split).total_seconds() / (365.25 * 86400)
    cagr = float((benchmark_wealth ** (1 / years) - 1) * 100)
    drawdown = reference["buy_hold_max_drawdown"]
    held_out.append(dict(selected_by="Buy and hold", buckets_per_day=np.nan, vpin_window=np.nan,
                         phase="test", start_time=split, end_time=stop, total_pnl=reference["buy_hold_pnl"],
                         cagr=cagr, max_drawdown=drawdown,
                         calmar=cagr / abs(drawdown) if drawdown < 0 else np.nan, num_exits=0))
    pd.DataFrame(held_out).to_csv(ASSETS / "held_out.csv", index=False)
    save_figures(results, shown_indicator, config["symbol"], stop)
    manifest.update(status="complete", illustrated_config=list(illustrated),
                    artifacts_sha256={name: sha256(ASSETS / name) for name in EVIDENCE_FILES})
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def latex(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}"}
    return "".join(replacements.get(char, char) for char in str(value))


def table(frame, columns, headings):
    lines = [r"\begin{tabular}{" + "l" + "r" * (len(columns) - 1) + "}",
             r"\toprule", " & ".join(headings) + r"\\", r"\midrule"]
    for _, row in frame.iterrows():
        values = []
        for name in columns:
            value = row[name]
            if pd.isna(value):
                values.append("--")
            elif name in ("buckets_per_day", "vpin_window", "num_exits"):
                values.append(str(int(value)))
            elif isinstance(value, (float, np.floating, int, np.integer)):
                values.append(f"{value:.3f}" if name == "calmar" else f"{value:.2f}")
            else:
                values.append(latex(value))
        lines.append(" & ".join(values) + r"\\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"])


def write_tables(manifest):
    complete = manifest["status"] == "complete"
    for name, expected in manifest["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Analysis source changed since the study: {name}")
    commands = [
        r"\newif\ifResultsAvailable",
        r"\ResultsAvailable" + ("true" if complete else "false"),
    ]
    values = dict(
        SnapshotStart=manifest["requested_start"],
        SnapshotEnd=manifest["data_end_exclusive"].replace("T", " ").replace("+00:00", " UTC"),
        RunStatus=manifest["status"].replace("_", " "),
        RunError=manifest.get("error", "No completed empirical study is available."),
    )
    for macro, key in dict(ReportSymbol="symbol", DefaultBuckets="bucket_target_bars_per_day",
                           DefaultWindow="vpin_window", ADVLookback="adv_lookback_days",
                           CDFLookback="cdf_lookback_days", FeeBps="fee_bps",
                           SlippageBps="slippage_bps").items():
        values[macro] = manifest["config"][key]
    if "coverage_evidence_sha256" in manifest:
        if sha256(ASSETS / "data_gaps.csv") != manifest["coverage_evidence_sha256"]:
            raise ValueError("Coverage evidence changed")
        gaps = pd.read_csv(ASSETS / "data_gaps.csv")
        if len(gaps) != manifest["gap_intervals"] or gaps.minutes.sum() != manifest["missing_candles"]:
            raise ValueError("Coverage counts disagree with the manifest")
    if complete:
        for name, expected in manifest["artifacts_sha256"].items():
            if sha256(ASSETS / name) != expected:
                raise ValueError(f"Saved evidence changed: {name}")
        grid = pd.read_csv(ASSETS / "grid_trials.csv")
        optuna = pd.read_csv(ASSETS / "optuna_trials.csv")
        rows = pd.concat([grid, optuna], ignore_index=True)
        best = (rows[rows.eligible.eq(True)].sort_values("calmar", ascending=False, kind="stable")
                .drop_duplicates(["buckets_per_day", "vpin_window"]).head(5))
        held_out = pd.read_csv(ASSETS / "held_out.csv")
        if not rows.phase.eq("train").all() or not held_out.phase.eq("test").all():
            raise ValueError("Result phases do not match the report protocol")
        values.update(
            DataRows=f"{manifest['data_rows']:,}", TrainingStart=manifest["bounds"][0][:16].replace("T", " "),
            SplitTime=manifest["bounds"][1][:16].replace("T", " "),
            GridEligible=manifest["grid_eligible"], OptunaEligible=manifest["optuna_eligible"],
            DataHash=manifest["data_sha256"], SourceRevision=manifest["git_head"],
            IllustrationConfig="/".join(map(str, manifest["illustrated_config"])),
        )
        headings = [r"$k$", r"$n$", r"Net \%", r"CAGR \%", r"DD \%", "Calmar", "Exits"]
        columns = ["buckets_per_day", "vpin_window", "total_pnl", "cagr", "max_drawdown", "calmar", "num_exits"]
        train_table = table(best, columns, headings) if not best.empty else "No training candidate passed eligibility."
        commands.append(r"\newcommand{\TrainingTable}{" + train_table + "}")
        held_out["configuration"] = held_out.apply(
            lambda row: "Buy and hold" if pd.isna(row.buckets_per_day)
            else f"{int(row.buckets_per_day)}/{int(row.vpin_window)} ({row.selected_by})", axis=1)
        test_table = table(held_out, ["configuration"] + columns[2:],
                           ["Configuration"] + headings[2:])
        commands.append(r"\newcommand{\HeldOutTable}{" + test_table + "}")
        champion = manifest["best_training"]
        winner_text = (f"Best training Calmar: {champion['buckets_per_day']} buckets/day and "
                       f"window {champion['vpin_window']} (Calmar {champion['calmar']:.3f})."
                       if champion else "No training candidate qualified; defaults are shown only as a reference.")
        values["WinnerText"] = winner_text
    commands.extend(r"\newcommand{\%s}{%s}" % (name, latex(value)) for name, value in values.items())
    (ASSETS / "tables.tex").write_text("\n".join(commands) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-backtests", action="store_true", help="Download a frozen snapshot and run both full searches")
    parser.add_argument("--start-date", help="Explicit study start override; does not modify config.json")
    parser.add_argument("--end-date", help="Exclusive UTC date for reproduction; defaults to today's midnight")
    args = parser.parse_args()
    if (args.start_date or args.end_date) and not args.run_backtests:
        parser.error("Date overrides require --run-backtests")
    os.chdir(ROOT)
    if args.run_backtests:
        end = (pd.to_datetime(args.end_date, format="%Y-%m-%d", utc=True)
               if args.end_date else pd.Timestamp.now(tz="UTC").normalize())
        if end > pd.Timestamp.now(tz="UTC").normalize():
            parser.error("The snapshot cannot include an incomplete or future UTC day")
        run_backtests(end, args.start_date)
    if not MANIFEST.exists():
        parser.error("No saved study; run with --run-backtests first")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    write_tables(manifest)
    save_explanation_figures()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    command = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
               f"-output-directory={OUTPUT}", "vpin_guide.tex"]
    for _ in range(2):
        subprocess.run(command, cwd=ROOT / "docs", check=True, stdout=subprocess.DEVNULL)
    print(f"Guide built: {OUTPUT / 'vpin_guide.pdf'} (study status: {manifest['status']})")


if __name__ == "__main__":
    main()
