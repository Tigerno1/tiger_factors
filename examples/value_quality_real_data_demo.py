"""Run the BM / F-score value-quality screen on a real long research frame.

The script expects a long table with at least ``date_``, ``code``,
``BM`` and ``FSCORE`` columns, plus a wide close panel for the same universe.
It reuses the Tiger-native value/quality backtest and produces a local
portfolio report plus a standard equity-curve / drawdown figure.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MATPLOTLIB_CACHE_DIR = PROJECT_ROOT / "tiger_analysis_outputs" / ".cache" / "matplotlib"
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

from tiger_factors.examples.value_quality_reporting import save_value_quality_backtest_plot
from tiger_factors.multifactor_evaluation import run_value_quality_long_short_backtest
from tiger_factors.multifactor_evaluation.reporting.portfolio import run_portfolio_from_backtest


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tiger_analysis_outputs" / "value_quality_real_data_demo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real-data BM/F-score long-short demo.")
    parser.add_argument("--panel", required=True, help="Path to a long table with date_, code, BM, FSCORE.")
    parser.add_argument("--close-panel", required=True, help="Path to a wide close panel with date_ as the index.")
    parser.add_argument("--bm-column", default="BM")
    parser.add_argument("--fscore-column", default="FSCORE")
    parser.add_argument("--high-quantile", type=float, default=0.5)
    parser.add_argument("--long-pct", type=float, default=0.25)
    parser.add_argument("--rebalance-freq", default="ME")
    parser.add_argument("--annual-trading-days", type=int, default=252)
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report-name", default="value_quality_real_data")
    return parser.parse_args()


def _load_frame(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(file_path)
    elif suffix == ".csv":
        frame = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    return frame


def _load_panel(path: str | Path) -> pd.DataFrame:
    frame = _load_frame(path)
    if "date_" in frame.columns:
        frame["date_"] = pd.to_datetime(frame["date_"], errors="coerce")
        frame = frame.dropna(subset=["date_"]).set_index("date_")
    elif not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index, errors="coerce")
        frame = frame.loc[~frame.index.isna()]
    frame.index = pd.DatetimeIndex(frame.index, name="date_")
    frame = frame.apply(pd.to_numeric, errors="coerce")
    return frame.sort_index()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    panel = _load_frame(args.panel)
    close_panel = _load_panel(args.close_panel)

    if args.bm_column not in panel.columns or args.fscore_column not in panel.columns:
        raise KeyError(
            f"panel must contain {args.bm_column!r} and {args.fscore_column!r} columns"
        )

    result = run_value_quality_long_short_backtest(
        panel,
        close_panel,
        bm_column=args.bm_column,
        fscore_column=args.fscore_column,
        high_quantile=args.high_quantile,
        long_pct=args.long_pct,
        rebalance_freq=args.rebalance_freq,
        long_short=True,
        annual_trading_days=args.annual_trading_days,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
    )

    report = run_portfolio_from_backtest(
        result["backtest"],
        output_dir=output_dir,
        report_name=args.report_name,
    )
    figure_path = save_value_quality_backtest_plot(
        result["backtest"],
        output_dir=output_dir,
        report_name=args.report_name,
    )

    print("value-quality combo head:")
    print(result["combo_frame"].head().to_string(index=False))
    print("\nbacktest stats:")
    print(pd.DataFrame(result["stats"]).T.to_string())
    if figure_path is not None:
        print(f"\nvalue-quality equity curve: {figure_path}")
    print("\nportfolio report:")
    if report is not None:
        print(report.to_summary())


if __name__ == "__main__":
    main()
