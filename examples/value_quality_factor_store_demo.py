"""Run the BM/F-Score value-quality screen from stored factor panels.

This demo loads precomputed BM and F-Score factor files from the Tiger factor
store, joins them into a long research frame, fetches matching close prices,
and runs the standard high-BM/high-F-Score versus low-BM/low-F-Score
long-short backtest.
"""

from __future__ import annotations

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
from tiger_factors.factor_store import DEFAULT_FACTOR_STORE_ROOT
from tiger_factors.factor_store import TigerFactorLibrary
from tiger_factors.multifactor_evaluation import run_value_quality_long_short_backtest
from tiger_factors.multifactor_evaluation.reporting.portfolio import run_portfolio_from_backtest


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tiger_analysis_outputs" / "value_quality_factor_store_demo"
DEFAULT_STORE_ROOT = DEFAULT_FACTOR_STORE_ROOT
DEFAULT_FACTOR_PROVIDER = "tiger"
DEFAULT_FACTOR_VARIANT: str | None = None
DEFAULT_BM_FACTOR = "BM"
DEFAULT_FSCORE_FACTOR = "FSCORE"
DEFAULT_PRICE_PROVIDER = "yahoo"
DEFAULT_CODES: list[str] | None = None
DEFAULT_START: str | None = None
DEFAULT_END: str | None = None
DEFAULT_HIGH_QUANTILE = 0.5
DEFAULT_LONG_PCT = 0.25
DEFAULT_REBALANCE_FREQ = "ME"
DEFAULT_ANNUAL_TRADING_DAYS = 252
DEFAULT_TRANSACTION_COST_BPS = 1.0
DEFAULT_SLIPPAGE_BPS = 1.0
DEFAULT_REPORT_NAME = "value_quality_factor_store"


def main() -> None:
    output_dir = Path(DEFAULT_OUTPUT_DIR)
    library = TigerFactorLibrary(output_dir=DEFAULT_STORE_ROOT, price_provider=DEFAULT_PRICE_PROVIDER, verbose=True)
    bm_panel = library.load_factor_panel(
        factor_name=DEFAULT_BM_FACTOR,
        provider=DEFAULT_FACTOR_PROVIDER,
        variant=DEFAULT_FACTOR_VARIANT,
    )
    fscore_panel = library.load_factor_panel(
        factor_name=DEFAULT_FSCORE_FACTOR,
        provider=DEFAULT_FACTOR_PROVIDER,
        variant=DEFAULT_FACTOR_VARIANT,
    )

    common_dates = bm_panel.index.intersection(fscore_panel.index)
    common_codes = [code for code in bm_panel.columns.intersection(fscore_panel.columns)]
    if DEFAULT_CODES:
        requested_codes = [str(code) for code in DEFAULT_CODES]
        common_codes = [code for code in requested_codes if code in common_codes]
    start = DEFAULT_START or str(common_dates.min().date())
    end = DEFAULT_END or str(common_dates.max().date())

    bm_panel = bm_panel.reindex(index=common_dates, columns=common_codes)
    fscore_panel = fscore_panel.reindex(index=common_dates, columns=common_codes)
    close_panel = library.price_panel(
        codes=common_codes,
        start=start,
        end=end,
        provider=DEFAULT_PRICE_PROVIDER,
        field="close",
    )

    panel = library.load_factor_frame(
        factor_names=[DEFAULT_BM_FACTOR, DEFAULT_FSCORE_FACTOR],
        provider=DEFAULT_FACTOR_PROVIDER,
        variant=DEFAULT_FACTOR_VARIANT,
        codes=common_codes,
        start=start,
        end=end,
    )
    panel = panel.rename(columns={DEFAULT_BM_FACTOR: "BM", DEFAULT_FSCORE_FACTOR: "FSCORE"})

    result = run_value_quality_long_short_backtest(
        panel,
        close_panel,
        bm_column="BM",
        fscore_column="FSCORE",
        high_quantile=DEFAULT_HIGH_QUANTILE,
        long_pct=DEFAULT_LONG_PCT,
        rebalance_freq=DEFAULT_REBALANCE_FREQ,
        long_short=True,
        annual_trading_days=DEFAULT_ANNUAL_TRADING_DAYS,
        transaction_cost_bps=DEFAULT_TRANSACTION_COST_BPS,
        slippage_bps=DEFAULT_SLIPPAGE_BPS,
    )

    report = run_portfolio_from_backtest(
        result["backtest"],
        output_dir=output_dir,
        report_name=DEFAULT_REPORT_NAME,
    )
    figure_path = save_value_quality_backtest_plot(
        result["backtest"],
        output_dir=output_dir,
        report_name=DEFAULT_REPORT_NAME,
    )

    print("loaded factor panels:")
    print(f"  BM panel shape: {bm_panel.shape}")
    print(f"  F-score panel shape: {fscore_panel.shape}")
    print(f"  overlapping codes: {len(common_codes)}")
    print(f"  date range: {start} -> {end}")
    print("\nvalue-quality combo head:")
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
