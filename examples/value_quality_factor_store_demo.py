"""Run the BM/F-Score value-quality screen from stored factor panels.

This demo loads precomputed BM and F-Score factor files from the Tiger factor
store, joins them into a long research frame, fetches matching close prices,
and runs the standard high-BM/high-F-Score versus low-BM/low-F-Score
long-short backtest.
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
from tiger_factors.factor_store import DEFAULT_FACTOR_STORE_ROOT
from tiger_factors.factor_store import TigerFactorLibrary
from tiger_factors.multifactor_evaluation import run_value_quality_long_short_backtest
from tiger_factors.multifactor_evaluation.reporting.portfolio import run_portfolio_from_backtest


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tiger_analysis_outputs" / "value_quality_factor_store_demo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BM/F-score value-quality screen from factor store panels.")
    parser.add_argument("--store-root", default=str(DEFAULT_FACTOR_STORE_ROOT), help="Root directory of the Tiger factor store.")
    parser.add_argument("--factor-provider", default="tiger", help="Provider namespace used when the factors were saved.")
    parser.add_argument("--factor-variant", default=None, help="Optional factor variant used when the factors were saved.")
    parser.add_argument("--bm-factor", default="BM", help="BM factor name as stored in the factor store.")
    parser.add_argument("--fscore-factor", default="FSCORE", help="F-Score factor name as stored in the factor store.")
    parser.add_argument("--price-provider", default="yahoo", help="Price provider used for the matching close panel.")
    parser.add_argument("--codes", nargs="*", default=None, help="Optional explicit universe to keep.")
    parser.add_argument("--start", default=None, help="Optional backtest start date.")
    parser.add_argument("--end", default=None, help="Optional backtest end date.")
    parser.add_argument("--high-quantile", type=float, default=0.5)
    parser.add_argument("--long-pct", type=float, default=0.25)
    parser.add_argument("--rebalance-freq", default="ME")
    parser.add_argument("--annual-trading-days", type=int, default=252)
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report-name", default="value_quality_factor_store")
    return parser.parse_args()


def _normalize_variant(value: str | None) -> str | None:
    if value is None:
        return None
    token = str(value).strip()
    if not token or token.lower() in {"none", "null", "na"}:
        return None
    return token


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    variant = _normalize_variant(args.factor_variant)

    library = TigerFactorLibrary(output_dir=args.store_root, price_provider=args.price_provider, verbose=True)
    bm_panel = library.load_factor_panel(
        factor_name=args.bm_factor,
        provider=args.factor_provider,
        variant=variant,
    )
    fscore_panel = library.load_factor_panel(
        factor_name=args.fscore_factor,
        provider=args.factor_provider,
        variant=variant,
    )

    if not isinstance(bm_panel, pd.DataFrame) or bm_panel.empty:
        raise ValueError(f"Could not load BM factor panel for {args.bm_factor!r} from factor store.")
    if not isinstance(fscore_panel, pd.DataFrame) or fscore_panel.empty:
        raise ValueError(f"Could not load F-score factor panel for {args.fscore_factor!r} from factor store.")

    common_dates = bm_panel.index.intersection(fscore_panel.index)
    common_codes = [code for code in bm_panel.columns.intersection(fscore_panel.columns)]
    if args.codes:
        requested_codes = [str(code) for code in args.codes]
        common_codes = [code for code in requested_codes if code in common_codes]
    if common_dates.empty:
        raise ValueError("No overlapping dates were found between the BM and F-score factor panels.")
    if not common_dates.empty:
        start = args.start or str(common_dates.min().date())
        end = args.end or str(common_dates.max().date())
    else:
        start = args.start
        end = args.end
    if not common_codes:
        raise ValueError("No overlapping codes were found between the BM and F-score factor panels.")
    if start is None or end is None:
        raise ValueError("Could not infer start/end dates from the loaded factor panels.")

    bm_panel = bm_panel.reindex(index=common_dates, columns=common_codes)
    fscore_panel = fscore_panel.reindex(index=common_dates, columns=common_codes)
    close_panel = library.price_panel(
        codes=common_codes,
        start=start,
        end=end,
        provider=args.price_provider,
        field="close",
    )
    if close_panel.empty:
        raise ValueError("Could not load a matching close panel for the selected factor universe.")

    panel = library.load_factor_frame(
        factor_names=[args.bm_factor, args.fscore_factor],
        provider=args.factor_provider,
        variant=variant,
        codes=common_codes,
        start=start,
        end=end,
    )
    panel = panel.rename(columns={args.bm_factor: "BM", args.fscore_factor: "FSCORE"})

    result = run_value_quality_long_short_backtest(
        panel,
        close_panel,
        bm_column="BM",
        fscore_column="FSCORE",
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
