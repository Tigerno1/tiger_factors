"""Self-contained demo for the factor_portfolio workflow.

This example shows the new factor -> stock weight -> positions -> backtest
pipeline on a small synthetic universe so it can run without external data.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MATPLOTLIB_CACHE_DIR = PROJECT_ROOT / "tiger_analysis_outputs" / ".cache" / "matplotlib"
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MATPLOTLIB_CACHE_DIR)
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd

from tiger_factors.factor_portfolio import run_factor_portfolio_workflow


def _build_sample_data() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    dates = pd.bdate_range("2024-01-01", periods=48)
    codes = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    rng = np.random.default_rng(7)

    momentum = pd.DataFrame(index=dates, columns=codes, dtype=float)
    value = pd.DataFrame(index=dates, columns=codes, dtype=float)
    close = pd.DataFrame(index=dates, columns=codes, dtype=float)

    base_levels = np.array([95.0, 100.0, 105.0, 110.0, 115.0, 120.0])
    drift = np.array([0.0015, 0.0008, 0.0002, -0.0002, -0.0008, -0.0012])

    for i, date in enumerate(dates):
        cross_section = rng.normal(0.0, 0.15, size=len(codes))
        momentum.loc[date] = cross_section + 0.03 * i
        value.loc[date] = cross_section[::-1] - 0.02 * i
        if i == 0:
            close.loc[date] = base_levels
        else:
            prev = close.iloc[i - 1].to_numpy(dtype=float)
            shock = rng.normal(0.0, 0.005, size=len(codes))
            close.loc[date] = prev * (1.0 + drift + 0.004 * cross_section + shock)

    factor_panels = {
        "momentum": momentum,
        "value": value,
    }
    return factor_panels, close


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the factor_portfolio workflow on synthetic data.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "tiger_analysis_outputs" / "factor_portfolio_workflow_demo"))
    parser.add_argument("--long-only", action="store_true", help="Build a long-only portfolio instead of long-short.")
    parser.add_argument("--skip-report", action="store_true", help="Skip rendering the portfolio tear sheet.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    factor_panels, close_panel = _build_sample_data()

    result = run_factor_portfolio_workflow(
        factor_panels,
        close_panel,
        factor_weights={"momentum": 0.65, "value": 0.35},
        long_only=args.long_only,
        standardize=True,
        rebalance_freq="ME",
        output_dir=None if args.skip_report else output_dir,
        report_name="factor_portfolio_workflow",
        open_report=False,
    )

    print("factor weights:")
    print(result.portfolio.factor_weights)
    print("\nbacktest stats:")
    print(result.stats)
    print("\npositions head:")
    print(result.positions.head().to_string())

    if not args.skip_report and result.report is not None:
        print(f"\nSaved report to: {output_dir}")


if __name__ == "__main__":
    main()
