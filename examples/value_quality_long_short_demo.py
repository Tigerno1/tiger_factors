"""Minimal BM/F-Score long-short demo.

This demo builds a tiny synthetic panel, feeds the BM and F-score columns
directly into the value-quality screen, and then runs the classic high B/M
+ high F-score versus low B/M + low F-score long/short backtest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MATPLOTLIB_CACHE_DIR = PROJECT_ROOT / "tiger_analysis_outputs" / ".cache" / "matplotlib"
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

from tiger_factors.multifactor_evaluation import run_value_quality_long_short_backtest
from tiger_factors.multifactor_evaluation.reporting.portfolio import run_portfolio_from_backtest
from tiger_factors.examples.value_quality_reporting import save_value_quality_backtest_plot


def _build_sample_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-01", periods=40)
    codes = ["A", "B", "C", "D", "E", "F"]
    rng = np.random.default_rng(17)

    rows: list[dict[str, object]] = []
    close_returns: dict[str, list[float]] = {code: [] for code in codes}

    value_ranking = {code: len(codes) - idx for idx, code in enumerate(codes)}
    fscore_ranking = {code: idx + 1 for idx, code in enumerate(codes)}
    for date_idx, date in enumerate(dates):
        market_shock = 0.0006 * np.sin(date_idx / 6.0)
        for code in codes:
            bm = float(value_ranking[code]) + rng.normal(0.0, 0.05)
            fscore = float(fscore_ranking[code]) + rng.normal(0.0, 0.05)
            rows.append(
                {
                    "date_": date,
                    "code": code,
                    "BM": bm,
                    "FSCORE": fscore,
                }
            )
            realized_return = (
                0.0005
                + 0.0008 * bm
                + 0.0006 * fscore
                + market_shock
                + rng.normal(0.0, 0.0035)
            )
            close_returns[code].append(realized_return)

    frame = pd.DataFrame(rows)
    close_panel = pd.DataFrame(close_returns, index=dates).sort_index()
    close_panel.index.name = "date_"
    close_panel = (1.0 + close_panel).cumprod() * 100.0
    return frame, close_panel


def main() -> None:
    output_dir = PROJECT_ROOT / "tiger_analysis_outputs" / "value_quality_long_short_demo"
    panel, close_panel = _build_sample_panel()

    result = run_value_quality_long_short_backtest(
        panel,
        close_panel,
        bm_column="BM",
        fscore_column="FSCORE",
        long_pct=0.25,
        rebalance_freq="ME",
        long_short=True,
        annual_trading_days=252,
        transaction_cost_bps=1.0,
        slippage_bps=1.0,
    )

    report = run_portfolio_from_backtest(
        result["backtest"],
        output_dir=output_dir,
        report_name="value_quality_long_short",
    )
    figure_path = save_value_quality_backtest_plot(
        result["backtest"],
        output_dir=output_dir,
        report_name="value_quality_long_short",
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
