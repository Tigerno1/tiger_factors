"""Minimal CSM research + portfolio report demo.

This script fits a cross-sectional selection model on a synthetic panel,
turns the selected scores into a wide factor panel, runs the multifactor
selection backtest, and renders local portfolio/trade/position reports.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MATPLOTLIB_CACHE_DIR = PROJECT_ROOT / "tiger_analysis_outputs" / ".cache" / "matplotlib"
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

from tiger_factors.factor_frame import build_csm_model
from tiger_factors.factor_frame import infer_csm_feature_columns
from tiger_factors.multifactor_evaluation import run_csm_selection_backtest
from tiger_factors.multifactor_evaluation.reporting.portfolio import run_portfolio_from_backtest


def _build_sample_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-01", periods=52)
    codes = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    rng = np.random.default_rng(29)

    factor_rows: list[dict[str, object]] = []
    close_returns: dict[str, list[float]] = {code: [] for code in codes}

    code_bias = {code: 0.0006 * idx for idx, code in enumerate(codes)}
    for date_idx, date in enumerate(dates):
        market_shock = 0.0005 * np.sin(date_idx / 5.0)
        for code_idx, code in enumerate(codes):
            momentum = code_idx + 0.08 * date_idx + rng.normal(0.0, 0.04)
            value = (len(codes) - code_idx) + 0.02 * date_idx + rng.normal(0.0, 0.04)
            quality = 0.4 * momentum - 0.22 * value + rng.normal(0.0, 0.03)
            forward_return = 0.5 * momentum - 0.42 * value + 0.18 * quality + rng.normal(0.0, 0.05)
            factor_rows.append(
                {
                    "date_": date,
                    "code": code,
                    "momentum": momentum,
                    "value": value,
                    "quality": quality,
                    "forward_return": forward_return,
                }
            )
            realized_return = (
                0.0004
                + code_bias[code]
                + 0.0008 * momentum
                - 0.0004 * value
                + 0.0002 * quality
                + market_shock
                + rng.normal(0.0, 0.004)
            )
            close_returns[code].append(realized_return)

    factor_frame = pd.DataFrame(factor_rows)
    close_panel = pd.DataFrame(close_returns, index=dates).sort_index()
    close_panel.index.name = "date_"
    close_panel = (1.0 + close_panel).cumprod() * 100.0
    return factor_frame, close_panel


def main() -> None:
    output_dir = PROJECT_ROOT / "tiger_analysis_outputs" / "csm_model_portfolio_demo"
    panel, close_panel = _build_sample_panel()
    train = panel[panel["date_"] <= pd.Timestamp("2024-01-30")].copy()
    feature_columns = infer_csm_feature_columns(train)

    model = build_csm_model(
        feature_columns,
        fit_method="ranknet",
        feature_transform="zscore",
        min_group_size=3,
        normalize_score_by_date=True,
        learning_rate=0.1,
        max_iter=100,
    )
    model.fit(train)

    selection_backtest = run_csm_selection_backtest(
        panel,
        close_panel,
        fit_method="ranknet",
        feature_transform="zscore",
        min_group_size=3,
        normalize_score_by_date=True,
        learning_rate=0.1,
        max_iter=100,
        top_n=2,
        bottom_n=2,
        long_only=False,
        annual_trading_days=252,
        transaction_cost_bps=1.0,
        slippage_bps=1.0,
    )

    report = run_portfolio_from_backtest(
        selection_backtest.backtest_returns,
        output_dir=output_dir,
        report_name="csm_model",
    )

    print("feature weights:")
    print(model.weights_)
    print("\nfeature stats:")
    print(model.feature_stats_.to_string(index=False))
    print("\nselection score panel shape:", selection_backtest.score_panel.shape)
    print("\nselection backtest stats:")
    print(pd.DataFrame(selection_backtest.backtest_stats).T.to_string())
    if report is not None:
        print("\nportfolio report:")
        print(report.to_summary())


if __name__ == "__main__":
    main()
