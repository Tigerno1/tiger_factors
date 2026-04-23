"""Minimal CSM research + backtest demo.

This script fits a cross-sectional selection model on a synthetic panel,
turns the predicted scores into a wide factor panel, and runs the
multifactor backtest helper on top of it.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tiger_factors.factor_frame import build_csm_model
from tiger_factors.factor_frame import infer_csm_feature_columns
from tiger_factors.multifactor_evaluation import run_csm_backtest
from tiger_factors.multifactor_evaluation import run_csm_selection_backtest
from tiger_factors.multifactor_evaluation.pipeline import run_factor_backtest


def _build_sample_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-01", periods=40)
    codes = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    rng = np.random.default_rng(17)

    factor_rows: list[dict[str, object]] = []
    close_returns: dict[str, list[float]] = {code: [] for code in codes}

    code_drift = {code: 0.0008 * idx for idx, code in enumerate(codes)}
    for date_idx, date in enumerate(dates):
        market_shock = 0.0004 * np.sin(date_idx / 4.0)
        for code_idx, code in enumerate(codes):
            momentum = code_idx + 0.07 * date_idx + rng.normal(0.0, 0.03)
            value = (len(codes) - code_idx) + 0.03 * date_idx + rng.normal(0.0, 0.03)
            quality = 0.35 * momentum - 0.18 * value + rng.normal(0.0, 0.03)
            forward_return = 0.55 * momentum - 0.4 * value + 0.2 * quality + rng.normal(0.0, 0.05)
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
                0.0005
                + code_drift[code]
                + 0.0007 * momentum
                - 0.0003 * value
                + 0.0002 * quality
                + market_shock
                + rng.normal(0.0, 0.005)
            )
            close_returns[code].append(realized_return)

    factor_frame = pd.DataFrame(factor_rows)
    close_panel = pd.DataFrame(close_returns, index=dates).sort_index()
    close_panel.index.name = "date_"
    close_panel = (1.0 + close_panel).cumprod() * 100.0
    return factor_frame, close_panel


def main() -> None:
    panel, close_panel = _build_sample_panel()
    train = panel[panel["date_"] <= pd.Timestamp("2024-01-26")].copy()
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

    score_panel = model.score_panel(panel)
    direct_backtest, direct_stats = run_factor_backtest(
        score_panel,
        close_panel,
        long_pct=0.25,
        rebalance_freq="W-FRI",
        long_short=True,
        annual_trading_days=252,
        transaction_cost_bps=1.0,
        slippage_bps=1.0,
    )
    csm_backtest = run_csm_backtest(
        panel,
        close_panel,
        fit_method="ranknet",
        feature_transform="zscore",
        min_group_size=3,
        normalize_score_by_date=True,
        learning_rate=0.1,
        max_iter=100,
        long_pct=0.25,
        rebalance_freq="W-FRI",
        long_short=True,
        annual_trading_days=252,
        transaction_cost_bps=1.0,
        slippage_bps=1.0,
    )
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

    print("feature weights:")
    print(model.weights_)
    print("\nfeature stats:")
    print(model.feature_stats_.to_string(index=False))
    print("\nscore panel shape:", score_panel.shape)
    print("\ndirect backtest stats:")
    print(pd.DataFrame(direct_stats).T.to_string())
    print("\nhelper backtest stats:")
    print(pd.DataFrame(csm_backtest.backtest_stats).T.to_string())
    print("\nselection backtest stats:")
    print(pd.DataFrame(selection_backtest.backtest_stats).T.to_string())
    print("\nbacktest head:")
    print(direct_backtest.head().to_string())


if __name__ == "__main__":
    main()
