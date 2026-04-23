from __future__ import annotations

import pandas as pd

from tiger_factors.factor_evaluation import calculate_benchmark_metrics
from tiger_factors.factor_evaluation import calculate_drawdown
from tiger_factors.factor_evaluation import calculate_monthly_returns
from tiger_factors.factor_evaluation import single_factor_backtest
from tiger_factors.multifactor_evaluation.backtest import multi_factor_backtest


def _sample_backtest_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=8)
    codes = ["A", "B", "C", "D"]
    rows = []
    for i, date in enumerate(dates):
        for j, code in enumerate(codes):
            rows.append(
                {
                    "date_": date,
                    "code": code,
                    "close": 100.0 + i * 1.5 + j * 0.8,
                    "factor_x": (i + 1) * (j + 1),
                    "factor_y": (i + 2) - j,
                }
            )
    return pd.DataFrame(rows)


def _sample_close_panel(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.pivot(index="date_", columns="code", values="close").sort_index()


def _sample_factor_panels(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "factor_x": frame.pivot(index="date_", columns="code", values="factor_x").sort_index(),
        "factor_y": frame.pivot(index="date_", columns="code", values="factor_y").sort_index(),
    }


def test_single_factor_backtest_and_helpers() -> None:
    frame = _sample_backtest_frame()
    result = single_factor_backtest(
        frame,
        factor_column="factor_x",
        price_column="close",
        forward_days=1,
        n_quantiles=3,
    )

    assert "portfolio_returns" in result
    assert "equity_curve" in result
    assert "metrics" in result
    assert "factor" in result["metrics"]
    assert "portfolio" in result["metrics"]
    assert "benchmark" in result["metrics"]

    monthly = calculate_monthly_returns(result["portfolio_returns"])
    drawdown = calculate_drawdown(result["equity_curve"])
    benchmark = calculate_benchmark_metrics(result["portfolio_returns"], result["portfolio_returns"] * 0.9 + 0.001)

    assert monthly.empty is False
    assert drawdown.empty is False
    assert "information_ratio" in benchmark


def test_multi_factor_backtest() -> None:
    frame = _sample_backtest_frame()
    close_panel = _sample_close_panel(frame)
    factor_panels = _sample_factor_panels(frame)

    result = multi_factor_backtest(
        factor_panels,
        close_panel,
        weights={"factor_x": 0.7, "factor_y": 0.3},
        rebalance_freq="B",
    )

    assert "composite_factor" in result
    assert "backtest" in result
    assert "portfolio_returns" in result
    assert result["portfolio_returns"].empty is False
