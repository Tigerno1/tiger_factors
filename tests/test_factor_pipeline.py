from __future__ import annotations

import numpy as np
import pandas as pd

from tiger_factors.multifactor_evaluation import FactorPipelineConfig, screen_factor_panels
from tiger_factors.multifactor_evaluation import create_multifactor_summary_report


def _build_synthetic_factor_panels() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    dates = pd.date_range("2022-01-01", periods=160, freq="D")
    codes = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    base_a = np.array([1.0, 0.6, 0.2, -0.2, -0.6], dtype=float)
    base_c = np.array([-0.3, 0.4, -0.1, 0.5, -0.2], dtype=float)

    factor_a_rows: list[list[float]] = []
    factor_b_rows: list[list[float]] = []
    factor_c_rows: list[list[float]] = []
    close_rows: list[list[float]] = []

    close = np.full(len(codes), 100.0, dtype=float)
    for idx, date in enumerate(dates):
        signal_a = base_a + 0.05 * np.sin(idx / 4.0)
        signal_c = base_c + 0.05 * np.cos(idx / 5.0)
        factor_a = signal_a + 0.01 * np.sin(idx / 3.0)
        factor_b = 0.97 * factor_a + 0.01 * np.cos(idx / 6.0)
        factor_c = signal_c + 0.01 * np.sin(idx / 7.0)

        factor_a_rows.append(factor_a.tolist())
        factor_b_rows.append(factor_b.tolist())
        factor_c_rows.append(factor_c.tolist())
        close_rows.append(close.tolist())

        if idx + 1 < len(dates):
            next_return = 0.002 * signal_a + 0.0015 * signal_c
            close = close * (1.0 + next_return)

    factor_panels = {
        "factor_a": pd.DataFrame(factor_a_rows, index=dates, columns=codes),
        "factor_b": pd.DataFrame(factor_b_rows, index=dates, columns=codes),
        "factor_c": pd.DataFrame(factor_c_rows, index=dates, columns=codes),
    }
    close_panel = pd.DataFrame(close_rows, index=dates, columns=codes)
    return factor_panels, close_panel


def test_screen_factor_panels_selects_low_correlation_factors_and_backtests(tmp_path) -> None:
    factor_panels, close_panel = _build_synthetic_factor_panels()
    result = screen_factor_panels(
        factor_panels,
        close_panel,
        config=FactorPipelineConfig(
            forward_days=1,
            top_n_initial=3,
            corr_threshold=0.8,
            selection_score_field="ic_ir",
            weight_method="positive",
            min_factor_weight=0.05,
            max_factor_weight=0.95,
            long_pct=0.4,
            long_short=True,
            transaction_cost_bps=5.0,
            slippage_bps=2.0,
            persist_outputs=False,
        ),
        output_dir=tmp_path / "outputs",
        report_dir=tmp_path / "reports",
        report_factor_name="synthetic_combined_factor",
    )

    assert result.summary is not None
    assert not result.summary.empty
    assert result.score_table is not None
    assert not result.score_table.empty
    assert {"metric_name", "metric_value", "metric_score", "combined_score"}.issubset(result.score_table.columns)
    assert result.score_table["factor_name"].nunique() == result.summary["factor_name"].nunique()
    assert result.score_table.groupby("factor_name").size().min() >= 2
    assert len(result.selected_factors) >= 2
    assert ("factor_a" in result.selected_factors) ^ ("factor_b" in result.selected_factors)
    assert "factor_c" in result.selected_factors
    assert abs(sum(result.factor_weights.values()) - 1.0) < 1e-9
    assert all(0.05 <= weight <= 0.95 for weight in result.factor_weights.values())
    assert not result.combined_factor.empty
    assert not result.backtest_returns.empty
    assert "portfolio" in result.backtest_returns.columns
    assert "benchmark" in result.backtest_returns.columns
    assert "turnover" in result.backtest_returns.columns
    assert "cost" in result.backtest_returns.columns
    assert result.backtest_stats["portfolio"]["sharpe"] == result.backtest_stats["portfolio"]["sharpe"]
    assert not (tmp_path / "outputs").exists()
    assert not (tmp_path / "reports").exists()


def test_multifactor_summary_report_writes_html_and_tables(tmp_path) -> None:
    dates = pd.bdate_range("2024-01-01", periods=80)
    backtest = pd.DataFrame(
        {
            "portfolio": np.linspace(0.001, 0.004, len(dates)) + 0.002 * np.sin(np.arange(len(dates)) / 5.0),
            "benchmark": np.linspace(0.0005, 0.0025, len(dates)) + 0.001 * np.cos(np.arange(len(dates)) / 6.0),
        },
        index=dates,
    )

    result = create_multifactor_summary_report(
        backtest,
        output_dir=tmp_path / "summary",
        report_name="multifactors_summary",
    )

    assert result is not None
    assert result.html_path.exists()
    assert result.summary_table_path.exists()
    assert result.comparison_table_path is not None
    assert result.comparison_table_path.exists()
    assert result.drawdown_table_path is not None
    assert result.drawdown_table_path.exists()
    assert result.monthly_returns_table_path is not None
    assert result.monthly_returns_table_path.exists()
    assert result.compare_table_paths is not None
    assert "monthly" in result.compare_table_paths
    assert result.compare_table_paths["monthly"].exists()
    assert result.montecarlo_summary_path is not None
    assert result.montecarlo_summary_path.exists()
    assert result.montecarlo_plot_path is not None
    assert result.montecarlo_plot_path.exists()
    assert result.manifest_path.exists()
    assert any(path.suffix == ".png" for path in result.figure_paths)
    html = result.html_path.read_text(encoding="utf-8")
    assert "Summary Metrics" in html
    assert "portfolio" in html
    assert "Monte Carlo" in html
    assert "Compare Tables" in html
