"""Shared reporting helpers for generic factor-store composite backtests."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _clean_series(values: pd.Series) -> pd.Series:
    series = pd.Series(values, copy=False)
    series = pd.to_numeric(series, errors="coerce")
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    series.index = pd.DatetimeIndex(pd.to_datetime(series.index, errors="coerce"))
    series = series[~series.index.isna()].sort_index()
    return series


def save_factor_backtest_plot(
    backtest: pd.DataFrame,
    *,
    output_dir: str | Path,
    report_name: str = "factor_store_multi_factor",
) -> Path | None:
    """Save a generic cumulative-return plus drawdown chart for a factor backtest."""

    if "portfolio" not in backtest.columns:
        return None
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    portfolio = _clean_series(backtest["portfolio"])
    benchmark = _clean_series(backtest["benchmark"]) if "benchmark" in backtest.columns else None
    if portfolio.empty:
        return None

    portfolio_equity = (1.0 + portfolio.fillna(0.0)).cumprod()
    portfolio_drawdown = portfolio_equity.div(portfolio_equity.cummax()).sub(1.0)
    benchmark_equity = None
    benchmark_drawdown = None
    if benchmark is not None and not benchmark.empty:
        common_index = portfolio_equity.index.intersection(benchmark.index)
        portfolio_equity = portfolio_equity.loc[common_index]
        portfolio_drawdown = portfolio_drawdown.loc[common_index]
        benchmark = benchmark.loc[common_index]
        benchmark_equity = (1.0 + benchmark.fillna(0.0)).cumprod()
        benchmark_drawdown = benchmark_equity.div(benchmark_equity.cummax()).sub(1.0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(portfolio_equity.index, portfolio_equity.values, label="portfolio", color="#1f77b4", linewidth=2.0)
    if benchmark_equity is not None:
        axes[0].plot(
            benchmark_equity.index,
            benchmark_equity.values,
            label="benchmark",
            color="#ff7f0e",
            linewidth=1.8,
            alpha=0.9,
        )
    axes[0].set_title("Generic Factor-Store Composite Equity Curve")
    axes[0].set_ylabel("Growth of $1")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper left")

    axes[1].fill_between(portfolio_drawdown.index, portfolio_drawdown.values, 0.0, color="#1f77b4", alpha=0.25)
    axes[1].plot(portfolio_drawdown.index, portfolio_drawdown.values, color="#1f77b4", linewidth=1.5, label="portfolio DD")
    if benchmark_drawdown is not None:
        axes[1].fill_between(benchmark_drawdown.index, benchmark_drawdown.values, 0.0, color="#ff7f0e", alpha=0.15)
        axes[1].plot(
            benchmark_drawdown.index,
            benchmark_drawdown.values,
            color="#ff7f0e",
            linewidth=1.3,
            label="benchmark DD",
        )
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="lower left")

    fig.tight_layout()
    figure_path = output_path / f"{report_name}_equity_curve.png"
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    curve_frame = pd.DataFrame(
        {
            "portfolio": portfolio,
            "portfolio_equity": portfolio_equity,
            "portfolio_drawdown": portfolio_drawdown,
        }
    )
    if benchmark_equity is not None and benchmark_drawdown is not None:
        curve_frame["benchmark"] = benchmark
        curve_frame["benchmark_equity"] = benchmark_equity
        curve_frame["benchmark_drawdown"] = benchmark_drawdown
    curve_frame.to_csv(output_path / f"{report_name}_equity_curve.csv")
    return figure_path


__all__ = ["save_factor_backtest_plot"]
