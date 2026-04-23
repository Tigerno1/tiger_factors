from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from tiger_factors.report_paths import figure_output_dir_for
from tiger_factors.factor_evaluation.core import (
    FactorEvaluation,
    evaluate_factor_groups,
    evaluate_factor_panel,
    factor_autocorrelation,
    rank_factor_autocorrelation,
)
from tiger_factors.factor_evaluation.plotting import plot_cumulative_returns
from tiger_factors.factor_evaluation.plotting import plot_cumulative_returns_by_quantile
from tiger_factors.factor_evaluation.plotting import plot_drawdown
from tiger_factors.factor_evaluation.plotting import plot_ic_hist
from tiger_factors.factor_evaluation.plotting import plot_ic_ts
from tiger_factors.factor_evaluation.plotting import plot_monthly_heatmap
from tiger_factors.factor_evaluation.plotting import plot_regression_scatter
from tiger_factors.factor_evaluation.plotting import plot_rolling_sharpe
from tiger_factors.factor_evaluation.plotting import plot_series_barh
from tiger_factors.factor_evaluation.plotting import plot_series_histogram
from tiger_factors.factor_evaluation.plotting import plot_series_line
from tiger_factors.factor_evaluation.plotting import save_figure


@dataclass(frozen=True)
class FactorTearSheetResult:
    factor_name: str
    evaluation: FactorEvaluation
    ic_series: pd.Series
    rank_ic_series: pd.Series
    factor_autocorr_series: pd.Series
    rank_factor_autocorr_series: pd.Series
    quantile_returns: pd.DataFrame
    long_short_returns: pd.Series
    output_dir: Path
    figure_paths: dict[str, Path]
    table_paths: dict[str, Path]
    figure_output_dir: Path | None = None
    group_summary: pd.DataFrame | None = None
    benchmark_regression: pd.DataFrame | None = None

    def to_summary(self) -> dict[str, Any]:
        payload = asdict(self.evaluation)
        payload["factor_name"] = self.factor_name
        payload["output_dir"] = str(self.output_dir)
        return payload


def _align_factor_and_forward_returns(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_index = factor.index.intersection(forward_returns.index)
    common_columns = factor.columns.intersection(forward_returns.columns)
    return factor.loc[common_index, common_columns], forward_returns.loc[common_index, common_columns]


def _safe_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _save_native_figure(
    figure_paths: dict[str, Path],
    key: str,
    ax,
    figure_output: Path,
    filename: str,
) -> None:
    path = figure_output / filename
    save_figure(ax, path)
    figure_paths[key] = path


def _save_native_table(
    table_paths: dict[str, Path],
    key: str,
    table: pd.DataFrame | pd.Series,
    output: Path,
    filename: str,
) -> None:
    path = output / filename
    if isinstance(table, pd.Series):
        table.to_frame().to_parquet(path)
    else:
        table.to_parquet(path)
    table_paths[key] = path


def monthly_returns_heatmap(returns: pd.Series) -> pd.DataFrame:
    series = _safe_series(returns).dropna()
    if series.empty:
        return pd.DataFrame()
    monthly = series.resample("ME").apply(lambda x: (1.0 + x).prod() - 1.0)
    heatmap = monthly.to_frame("return")
    heatmap["year"] = heatmap.index.year
    heatmap["month"] = heatmap.index.month
    table = heatmap.pivot(index="year", columns="month", values="return").sort_index()
    table.columns = [pd.Timestamp(2000, int(col), 1).strftime("%b") for col in table.columns]
    return table


def _build_quantile_returns(factor: pd.DataFrame, forward_returns: pd.DataFrame, quantiles: int) -> tuple[pd.DataFrame, pd.Series]:
    factor, forward_returns = _align_factor_and_forward_returns(factor, forward_returns)
    if factor.empty:
        empty = pd.DataFrame(index=factor.index)
        return empty, pd.Series(dtype=float)

    factor_long = factor.stack().rename("factor")
    returns_long = forward_returns.stack().rename("forward_return")
    joined = pd.concat([factor_long, returns_long], axis=1).dropna()
    if joined.empty:
        empty = pd.DataFrame(index=factor.index)
        return empty, pd.Series(dtype=float)

    joined = joined.reset_index()
    date_col = joined.columns[0]
    code_col = joined.columns[1]

    def _assign_quantiles(group: pd.DataFrame) -> pd.DataFrame:
        ranks = group["factor"].rank(method="first", pct=True)
        q = np.ceil(ranks * quantiles).astype(int).clip(1, quantiles)
        group = group.copy()
        group["quantile"] = q
        return group

    joined = pd.concat(
        [_assign_quantiles(group) for _, group in joined.groupby(date_col, sort=False)],
        ignore_index=True,
    )
    quantile_returns = (
        joined.groupby([date_col, "quantile"])["forward_return"]
        .mean()
        .unstack("quantile")
        .sort_index()
    )
    quantile_returns.index = pd.DatetimeIndex(quantile_returns.index)

    long_short = quantile_returns.iloc[:, -1] - quantile_returns.iloc[:, 0]
    long_short.name = "long_short"
    return quantile_returns, long_short


def create_native_full_tear_sheet(
    factor_name: str,
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    *,
    output_dir: str | Path,
    figure_output_dir: str | Path | None = None,
    quantiles: int = 5,
    portfolio_returns: pd.Series | None = None,
    benchmark_returns: pd.Series | None = None,
    group_labels: pd.DataFrame | pd.Series | None = None,
) -> FactorTearSheetResult:
    factor, forward_returns = _align_factor_and_forward_returns(factor, forward_returns)
    benchmark_returns = _safe_series(benchmark_returns).dropna().sort_index() if benchmark_returns is not None else None
    evaluation = evaluate_factor_panel(factor, forward_returns, benchmark_returns=benchmark_returns)
    date_index = factor.index.intersection(forward_returns.index)
    ic_series = factor.loc[date_index].apply(lambda row: row.corr(forward_returns.loc[row.name]), axis=1)
    rank_ic_series = factor.loc[date_index].apply(
        lambda row: row.rank().corr(forward_returns.loc[row.name].rank()),
        axis=1,
    )
    ic_series = _safe_series(ic_series)
    rank_ic_series = _safe_series(rank_ic_series)
    factor_autocorr_series = factor_autocorrelation(factor)
    rank_factor_autocorr_series = rank_factor_autocorrelation(factor)

    quantile_returns, long_short_returns = _build_quantile_returns(factor, forward_returns, quantiles=quantiles)
    group_summary = evaluate_factor_groups(factor, forward_returns, group_labels, benchmark_returns=benchmark_returns) if group_labels is not None else None
    benchmark_regression = None
    if benchmark_returns is not None:
        benchmark_regression = pd.DataFrame(
            [
                {
                    "alpha": evaluation.benchmark_alpha,
                    "beta": evaluation.benchmark_beta,
                    "r2": evaluation.benchmark_r2,
                    "n_obs": evaluation.benchmark_n_obs,
                }
            ]
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figure_output = Path(figure_output_dir) if figure_output_dir is not None else output
    figure_output.mkdir(parents=True, exist_ok=True)

    figure_paths: dict[str, Path] = {}
    table_paths: dict[str, Path] = {}

    tables = {
        "evaluation": pd.DataFrame([evaluation.__dict__]),
        "ic_series": ic_series.to_frame(name="ic"),
        "rank_ic_series": rank_ic_series.to_frame(name="rank_ic"),
        "factor_autocorr_series": factor_autocorr_series.to_frame(name="factor_autocorr"),
        "rank_factor_autocorr_series": rank_factor_autocorr_series.to_frame(name="rank_factor_autocorr"),
        "quantile_returns": quantile_returns,
        "long_short_returns": long_short_returns.to_frame(name="long_short"),
    }
    if group_summary is not None and not group_summary.empty:
        tables["group_summary"] = group_summary
    if benchmark_regression is not None:
        tables["benchmark_regression"] = benchmark_regression
    for name, table in tables.items():
        _save_native_table(table_paths, name, table, output, f"{name}.parquet")

    ax = plot_ic_ts(ic_series.to_frame(name="ic"))
    ax.set_title(f"{factor_name} - IC series")
    _save_native_figure(figure_paths, "ic_series", ax, figure_output, "ic_series.png")
    ax = plot_ic_hist(ic_series.to_frame(name="ic"))
    ax.set_title(f"{factor_name} - IC histogram")
    _save_native_figure(figure_paths, "ic_histogram", ax, figure_output, "ic_histogram.png")
    if not factor_autocorr_series.empty:
        ax = plot_series_line(
            factor_autocorr_series,
            title=f"{factor_name} - Factor autocorrelation",
            color="#4b8bbe",
            label="factor autocorr",
        )
        _save_native_figure(figure_paths, "factor_autocorr_series", ax, figure_output, "factor_autocorr_series.png")
        ax = plot_series_histogram(
            factor_autocorr_series,
            title=f"{factor_name} - Factor autocorrelation histogram",
            color="#4b8bbe",
        )
        _save_native_figure(figure_paths, "factor_autocorr_histogram", ax, figure_output, "factor_autocorr_histogram.png")
    if not rank_factor_autocorr_series.empty:
        ax = plot_series_line(
            rank_factor_autocorr_series,
            title=f"{factor_name} - Rank factor autocorrelation",
            color="#2aa876",
            label="rank factor autocorr",
        )
        _save_native_figure(figure_paths, "rank_factor_autocorr_series", ax, figure_output, "rank_factor_autocorr_series.png")
        ax = plot_series_histogram(
            rank_factor_autocorr_series,
            title=f"{factor_name} - Rank factor autocorrelation histogram",
            color="#2aa876",
        )
        _save_native_figure(figure_paths, "rank_factor_autocorr_histogram", ax, figure_output, "rank_factor_autocorr_histogram.png")
    if not quantile_returns.empty:
        ax = plot_cumulative_returns_by_quantile(quantile_returns, period=None)
        ax.set_title(f"{factor_name} - Quantile cumulative returns")
        _save_native_figure(figure_paths, "quantile_cumulative_returns_overview", ax, figure_output, "quantile_cumulative_returns_overview.png")
    if not long_short_returns.empty:
        ax = plot_cumulative_returns(long_short_returns, period=None, title=f"{factor_name} - Top minus bottom cumulative returns")
        _save_native_figure(figure_paths, "long_short_cumulative_returns_overview", ax, figure_output, "long_short_cumulative_returns_overview.png")
        ax = plot_drawdown(long_short_returns, title=f"{factor_name} - Long short drawdown")
        _save_native_figure(figure_paths, "long_short_drawdown", ax, figure_output, "long_short_drawdown.png")
        ax = plot_rolling_sharpe(long_short_returns, title=f"{factor_name} - Long short rolling Sharpe")
        _save_native_figure(figure_paths, "long_short_rolling_sharpe", ax, figure_output, "long_short_rolling_sharpe.png")
        ax = plot_monthly_heatmap(long_short_returns, title=f"{factor_name} - Long short monthly heatmap")
        _save_native_figure(figure_paths, "long_short_monthly_heatmap", ax, figure_output, "long_short_monthly_heatmap.png")
    if benchmark_returns is not None and not long_short_returns.empty:
        ax = plot_regression_scatter(
            benchmark_returns.loc[long_short_returns.index.intersection(benchmark_returns.index)],
            long_short_returns.loc[long_short_returns.index.intersection(benchmark_returns.index)],
            alpha=evaluation.benchmark_alpha,
            beta=evaluation.benchmark_beta,
            title=f"{factor_name} - Long short vs benchmark regression",
        )
        _save_native_figure(figure_paths, "benchmark_regression_scatter", ax, figure_output, "benchmark_regression_scatter.png")
    if group_summary is not None and not group_summary.empty and "ic_mean" in group_summary.columns:
        ax = plot_series_barh(
            group_summary.set_index("group")["ic_mean"],
            title=f"{factor_name} - Group IC mean",
            color="#2a6fdb",
            xlabel="IC mean",
        )
        _save_native_figure(figure_paths, "group_ic_mean", ax, figure_output, "group_ic_mean.png")
        if "fitness" in group_summary.columns:
            ax = plot_series_barh(
                group_summary.set_index("group")["fitness"],
                title=f"{factor_name} - Group fitness",
                color="#c44536",
                xlabel="Fitness",
            )
            _save_native_figure(figure_paths, "group_fitness", ax, figure_output, "group_fitness.png")

    if portfolio_returns is not None:
        portfolio_returns = _safe_series(portfolio_returns).dropna()
        if not portfolio_returns.empty:
            portfolio_returns = portfolio_returns.sort_index()
            cumulative = _series_to_cumulative(portfolio_returns)
            drawdown = _drawdown_from_returns(portfolio_returns)
            rolling_sharpe = _annualized_sharpe(portfolio_returns)

            _save_native_table(table_paths, "portfolio_cumulative_returns", cumulative, output, "portfolio_cumulative_returns.parquet")
            _save_native_table(table_paths, "portfolio_drawdown", drawdown, output, "portfolio_drawdown.parquet")
            _save_native_table(table_paths, "portfolio_rolling_sharpe", rolling_sharpe, output, "portfolio_rolling_sharpe.parquet")

            ax = plot_cumulative_returns(portfolio_returns, title=f"{factor_name} - Portfolio cumulative returns")
            _save_native_figure(figure_paths, "portfolio_cumulative_returns", ax, figure_output, "portfolio_cumulative_returns.png")
            ax = plot_drawdown(portfolio_returns, title=f"{factor_name} - Portfolio drawdown")
            _save_native_figure(figure_paths, "portfolio_drawdown", ax, figure_output, "portfolio_drawdown.png")
            ax = plot_rolling_sharpe(portfolio_returns, title=f"{factor_name} - Portfolio rolling Sharpe")
            _save_native_figure(figure_paths, "portfolio_rolling_sharpe", ax, figure_output, "portfolio_rolling_sharpe.png")
            ax = plot_monthly_heatmap(portfolio_returns, title=f"{factor_name} - Portfolio monthly heatmap")
            _save_native_figure(figure_paths, "portfolio_monthly_heatmap", ax, figure_output, "portfolio_monthly_heatmap.png")

    if benchmark_returns is not None:
        benchmark_returns = _safe_series(benchmark_returns).dropna()
        if not benchmark_returns.empty:
            benchmark_returns = benchmark_returns.sort_index()
            _save_native_table(table_paths, "benchmark_returns", benchmark_returns, output, "benchmark_returns.parquet")

    return FactorTearSheetResult(
        factor_name=factor_name,
        evaluation=evaluation,
        ic_series=ic_series,
        rank_ic_series=rank_ic_series,
        factor_autocorr_series=factor_autocorr_series,
        rank_factor_autocorr_series=rank_factor_autocorr_series,
        quantile_returns=quantile_returns,
        long_short_returns=long_short_returns,
        output_dir=output,
        figure_output_dir=figure_output,
        figure_paths=figure_paths,
        table_paths=table_paths,
        group_summary=group_summary,
        benchmark_regression=benchmark_regression,
    )
