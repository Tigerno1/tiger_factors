from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tiger_factors.multifactor_evaluation.analysis import MultifactorAnalysisResult
from tiger_factors.multifactor_evaluation.analysis import analyze_multifactor
from tiger_factors.multifactor_evaluation.common.matplotlib_config import configure_matplotlib
from tiger_factors.multifactor_evaluation.common.parquet_utils import to_parquet_clean
from tiger_factors.multifactor_evaluation.common.plotting import finalize_quantstats_axis
from tiger_factors.multifactor_evaluation.common.plotting import save_quantstats_figure
from tiger_factors.multifactor_evaluation.common.results import TearSheetResultMixin
from tiger_factors.multifactor_evaluation.reporting.html_report import render_artifact_report

configure_matplotlib()

import matplotlib.pyplot as plt
import seaborn as sns


def _ensure_output_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _flatten_summary(summary: dict[str, Any]) -> pd.DataFrame:
    frame = pd.json_normalize(summary, sep=".")
    if frame.empty:
        return pd.DataFrame([{}])
    return frame


def _dict_frame(data: dict[str, Any] | None, *, key_name: str = "metric", value_name: str = "value") -> pd.DataFrame:
    if not data:
        return pd.DataFrame(columns=[key_name, value_name])
    items = []
    for key, value in data.items():
        if isinstance(value, (dict, list, tuple, set)):
            items.append({key_name: key, value_name: json.dumps(value, default=str, ensure_ascii=False)})
        else:
            items.append({key_name: key, value_name: value})
    return pd.DataFrame(items)


def _save_frame_bundle(frame: pd.DataFrame | pd.Series | None, output_dir: Path, stem: str) -> Path | None:
    if frame is None:
        return None
    table = pd.DataFrame(frame)
    if table.empty:
        return None
    parquet_path = output_dir / f"{stem}.parquet"
    csv_path = output_dir / f"{stem}.csv"
    table.to_csv(csv_path, index=True)
    try:
        to_parquet_clean(table, parquet_path)
        return parquet_path
    except Exception:
        return csv_path


def _save_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    return path


def _as_series(values: pd.Series | pd.DataFrame | None) -> pd.Series:
    if values is None:
        return pd.Series(dtype=float)
    if isinstance(values, pd.DataFrame):
        if values.empty:
            return pd.Series(dtype=float)
        for candidate in ("portfolio", "returns", "return", "equity_curve"):
            if candidate in values.columns:
                values = values[candidate]
                break
        else:
            numeric = values.select_dtypes(include=[np.number])
            if numeric.empty:
                return pd.Series(dtype=float)
            values = numeric.iloc[:, 0]
    series = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    series.index = pd.DatetimeIndex(series.index, name="date_")
    return series.sort_index()


def _figure_dir(output_dir: Path) -> Path:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir


def _save_equity_curve(returns: pd.Series, path: Path, *, benchmark: pd.Series | None = None) -> Path | None:
    if returns.empty:
        return None
    cumulative = (1.0 + returns.fillna(0.0)).cumprod() - 1.0
    plt.figure(figsize=(12, 4))
    plt.plot(cumulative.index, cumulative.values, color="#1d4ed8", linewidth=2.0, label="portfolio")
    if benchmark is not None and not benchmark.empty:
        aligned = benchmark.reindex(cumulative.index).ffill().fillna(0.0)
        benchmark_curve = (1.0 + aligned.fillna(0.0)).cumprod() - 1.0
        plt.plot(benchmark_curve.index, benchmark_curve.values, color="#64748b", linewidth=1.8, label="benchmark")
    finalize_quantstats_axis(plt.gca(), title="Equity Curve", ylabel="Cumulative Return", legend=True)
    save_quantstats_figure(path)
    return path


def _save_benchmark_compare(returns: pd.Series, benchmark: pd.Series | None, path: Path) -> Path | None:
    if returns.empty or benchmark is None or benchmark.empty:
        return None
    aligned_index = returns.index.intersection(benchmark.index)
    if aligned_index.empty:
        return None
    left = (1.0 + returns.reindex(aligned_index).fillna(0.0)).cumprod() - 1.0
    right = (1.0 + benchmark.reindex(aligned_index).fillna(0.0)).cumprod() - 1.0
    plt.figure(figsize=(12, 4))
    plt.plot(left.index, left.values, label="portfolio", color="#0b5cab", linewidth=2.0)
    plt.plot(right.index, right.values, label="benchmark", color="#9ca3af", linewidth=1.8)
    finalize_quantstats_axis(plt.gca(), title="Benchmark Compare", ylabel="Cumulative Return", legend=True)
    save_quantstats_figure(path)
    return path


def _save_drawdown_plot(drawdown: pd.Series, path: Path) -> Path | None:
    if drawdown.empty:
        return None
    plt.figure(figsize=(12, 4))
    plt.fill_between(drawdown.index, drawdown.values, 0.0, color="#ef4444", alpha=0.35)
    plt.plot(drawdown.index, drawdown.values, color="#b91c1c", linewidth=1.8)
    finalize_quantstats_axis(plt.gca(), title="Drawdown", ylabel="Drawdown")
    save_quantstats_figure(path)
    return path


def _save_rolling_plot(series: pd.Series, path: Path, *, title: str, ylabel: str, color: str) -> Path | None:
    if series.empty:
        return None
    plt.figure(figsize=(12, 3))
    plt.plot(series.index, series.values, color=color, linewidth=1.6)
    finalize_quantstats_axis(plt.gca(), title=title, ylabel=ylabel)
    save_quantstats_figure(path)
    return path


def _save_distribution_plot(returns: pd.Series, path: Path) -> Path | None:
    if returns.empty:
        return None
    plt.figure(figsize=(10, 4))
    sns.histplot(returns.dropna().values, bins=40, kde=True, color="#2563eb", stat="density")
    finalize_quantstats_axis(plt.gca(), title="Returns Distribution", xlabel="Return", ylabel="Density")
    save_quantstats_figure(path)
    return path


def _save_monthly_heatmap(monthly_heatmap: pd.DataFrame, path: Path) -> Path | None:
    if monthly_heatmap.empty:
        return None
    plt.figure(figsize=(12, max(4, int(monthly_heatmap.shape[0] * 0.55))))
    sns.heatmap(monthly_heatmap, annot=False, cmap="RdYlGn", center=0.0, cbar_kws={"label": "Return"})
    finalize_quantstats_axis(plt.gca(), title="Monthly Returns Heatmap", xlabel="Month", ylabel="Year")
    save_quantstats_figure(path)
    return path


@dataclass(frozen=True)
class MultifactorAnalysisReportSpec:
    save_html: bool = True
    save_tables: bool = True
    save_figures: bool = True
    save_summary_table: bool = True
    save_metric_table: bool = True
    save_compare_table: bool = True
    save_drawdown_table: bool = True
    save_monthly_returns_table: bool = True
    save_monthly_heatmap_table: bool = True
    save_positions_summary: bool = True
    save_latest_holdings: bool = True
    save_transaction_summary: bool = True
    save_round_trip_summary: bool = True
    save_round_trips: bool = True
    save_equity_curve_figure: bool = True
    save_benchmark_compare_figure: bool = True
    save_drawdown_figure: bool = True
    save_rolling_sharpe_figure: bool = True
    save_rolling_sortino_figure: bool = True
    save_rolling_volatility_figure: bool = True
    save_rolling_beta_figure: bool = True
    save_returns_distribution_figure: bool = True
    save_monthly_heatmap_figure: bool = True

    def wants_table(self, name: str) -> bool:
        mapping = {
            "summary": self.save_summary_table,
            "metrics": self.save_metric_table,
            "compare": self.save_compare_table,
            "drawdown": self.save_drawdown_table,
            "monthly_returns": self.save_monthly_returns_table,
            "monthly_heatmap": self.save_monthly_heatmap_table,
            "positions_summary": self.save_positions_summary,
            "latest_holdings": self.save_latest_holdings,
            "transaction_summary": self.save_transaction_summary,
            "round_trip_summary": self.save_round_trip_summary,
            "round_trips": self.save_round_trips,
        }
        return mapping.get(name, False)

    def wants_figure(self, name: str) -> bool:
        mapping = {
            "equity_curve": self.save_equity_curve_figure,
            "benchmark_compare": self.save_benchmark_compare_figure,
            "drawdown": self.save_drawdown_figure,
            "rolling_sharpe": self.save_rolling_sharpe_figure,
            "rolling_sortino": self.save_rolling_sortino_figure,
            "rolling_volatility": self.save_rolling_volatility_figure,
            "rolling_beta": self.save_rolling_beta_figure,
            "returns_distribution": self.save_returns_distribution_figure,
            "monthly_heatmap": self.save_monthly_heatmap_figure,
        }
        return mapping.get(name, False)


@dataclass(frozen=True)
class MultifactorAnalysisReportResult(TearSheetResultMixin):
    output_dir: Path
    report_name: str
    html_path: Path
    summary_table_path: Path
    metric_table_path: Path
    compare_table_path: Path | None
    drawdown_table_path: Path | None
    monthly_returns_table_path: Path | None
    monthly_heatmap_table_path: Path | None
    positions_summary_path: Path | None
    latest_holdings_path: Path | None
    transaction_summary_path: Path | None
    round_trip_summary_path: Path | None
    round_trips_path: Path | None
    figure_paths: list[Path]
    manifest_path: Path
    analysis: MultifactorAnalysisResult
    spec: MultifactorAnalysisReportSpec

    def _table_paths(self) -> dict[str, Path]:
        paths: dict[str, Path] = {
            "summary": self.summary_table_path,
            "metrics": self.metric_table_path,
        }
        if self.compare_table_path is not None:
            paths["compare"] = self.compare_table_path
        if self.drawdown_table_path is not None:
            paths["drawdown"] = self.drawdown_table_path
        if self.monthly_returns_table_path is not None:
            paths["monthly_returns"] = self.monthly_returns_table_path
        if self.monthly_heatmap_table_path is not None:
            paths["monthly_heatmap"] = self.monthly_heatmap_table_path
        if self.positions_summary_path is not None:
            paths["positions_summary"] = self.positions_summary_path
        if self.latest_holdings_path is not None:
            paths["latest_holdings"] = self.latest_holdings_path
        if self.transaction_summary_path is not None:
            paths["transaction_summary"] = self.transaction_summary_path
        if self.round_trip_summary_path is not None:
            paths["round_trip_summary"] = self.round_trip_summary_path
        if self.round_trips_path is not None:
            paths["round_trips"] = self.round_trips_path
        return paths

    def report(self) -> str | None:
        return self.html_path.stem if self.html_path.exists() else None

    def preferred_table_order(self) -> list[str]:
        return [
            "summary",
            "metrics",
            "compare",
            "drawdown",
            "monthly_returns",
            "monthly_heatmap",
            "positions_summary",
            "latest_holdings",
            "transaction_summary",
            "round_trip_summary",
            "round_trips",
        ]

    def get_report(self, *, open_browser: bool = True) -> Path:
        if not self.html_path.exists():
            raise FileNotFoundError(f"analysis report not found: {self.html_path}")
        if open_browser:
            import webbrowser

            webbrowser.open(self.html_path.as_uri())
        return self.html_path

    def to_summary(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "report_name": self.report_name,
            "html_path": str(self.html_path),
            "summary_table_path": str(self.summary_table_path),
            "metric_table_path": str(self.metric_table_path),
            "compare_table_path": str(self.compare_table_path) if self.compare_table_path else None,
            "drawdown_table_path": str(self.drawdown_table_path) if self.drawdown_table_path else None,
            "monthly_returns_table_path": str(self.monthly_returns_table_path) if self.monthly_returns_table_path else None,
            "monthly_heatmap_table_path": str(self.monthly_heatmap_table_path) if self.monthly_heatmap_table_path else None,
            "positions_summary_path": str(self.positions_summary_path) if self.positions_summary_path else None,
            "latest_holdings_path": str(self.latest_holdings_path) if self.latest_holdings_path else None,
            "transaction_summary_path": str(self.transaction_summary_path) if self.transaction_summary_path else None,
            "round_trip_summary_path": str(self.round_trip_summary_path) if self.round_trip_summary_path else None,
            "round_trips_path": str(self.round_trips_path) if self.round_trips_path else None,
            "figure_paths": [str(path) for path in self.figure_paths],
            "manifest_path": str(self.manifest_path),
            "analysis": self.analysis.to_summary(),
            "spec": asdict(self.spec),
        }


def create_analysis_report(
    returns: pd.Series | pd.DataFrame | None = None,
    *,
    spec: MultifactorAnalysisReportSpec | None = None,
    backtest: pd.DataFrame | None = None,
    positions: pd.DataFrame | None = None,
    transactions: pd.DataFrame | None = None,
    benchmark_returns: pd.Series | pd.DataFrame | None = None,
    market_data: pd.DataFrame | None = None,
    close_panel: pd.DataFrame | None = None,
    factor_data: dict[str, pd.Series | pd.DataFrame | float | int] | None = None,
    previous_positions: pd.DataFrame | None = None,
    live_start_date: str | pd.Timestamp | None = None,
    annual_trading_days: int = 252,
    rolling_window: int = 126,
    output_dir: str | Path = Path("multifactor_evaluation"),
    report_name: str = "analysis",
    open_browser: bool = False,
) -> MultifactorAnalysisReportResult:
    report_spec = spec or MultifactorAnalysisReportSpec()
    output_path = _ensure_output_dir(output_dir)
    figure_dir = _figure_dir(output_path)

    returns_frame = returns
    benchmark_frame = benchmark_returns
    if returns_frame is None and backtest is not None:
        backtest_frame = pd.DataFrame(backtest).copy()
        if "portfolio" in backtest_frame.columns:
            returns_frame = backtest_frame["portfolio"]
        elif "returns" in backtest_frame.columns:
            returns_frame = backtest_frame["returns"]
        elif "return" in backtest_frame.columns:
            returns_frame = backtest_frame["return"]
        if benchmark_frame is None and "benchmark" in backtest_frame.columns:
            benchmark_frame = backtest_frame["benchmark"]

    analysis = analyze_multifactor(
        returns=returns_frame,
        positions=positions,
        transactions=transactions,
        benchmark_returns=benchmark_frame,
        market_data=market_data,
        close_panel=close_panel,
        factor_data=factor_data,
        previous_positions=previous_positions,
        live_start_date=live_start_date,
        annual_trading_days=annual_trading_days,
        rolling_window=rolling_window,
    )

    if analysis.returns is None:
        raise ValueError("create_analysis_report requires returns or backtest data with a portfolio column")

    summary_table = _flatten_summary(analysis.to_summary())
    metric_table = analysis.returns.metric_table
    compare_table = analysis.returns.compare_table if analysis.returns.compare_table is not None and not analysis.returns.compare_table.empty else None
    drawdown_table = analysis.returns.drawdown_details
    monthly_returns_table = analysis.returns.monthly_returns.to_frame("return")
    monthly_heatmap_table = analysis.returns.monthly_returns_heatmap
    positions_summary = _dict_frame(analysis.positions.to_summary()) if analysis.positions is not None else None
    latest_holdings = analysis.positions.latest_holdings if analysis.positions is not None else None
    transaction_summary = _dict_frame(analysis.transactions.summary) if analysis.transactions is not None else None
    round_trip_summary = analysis.transactions.round_trip_summary if analysis.transactions is not None else None
    round_trips = analysis.transactions.round_trips if analysis.transactions is not None else None

    summary_table_path = _save_frame_bundle(summary_table, output_path, f"{report_name}_summary") if report_spec.save_tables and report_spec.save_summary_table else None
    metric_table_path = _save_frame_bundle(metric_table, output_path, f"{report_name}_metrics") if report_spec.save_tables and report_spec.save_metric_table else None
    compare_table_path = _save_frame_bundle(compare_table, output_path, f"{report_name}_compare") if compare_table is not None and report_spec.save_tables and report_spec.save_compare_table else None
    drawdown_table_path = _save_frame_bundle(drawdown_table, output_path, f"{report_name}_drawdown") if report_spec.save_tables and report_spec.save_drawdown_table else None
    monthly_returns_table_path = _save_frame_bundle(monthly_returns_table, output_path, f"{report_name}_monthly_returns") if report_spec.save_tables and report_spec.save_monthly_returns_table else None
    monthly_heatmap_table_path = _save_frame_bundle(monthly_heatmap_table, output_path, f"{report_name}_monthly_heatmap") if report_spec.save_tables and report_spec.save_monthly_heatmap_table else None
    positions_summary_path = _save_frame_bundle(positions_summary, output_path, f"{report_name}_positions_summary") if positions_summary is not None and report_spec.save_tables and report_spec.save_positions_summary else None
    latest_holdings_path = _save_frame_bundle(latest_holdings, output_path, f"{report_name}_latest_holdings") if latest_holdings is not None and report_spec.save_tables and report_spec.save_latest_holdings else None
    transaction_summary_path = _save_frame_bundle(transaction_summary, output_path, f"{report_name}_transaction_summary") if transaction_summary is not None and report_spec.save_tables and report_spec.save_transaction_summary else None
    round_trip_summary_path = _save_frame_bundle(round_trip_summary, output_path, f"{report_name}_round_trip_summary") if round_trip_summary is not None and report_spec.save_tables and report_spec.save_round_trip_summary else None
    round_trips_path = _save_frame_bundle(round_trips, output_path, f"{report_name}_round_trips") if round_trips is not None and report_spec.save_tables and report_spec.save_round_trips else None

    figures: list[Path] = []
    returns_series = analysis.returns.returns
    benchmark_series = analysis.returns.benchmark_returns
    maybe = _save_equity_curve(returns_series, figure_dir / f"{report_name}_equity_curve.png", benchmark=benchmark_series) if report_spec.save_figures and report_spec.save_equity_curve_figure else None
    if maybe is not None:
        figures.append(maybe)
    maybe = _save_benchmark_compare(returns_series, benchmark_series, figure_dir / f"{report_name}_benchmark_compare.png") if report_spec.save_figures and report_spec.save_benchmark_compare_figure else None
    if maybe is not None:
        figures.append(maybe)
    maybe = _save_drawdown_plot(analysis.returns.drawdown, figure_dir / f"{report_name}_drawdown.png") if report_spec.save_figures and report_spec.save_drawdown_figure else None
    if maybe is not None:
        figures.append(maybe)
    maybe = _save_rolling_plot(analysis.returns.rolling_sharpe, figure_dir / f"{report_name}_rolling_sharpe.png", title="Rolling Sharpe", ylabel="Sharpe", color="#1d4ed8") if report_spec.save_figures and report_spec.save_rolling_sharpe_figure else None
    if maybe is not None:
        figures.append(maybe)
    maybe = _save_rolling_plot(analysis.returns.rolling_sortino, figure_dir / f"{report_name}_rolling_sortino.png", title="Rolling Sortino", ylabel="Sortino", color="#7c3aed") if report_spec.save_figures and report_spec.save_rolling_sortino_figure else None
    if maybe is not None:
        figures.append(maybe)
    maybe = _save_rolling_plot(analysis.returns.rolling_volatility, figure_dir / f"{report_name}_rolling_volatility.png", title="Rolling Volatility", ylabel="Volatility", color="#0f766e") if report_spec.save_figures and report_spec.save_rolling_volatility_figure else None
    if maybe is not None:
        figures.append(maybe)
    maybe = _save_rolling_plot(analysis.returns.rolling_beta, figure_dir / f"{report_name}_rolling_beta.png", title="Rolling Beta", ylabel="Beta", color="#457b9d") if report_spec.save_figures and report_spec.save_rolling_beta_figure else None
    if maybe is not None:
        figures.append(maybe)
    maybe = _save_distribution_plot(returns_series, figure_dir / f"{report_name}_returns_distribution.png") if report_spec.save_figures and report_spec.save_returns_distribution_figure else None
    if maybe is not None:
        figures.append(maybe)
    maybe = _save_monthly_heatmap(analysis.returns.monthly_returns_heatmap, figure_dir / f"{report_name}_monthly_heatmap.png") if report_spec.save_figures and report_spec.save_monthly_heatmap_figure else None
    if maybe is not None:
        figures.append(maybe)

    tables: dict[str, pd.DataFrame] = {
        key: table
        for key, table in {
            "summary": summary_table,
            "metrics": metric_table,
            "monthly_returns": monthly_returns_table,
            "monthly_heatmap": monthly_heatmap_table,
        }.items()
        if report_spec.save_tables and report_spec.wants_table(key)
    }
    if compare_table is not None and not compare_table.empty and report_spec.save_tables and report_spec.wants_table("compare"):
        tables["compare"] = compare_table
    if drawdown_table is not None and not drawdown_table.empty and report_spec.save_tables and report_spec.wants_table("drawdown"):
        tables["drawdown"] = drawdown_table
    if positions_summary is not None and not positions_summary.empty and report_spec.save_tables and report_spec.wants_table("positions_summary"):
        tables["positions_summary"] = positions_summary
    if latest_holdings is not None and not latest_holdings.empty and report_spec.save_tables and report_spec.wants_table("latest_holdings"):
        tables["latest_holdings"] = pd.DataFrame(latest_holdings)
    if transaction_summary is not None and not transaction_summary.empty and report_spec.save_tables and report_spec.wants_table("transaction_summary"):
        tables["transaction_summary"] = transaction_summary
    if round_trip_summary is not None and not round_trip_summary.empty and report_spec.save_tables and report_spec.wants_table("round_trip_summary"):
        tables["round_trip_summary"] = round_trip_summary
    if round_trips is not None and not round_trips.empty and report_spec.save_tables and report_spec.wants_table("round_trips"):
        tables["round_trips"] = round_trips

    html_path = output_path / f"{report_name}_analysis_report.html"
    if report_spec.save_html:
        html_path = render_artifact_report(
            title=f"{report_name} Analysis Report",
            output_dir=output_path,
            report_name=f"{report_name}_analysis_report",
            tables=tables,
            figure_paths=figures,
            open_browser=open_browser,
            subtitle="Combined return, positions, transaction, benchmark, and rolling analysis.",
        )

    manifest_path = output_path / f"{report_name}_analysis_manifest.json"
    _save_json(
        manifest_path,
        {
            "output_dir": str(output_path),
            "report_name": report_name,
            "html_path": str(html_path),
            "summary_table_path": str(summary_table_path) if summary_table_path else None,
            "metric_table_path": str(metric_table_path) if metric_table_path else None,
            "compare_table_path": str(compare_table_path) if compare_table_path else None,
            "drawdown_table_path": str(drawdown_table_path) if drawdown_table_path else None,
            "monthly_returns_table_path": str(monthly_returns_table_path) if monthly_returns_table_path else None,
            "monthly_heatmap_table_path": str(monthly_heatmap_table_path) if monthly_heatmap_table_path else None,
            "positions_summary_path": str(positions_summary_path) if positions_summary_path else None,
            "latest_holdings_path": str(latest_holdings_path) if latest_holdings_path else None,
            "transaction_summary_path": str(transaction_summary_path) if transaction_summary_path else None,
            "round_trip_summary_path": str(round_trip_summary_path) if round_trip_summary_path else None,
            "round_trips_path": str(round_trips_path) if round_trips_path else None,
            "figure_paths": [str(path) for path in figures],
            "spec": asdict(report_spec),
        },
    )

    return MultifactorAnalysisReportResult(
        output_dir=output_path,
        report_name=report_name,
        html_path=html_path,
        summary_table_path=summary_table_path,
        metric_table_path=metric_table_path,
        compare_table_path=compare_table_path,
        drawdown_table_path=drawdown_table_path,
        monthly_returns_table_path=monthly_returns_table_path,
        monthly_heatmap_table_path=monthly_heatmap_table_path,
        positions_summary_path=positions_summary_path,
        latest_holdings_path=latest_holdings_path,
        transaction_summary_path=transaction_summary_path,
        round_trip_summary_path=round_trip_summary_path,
        round_trips_path=round_trips_path,
        figure_paths=figures,
        manifest_path=manifest_path,
        analysis=analysis,
        spec=report_spec,
    )


__all__ = [
    "MultifactorAnalysisReportResult",
    "create_analysis_report",
]
