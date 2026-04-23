from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tiger_factors.multifactor_evaluation.common.matplotlib_config import configure_matplotlib
from tiger_factors.multifactor_evaluation.common.parquet_utils import to_parquet_clean
from tiger_factors.multifactor_evaluation.common.plotting import finalize_quantstats_axis
from tiger_factors.multifactor_evaluation.common.plotting import save_quantstats_figure
from tiger_factors.multifactor_evaluation.common.results import TearSheetResultMixin
from tiger_factors.multifactor_evaluation.reporting.html_report import render_portfolio_report_html
from tiger_factors.multifactor_evaluation.reporting.html_report import render_position_report_html
from tiger_factors.multifactor_evaluation.reporting.trades import create_trade_report
configure_matplotlib()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tiger_factors.utils.returns_analysis import align_returns as shared_align_returns
from tiger_factors.utils.returns_analysis import annualized_return as shared_annualized_return
from tiger_factors.utils.returns_analysis import annualized_volatility_value as shared_annualized_volatility_value
from tiger_factors.utils.returns_analysis import clean_returns as shared_clean_returns
from tiger_factors.utils.returns_analysis import cumulative_returns as shared_cumulative_returns
from tiger_factors.utils.returns_analysis import drawdown_series as shared_drawdown_series
from tiger_factors.utils.returns_analysis import max_drawdown as shared_max_drawdown
from tiger_factors.utils.returns_analysis import monthly_returns_heatmap as shared_monthly_returns_heatmap
from tiger_factors.utils.returns_analysis import rolling_beta as shared_rolling_beta
from tiger_factors.utils.returns_analysis import rolling_volatility as shared_rolling_volatility
from tiger_factors.utils.returns_analysis import sharpe_ratio as shared_sharpe_ratio
from tiger_factors.utils.returns_analysis import win_rate as shared_win_rate


@dataclass(frozen=True)
class PositionReportResult(TearSheetResultMixin):
    output_dir: Path
    figure_paths: list[Path]
    positions_path: Path | None
    positions_summary_path: Path | None
    latest_holdings_path: Path | None
    concentration_path: Path | None
    sector_allocations_path: Path | None
    report_name: str

    def _table_paths(self) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        if self.positions_path is not None:
            paths["positions"] = self.positions_path
        if self.positions_summary_path is not None:
            paths["summary"] = self.positions_summary_path
            paths["position_summary"] = self.positions_summary_path
        if self.latest_holdings_path is not None:
            paths["latest_holdings"] = self.latest_holdings_path
        if self.concentration_path is not None:
            paths["concentration"] = self.concentration_path
        if self.sector_allocations_path is not None:
            paths["sector_allocations"] = self.sector_allocations_path
        return paths

    def report(self) -> str | None:
        report_path = self.output_dir / f"{self.report_name}_positions_report.html"
        return report_path.stem if report_path.exists() else None

    def preferred_table_order(self) -> list[str]:
        return ["summary", "position_summary", "positions", "latest_holdings", "concentration", "sector_allocations"]

    def get_report(self, *, open_browser: bool = True) -> Path:
        tables = {}
        for name in self.ordered_tables():
            tables[name] = self.get_table(name)
        html_path = render_position_report_html(
            output_dir=self.output_dir,
            report_name=f"{self.report_name}_positions_report",
            tables=tables,
            figure_paths=self.figure_paths,
            open_browser=open_browser,
            subtitle=f"Position report for {self.report_name}",
        )
        return html_path

    def to_summary(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["figure_paths"] = [str(path) for path in self.figure_paths]
        payload["positions_path"] = str(self.positions_path) if self.positions_path is not None else None
        payload["positions_summary_path"] = (
            str(self.positions_summary_path) if self.positions_summary_path is not None else None
        )
        payload["latest_holdings_path"] = str(self.latest_holdings_path) if self.latest_holdings_path is not None else None
        payload["concentration_path"] = str(self.concentration_path) if self.concentration_path is not None else None
        payload["sector_allocations_path"] = (
            str(self.sector_allocations_path) if self.sector_allocations_path is not None else None
        )
        return payload


@dataclass(frozen=True)
class PortfolioTearSheetResult(TearSheetResultMixin):
    output_dir: Path
    figure_paths: list[Path]
    portfolio_returns_path: Path | None
    benchmark_returns_path: Path | None
    positions_path: Path | None
    positions_summary_path: Path | None
    latest_holdings_path: Path | None
    concentration_path: Path | None
    sector_allocations_path: Path | None
    transactions_path: Path | None
    round_trips_path: Path | None
    capacity_summary_path: Path | None
    factor_attribution_path: Path | None
    transaction_summary_path: Path | None
    round_trip_summary_path: Path | None
    report_name: str

    def _table_paths(self) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        summary_path = self.output_dir / f"{self.report_name}_summary.csv"
        if summary_path.exists():
            paths["summary"] = summary_path
        if self.portfolio_returns_path is not None:
            paths["portfolio_returns"] = self.portfolio_returns_path
        if self.benchmark_returns_path is not None:
            paths["benchmark_returns"] = self.benchmark_returns_path
        if self.positions_path is not None:
            paths["positions"] = self.positions_path
        if self.positions_summary_path is not None:
            paths["position_summary"] = self.positions_summary_path
        if self.latest_holdings_path is not None:
            paths["latest_holdings"] = self.latest_holdings_path
        if self.concentration_path is not None:
            paths["concentration"] = self.concentration_path
        if self.sector_allocations_path is not None:
            paths["sector_allocations"] = self.sector_allocations_path
        if self.transactions_path is not None:
            paths["transactions"] = self.transactions_path
        if self.round_trips_path is not None:
            paths["round_trips"] = self.round_trips_path
        if self.capacity_summary_path is not None:
            paths["capacity_summary"] = self.capacity_summary_path
        if self.factor_attribution_path is not None:
            paths["factor_attribution"] = self.factor_attribution_path
        if self.transaction_summary_path is not None:
            paths["transaction_summary"] = self.transaction_summary_path
        if self.round_trip_summary_path is not None:
            paths["round_trip_summary"] = self.round_trip_summary_path
        return paths

    def report(self) -> str | None:
        report_path = self.output_dir / f"{self.report_name}_portfolio_report.html"
        return report_path.stem if report_path.exists() else None

    def preferred_table_order(self) -> list[str]:
        return [
            "summary",
            "portfolio_returns",
            "benchmark_returns",
            "positions",
            "position_summary",
            "latest_holdings",
            "concentration",
            "sector_allocations",
            "transactions",
            "transaction_summary",
            "round_trips",
            "round_trip_summary",
            "capacity_summary",
            "factor_attribution",
        ]

    def get_report(self, *, open_browser: bool = True) -> Path:
        tables = {}
        for name in self.ordered_tables():
            tables[name] = self.get_table(name)
        html_path = render_portfolio_report_html(
            output_dir=self.output_dir,
            report_name=f"{self.report_name}_portfolio_report",
            tables=tables,
            figure_paths=self.figure_paths,
            open_browser=open_browser,
            subtitle=f"Portfolio tear sheet for {self.report_name}",
        )
        return html_path

    def to_summary(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["figure_paths"] = [str(path) for path in self.figure_paths]
        payload["portfolio_returns_path"] = (
            str(self.portfolio_returns_path) if self.portfolio_returns_path is not None else None
        )
        payload["benchmark_returns_path"] = (
            str(self.benchmark_returns_path) if self.benchmark_returns_path is not None else None
        )
        payload["positions_path"] = str(self.positions_path) if self.positions_path is not None else None
        payload["positions_summary_path"] = (
            str(self.positions_summary_path) if self.positions_summary_path is not None else None
        )
        payload["latest_holdings_path"] = str(self.latest_holdings_path) if self.latest_holdings_path is not None else None
        payload["concentration_path"] = str(self.concentration_path) if self.concentration_path is not None else None
        payload["sector_allocations_path"] = (
            str(self.sector_allocations_path) if self.sector_allocations_path is not None else None
        )
        payload["transactions_path"] = str(self.transactions_path) if self.transactions_path is not None else None
        payload["round_trips_path"] = str(self.round_trips_path) if self.round_trips_path is not None else None
        payload["capacity_summary_path"] = (
            str(self.capacity_summary_path) if self.capacity_summary_path is not None else None
        )
        payload["factor_attribution_path"] = (
            str(self.factor_attribution_path) if self.factor_attribution_path is not None else None
        )
        payload["transaction_summary_path"] = (
            str(self.transaction_summary_path) if self.transaction_summary_path is not None else None
        )
        payload["round_trip_summary_path"] = (
            str(self.round_trip_summary_path) if self.round_trip_summary_path is not None else None
        )
        return payload


def _to_clean_series(values: pd.Series) -> pd.Series:
    return shared_clean_returns(values)


def _align_series(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None,
) -> tuple[pd.Series, pd.Series | None]:
    return shared_align_returns(portfolio_returns, benchmark_returns)


def _coerce_positions_frame(positions: pd.DataFrame | None) -> pd.DataFrame | None:
    if positions is None:
        return None
    frame = pd.DataFrame(positions).copy()
    if frame.empty:
        return frame
    if {"date_", "code", "weight"}.issubset(frame.columns):
        frame["date_"] = pd.to_datetime(frame["date_"], errors="coerce")
        frame = frame.dropna(subset=["date_", "code"])
        frame["code"] = frame["code"].astype(str)
        frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")
        frame = frame.dropna(subset=["weight"])
        wide = frame.pivot_table(index="date_", columns="code", values="weight", aggfunc="sum").sort_index()
        wide.index = pd.DatetimeIndex(wide.index, name="date_")
        wide.columns = wide.columns.astype(str)
        frame = wide
    else:
        if "date_" in frame.columns:
            frame["date_"] = pd.to_datetime(frame["date_"], errors="coerce")
            frame = frame.set_index("date_")
        elif "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame = frame.set_index("date")
        if not isinstance(frame.index, pd.DatetimeIndex):
            frame.index = pd.to_datetime(frame.index, errors="coerce")
        frame = frame.loc[~frame.index.isna()].sort_index()
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        frame = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        frame.columns = frame.columns.astype(str)
    if frame.empty:
        return frame
    if "cash" not in frame.columns:
        asset_cols = [column for column in frame.columns if column != "cash"]
        frame["cash"] = 1.0 - frame[asset_cols].sum(axis=1)
    frame.index = pd.DatetimeIndex(frame.index, name="date_")
    return frame.sort_index()


def _position_columns(positions: pd.DataFrame) -> list[str]:
    return [column for column in positions.columns if column != "cash"]


def _position_gross_leverage(positions: pd.DataFrame) -> pd.Series:
    asset_cols = _position_columns(positions)
    if not asset_cols:
        return pd.Series(dtype=float)
    return positions[asset_cols].abs().sum(axis=1)


def _position_net_exposure(positions: pd.DataFrame) -> pd.Series:
    asset_cols = _position_columns(positions)
    if not asset_cols:
        return pd.Series(dtype=float)
    return positions[asset_cols].sum(axis=1)


def _position_long_exposure(positions: pd.DataFrame) -> pd.Series:
    asset_cols = _position_columns(positions)
    if not asset_cols:
        return pd.Series(dtype=float)
    return positions[asset_cols].clip(lower=0.0).sum(axis=1)


def _position_short_exposure(positions: pd.DataFrame) -> pd.Series:
    asset_cols = _position_columns(positions)
    if not asset_cols:
        return pd.Series(dtype=float)
    return -positions[asset_cols].clip(upper=0.0).sum(axis=1)


def _position_concentration(positions: pd.DataFrame) -> pd.DataFrame:
    asset_cols = _position_columns(positions)
    if not asset_cols:
        return pd.DataFrame()
    abs_weights = positions[asset_cols].abs()
    total = abs_weights.sum(axis=1).replace(0.0, np.nan)
    top10_share = abs_weights.apply(lambda row: row.nlargest(min(10, len(row))).sum(), axis=1) / total
    hhi = abs_weights.pow(2).sum(axis=1) / total.pow(2)
    concentration = pd.DataFrame({"top10_share": top10_share, "herfindahl_index": hhi}).replace([np.inf, -np.inf], np.nan)
    concentration.index = pd.DatetimeIndex(concentration.index, name="date_")
    return concentration


def _latest_position_table(positions: pd.DataFrame, *, top_n: int = 10) -> pd.DataFrame:
    asset_cols = _position_columns(positions)
    if not asset_cols or positions.empty:
        return pd.DataFrame(columns=["position", "abs_position"])
    latest = positions.iloc[-1][asset_cols].sort_values(key=lambda series: series.abs(), ascending=False).head(top_n)
    table = pd.DataFrame({"position": latest, "abs_position": latest.abs()})
    table.index.name = "asset"
    return table


def _plot_positions_holdings(positions: pd.DataFrame, *, output_dir: Path, report_name: str, top_n: int = 10) -> Path | None:
    asset_cols = _position_columns(positions)
    if not asset_cols:
        return None
    selected = positions[asset_cols].abs().mean().sort_values(ascending=False).head(top_n).index.tolist()
    if not selected:
        return None
    plt.figure(figsize=(12, 5))
    positions[selected].plot(ax=plt.gca(), linewidth=1.2)
    if "cash" in positions.columns:
        positions["cash"].plot(ax=plt.gca(), color="#444444", linewidth=1.5, label="cash")
    finalize_quantstats_axis(plt.gca(), title="Holdings", ylabel="Weight", legend=True, legend_loc="upper left")
    path = output_dir / f"{report_name}_positions_holdings.png"
    save_quantstats_figure(path)
    return path


def _plot_positions_long_short_holdings(positions: pd.DataFrame, *, output_dir: Path, report_name: str) -> Path | None:
    asset_cols = _position_columns(positions)
    if not asset_cols:
        return None
    long_exposure = _position_long_exposure(positions)
    short_exposure = _position_short_exposure(positions)
    if long_exposure.empty and short_exposure.empty:
        return None
    plt.figure(figsize=(12, 4))
    long_exposure.plot(color="#2a6fdb", linewidth=1.8, label="long exposure")
    short_exposure.plot(color="#c44536", linewidth=1.8, label="short exposure")
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    finalize_quantstats_axis(plt.gca(), title="Long/short holdings", ylabel="Weight", legend=True)
    path = output_dir / f"{report_name}_positions_long_short_holdings.png"
    save_quantstats_figure(path)
    return path


def _plot_positions_exposure(positions: pd.DataFrame, *, output_dir: Path, report_name: str) -> Path | None:
    gross = _position_gross_leverage(positions)
    net = _position_net_exposure(positions)
    if gross.empty and net.empty:
        return None
    plt.figure(figsize=(12, 4))
    gross.plot(color="#1f7a8c", linewidth=1.8, label="gross leverage")
    net.plot(color="#5a189a", linewidth=1.6, label="net exposure")
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    finalize_quantstats_axis(plt.gca(), title="Gross leverage / exposure", ylabel="Weight", legend=True)
    path = output_dir / f"{report_name}_positions_exposure.png"
    save_quantstats_figure(path)
    return path


def _plot_positions_top_holdings(positions: pd.DataFrame, *, output_dir: Path, report_name: str, top_n: int = 10) -> Path | None:
    table = _latest_position_table(positions, top_n=top_n)
    if table.empty:
        return None
    plt.figure(figsize=(10, 4 + 0.35 * len(table)))
    table["position"].sort_values().plot(kind="barh", color="#2a6fdb")
    plt.axvline(0.0, color="black", linewidth=0.8, alpha=0.7)
    finalize_quantstats_axis(plt.gca(), title="Top positions", xlabel="Weight")
    path = output_dir / f"{report_name}_positions_top_holdings.png"
    save_quantstats_figure(path)
    return path


def _plot_positions_concentration(positions: pd.DataFrame, *, output_dir: Path, report_name: str) -> Path | None:
    concentration = _position_concentration(positions)
    if concentration.empty:
        return None
    plt.figure(figsize=(12, 4))
    concentration["top10_share"].plot(color="#c44536", linewidth=1.8, label="top10 share")
    concentration["herfindahl_index"].plot(color="#2a6fdb", linewidth=1.8, label="herfindahl")
    finalize_quantstats_axis(plt.gca(), title="Position concentration", ylabel="Concentration", legend=True)
    path = output_dir / f"{report_name}_positions_concentration.png"
    save_quantstats_figure(path)
    return path


def _position_summary_table(positions: pd.DataFrame) -> pd.DataFrame:
    asset_cols = _position_columns(positions)
    if not asset_cols:
        return pd.DataFrame(columns=["value"]).rename_axis("metric")
    concentration = _position_concentration(positions)
    gross = _position_gross_leverage(positions)
    net = _position_net_exposure(positions)
    long_exposure = _position_long_exposure(positions)
    short_exposure = _position_short_exposure(positions)
    summary = {
        "observations": float(len(positions)),
        "asset_count": float(len(asset_cols)),
        "mean_gross_leverage": float(gross.mean()) if not gross.empty else 0.0,
        "std_gross_leverage": float(gross.std(ddof=0)) if not gross.empty else 0.0,
        "max_gross_leverage": float(gross.max()) if not gross.empty else 0.0,
        "mean_net_exposure": float(net.mean()) if not net.empty else 0.0,
        "std_net_exposure": float(net.std(ddof=0)) if not net.empty else 0.0,
        "mean_long_exposure": float(long_exposure.mean()) if not long_exposure.empty else 0.0,
        "mean_short_exposure": float(short_exposure.mean()) if not short_exposure.empty else 0.0,
        "mean_top10_share": float(concentration["top10_share"].mean()) if not concentration.empty else 0.0,
        "max_top10_share": float(concentration["top10_share"].max()) if not concentration.empty else 0.0,
        "mean_herfindahl_index": float(concentration["herfindahl_index"].mean()) if not concentration.empty else 0.0,
        "max_herfindahl_index": float(concentration["herfindahl_index"].max()) if not concentration.empty else 0.0,
    }
    table = pd.DataFrame.from_dict(summary, orient="index", columns=["value"])
    table.index.name = "metric"
    return table


def _sector_allocation_table(
    positions: pd.DataFrame,
    *,
    sector_mappings: pd.Series | dict[str, str] | None = None,
    top_n: int = 10,
) -> pd.DataFrame | None:
    if sector_mappings is None:
        return None
    sector_lookup = pd.Series(sector_mappings, dtype="object").astype(str)
    sector_lookup.index = sector_lookup.index.astype(str)
    latest_assets = _latest_position_table(positions, top_n=top_n).index.tolist()
    if not latest_assets:
        return None
    latest = positions.iloc[-1]
    sector_weights = (
        pd.Series(
            {
                sector_lookup.get(asset, "unknown"): float(latest.get(asset, 0.0))
                for asset in latest_assets
            }
        )
        .groupby(level=0)
        .sum()
        .sort_values()
    )
    if sector_weights.empty:
        return None
    table = sector_weights.to_frame(name="weight")
    table.index.name = "sector"
    return table


def _plot_sector_allocations(
    positions: pd.DataFrame,
    *,
    output_dir: Path,
    report_name: str,
    sector_mappings: pd.Series | dict[str, str] | None = None,
    top_n: int = 10,
) -> tuple[Path | None, pd.DataFrame | None]:
    table = _sector_allocation_table(positions, sector_mappings=sector_mappings, top_n=top_n)
    if table is None or table.empty:
        return None, table
    plt.figure(figsize=(10, 4))
    table["weight"].plot(kind="barh", color="#2a6fdb")
    plt.axvline(0.0, color="black", linewidth=0.8, alpha=0.7)
    finalize_quantstats_axis(plt.gca(), title="Sector allocations")
    path = output_dir / f"{report_name}_sector_allocations.png"
    save_quantstats_figure(path)
    return path, table


DEFAULT_INTERESTING_TIMES: tuple[tuple[str, str, str], ...] = (
    ("GFC", "2008-09-01", "2009-03-31"),
    ("Flash Crash", "2010-05-01", "2010-06-30"),
    ("Euro Debt", "2011-07-01", "2011-12-31"),
    ("China Selloff", "2015-08-01", "2015-10-31"),
    ("Q4 2018", "2018-10-01", "2018-12-31"),
    ("COVID Crash", "2020-02-15", "2020-05-31"),
    ("Inflation", "2022-01-01", "2022-12-31"),
)


def _plot_interesting_times(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None,
    *,
    output_dir: Path,
    report_name: str,
    interesting_times: tuple[tuple[str, str, str], ...] = DEFAULT_INTERESTING_TIMES,
) -> Path | None:
    portfolio = _to_clean_series(portfolio_returns)
    benchmark = _to_clean_series(benchmark_returns) if benchmark_returns is not None else None
    if portfolio.empty:
        return None
    events: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for label, start, end in interesting_times:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        overlap = portfolio.loc[(portfolio.index >= start_ts) & (portfolio.index <= end_ts)]
        if overlap.empty:
            continue
        events.append((label, start_ts, end_ts))
    if not events:
        return None
    fig, axes = plt.subplots(len(events), 1, figsize=(12, max(3.5, 2.8 * len(events))), sharex=False)
    if len(events) == 1:
        axes = [axes]
    for ax, (label, start_ts, end_ts) in zip(axes, events):
        port_slice = portfolio.loc[(portfolio.index >= start_ts) & (portfolio.index <= end_ts)]
        port_curve = (1.0 + port_slice.fillna(0.0)).cumprod() - 1.0
        port_curve.plot(ax=ax, color="#2a6fdb", linewidth=1.8, label="portfolio")
        if benchmark is not None:
            bench_slice = benchmark.loc[(benchmark.index >= start_ts) & (benchmark.index <= end_ts)]
            if not bench_slice.empty:
                bench_curve = (1.0 + bench_slice.fillna(0.0)).cumprod() - 1.0
                bench_curve.plot(ax=ax, color="#6c757d", linewidth=1.5, label="benchmark")
        finalize_quantstats_axis(ax, title=label, legend=True)
    plt.tight_layout()
    path = output_dir / f"{report_name}_interesting_times.png"
    save_quantstats_figure(path)
    return path


def _cumulative_returns(returns: pd.Series) -> pd.Series:
    return shared_cumulative_returns(returns)


def _drawdown_series(returns: pd.Series) -> pd.Series:
    return shared_drawdown_series(returns)


def _annualized_return(returns: pd.Series, annualization: int = 252) -> float:
    return float(shared_annualized_return(returns, annualization=annualization))


def _annualized_volatility(returns: pd.Series, annualization: int = 252) -> float:
    return float(shared_annualized_volatility_value(returns, annualization=annualization))


def _sharpe_ratio(returns: pd.Series, annualization: int = 252) -> float:
    return shared_sharpe_ratio(returns, annualization=annualization)


def _max_drawdown(returns: pd.Series) -> float:
    return shared_max_drawdown(returns)


def _win_rate(returns: pd.Series) -> float:
    return shared_win_rate(returns)


def _summary_table(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None,
    *,
    annualization: int = 252,
) -> pd.DataFrame:
    portfolio = _to_clean_series(portfolio_returns)
    benchmark = _to_clean_series(benchmark_returns) if benchmark_returns is not None else None
    rows: list[dict[str, Any]] = [
        {
            "series": "portfolio",
            "total_return": float((1.0 + portfolio.fillna(0.0)).prod() - 1.0) if not portfolio.empty else float("nan"),
            "annual_return": _annualized_return(portfolio, annualization=annualization),
            "annual_volatility": _annualized_volatility(portfolio, annualization=annualization),
            "sharpe": _sharpe_ratio(portfolio, annualization=annualization),
            "max_drawdown": _max_drawdown(portfolio),
            "win_rate": _win_rate(portfolio),
            "observations": int(portfolio.shape[0]),
        }
    ]
    if benchmark is not None:
        rows.append(
            {
                "series": "benchmark",
                "total_return": float((1.0 + benchmark.fillna(0.0)).prod() - 1.0) if not benchmark.empty else float("nan"),
                "annual_return": _annualized_return(benchmark, annualization=annualization),
                "annual_volatility": _annualized_volatility(benchmark, annualization=annualization),
                "sharpe": _sharpe_ratio(benchmark, annualization=annualization),
                "max_drawdown": _max_drawdown(benchmark),
                "win_rate": _win_rate(benchmark),
                "observations": int(benchmark.shape[0]),
            }
        )
        active = portfolio.align(benchmark, join="inner")
        active_returns = active[0] - active[1]
        rows.append(
            {
                "series": "active",
                "total_return": float((1.0 + active_returns.fillna(0.0)).prod() - 1.0) if not active_returns.empty else float("nan"),
                "annual_return": _annualized_return(active_returns, annualization=annualization),
                "annual_volatility": _annualized_volatility(active_returns, annualization=annualization),
                "sharpe": _sharpe_ratio(active_returns, annualization=annualization),
                "max_drawdown": _max_drawdown(active_returns),
                "win_rate": _win_rate(active_returns),
                "observations": int(active_returns.shape[0]),
            }
        )
    table = pd.DataFrame.from_records(rows).set_index("series")
    return table


def _annual_returns(returns: pd.Series) -> pd.Series:
    clean = _to_clean_series(returns)
    if clean.empty:
        return pd.Series(dtype=float)
    annual = clean.resample("YE").apply(lambda x: (1.0 + x.fillna(0.0)).prod() - 1.0)
    annual.index = annual.index.year.astype(str)
    annual.name = "return"
    return annual


def _rolling_window(returns: pd.Series, desired: int = 126, minimum: int = 20) -> int:
    clean = _to_clean_series(returns)
    if clean.empty:
        return desired
    window = min(desired, max(int(clean.shape[0] // 4), minimum))
    return max(window, minimum)


def _rolling_volatility(returns: pd.Series, window: int) -> pd.Series:
    return shared_rolling_volatility(returns, window=window, annualization=252)


def _rolling_beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series, window: int) -> pd.Series:
    return shared_rolling_beta(portfolio_returns, benchmark_returns, window=window)


def _drawdown_periods(returns: pd.Series, top_n: int = 5) -> pd.DataFrame:
    clean = _to_clean_series(returns).fillna(0.0)
    if clean.empty:
        return pd.DataFrame(columns=["start", "end", "days", "drawdown"])
    wealth = (1.0 + clean).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    in_drawdown = drawdown < 0.0
    periods: list[dict[str, Any]] = []
    start = None
    trough = None
    trough_value = 0.0
    for date, is_down, dd in zip(drawdown.index, in_drawdown, drawdown):
        if is_down and start is None:
            start = date
            trough = date
            trough_value = float(dd)
        elif is_down and start is not None and dd < trough_value:
            trough = date
            trough_value = float(dd)
        elif not is_down and start is not None:
            periods.append(
                {
                    "start": start,
                    "end": date,
                    "days": int((date - start).days) if hasattr(date - start, "days") else 0,
                    "drawdown": trough_value,
                    "trough": trough,
                }
            )
            start = None
            trough = None
            trough_value = 0.0
    if start is not None:
        end = drawdown.index[-1]
        periods.append(
            {
                "start": start,
                "end": end,
                "days": int((end - start).days) if hasattr(end - start, "days") else 0,
                "drawdown": trough_value,
                "trough": trough,
            }
        )
    if not periods:
        return pd.DataFrame(columns=["start", "end", "days", "drawdown"])
    table = pd.DataFrame.from_records(periods).sort_values("drawdown").head(top_n)
    return table.reset_index(drop=True)


def _monthly_returns_heatmap(returns: pd.Series) -> pd.DataFrame:
    return shared_monthly_returns_heatmap(returns)


def _save_current_figures(output_dir: Path, *, report_name: str, prefix: str = "tear_sheet") -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: list[Path] = []
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        stem = f"{report_name}_{prefix}" if report_name else prefix
        figure_path = output_dir / f"{stem}_{fig_num:02d}.png"
        fig.savefig(figure_path, dpi=150, bbox_inches="tight")
        figure_paths.append(figure_path)
    plt.close("all")
    return figure_paths


def _plot_cumulative_returns(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None,
    *,
    output_dir: Path,
    report_name: str,
) -> None:
    plt.figure(figsize=(12, 5))
    cumulative = _cumulative_returns(portfolio_returns)
    cumulative.plot(color="#2a6fdb", linewidth=2.0, label="portfolio")
    if benchmark_returns is not None:
        benchmark_cumulative = _cumulative_returns(benchmark_returns)
        benchmark_cumulative.plot(color="#6c757d", linewidth=1.8, label="benchmark")
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    finalize_quantstats_axis(plt.gca(), title="Cumulative returns", ylabel="Cumulative return", legend=True)


def _plot_cumulative_returns_log(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None,
) -> None:
    plt.figure(figsize=(12, 5))
    portfolio_wealth = (1.0 + _to_clean_series(portfolio_returns).fillna(0.0)).cumprod()
    portfolio_wealth = portfolio_wealth.replace([np.inf, -np.inf], np.nan).dropna()
    portfolio_wealth[portfolio_wealth <= 0.0] = np.nan
    portfolio_wealth.plot(color="#2a6fdb", linewidth=2.0, label="portfolio")
    if benchmark_returns is not None:
        benchmark_wealth = (1.0 + _to_clean_series(benchmark_returns).fillna(0.0)).cumprod()
        benchmark_wealth = benchmark_wealth.replace([np.inf, -np.inf], np.nan).dropna()
        benchmark_wealth[benchmark_wealth <= 0.0] = np.nan
        benchmark_wealth.plot(color="#6c757d", linewidth=1.8, label="benchmark")
    plt.yscale("log")
    finalize_quantstats_axis(plt.gca(), title="Cumulative returns (log scale)", ylabel="Wealth index", legend=True)
    plt.yscale("log")
    plt.grid(True, alpha=0.25, which="both")
    plt.tight_layout()


def _plot_drawdown(portfolio_returns: pd.Series) -> None:
    plt.figure(figsize=(12, 4))
    drawdown = _drawdown_series(portfolio_returns)
    drawdown.plot(color="#6a4c93", linewidth=1.8)
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    finalize_quantstats_axis(plt.gca(), title="Drawdown", ylabel="Drawdown")


def _plot_top_drawdown_periods(portfolio_returns: pd.Series, top_n: int = 5) -> None:
    periods = _drawdown_periods(portfolio_returns, top_n=top_n)
    plt.figure(figsize=(12, 4 + 0.6 * max(len(periods), 1)))
    if periods.empty:
        plt.text(0.5, 0.5, "No drawdown periods found", ha="center", va="center")
        plt.axis("off")
    else:
        display = periods.copy()
        display["start"] = pd.to_datetime(display["start"]).dt.strftime("%Y-%m-%d")
        display["end"] = pd.to_datetime(display["end"]).dt.strftime("%Y-%m-%d")
        display["trough"] = pd.to_datetime(display["trough"]).dt.strftime("%Y-%m-%d")
        display["drawdown"] = display["drawdown"].map(lambda value: f"{value:.2%}")
        display = display.rename(columns={"days": "duration_days"})
        plt.axis("off")
        table = plt.table(
            cellText=display[["start", "trough", "end", "duration_days", "drawdown"]].values,
            colLabels=["Start", "Trough", "End", "Days", "Drawdown"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.5)
    finalize_quantstats_axis(plt.gca(), title="Top drawdown periods")


def _plot_rolling_beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series | None) -> None:
    if benchmark_returns is None:
        return
    window = _rolling_window(portfolio_returns)
    beta = _rolling_beta(portfolio_returns, benchmark_returns, window)
    if beta.empty:
        return
    plt.figure(figsize=(12, 4))
    beta.plot(color="#5a189a", linewidth=1.8)
    plt.axhline(1.0, color="black", linestyle="--", linewidth=0.9, alpha=0.6)
    finalize_quantstats_axis(plt.gca(), title=f"{window}D rolling beta to benchmark", ylabel="Beta")


def _plot_rolling_volatility_vs_benchmark(portfolio_returns: pd.Series, benchmark_returns: pd.Series | None) -> None:
    window = _rolling_window(portfolio_returns)
    portfolio_vol = _rolling_volatility(portfolio_returns, window)
    if portfolio_vol.empty:
        return
    plt.figure(figsize=(12, 4))
    portfolio_vol.plot(color="#1f7a8c", linewidth=1.8, label="portfolio")
    if benchmark_returns is not None:
        benchmark_vol = _rolling_volatility(benchmark_returns, window)
        if not benchmark_vol.empty:
            benchmark_vol.plot(color="#6c757d", linewidth=1.6, label="benchmark")
    finalize_quantstats_axis(plt.gca(), title=f"{window}D rolling volatility", ylabel="Annualized vol", legend=True)


def _plot_rolling_statistics(portfolio_returns: pd.Series) -> None:
    clean = _to_clean_series(portfolio_returns).fillna(0.0)
    window = min(63, max(int(clean.shape[0] // 3), 20))
    rolling_mean = clean.rolling(window).mean()
    rolling_vol = clean.rolling(window).std(ddof=0) * np.sqrt(252)
    rolling_sharpe = rolling_mean / clean.rolling(window).std(ddof=0) * np.sqrt(252)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    rolling_vol.plot(ax=axes[0], color="#1f7a8c", linewidth=1.8)
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    finalize_quantstats_axis(axes[0], title=f"{window}D rolling volatility")

    rolling_sharpe.replace([np.inf, -np.inf], np.nan).plot(ax=axes[1], color="#c44536", linewidth=1.8)
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    finalize_quantstats_axis(axes[1], title=f"{window}D rolling sharpe")
    plt.tight_layout()


def _plot_return_quantiles(portfolio_returns: pd.Series) -> None:
    clean = _to_clean_series(portfolio_returns)
    if clean.empty:
        return
    daily = clean
    weekly = clean.resample("W-FRI").apply(lambda x: float((1.0 + x.fillna(0.0)).prod() - 1.0))
    monthly = clean.resample("ME").apply(lambda x: float((1.0 + x.fillna(0.0)).prod() - 1.0))
    plt.figure(figsize=(10, 4))
    plt.boxplot(
        [daily.values, weekly.values, monthly.values],
        tick_labels=["Daily", "Weekly", "Monthly"],
        patch_artist=True,
        boxprops={"facecolor": "#dce6f1"},
        medianprops={"color": "#2a6fdb", "linewidth": 1.4},
        whiskerprops={"color": "#444444"},
        capprops={"color": "#444444"},
    )
    finalize_quantstats_axis(plt.gca(), title="Return quantiles")


def _plot_monthly_returns_bar(portfolio_returns: pd.Series) -> None:
    clean = _to_clean_series(portfolio_returns)
    if clean.empty:
        return
    monthly = clean.resample("ME").apply(lambda x: float((1.0 + x.fillna(0.0)).prod() - 1.0))
    if monthly.empty:
        return
    plt.figure(figsize=(14, 4))
    monthly.plot(kind="bar", color="#2a6fdb")
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    finalize_quantstats_axis(plt.gca(), title="Monthly returns", ylabel="Return")


def _plot_monthly_return_distribution(portfolio_returns: pd.Series) -> None:
    clean = _to_clean_series(portfolio_returns)
    if clean.empty:
        return
    monthly = clean.resample("ME").apply(lambda x: float((1.0 + x.fillna(0.0)).prod() - 1.0))
    if monthly.empty:
        return
    plt.figure(figsize=(10, 4))
    sns.histplot(monthly, bins=20, color="#6c757d", kde=True)
    plt.axvline(monthly.mean(), color="black", linestyle="--", linewidth=1.1, label="mean")
    finalize_quantstats_axis(plt.gca(), title="Distribution of monthly returns", legend=True)


def _plot_monthly_heatmap(portfolio_returns: pd.Series) -> None:
    heatmap = _monthly_returns_heatmap(portfolio_returns)
    if heatmap.empty:
        return
    plt.figure(figsize=(12, max(4, 0.35 * len(heatmap))))
    sns.heatmap(heatmap, annot=True, fmt=".1%", cmap="RdYlGn", center=0.0, cbar_kws={"label": "Return"})
    plt.title("Monthly returns")
    plt.tight_layout()


def _plot_return_histogram(portfolio_returns: pd.Series) -> None:
    clean = _to_clean_series(portfolio_returns)
    if clean.empty:
        return
    plt.figure(figsize=(10, 4))
    sns.histplot(clean, bins=30, color="#2a6fdb", kde=True)
    plt.axvline(clean.mean(), color="black", linestyle="--", linewidth=1.1, label="mean")
    finalize_quantstats_axis(plt.gca(), title="Return distribution", legend=True)


def _plot_annual_returns(portfolio_returns: pd.Series, benchmark_returns: pd.Series | None) -> None:
    plt.figure(figsize=(10, 4))
    annual = _annual_returns(portfolio_returns)
    if annual.empty:
        return
    annual.plot(kind="bar", color="#2a6fdb", label="portfolio")
    if benchmark_returns is not None:
        benchmark_annual = _annual_returns(benchmark_returns)
        if not benchmark_annual.empty:
            benchmark_annual.reindex(annual.index).plot(kind="bar", color="#6c757d", alpha=0.5, label="benchmark")
    finalize_quantstats_axis(plt.gca(), title="Annual returns", ylabel="Return", legend=True)


def create_position_report(
    positions: pd.DataFrame,
    *,
    output_dir: str | Path,
    report_name: str = "positions",
    sector_mappings: pd.Series | dict[str, str] | None = None,
) -> PositionReportResult | None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    positions_frame = _coerce_positions_frame(positions)
    if positions_frame is None or positions_frame.empty:
        print("Position data is missing, skipping position report.")
        return None

    print("\nRunning position report...")

    stem = f"{report_name}_" if report_name else ""
    positions_path = output_dir / f"{stem}positions.parquet"
    to_parquet_clean(positions_frame, positions_path)

    summary_table = _position_summary_table(positions_frame)
    positions_summary_path = output_dir / f"{stem}positions_summary.csv"
    summary_table.to_csv(positions_summary_path)

    latest_holdings = _latest_position_table(positions_frame, top_n=10)
    latest_holdings_path = output_dir / f"{stem}latest_holdings.csv"
    latest_holdings.to_csv(latest_holdings_path)

    concentration_table = _position_concentration(positions_frame)
    concentration_path = output_dir / f"{stem}position_concentration.csv"
    concentration_table.to_csv(concentration_path)

    figure_paths: list[Path] = []
    for maybe_path in (
        _plot_positions_holdings(positions_frame, output_dir=output_dir, report_name=report_name),
        _plot_positions_long_short_holdings(positions_frame, output_dir=output_dir, report_name=report_name),
        _plot_positions_exposure(positions_frame, output_dir=output_dir, report_name=report_name),
        _plot_positions_top_holdings(positions_frame, output_dir=output_dir, report_name=report_name),
        _plot_positions_concentration(positions_frame, output_dir=output_dir, report_name=report_name),
    ):
        if maybe_path is not None:
            figure_paths.append(maybe_path)

    sector_allocations_path = None
    sector_path, sector_table = _plot_sector_allocations(
        positions_frame,
        output_dir=output_dir,
        report_name=report_name,
        sector_mappings=sector_mappings,
    )
    if sector_path is not None:
        figure_paths.append(sector_path)
    if sector_table is not None and not sector_table.empty:
        sector_allocations_path = output_dir / f"{stem}sector_allocations.csv"
        sector_table.to_csv(sector_allocations_path)

    return PositionReportResult(
        output_dir=output_dir,
        figure_paths=figure_paths,
        positions_path=positions_path,
        positions_summary_path=positions_summary_path,
        latest_holdings_path=latest_holdings_path,
        concentration_path=concentration_path,
        sector_allocations_path=sector_allocations_path,
        report_name=report_name,
    )


def run_portfolio_tear_sheet(
    portfolio_returns: pd.Series,
    *,
    benchmark_returns: pd.Series | None = None,
    positions: pd.DataFrame | None = None,
    sector_mappings: pd.Series | dict[str, str] | None = None,
    transactions: pd.DataFrame | None = None,
    market_data: pd.DataFrame | None = None,
    close_panel: pd.DataFrame | None = None,
    factor_data: dict[str, pd.Series | pd.DataFrame | float | int] | None = None,
    capital_base: float = 1_000_000.0,
    output_dir: str | Path,
    report_name: str = "portfolio",
) -> PortfolioTearSheetResult | None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    portfolio, benchmark = _align_series(portfolio_returns, benchmark_returns)
    positions_frame = _coerce_positions_frame(positions)
    if portfolio.empty:
        print("Backtest returns are missing, skipping portfolio tear sheet.")
        return None
    if benchmark_returns is not None and (benchmark is None or benchmark.empty):
        print("Backtest returns are missing, skipping portfolio tear sheet.")
        return None

    print("\nRunning portfolio tear sheet...")

    stem = f"{report_name}_" if report_name else ""
    portfolio_returns_path = output_dir / f"{stem}portfolio_returns.csv"
    benchmark_returns_path = output_dir / f"{stem}benchmark_returns.csv" if benchmark is not None else None
    portfolio.rename("portfolio").to_csv(portfolio_returns_path)
    if benchmark is not None and benchmark_returns_path is not None:
        benchmark.rename("benchmark").to_csv(benchmark_returns_path)

    summary_table = _summary_table(portfolio, benchmark)
    summary_path = output_dir / f"{stem}summary.csv"
    summary_table.to_csv(summary_path)

    _plot_cumulative_returns(portfolio, benchmark, output_dir=output_dir, report_name=report_name)
    _plot_cumulative_returns_log(portfolio, benchmark)
    _plot_drawdown(portfolio)
    _plot_top_drawdown_periods(portfolio)
    _plot_rolling_beta(portfolio, benchmark)
    _plot_rolling_volatility_vs_benchmark(portfolio, benchmark)
    _plot_rolling_statistics(portfolio)
    _plot_monthly_returns_bar(portfolio)
    _plot_monthly_return_distribution(portfolio)
    _plot_return_quantiles(portfolio)
    _plot_monthly_heatmap(portfolio)
    _plot_return_histogram(portfolio)
    _plot_annual_returns(portfolio, benchmark)
    extra_figure_paths: list[Path] = []
    interesting_times_path = _plot_interesting_times(portfolio, benchmark, output_dir=output_dir, report_name=report_name)
    if interesting_times_path is not None:
        extra_figure_paths.append(interesting_times_path)

    position_result = None
    if positions_frame is not None and not positions_frame.empty:
        position_result = create_position_report(
            positions_frame,
            output_dir=output_dir / "positions",
            report_name=report_name,
            sector_mappings=sector_mappings,
        )
    positions_path = position_result.positions_path if position_result is not None else None
    positions_summary_path = position_result.positions_summary_path if position_result is not None else None
    latest_holdings_path = position_result.latest_holdings_path if position_result is not None else None
    concentration_path = position_result.concentration_path if position_result is not None else None
    sector_allocations_path = position_result.sector_allocations_path if position_result is not None else None
    if position_result is not None and position_result.figure_paths:
        extra_figure_paths.extend(position_result.figure_paths)

    trade_result = create_trade_report(
        portfolio,
        positions=positions_frame,
        transactions=transactions,
        market_data=market_data,
        close_panel=close_panel,
        factor_data=factor_data,
        output_dir=output_dir,
        report_name=report_name,
        capital_base=capital_base,
    )
    transactions_path = trade_result.transactions_path if trade_result is not None else None
    round_trips_path = trade_result.round_trips_path if trade_result is not None else None
    capacity_summary_path = trade_result.capacity_summary_path if trade_result is not None else None
    factor_attribution_path = trade_result.factor_attribution_path if trade_result is not None else None
    transaction_summary_path = trade_result.transaction_summary_path if trade_result is not None else None
    round_trip_summary_path = trade_result.round_trip_summary_path if trade_result is not None else None
    if trade_result is not None and trade_result.figure_paths:
        extra_figure_paths.extend(trade_result.figure_paths)

    figure_paths = extra_figure_paths + _save_current_figures(output_dir, report_name=report_name)
    return PortfolioTearSheetResult(
        output_dir=output_dir,
        figure_paths=figure_paths,
        portfolio_returns_path=portfolio_returns_path,
        benchmark_returns_path=benchmark_returns_path,
        positions_path=positions_path,
        positions_summary_path=positions_summary_path,
        latest_holdings_path=latest_holdings_path,
        concentration_path=concentration_path,
        sector_allocations_path=sector_allocations_path,
        transactions_path=transactions_path,
        round_trips_path=round_trips_path,
        capacity_summary_path=capacity_summary_path,
        factor_attribution_path=factor_attribution_path,
        transaction_summary_path=transaction_summary_path,
        round_trip_summary_path=round_trip_summary_path,
        report_name=report_name,
    )


def run_portfolio_from_backtest(
    backtest: pd.DataFrame,
    *,
    output_dir: str | Path,
    report_name: str = "portfolio",
    portfolio_column: str = "portfolio",
    benchmark_column: str = "benchmark",
    positions: pd.DataFrame | None = None,
    sector_mappings: pd.Series | dict[str, str] | None = None,
    transactions: pd.DataFrame | None = None,
    market_data: pd.DataFrame | None = None,
    factor_data: dict[str, pd.Series | pd.DataFrame | float | int] | None = None,
    capital_base: float = 1_000_000.0,
) -> PortfolioTearSheetResult | None:
    if portfolio_column not in backtest.columns or benchmark_column not in backtest.columns:
        print("Backtest returns are missing, skipping portfolio tear sheet.")
        return None
    positions_frame = positions
    if positions_frame is None:
        positions_frame = backtest.attrs.get("positions")
    close_panel = backtest.attrs.get("close_panel")
    market_data_frame = market_data if market_data is not None else backtest.attrs.get("market_data")
    transactions_frame = transactions if transactions is not None else backtest.attrs.get("transactions")
    factor_data = factor_data if factor_data is not None else backtest.attrs.get("factor_data")
    return run_portfolio_tear_sheet(
        backtest[portfolio_column],
        benchmark_returns=backtest[benchmark_column],
        positions=positions_frame,
        sector_mappings=sector_mappings,
        transactions=transactions_frame,
        market_data=market_data_frame,
        close_panel=close_panel,
        factor_data=factor_data,
        capital_base=capital_base,
        output_dir=output_dir,
        report_name=report_name,
    )


def create_position_tear_sheet(
    positions: pd.DataFrame,
    *,
    output_dir: str | Path,
    report_name: str = "positions",
    sector_mappings: pd.Series | dict[str, str] | None = None,
) -> PositionReportResult | None:
    return create_position_report(
        positions,
        output_dir=output_dir,
        report_name=report_name,
        sector_mappings=sector_mappings,
    )


def create_portfolio_tear_sheet(
    portfolio_returns: pd.Series,
    *,
    benchmark_returns: pd.Series | None = None,
    positions: pd.DataFrame | None = None,
    sector_mappings: pd.Series | dict[str, str] | None = None,
    transactions: pd.DataFrame | None = None,
    market_data: pd.DataFrame | None = None,
    close_panel: pd.DataFrame | None = None,
    factor_data: dict[str, pd.Series | pd.DataFrame | float | int] | None = None,
    capital_base: float = 1_000_000.0,
    output_dir: str | Path,
    report_name: str = "portfolio",
) -> PortfolioTearSheetResult | None:
    return run_portfolio_tear_sheet(
        portfolio_returns,
        benchmark_returns=benchmark_returns,
        positions=positions,
        sector_mappings=sector_mappings,
        transactions=transactions,
        market_data=market_data,
        close_panel=close_panel,
        factor_data=factor_data,
        capital_base=capital_base,
        output_dir=output_dir,
        report_name=report_name,
    )


__all__ = [
    "PositionReportResult",
    "PortfolioTearSheetResult",
    "create_position_report",
    "create_position_tear_sheet",
    "create_portfolio_tear_sheet",
    "run_portfolio_tear_sheet",
    "run_portfolio_from_backtest",
]
