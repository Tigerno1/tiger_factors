from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from tiger_factors.multifactor_evaluation.reporting import MultifactorSummaryReportResult
from tiger_factors.multifactor_evaluation.reporting.html_report import render_multifactor_index_report
from tiger_factors.multifactor_evaluation.tearsheets import create_portfolio_tear_sheet as _create_portfolio_tear_sheet
from tiger_factors.multifactor_evaluation.tearsheets import create_position_tear_sheet as _create_position_tear_sheet
from tiger_factors.multifactor_evaluation.tearsheets import create_summary_tear_sheet as _create_summary_tear_sheet
from tiger_factors.multifactor_evaluation.tearsheets import create_trade_tear_sheet as _create_trade_tear_sheet
from tiger_factors.multifactor_evaluation.tearsheets.portfolio import PortfolioTearSheetResult
from tiger_factors.multifactor_evaluation.tearsheets.positions import PositionReportResult
from tiger_factors.multifactor_evaluation.tearsheets.trades import PortfolioTradeAnalysisResult


def _resolve_output_dir(base: str | Path | None, default_root: Path, name: str) -> Path:
    if base is None:
        return default_root / name
    return Path(base)


def _resolve_frame(frame: pd.DataFrame | None, fallback: pd.DataFrame | None, *, name: str) -> pd.DataFrame:
    if frame is not None:
        return frame
    if fallback is not None:
        return fallback
    raise ValueError(f"{name} is required")


@dataclass(frozen=True)
class MultifactorEvaluationBundle:
    summary_result: MultifactorSummaryReportResult | None = None
    position_result: PositionReportResult | None = None
    trade_result: PortfolioTradeAnalysisResult | None = None
    portfolio_result: PortfolioTearSheetResult | None = None
    report_path: Path | None = None

    def to_summary(self) -> dict[str, Any]:
        return {
            "summary_result": None if self.summary_result is None else self.summary_result.to_summary(),
            "position_result": None if self.position_result is None else self.position_result.to_summary(),
            "trade_result": None if self.trade_result is None else self.trade_result.to_summary(),
            "portfolio_result": None if self.portfolio_result is None else self.portfolio_result.to_summary(),
            "report_path": None if self.report_path is None else str(self.report_path),
        }


@dataclass(frozen=True)
class MultifactorEvaluation:
    backtest: pd.DataFrame | None = None
    positions_frame: pd.DataFrame | None = None
    transactions_frame: pd.DataFrame | None = None
    market_data_frame: pd.DataFrame | None = None
    close_panel_frame: pd.DataFrame | None = None
    factor_data: dict[str, pd.Series | pd.DataFrame | float | int] | None = None
    sector_mappings: pd.Series | dict[str, str] | None = None
    capital_base: float = 1_000_000.0
    output_dir: str | Path = Path("multifactor_evaluation")
    report_name: str = "multifactor"
    portfolio_column: str = "portfolio"
    benchmark_column: str = "benchmark"

    def _base_dir(self, output_dir: str | Path | None) -> Path:
        return Path(output_dir) if output_dir is not None else Path(self.output_dir)

    def _section_dir(self, output_dir: str | Path | None, name: str) -> Path:
        return _resolve_output_dir(output_dir, Path(self.output_dir), name)

    def _backtest_frame(self, backtest: pd.DataFrame | None) -> pd.DataFrame:
        return _resolve_frame(backtest, self.backtest, name="backtest")

    def _positions_frame(self, positions: pd.DataFrame | None) -> pd.DataFrame | None:
        return positions if positions is not None else self.positions_frame

    def _transactions_frame(self, transactions: pd.DataFrame | None) -> pd.DataFrame | None:
        return transactions if transactions is not None else self.transactions_frame

    def _market_data_frame(self, market_data: pd.DataFrame | None) -> pd.DataFrame | None:
        return market_data if market_data is not None else self.market_data_frame

    def _close_panel_frame(self, close_panel: pd.DataFrame | None) -> pd.DataFrame | None:
        return close_panel if close_panel is not None else self.close_panel_frame

    def summary(
        self,
        backtest: pd.DataFrame | None = None,
        *,
        output_dir: str | Path | None = None,
        report_name: str | None = None,
        portfolio_column: str | None = None,
        benchmark_column: str | None = None,
        annualization: int = 252,
        format: str = "all",
        open_browser: bool = False,
    ) -> MultifactorSummaryReportResult | None:
        return _create_summary_tear_sheet(
            self._backtest_frame(backtest),
            output_dir=self._section_dir(output_dir, "summary"),
            report_name=report_name or f"{self.report_name}_summary_report",
            portfolio_column=portfolio_column or self.portfolio_column,
            benchmark_column=benchmark_column or self.benchmark_column,
            annualization=annualization,
            format=format,
            open_browser=open_browser,
        )

    def positions(
        self,
        positions: pd.DataFrame | None = None,
        *,
        output_dir: str | Path | None = None,
        report_name: str | None = None,
        sector_mappings: pd.Series | dict[str, str] | None = None,
    ) -> PositionReportResult | None:
        return _create_position_tear_sheet(
            _resolve_frame(positions, self.positions_frame, name="positions"),
            output_dir=self._section_dir(output_dir, "positions"),
            report_name=report_name or f"{self.report_name}_positions",
            sector_mappings=sector_mappings if sector_mappings is not None else self.sector_mappings,
        )

    def trades(
        self,
        returns: pd.Series | None = None,
        *,
        output_dir: str | Path | None = None,
        report_name: str | None = None,
        positions: pd.DataFrame | None = None,
        transactions: pd.DataFrame | None = None,
        market_data: pd.DataFrame | None = None,
        close_panel: pd.DataFrame | None = None,
        factor_data: dict[str, pd.Series | pd.DataFrame | float | int] | None = None,
        capital_base: float | None = None,
    ) -> PortfolioTradeAnalysisResult | None:
        if returns is None:
            backtest = self._backtest_frame(None)
            if self.portfolio_column not in backtest.columns:
                raise ValueError("returns or backtest with portfolio column is required")
            returns = backtest[self.portfolio_column]
        return _create_trade_tear_sheet(
            returns,
            positions=self._positions_frame(positions),
            transactions=self._transactions_frame(transactions),
            market_data=self._market_data_frame(market_data),
            close_panel=self._close_panel_frame(close_panel),
            factor_data=factor_data if factor_data is not None else self.factor_data,
            output_dir=self._section_dir(output_dir, "trades"),
            report_name=report_name or f"{self.report_name}_trades",
            capital_base=self.capital_base if capital_base is None else capital_base,
        )

    def portfolio(
        self,
        backtest: pd.DataFrame | None = None,
        *,
        output_dir: str | Path | None = None,
        report_name: str | None = None,
        portfolio_column: str | None = None,
        benchmark_column: str | None = None,
        positions: pd.DataFrame | None = None,
        sector_mappings: pd.Series | dict[str, str] | None = None,
        transactions: pd.DataFrame | None = None,
        market_data: pd.DataFrame | None = None,
        close_panel: pd.DataFrame | None = None,
        factor_data: dict[str, pd.Series | pd.DataFrame | float | int] | None = None,
        capital_base: float | None = None,
    ) -> PortfolioTearSheetResult | None:
        backtest_frame = self._backtest_frame(backtest)
        portfolio_col = portfolio_column or self.portfolio_column
        benchmark_col = benchmark_column or self.benchmark_column
        if portfolio_col not in backtest_frame.columns:
            raise ValueError(f"backtest must contain portfolio column '{portfolio_col}'")
        portfolio_returns = backtest_frame[portfolio_col]
        benchmark_returns = backtest_frame[benchmark_col] if benchmark_col in backtest_frame.columns else None
        return _create_portfolio_tear_sheet(
            portfolio_returns,
            benchmark_returns=benchmark_returns,
            positions=self._positions_frame(positions),
            sector_mappings=sector_mappings if sector_mappings is not None else self.sector_mappings,
            transactions=self._transactions_frame(transactions),
            market_data=self._market_data_frame(market_data),
            close_panel=self._close_panel_frame(close_panel),
            factor_data=factor_data if factor_data is not None else self.factor_data,
            capital_base=self.capital_base if capital_base is None else capital_base,
            output_dir=self._section_dir(output_dir, "portfolio"),
            report_name=report_name or f"{self.report_name}_portfolio",
        )

    def full(
        self,
        backtest: pd.DataFrame | None = None,
        *,
        output_dir: str | Path | None = None,
        report_name: str | None = None,
        open_browser: bool = False,
    ) -> MultifactorEvaluationBundle:
        base_dir = self._base_dir(output_dir)
        name = report_name or self.report_name
        backtest_frame = self._backtest_frame(backtest)
        summary_result = self.summary(
            backtest_frame,
            output_dir=base_dir / "summary",
            report_name=f"{name}_summary_report",
            format="all",
            open_browser=open_browser,
        )
        position_result = None
        if self.positions_frame is not None:
            position_result = self.positions(
                self.positions_frame,
                output_dir=base_dir / "positions",
                report_name=f"{name}_positions",
                sector_mappings=self.sector_mappings,
            )
        trade_result = None
        if self.transactions_frame is not None or backtest_frame is not None:
            trade_result = self.trades(
                returns=backtest_frame[self.portfolio_column] if self.portfolio_column in backtest_frame.columns else None,
                output_dir=base_dir / "trades",
                report_name=f"{name}_trades",
                positions=self.positions_frame,
                transactions=self.transactions_frame,
                market_data=self.market_data_frame,
                close_panel=self.close_panel_frame,
                factor_data=self.factor_data,
            )
        portfolio_result = self.portfolio(
            backtest_frame,
            output_dir=base_dir / "portfolio",
            report_name=f"{name}_portfolio",
            positions=self.positions_frame,
            sector_mappings=self.sector_mappings,
            transactions=self.transactions_frame,
            market_data=self.market_data_frame,
            close_panel=self.close_panel_frame,
            factor_data=self.factor_data,
        )
        module_entries: list[dict[str, Any]] = []
        for module_name, result in (
            ("summary", summary_result),
            ("positions", position_result),
            ("trades", trade_result),
            ("portfolio", portfolio_result),
        ):
            if result is None:
                continue
            report_path = None
            if hasattr(result, "get_report"):
                report_path = result.get_report(open_browser=False)
            tables = {}
            if hasattr(result, "tables") and hasattr(result, "get_table"):
                ordered_table_names = result.ordered_tables() if hasattr(result, "ordered_tables") else result.tables()
                for table_name in ordered_table_names:
                    try:
                        tables[table_name] = result.get_table(table_name)
                    except Exception:
                        continue
            module_entries.append(
                {
                    "name": module_name,
                    "title": module_name.replace("_", " ").title(),
                    "report_path": report_path,
                    "tables": tables,
                    "figure_paths": getattr(result, "figure_paths", []),
                    "summary": getattr(result, "report_name", ""),
                }
            )
        report_path = render_multifactor_index_report(
            title=f"{name} Multifactor Report",
            output_dir=base_dir,
            report_name="report",
            modules=module_entries,
            open_browser=open_browser,
            subtitle="Combined multifactor summary, positions, trades, and portfolio tears.",
        ) if module_entries else None
        return MultifactorEvaluationBundle(
            summary_result=summary_result,
            position_result=position_result,
            trade_result=trade_result,
            portfolio_result=portfolio_result,
            report_path=report_path,
        )


def create_multifactor_evaluation(**kwargs: Any) -> MultifactorEvaluation:
    return MultifactorEvaluation(**kwargs)


def create_summary_tear_sheet(
    backtest: pd.DataFrame,
    *,
    output_dir: str | Path,
    report_name: str = "summary_report",
    portfolio_column: str = "portfolio",
    benchmark_column: str = "benchmark",
    annualization: int = 252,
    format: str = "all",
    open_browser: bool = False,
) -> MultifactorSummaryReportResult | None:
    return _create_summary_tear_sheet(
        backtest,
        output_dir=output_dir,
        report_name=report_name,
        portfolio_column=portfolio_column,
        benchmark_column=benchmark_column,
        annualization=annualization,
        format=format,
        open_browser=open_browser,
    )


def create_position_tear_sheet(*args: Any, **kwargs: Any) -> PositionReportResult | None:
    return _create_position_tear_sheet(*args, **kwargs)


def create_trade_tear_sheet(*args: Any, **kwargs: Any) -> PortfolioTradeAnalysisResult | None:
    return _create_trade_tear_sheet(*args, **kwargs)


def create_portfolio_tear_sheet(*args: Any, **kwargs: Any) -> PortfolioTearSheetResult | None:
    return _create_portfolio_tear_sheet(*args, **kwargs)


def create_full_tear_sheet(
    backtest: pd.DataFrame | None = None,
    *,
    positions: pd.DataFrame | None = None,
    transactions: pd.DataFrame | None = None,
    market_data: pd.DataFrame | None = None,
    close_panel: pd.DataFrame | None = None,
    factor_data: dict[str, pd.Series | pd.DataFrame | float | int] | None = None,
    sector_mappings: pd.Series | dict[str, str] | None = None,
    capital_base: float = 1_000_000.0,
    output_dir: str | Path = Path("multifactor_evaluation"),
    report_name: str = "multifactor",
    portfolio_column: str = "portfolio",
    benchmark_column: str = "benchmark",
    open_browser: bool = False,
) -> MultifactorEvaluationBundle:
    evaluation = MultifactorEvaluation(
        backtest=backtest,
        positions_frame=positions,
        transactions_frame=transactions,
        market_data_frame=market_data,
        close_panel_frame=close_panel,
        factor_data=factor_data,
        sector_mappings=sector_mappings,
        capital_base=capital_base,
        output_dir=output_dir,
        report_name=report_name,
        portfolio_column=portfolio_column,
        benchmark_column=benchmark_column,
    )
    return evaluation.full(
        backtest,
        output_dir=output_dir,
        report_name=report_name,
        open_browser=open_browser,
    )


__all__ = [
    "MultifactorEvaluation",
    "MultifactorEvaluationBundle",
    "create_multifactor_evaluation",
    "create_summary_tear_sheet",
    "create_position_tear_sheet",
    "create_trade_tear_sheet",
    "create_portfolio_tear_sheet",
    "create_full_tear_sheet",
]
