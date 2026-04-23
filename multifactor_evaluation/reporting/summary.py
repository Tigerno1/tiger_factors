from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import MultifactorSummaryReportResult
from tiger_factors.multifactor_evaluation.reporting.summary_report_quantstats import (
    create_multifactor_summary_report as _create_multifactor_summary_report,
)


def create_summary_tear_sheet(
    backtest,
    *,
    output_dir,
    report_name: str = "summary_report",
    portfolio_column: str = "portfolio",
    benchmark_column: str = "benchmark",
    annualization: int = 252,
    format: str = "all",
    open_browser: bool = False,
):
    return _create_multifactor_summary_report(
        backtest,
        output_dir=output_dir,
        report_name=report_name,
        portfolio_column=portfolio_column,
        benchmark_column=benchmark_column,
        annualization=annualization,
        format=format,
        open_browser=open_browser,
    )

__all__ = [
    "MultifactorSummaryReportResult",
    "create_summary_tear_sheet",
]
