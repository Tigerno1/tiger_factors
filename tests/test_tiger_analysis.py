from __future__ import annotations

import pandas as pd

from tiger_factors.multifactor_evaluation.reporting.portfolio import PortfolioTearSheetResult
from tiger_factors.multifactor_evaluation.reporting.portfolio import PositionReportResult
from tiger_factors.multifactor_evaluation.reporting.trades import PortfolioTradeAnalysisResult
from tiger_factors.multifactor_evaluation.reporting.portfolio import create_position_report
from tiger_factors.multifactor_evaluation.reporting.trades import create_trade_report
from tiger_factors.multifactor_evaluation.reporting.portfolio import run_portfolio_from_backtest
from tiger_factors.multifactor_evaluation.reporting.portfolio import run_portfolio_tear_sheet


def test_tiger_analysis_exports() -> None:
    assert callable(run_portfolio_from_backtest)
    assert callable(run_portfolio_tear_sheet)
    assert callable(create_position_report)
    assert callable(create_trade_report)
    assert PortfolioTearSheetResult.__name__ == "PortfolioTearSheetResult"
    assert PositionReportResult.__name__ == "PositionReportResult"
    assert PortfolioTradeAnalysisResult.__name__ == "PortfolioTradeAnalysisResult"


def test_position_report_outputs(tmp_path) -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    positions = pd.DataFrame(
        {
            "AAA": [0.6, 0.5, 0.3, 0.1],
            "BBB": [0.4, 0.5, 0.7, 0.9],
            "cash": [0.0, 0.0, 0.0, 0.0],
        },
        index=index,
    )
    result = create_position_report(
        positions,
        output_dir=tmp_path,
        report_name="position_demo",
        sector_mappings={"AAA": "tech", "BBB": "financials"},
    )

    assert result is not None
    assert result.positions_path is not None and result.positions_path.exists()
    assert result.positions_summary_path is not None and result.positions_summary_path.exists()
    assert result.latest_holdings_path is not None and result.latest_holdings_path.exists()
    assert result.concentration_path is not None and result.concentration_path.exists()
    assert result.sector_allocations_path is not None and result.sector_allocations_path.exists()
    assert len(result.figure_paths) >= 5
    html_path = result.get_report(open_browser=False)
    html_text = html_path.read_text(encoding="utf-8")
    assert "Holdings Overview" in html_text
    assert "Exposure Detail" in html_text
