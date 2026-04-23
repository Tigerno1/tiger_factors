from __future__ import annotations

import pandas as pd

from tiger_factors.multifactor_evaluation.reporting.trades import create_trade_report
from tiger_factors.multifactor_evaluation.reporting.trades import synthesize_transactions_from_positions
from tiger_factors.multifactor_evaluation.reporting.portfolio import run_portfolio_from_backtest


def test_tiger_analysis_trade_sections(tmp_path) -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    portfolio_returns = pd.Series([0.01, -0.005, 0.02, 0.0], index=index)
    benchmark_returns = pd.Series([0.005, -0.0025, 0.01, 0.0], index=index)
    backtest = pd.DataFrame({"portfolio": portfolio_returns, "benchmark": benchmark_returns}, index=index)
    backtest.attrs["positions"] = pd.DataFrame(
        {
            "AAA": [0.6, 0.6, 0.2, 0.0],
            "BBB": [0.4, 0.4, 0.8, 1.0],
            "cash": [0.0, 0.0, 0.0, 0.0],
        },
        index=index,
    )
    backtest.attrs["close_panel"] = pd.DataFrame(
        {
            "AAA": [10.0, 11.0, 12.0, 13.0],
            "BBB": [20.0, 19.0, 18.0, 17.0],
        },
        index=index,
    )
    market_data = pd.DataFrame(
        {
            "date_": list(index) + list(index),
            "symbol": ["AAA"] * len(index) + ["BBB"] * len(index),
            "price": [10.0, 11.0, 12.0, 13.0, 20.0, 19.0, 18.0, 17.0],
            "volume": [1000, 1100, 1200, 1300, 2000, 1900, 1800, 1700],
        }
    )

    result = run_portfolio_from_backtest(
        backtest,
        output_dir=tmp_path,
        report_name="trade_demo",
        market_data=market_data,
    )

    assert result is not None
    assert result.transactions_path is not None and result.transactions_path.exists()
    assert result.round_trips_path is not None and result.round_trips_path.exists()
    assert result.capacity_summary_path is not None and result.capacity_summary_path.exists()
    assert result.transaction_summary_path is not None and result.transaction_summary_path.exists()
    assert result.round_trip_summary_path is not None and result.round_trip_summary_path.exists()
    assert len(result.figure_paths) >= 20
    html_path = result.get_report(open_browser=False)
    html_text = html_path.read_text(encoding="utf-8")
    assert "Performance Overview" in html_text
    assert "Trading Detail" in html_text


def test_trade_report_entrypoint(tmp_path) -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    returns = pd.Series([0.01, -0.005, 0.02, 0.0], index=index)
    positions = pd.DataFrame(
        {
            "AAA": [0.6, 0.6, 0.2, 0.0],
            "BBB": [0.4, 0.4, 0.8, 1.0],
            "cash": [0.0, 0.0, 0.0, 0.0],
        },
        index=index,
    )
    close_panel = pd.DataFrame(
        {
            "AAA": [10.0, 11.0, 12.0, 13.0],
            "BBB": [20.0, 19.0, 18.0, 17.0],
        },
        index=index,
    )

    result = create_trade_report(
        returns,
        positions=positions,
        close_panel=close_panel,
        output_dir=tmp_path,
        report_name="trade_demo",
    )

    assert result is not None
    assert result.transactions_path is not None and result.transactions_path.exists()
    assert result.round_trips_path is not None and result.round_trips_path.exists()
    assert result.transaction_summary_path is not None and result.transaction_summary_path.exists()
    assert result.round_trip_summary_path is not None and result.round_trip_summary_path.exists()
    html_path = result.get_report(open_browser=False)
    html_text = html_path.read_text(encoding="utf-8")
    assert "Trade Summary" in html_text
    assert "Trade Ledger" in html_text


def test_synthesize_transactions_handles_partial_close_columns() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    positions = pd.DataFrame(
        {
            "AAA": [0.5, 0.4, 0.0],
            "BBB": [0.5, 0.6, 1.0],
            "cash": [0.0, 0.0, 0.0],
        },
        index=index,
    )
    close_panel = pd.DataFrame(
        {
            "AAA": [10.0, 11.0, 12.0],
        },
        index=index,
    )

    transactions = synthesize_transactions_from_positions(positions, close_panel)

    assert not transactions.empty
    assert set(transactions["symbol"].astype(str)) == {"AAA"}
