from __future__ import annotations

import numpy as np
import pandas as pd

from tiger_factors.factor_portfolio import factor_to_stock_portfolio
from tiger_factors.factor_portfolio import multi_factor_to_stock_portfolio
from tiger_factors.factor_portfolio import run_factor_portfolio_workflow
from tiger_factors.factor_portfolio import run_weight_panel_backtest
from tiger_factors.factor_portfolio import summarize_factor_portfolio_holdings
from tiger_factors.factor_portfolio import weights_to_positions_frame


def test_factor_to_stock_portfolio_long_only_normalizes_each_row() -> None:
    panel = pd.DataFrame(
        {
            "AAA": [1.0, -1.0],
            "BBB": [2.0, 1.0],
            "CCC": [-1.0, 0.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    result = factor_to_stock_portfolio(panel, long_only=True, standardize=False)

    expected = pd.DataFrame(
        {
            "AAA": [1.0 / 3.0, 0.0],
            "BBB": [2.0 / 3.0, 1.0],
            "CCC": [0.0, 0.0],
        },
        index=panel.index,
    )
    expected.index.name = "date_"
    pd.testing.assert_frame_equal(result.weights, expected)
    assert result.weights.sum(axis=1).tolist() == [1.0, 1.0]


def test_factor_to_stock_portfolio_long_short_is_dollar_neutral() -> None:
    panel = pd.DataFrame(
        {
            "AAA": [3.0, 1.0],
            "BBB": [2.0, 4.0],
            "CCC": [1.0, 2.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    result = factor_to_stock_portfolio(panel, long_only=False, standardize=False)

    np.testing.assert_allclose(result.weights.sum(axis=1).to_numpy(), np.zeros(len(panel)), atol=1e-12)
    np.testing.assert_allclose(result.weights.abs().sum(axis=1).to_numpy(), np.ones(len(panel)), atol=1e-12)


def test_multi_factor_to_stock_portfolio_uses_factor_weights() -> None:
    factor_panels = {
        "value": pd.DataFrame(
            {
                "AAA": [1.0, 0.0],
                "BBB": [0.0, 1.0],
                "CCC": [0.0, 0.0],
            },
            index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        ),
        "momentum": pd.DataFrame(
            {
                "AAA": [0.0, 1.0],
                "BBB": [1.0, 0.0],
                "CCC": [0.0, 0.0],
            },
            index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        ),
    }

    result = multi_factor_to_stock_portfolio(
        factor_panels,
        factor_weights={"value": 3.0, "momentum": 1.0},
        long_only=True,
        standardize=False,
    )

    expected = pd.DataFrame(
        {
            "AAA": [0.75, 0.25],
            "BBB": [0.25, 0.75],
            "CCC": [0.0, 0.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    expected.index.name = "date_"
    pd.testing.assert_frame_equal(result.signal, expected)
    pd.testing.assert_frame_equal(result.weights, expected)
    assert result.factor_weights == {"value": 0.75, "momentum": 0.25}


def test_weights_to_positions_frame_flattens_wide_panel() -> None:
    weights = pd.DataFrame(
        {
            "AAA": [0.6, 0.4],
            "BBB": [0.4, 0.6],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    positions = weights_to_positions_frame(weights)

    expected = pd.DataFrame(
        {
            "date_": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
            "code": ["AAA", "BBB", "AAA", "BBB"],
            "weight": [0.6, 0.4, 0.4, 0.6],
        }
    )
    pd.testing.assert_frame_equal(positions, expected)


def test_run_weight_panel_backtest_attaches_positions() -> None:
    weights = pd.DataFrame(
        {
            "AAA": [0.6, 0.5, 0.4],
            "BBB": [0.4, 0.5, 0.6],
        },
        index=pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"]),
    )
    close = pd.DataFrame(
        {
            "AAA": [100.0, 110.0, 121.0, 133.1],
            "BBB": [100.0, 101.0, 102.0, 103.0],
        },
        index=pd.to_datetime(["2024-01-31", "2024-02-15", "2024-03-15", "2024-04-15"]),
    )

    backtest, stats = run_weight_panel_backtest(weights, close)

    assert not backtest.empty
    assert "portfolio" in backtest.columns
    assert "benchmark" in backtest.columns
    assert "positions" in backtest.attrs
    assert not backtest.attrs["positions"].empty
    assert stats["portfolio"]["total_return"] != 0.0


def test_run_factor_portfolio_workflow_handles_single_factor() -> None:
    factor = pd.DataFrame(
        {
            "AAA": [1.0, 0.5],
            "BBB": [0.0, 1.0],
            "CCC": [0.2, 0.1],
        },
        index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
    )
    close = pd.DataFrame(
        {
            "AAA": [100.0, 101.0, 102.0],
            "BBB": [100.0, 99.0, 98.0],
            "CCC": [100.0, 100.5, 101.0],
        },
        index=pd.to_datetime(["2024-01-31", "2024-02-15", "2024-03-15"]),
    )

    result = run_factor_portfolio_workflow(
        factor,
        close,
        long_only=True,
        standardize=False,
    )

    assert result.report is None
    assert not result.backtest.empty
    assert not result.positions.empty
    assert result.backtest.attrs["positions"].equals(result.positions)


def test_summarize_factor_portfolio_holdings_returns_latest_holdings() -> None:
    factor_panels = {
        "momentum": pd.DataFrame(
            {
                "AAA": [1.0, 2.0],
                "BBB": [0.0, 1.0],
                "CCC": [0.5, 0.2],
            },
            index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
        ),
        "value": pd.DataFrame(
            {
                "AAA": [0.2, 0.1],
                "BBB": [1.0, 0.5],
                "CCC": [0.3, 0.7],
            },
            index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
        ),
    }

    summary = summarize_factor_portfolio_holdings(
        factor_panels,
        factor_weights={"momentum": 0.7, "value": 0.3},
        long_only=True,
        standardize=False,
        top_n=2,
    )

    assert summary["latest_date"] == pd.Timestamp("2024-02-29")
    assert list(summary["latest_holdings"]["code"]) == ["AAA", "BBB"]
    assert abs(float(summary["stock_weights"].iloc[-1].sum()) - 1.0) < 1e-12
    assert summary["factor_weights"] == {"momentum": 0.7, "value": 0.3}
