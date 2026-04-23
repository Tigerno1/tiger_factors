from __future__ import annotations

import pandas as pd

from tiger_factors.multifactor_evaluation import StrategyComparisonConfig
from tiger_factors.multifactor_evaluation import calculate_combined_factor_score
from tiger_factors.multifactor_evaluation import analyze_portfolio_comprehensive
from tiger_factors.multifactor_evaluation import calculate_concentration
from tiger_factors.multifactor_evaluation import calculate_factor_exposure
from tiger_factors.multifactor_evaluation import calculate_industry_exposure
from tiger_factors.multifactor_evaluation import calculate_market_cap_weights
from tiger_factors.multifactor_evaluation import calculate_risk_metrics
from tiger_factors.multifactor_evaluation import calculate_turnover
from tiger_factors.multifactor_evaluation import compare_and_rank
from tiger_factors.multifactor_evaluation import compare_strategies
from tiger_factors.multifactor_evaluation import generate_comparison_report
from tiger_factors.multifactor_evaluation import neutralize_both
from tiger_factors.multifactor_evaluation import neutralize_industry
from tiger_factors.multifactor_evaluation import neutralize_market_cap
from tiger_factors.multifactor_evaluation import optimize_factor_weights
from tiger_factors.multifactor_evaluation import score_factor
from tiger_factors.multifactor_evaluation import score_portfolio
from tiger_factors.multifactor_evaluation import score_strategy


def _sample_positions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stock_code": ["A", "B", "C", "D"],
            "weight": [0.40, 0.30, -0.20, 0.10],
            "industry": ["Tech", "Tech", "Finance", "Energy"],
        }
    )


def _sample_returns() -> pd.Series:
    index = pd.bdate_range("2024-01-01", periods=6)
    return pd.Series([0.01, -0.005, 0.012, 0.004, -0.002, 0.008], index=index, name="portfolio")


def test_industry_exposure_and_concentration() -> None:
    positions = _sample_positions()

    exposure = calculate_industry_exposure(positions)
    concentration = calculate_concentration(positions)
    market_cap_weights = calculate_market_cap_weights(
        positions.assign(market_cap=[100.0, 80.0, 60.0, 40.0])
    )

    assert "industry_exposure" in exposure
    assert exposure["gross_weight"] > 0
    assert "top10_concentration" in concentration
    assert concentration["top10_concentration"] > 0
    assert "market_cap_weights" in market_cap_weights
    assert market_cap_weights["total_market_cap"] > 0


def test_factor_exposure_and_turnover() -> None:
    positions = _sample_positions()
    factor_data = {
        "value_factor": pd.Series({"A": 1.0, "B": 0.5, "C": -1.0, "D": 0.25}),
    }
    previous_positions = pd.DataFrame(
        {
            "stock_code": ["A", "B", "E"],
            "weight": [0.20, 0.10, 0.05],
        }
    )

    factor_exposure = calculate_factor_exposure(positions, factor_data)
    turnover = calculate_turnover(positions, previous_positions)

    assert "factor_exposures" in factor_exposure
    assert "value_factor" in factor_exposure["factor_exposures"]
    assert turnover["turnover"] > 0


def test_neutralization_helpers() -> None:
    frame = pd.DataFrame(
        {
            "stock_code": ["A", "B", "C", "D"],
            "factor_x": [10.0, 12.0, 8.0, 9.0],
            "market_cap": [100.0, 120.0, 80.0, 90.0],
            "industry": ["Tech", "Tech", "Finance", "Energy"],
        }
    )

    mc = neutralize_market_cap(frame, "factor_x")
    industry = neutralize_industry(frame, "factor_x")
    both = neutralize_both(frame, "factor_x")

    assert len(mc) == len(frame)
    assert len(industry) == len(frame)
    assert len(both) == len(frame)
    assert mc.notna().sum() > 0
    assert industry.notna().sum() > 0
    assert both.notna().sum() > 0


def test_risk_metrics_with_benchmark() -> None:
    returns = _sample_returns()
    benchmark = pd.Series([0.008, -0.003, 0.010, 0.003, -0.001, 0.006], index=returns.index, name="benchmark")

    metrics = calculate_risk_metrics(returns, benchmark_returns=benchmark)

    assert metrics["total_return"] != 0.0
    assert "information_ratio" in metrics
    assert "tracking_error" in metrics


def test_compare_strategies_and_report() -> None:
    index = pd.bdate_range("2024-01-01", periods=5)
    strategies = {
        "alpha": pd.Series([0.01, 0.005, -0.002, 0.008, 0.004], index=index),
        "beta": pd.Series([0.008, 0.004, -0.001, 0.006, 0.003], index=index),
    }
    benchmark = pd.Series([0.006, 0.002, -0.001, 0.004, 0.002], index=index)

    comparison = compare_strategies(strategies, benchmark_returns=benchmark)
    report = generate_comparison_report(comparison)

    assert isinstance(comparison["metrics_table"], pd.DataFrame)
    assert list(comparison["metrics_table"].index) == ["alpha", "beta"]
    assert "correlation_matrix" in comparison
    assert "策略对比报告" in report


def test_comprehensive_portfolio_analysis() -> None:
    positions = _sample_positions()
    returns = _sample_returns()
    previous_positions = pd.DataFrame(
        {
            "stock_code": ["A", "B", "C"],
            "weight": [0.25, 0.15, 0.05],
        }
    )
    factor_data = {
        "value_factor": pd.Series({"A": 1.0, "B": 0.5, "C": -1.0, "D": 0.25}),
    }

    result = analyze_portfolio_comprehensive(
        positions,
        returns,
        factor_data=factor_data,
        benchmark_returns=returns * 0.8,
        previous_positions=previous_positions,
    )

    assert "industry_exposure" in result
    assert "factor_exposure" in result
    assert "concentration" in result
    assert "risk_metrics" in result
    assert "turnover" in result


def test_comprehensive_scoring_helpers() -> None:
    factor_score = score_factor(
        {
            "ic_mean": 0.03,
            "ic_ir": 0.7,
            "stability_score": 0.8,
            "turnover": 0.2,
        }
    )
    strategy_score = score_strategy(
        {
            "annual_return": 0.18,
            "max_drawdown": -0.08,
            "sharpe_ratio": 1.5,
            "win_rate": 0.6,
            "turnover": 0.3,
        }
    )
    portfolio_score = score_portfolio(
        {
            "annual_return": 0.12,
            "volatility": 0.10,
            "max_drawdown": -0.05,
            "sharpe_ratio": 1.4,
            "herfindahl_index": 0.15,
        },
        benchmark_metrics={"annual_return": 0.08},
    )
    ranking = compare_and_rank(
        [
            {"name": "alpha", "metrics": {"annual_return": 0.2, "max_drawdown": -0.05, "sharpe_ratio": 1.8, "win_rate": 0.65, "turnover": 0.25}},
            {"name": "beta", "metrics": {"annual_return": 0.1, "max_drawdown": -0.08, "sharpe_ratio": 1.1, "win_rate": 0.55, "turnover": 0.35}},
        ],
        scoring_type="strategy",
    )

    assert factor_score["total_score"] > 0
    assert strategy_score["grade"] in {"A+", "A", "B+", "B", "C+", "C", "D"}
    assert portfolio_score["total_score"] > 0
    assert ranking[0]["name"] == "alpha"


def test_combined_factor_score_and_heuristic_weights() -> None:
    dates = pd.bdate_range("2024-01-01", periods=4)
    factor_data = {
        "value_a": pd.Series([1.0, 2.0, 3.0, 4.0], index=dates),
        "value_b": pd.Series([4.0, 3.0, 2.0, 1.0], index=dates),
    }
    composite = calculate_combined_factor_score(factor_data, {"value_a": 0.6, "value_b": 0.4})

    returns_panel = pd.DataFrame(
        {
            "a": [0.01, 0.02, -0.01, 0.015],
            "b": [0.005, -0.002, 0.004, 0.006],
        },
        index=dates,
    )
    eq = optimize_factor_weights(returns_panel, method="equal_weight")
    ic = optimize_factor_weights(returns_panel, method="ic_weight")
    methods = compare_and_rank(
        [
            {"name": "equal", "metrics": {"annual_return": eq["expected_return"], "max_drawdown": -0.05, "sharpe_ratio": eq["sharpe_ratio"], "win_rate": 0.6, "turnover": 0.2}},
            {"name": "ic", "metrics": {"annual_return": ic["expected_return"], "max_drawdown": -0.03, "sharpe_ratio": ic["sharpe_ratio"], "win_rate": 0.62, "turnover": 0.18}},
        ],
        scoring_type="strategy",
    )

    assert isinstance(composite, pd.Series)
    assert composite.name == "composite_score"
    assert set(eq["weights"].keys()) == {"a", "b"}
    assert set(ic["weights"].keys()) == {"a", "b"}
    assert methods[0]["rank"] == 1
