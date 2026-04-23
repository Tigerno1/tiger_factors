from __future__ import annotations

import pandas as pd
import pytest

from tiger_factors.factor_screener import FactorSelectionConfig
from tiger_factors.factor_screener import FactorMetricFilterConfig
from tiger_factors.factor_screener import build_factor_registry_from_root
from tiger_factors.factor_screener import add_cost_analysis
from tiger_factors.factor_screener import evaluate_factor
from tiger_factors.factor_screener import screen_factor_metrics
from tiger_factors.factor_screener import screen_factor_registry
from tiger_factors.factor_screener import screen_factor_results


def _sample_result() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "horizon": [1, 5, 10],
            "mean_ic": [0.03, 0.02, 0.015],
            "ic_std": [0.01, 0.012, 0.013],
            "ic_ir": [3.0, 1.67, 1.15],
            "mean_spread": [0.01, 0.015, 0.012],
            "spread_std": [0.02, 0.02, 0.025],
            "spread_ir": [0.5, 0.75, 0.48],
            "ann_return": [0.24, 0.34, 0.26],
            "ann_vol": [0.12, 0.15, 0.14],
            "sharpe": [2.0, 2.27, 1.86],
            "max_drawdown": [-0.08, -0.09, -0.1],
            "avg_turnover": [0.25, 0.35, 0.45],
            "rank_autocorr": [0.6, 0.52, 0.44],
            "direction": ["positive", "positive", "positive"],
        }
    )


def test_add_cost_analysis_adds_net_columns() -> None:
    result = _sample_result()

    out = add_cost_analysis(result, cost_rate=0.001)

    assert {"cost_penalty", "net_ann_return", "net_sharpe"}.issubset(out.columns)
    assert out.loc[0, "cost_penalty"] == pytest.approx(0.25 * 0.001 * 252)
    assert out.loc[0, "net_ann_return"] == pytest.approx(0.24 - 0.25 * 0.001 * 252)
    assert out.loc[0, "net_sharpe"] == pytest.approx(out.loc[0, "net_ann_return"] / 0.12)


def test_evaluate_factor_returns_use_as_is_for_positive_factor() -> None:
    result = add_cost_analysis(_sample_result(), cost_rate=0.0005)

    evaluation = evaluate_factor(result)

    assert evaluation["usable"] == True
    assert evaluation["direction"] == "use_as_is"
    assert evaluation["quality"] == "strong"
    assert evaluation["best_ic_horizon"] == 1
    assert evaluation["best_sharpe_horizon"] == 5
    assert "5D holding" in evaluation["recommendation"]


def test_evaluate_factor_returns_reverse_factor_for_negative_ic() -> None:
    result = _sample_result()
    result["mean_ic"] = [-0.02, -0.015, -0.01]

    evaluation = evaluate_factor(result)

    assert evaluation["usable"] == True
    assert evaluation["direction"] == "reverse_factor"
    assert evaluation["quality"] == "medium"


def test_evaluate_factor_handles_missing_ic() -> None:
    result = _sample_result()
    result["mean_ic"] = [float("nan"), float("nan"), float("nan")]

    evaluation = evaluate_factor(result)

    assert evaluation["usable"] == False
    assert evaluation["reason"] == "No valid IC observations"
    assert evaluation["recommendation"] == "DO NOT USE"


def test_screen_factor_results_builds_sorted_summary() -> None:
    strong = _sample_result()
    weak = _sample_result()
    weak["mean_ic"] = [0.002, 0.003, 0.004]
    weak["sharpe"] = [0.4, 0.5, 0.45]

    summary = screen_factor_results(
        {"alpha_strong": strong, "alpha_weak": weak},
        config=FactorSelectionConfig(cost_rate=0.0005),
    )

    assert list(summary["factor_name"]) == ["alpha_strong", "alpha_weak"]
    assert summary.loc[0, "usable"] == True
    assert summary.loc[1, "usable"] == False
    assert summary.loc[0, "selected_horizon"] == 5
    assert summary.loc[0, "selected_net_sharpe"] == pytest.approx(
        add_cost_analysis(strong, cost_rate=0.0005).loc[1, "net_sharpe"]
    )


def test_screen_factor_metrics_filters_core_evaluation_summary() -> None:
    summary = pd.DataFrame(
        {
            "factor_name": ["alpha_strong", "alpha_weak"],
            "ic_mean": [0.03, 0.002],
            "ic_ir": [2.8, 0.12],
            "rank_ic_mean": [0.025, 0.001],
            "sharpe": [1.8, 0.1],
            "turnover": [0.25, 0.55],
            "fitness": [1.2, -0.3],
            "decay_score": [0.8, 0.1],
            "capacity_score": [0.7, 0.05],
            "correlation_penalty": [0.2, 0.6],
            "regime_robustness": [0.4, 0.05],
            "out_of_sample_stability": [0.5, 0.02],
        }
    )

    screened = screen_factor_metrics(
        summary,
        config=FactorMetricFilterConfig(
            min_ic_mean=0.005,
            min_rank_ic_mean=0.005,
            min_sharpe=0.3,
            max_turnover=0.5,
            min_decay_score=0.2,
            min_capacity_score=0.2,
            max_correlation_penalty=0.6,
            min_regime_robustness=0.4,
            min_out_of_sample_stability=0.5,
            sort_field="fitness",
            tie_breaker_field="ic_ir",
        ),
    )

    assert list(screened["factor_name"]) == ["alpha_strong", "alpha_weak"]
    assert bool(screened.loc[0, "usable"]) is True
    assert bool(screened.loc[1, "usable"]) is False
    assert "turnover>0.5" in screened.loc[1, "failed_rules"]


def test_screen_factor_metrics_keeps_reverse_direction_factors() -> None:
    summary = pd.DataFrame(
        {
            "factor_name": ["alpha_reverse", "alpha_positive"],
            "ic_mean": [-0.03, 0.012],
            "ic_ir": [-2.8, 0.9],
            "rank_ic_mean": [-0.025, 0.01],
            "sharpe": [-1.9, 0.7],
            "turnover": [0.25, 0.22],
            "fitness": [-1.2, 0.4],
            "decay_score": [0.8, 0.7],
            "capacity_score": [0.7, 0.6],
            "correlation_penalty": [0.2, 0.1],
            "regime_robustness": [0.4, 0.5],
            "out_of_sample_stability": [0.5, 0.6],
        }
    )

    screened = screen_factor_metrics(
        summary,
        config=FactorMetricFilterConfig(
            min_fitness=0.0,
            min_ic_mean=0.005,
            min_rank_ic_mean=0.005,
            min_sharpe=0.3,
            max_turnover=0.5,
            min_decay_score=0.2,
            min_capacity_score=0.2,
            max_correlation_penalty=0.6,
            min_regime_robustness=0.4,
            min_out_of_sample_stability=0.5,
            sort_field="fitness",
            tie_breaker_field="ic_ir",
        ),
    )

    assert list(screened["factor_name"]) == ["alpha_reverse", "alpha_positive"]
    assert bool(screened.loc[0, "usable"]) is True
    assert bool(screened.loc[1, "usable"]) is True
    assert screened.loc[0, "direction_hint"] == "reverse_factor"
    assert screened.loc[0, "directional_fitness"] == pytest.approx(1.2)
    assert screened.loc[0, "directional_sharpe"] == pytest.approx(1.9)
    assert screened.loc[0, "directional_ic_ir"] == pytest.approx(2.8)


def test_build_factor_registry_from_strategy_summaries(tmp_path) -> None:
    alpha_dir = tmp_path / "alpha_strong" / "summary"
    beta_dir = tmp_path / "alpha_weak" / "summary"
    alpha_dir.mkdir(parents=True)
    beta_dir.mkdir(parents=True)

    strong = pd.DataFrame(
        [
            {
                "factor_name": "alpha_strong",
                "ic_mean": 0.03,
                "ic_ir": 2.8,
                "rank_ic_mean": 0.025,
                "sharpe": 1.8,
                "turnover": 0.25,
                "fitness": 1.2,
            }
        ]
    )
    weak = pd.DataFrame(
        [
            {
                "factor_name": "alpha_weak",
                "ic_mean": 0.002,
                "ic_ir": 0.12,
                "rank_ic_mean": 0.001,
                "sharpe": 0.1,
                "turnover": 0.55,
                "fitness": -0.3,
            }
        ]
    )
    strong.to_parquet(alpha_dir / "evaluation.parquet")
    weak.to_parquet(beta_dir / "evaluation.parquet")

    registry = build_factor_registry_from_root(tmp_path)
    assert list(registry["strategy_name"]) == ["alpha_strong", "alpha_weak"]
    assert list(registry["factor_name"]) == ["alpha_strong", "alpha_weak"]

    screened = screen_factor_registry(
        registry,
        config=FactorMetricFilterConfig(
            min_ic_mean=0.005,
            min_rank_ic_mean=0.005,
            min_sharpe=0.3,
            max_turnover=0.5,
            min_decay_score=0.2,
            min_capacity_score=0.2,
            max_correlation_penalty=0.6,
            min_regime_robustness=0.4,
            min_out_of_sample_stability=0.5,
            sort_field="fitness",
            tie_breaker_field="ic_ir",
        ),
    )
    assert list(screened["strategy_name"]) == ["alpha_strong", "alpha_weak"]
