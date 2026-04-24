from __future__ import annotations

from tiger_factors.factor_screener.screening import FactorFilterConfig
from tiger_factors.factor_screener.screening import FactorMetricFilterConfig
from tiger_factors.factor_screener.screening import FactorSelectionConfig
from tiger_factors.factor_screener.screening import add_cost_analysis
from tiger_factors.factor_screener.screening import classify_quality
from tiger_factors.factor_screener.screening import compute_factor_score
from tiger_factors.factor_screener.screening import evaluate_factor
from tiger_factors.factor_screener.screening import evaluate_factor_with_filter
from tiger_factors.factor_screener.screening import factor_metric_frame
from tiger_factors.factor_screener.screening import screen_factor_metrics
from tiger_factors.factor_screener.screening import screen_factor_results


def run_single_factor_screening(
    results,
    *,
    config: FactorMetricFilterConfig | None = None,
):
    return screen_factor_metrics(results, config=config)


__all__ = [
    "FactorFilterConfig",
    "FactorMetricFilterConfig",
    "FactorSelectionConfig",
    "add_cost_analysis",
    "classify_quality",
    "compute_factor_score",
    "evaluate_factor",
    "evaluate_factor_with_filter",
    "factor_metric_frame",
    "run_single_factor_screening",
    "screen_factor_metrics",
    "screen_factor_results",
]
