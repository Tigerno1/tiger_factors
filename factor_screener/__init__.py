from __future__ import annotations

from tiger_factors.multifactor_evaluation.screening import FactorFilterConfig
from tiger_factors.multifactor_evaluation.screening import FactorMetricFilterConfig
from tiger_factors.multifactor_evaluation.screening import FactorSelectionConfig
from tiger_factors.multifactor_evaluation.screening import add_cost_analysis
from tiger_factors.multifactor_evaluation.screening import classify_quality
from tiger_factors.multifactor_evaluation.screening import compute_factor_score
from tiger_factors.multifactor_evaluation.screening import evaluate_factor
from tiger_factors.multifactor_evaluation.screening import evaluate_factor_with_filter
from tiger_factors.multifactor_evaluation.screening import factor_metric_frame
from tiger_factors.multifactor_evaluation.screening import screen_factor_metrics
from tiger_factors.multifactor_evaluation.screening import screen_factor_results
from tiger_factors.multifactor_evaluation.selection import cluster_factors
from tiger_factors.multifactor_evaluation.selection import factor_correlation_matrix
from tiger_factors.multifactor_evaluation.selection import greedy_select_by_correlation
from tiger_factors.multifactor_evaluation.selection import ic_correlation_matrix
from tiger_factors.multifactor_evaluation.selection import ic_time_series
from tiger_factors.multifactor_evaluation.batch import FactorScreeningEngine
from tiger_factors.multifactor_evaluation.registry import FactorRegistryConfig
from tiger_factors.multifactor_evaluation.registry import build_factor_registry
from tiger_factors.multifactor_evaluation.registry import build_factor_registry_from_root
from tiger_factors.multifactor_evaluation.registry import screen_factor_registry
from tiger_factors.multifactor_evaluation.selection import select_ic_coherent_factors
from tiger_factors.multifactor_evaluation.selection import select_non_redundant_factors
from tiger_factors.tiger_screener import FactorScreener as _FactorScreener
from tiger_factors.tiger_screener import FactorScreenerResult as _FactorScreenerResult
from tiger_factors.tiger_screener import FactorScreenerSpec as _FactorScreenerSpec
from tiger_factors.tiger_screener import run_factor_screener as _run_factor_screener

__all__ = [
    "FactorFilterConfig",
    "FactorMetricFilterConfig",
    "FactorSelectionConfig",
    "FactorScreeningEngine",
    "FactorRegistryConfig",
    "FactorScreener",
    "FactorScreenerResult",
    "FactorScreenerSpec",
    "add_cost_analysis",
    "classify_quality",
    "cluster_factors",
    "compute_factor_score",
    "evaluate_factor",
    "evaluate_factor_with_filter",
    "factor_correlation_matrix",
    "factor_metric_frame",
    "greedy_select_by_correlation",
    "ic_correlation_matrix",
    "ic_time_series",
    "run_factor_screener",
    "build_factor_registry",
    "build_factor_registry_from_root",
    "screen_factor_metrics",
    "screen_factor_registry",
    "screen_factor_results",
    "select_ic_coherent_factors",
    "select_non_redundant_factors",
]

FactorScreener = _FactorScreener
FactorScreenerResult = _FactorScreenerResult
FactorScreenerSpec = _FactorScreenerSpec
run_factor_screener = _run_factor_screener
