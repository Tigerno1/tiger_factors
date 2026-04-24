from __future__ import annotations

from tiger_factors.factor_screener.registry import FactorRegistryConfig
from tiger_factors.factor_screener.registry import build_factor_registry
from tiger_factors.factor_screener.registry import build_factor_registry_from_root
from tiger_factors.factor_screener.registry import screen_factor_registry
from tiger_factors.factor_screener.batch import FactorScreenerBatch
from tiger_factors.factor_screener.batch import FactorScreenerBatchItem
from tiger_factors.factor_screener.batch import FactorScreenerBatchResult
from tiger_factors.factor_screener.batch import FactorScreenerBatchSpec
from tiger_factors.factor_screener.batch import FactorScreenerDetailManifest
from tiger_factors.factor_screener.batch import FactorReturnGainSelectionConfig
from tiger_factors.factor_screener.batch import FactorSelectionMode
from tiger_factors.factor_screener.batch import run_factor_screener_batch
from tiger_factors.factor_screener.batch import run_factor_screener_flow
from tiger_factors.factor_screener._screener import FactorScreener
from tiger_factors.factor_screener._screener import FactorScreenerResult
from tiger_factors.factor_screener._screener import FactorScreenerSpec
from tiger_factors.factor_screener._screener import run_factor_screener
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
from tiger_factors.factor_screener.selection import cluster_factors
from tiger_factors.factor_screener.selection import factor_correlation_matrix
from tiger_factors.factor_screener.selection import FactorMarginalSelectionConfig
from tiger_factors.factor_screener.selection import greedy_select_by_correlation
from tiger_factors.factor_screener.selection import ic_correlation_matrix
from tiger_factors.factor_screener.selection import ic_time_series
from tiger_factors.factor_screener.selection import select_factors_by_marginal_gain
from tiger_factors.factor_screener.selection import select_ic_coherent_factors
from tiger_factors.factor_screener.selection import select_non_redundant_factors
from . import bayes_validation as _bayes_validation
from . import validation as _validation
from tiger_factors.factor_screener.validation import *  # noqa: F401,F403
from tiger_factors.factor_screener.bayes_validation import *  # noqa: F401,F403

__all__ = [
    "FactorFilterConfig",
    "FactorMetricFilterConfig",
    "FactorSelectionConfig",
    "FactorRegistryConfig",
    "FactorMarginalSelectionConfig",
    "FactorScreenerBatch",
    "FactorScreenerBatchItem",
    "FactorScreenerBatchResult",
    "FactorScreenerBatchSpec",
    "FactorScreenerDetailManifest",
    "FactorReturnGainSelectionConfig",
    "FactorSelectionMode",
    "FactorScreener",
    "FactorScreenerResult",
    "FactorScreenerSpec",
    "add_cost_analysis",
    "build_factor_registry",
    "build_factor_registry_from_root",
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
    "run_factor_screener_batch",
    "run_factor_screener_flow",
    "screen_factor_metrics",
    "screen_factor_registry",
    "screen_factor_results",
    "select_factors_by_marginal_gain",
    "select_ic_coherent_factors",
    "select_non_redundant_factors",
]

__all__ += list(getattr(_validation, "__all__", ()))
__all__ += list(getattr(_bayes_validation, "__all__", ()))
