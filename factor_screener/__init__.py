from __future__ import annotations

from tiger_factors.factor_screener.registry import FactorRegistryConfig
from tiger_factors.factor_screener.registry import build_factor_registry
from tiger_factors.factor_screener.registry import build_factor_registry_from_root
from tiger_factors.factor_screener.registry import screen_factor_registry
from tiger_factors.factor_screener.factor_screener import FactorScreener
from tiger_factors.factor_screener.factor_screener import FactorScreenerResult
from tiger_factors.factor_screener.factor_screener import FactorScreenerSpec
from tiger_factors.factor_screener.factor_screener import run_factor_screener
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
from tiger_factors.factor_screener.correlation_screener import CorrelationScreener
from tiger_factors.factor_screener.correlation_screener import CorrelationScreenerResult
from tiger_factors.factor_screener.correlation_screener import CorrelationScreenerSpec
from tiger_factors.factor_screener.correlation_screener import run_correlation_screener
from tiger_factors.factor_screener.backtest_marginal_screener import BacktestMarginalScreener
from tiger_factors.factor_screener.backtest_marginal_screener import BacktestMarginalScreenerResult
from tiger_factors.factor_screener.backtest_marginal_screener import BacktestMarginalScreenerSpec
from tiger_factors.factor_screener.backtest_marginal_screener import run_backtest_marginal_screener
from tiger_factors.factor_screener.return_adapter import ReturnAdapter
from tiger_factors.factor_screener.return_adapter import ReturnAdapterResult
from tiger_factors.factor_screener.return_adapter import ReturnAdapterSpec
from tiger_factors.factor_screener.return_adapter import run_return_adapter
from tiger_factors.factor_screener.screener import Screener
from tiger_factors.factor_screener.screener import ScreenerFinalResult
from tiger_factors.factor_screener.screener import ScreenerResult
from tiger_factors.factor_screener.screener import run_screener
from tiger_factors.factor_screener.selection import cluster_factors
from tiger_factors.factor_screener.selection import factor_correlation_matrix
from tiger_factors.factor_screener.selection import greedy_select_by_correlation
from tiger_factors.factor_screener.selection import ic_correlation_matrix
from tiger_factors.factor_screener.selection import ic_time_series
from tiger_factors.factor_screener.selection import select_by_average_correlation
from tiger_factors.factor_screener.selection import select_by_graph_independent_set
from tiger_factors.factor_screener.selection import select_ic_by_average_correlation
from tiger_factors.factor_screener.selection import select_ic_by_graph_independent_set
from tiger_factors.factor_screener.selection import select_non_redundant_factors
from tiger_factors.factor_screener.marginal_screener import MarginalScreener
from tiger_factors.factor_screener.marginal_screener import MarginalScreenerResult
from tiger_factors.factor_screener.marginal_screener import MarginalScreenerSpec
from tiger_factors.factor_screener.marginal_screener import run_marginal_screener
from tiger_factors.factor_screener.single_factor import run_single_factor_screening
from . import bayes_validation as _bayes_validation
from . import validation as _validation
from tiger_factors.factor_screener.validation import *  # noqa: F401,F403
from tiger_factors.factor_screener.bayes_validation import *  # noqa: F401,F403

__all__ = [
    "FactorFilterConfig",
    "FactorMetricFilterConfig",
    "FactorSelectionConfig",
    "FactorRegistryConfig",
    "CorrelationScreener",
    "CorrelationScreenerResult",
    "CorrelationScreenerSpec",
    "FactorScreener",
    "FactorScreenerResult",
    "FactorScreenerSpec",
    "MarginalScreener",
    "MarginalScreenerResult",
    "MarginalScreenerSpec",
    "BacktestMarginalScreener",
    "BacktestMarginalScreenerResult",
    "BacktestMarginalScreenerSpec",
    "ReturnAdapter",
    "ReturnAdapterResult",
    "ReturnAdapterSpec",
    "ScreenerFinalResult",
    "Screener",
    "ScreenerResult",
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
    "select_by_average_correlation",
    "select_by_graph_independent_set",
    "run_factor_screener",
    "run_screener",
    "run_single_factor_screening",
    "run_return_adapter",
    "run_correlation_screener",
    "run_marginal_screener",
    "run_backtest_marginal_screener",
    "screen_factor_metrics",
    "screen_factor_registry",
    "screen_factor_results",
    "select_ic_by_average_correlation",
    "select_ic_by_graph_independent_set",
    "select_non_redundant_factors",
]

__all__ += list(getattr(_validation, "__all__", ()))
__all__ += list(getattr(_bayes_validation, "__all__", ()))
