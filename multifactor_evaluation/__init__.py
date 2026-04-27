from __future__ import annotations

from tiger_factors.multifactor_evaluation.evaluation import MultifactorEvaluation
from tiger_factors.multifactor_evaluation.evaluation import MultifactorEvaluationBundle
from tiger_factors.multifactor_evaluation.evaluation import create_full_tear_sheet
from tiger_factors.multifactor_evaluation.evaluation import create_multifactor_evaluation
from tiger_factors.multifactor_evaluation.analysis import MultifactorAnalysisResult
from tiger_factors.multifactor_evaluation.analysis import PositionAnalysisResult
from tiger_factors.multifactor_evaluation.analysis import ReturnAnalysisResult
from tiger_factors.multifactor_evaluation.analysis import TransactionAnalysisResult
from tiger_factors.multifactor_evaluation.analysis import analyze_multifactor
from tiger_factors.multifactor_evaluation.analysis import analyze_positions
from tiger_factors.multifactor_evaluation.analysis import analyze_returns
from tiger_factors.multifactor_evaluation.analysis import analyze_transactions
from tiger_factors.multifactor_evaluation.analysis import best
from tiger_factors.multifactor_evaluation.analysis import calmar
from tiger_factors.multifactor_evaluation.analysis import cagr
from tiger_factors.multifactor_evaluation.analysis import common_sense_ratio
from tiger_factors.multifactor_evaluation.analysis import consecutive_losses
from tiger_factors.multifactor_evaluation.analysis import consecutive_wins
from tiger_factors.multifactor_evaluation.analysis import cpc_index
from tiger_factors.multifactor_evaluation.analysis import cvar
from tiger_factors.multifactor_evaluation.analysis import expected_shortfall
from tiger_factors.multifactor_evaluation.analysis import gain_to_pain_ratio
from tiger_factors.multifactor_evaluation.analysis import information_ratio
from tiger_factors.multifactor_evaluation.analysis import kelly_criterion
from tiger_factors.multifactor_evaluation.analysis import exposure
from tiger_factors.multifactor_evaluation.analysis import kurtosis
from tiger_factors.multifactor_evaluation.analysis import monthly_returns_heatmap
from tiger_factors.multifactor_evaluation.analysis import omega
from tiger_factors.multifactor_evaluation.analysis import payoff_ratio
from tiger_factors.multifactor_evaluation.analysis import probabilistic_adjusted_sortino_ratio
from tiger_factors.multifactor_evaluation.analysis import probabilistic_ratio
from tiger_factors.multifactor_evaluation.analysis import probabilistic_sharpe_ratio
from tiger_factors.multifactor_evaluation.analysis import probabilistic_sortino_ratio
from tiger_factors.multifactor_evaluation.analysis import profit_factor
from tiger_factors.multifactor_evaluation.analysis import profit_ratio
from tiger_factors.multifactor_evaluation.analysis import recovery_factor
from tiger_factors.multifactor_evaluation.analysis import r2
from tiger_factors.multifactor_evaluation.analysis import risk_return_ratio
from tiger_factors.multifactor_evaluation.analysis import serenity_index
from tiger_factors.multifactor_evaluation.analysis import skew
from tiger_factors.multifactor_evaluation.analysis import smart_sharpe
from tiger_factors.multifactor_evaluation.analysis import smart_sortino
from tiger_factors.multifactor_evaluation.analysis import tail_ratio
from tiger_factors.multifactor_evaluation.analysis import treynor_ratio
from tiger_factors.multifactor_evaluation.analysis import ulcer_index
from tiger_factors.multifactor_evaluation.analysis import ulcer_performance_index
from tiger_factors.multifactor_evaluation.analysis import var
from tiger_factors.multifactor_evaluation.analysis import value_at_risk
from tiger_factors.multifactor_evaluation.analysis import win_loss_ratio
from tiger_factors.multifactor_evaluation.analysis import worst
from tiger_factors.multifactor_evaluation.reporting.analysis_report import MultifactorAnalysisReportResult
from tiger_factors.multifactor_evaluation.reporting.analysis_report import MultifactorAnalysisReportSpec
from tiger_factors.multifactor_evaluation.reporting.analysis_report import create_analysis_report

__all__ = [
    "MultifactorEvaluation",
    "MultifactorEvaluationBundle",
    "MultifactorAnalysisResult",
    "MultifactorAnalysisReportResult",
    "MultifactorAnalysisReportSpec",
    "MultipleTestingResult",
    "ValidationResult",
    "BayesianMixtureResult",
    "HierarchicalBayesianResult",
    "RollingBayesianAlphaResult",
    "DynamicBayesianAlphaResult",
    "AlphaHackingBayesianResult",
    "FactorConvexityResult",
    "FactorDecayResult",
    "FactorRecentICResult",
    "FactorEffectivenessConfig",
    "PositionAnalysisResult",
    "ReturnAnalysisResult",
    "TransactionAnalysisResult",
    "analyze_multifactor",
    "analyze_positions",
    "analyze_returns",
    "analyze_transactions",
    "adjust_p_values",
    "best",
    "cagr",
    "calmar",
    "alpha_hacking_bayesian_update",
    "common_sense_ratio",
    "consecutive_losses",
    "consecutive_wins",
    "cpc_index",
    "cvar",
    "bayesian_fdr",
    "bayesian_fwer",
    "benjamini_hochberg",
    "benjamini_yekutieli",
    "bonferroni_adjust",
    "bootstrap_confidence_interval",
    "dynamic_bayesian_alpha",
    "estimate_pi0",
    "expected_shortfall",
    "create_full_tear_sheet",
    "create_analysis_report",
    "create_multifactor_evaluation",
    "factor_convexity_test",
    "factor_decay_test",
    "factor_effectiveness_test",
    "factor_recent_ic_test",
    "factor_stability_test",
    "gain_to_pain_ratio",
    "exposure",
    "information_ratio",
    "kelly_criterion",
    "fit_bayesian_mixture",
    "fit_hierarchical_bayesian_mixture",
    "hierarchical_bayesian_fdr",
    "holm_adjust",
    "omega",
    "payoff_ratio",
    "permutation_test",
    "probabilistic_adjusted_sortino_ratio",
    "probabilistic_ratio",
    "probabilistic_sharpe_ratio",
    "probabilistic_sortino_ratio",
    "profit_factor",
    "profit_ratio",
    "recovery_factor",
    "r2",
    "risk_return_ratio",
    "serenity_index",
    "kurtosis",
    "monthly_returns_heatmap",
    "rolling_bayesian_alpha",
    "smart_sharpe",
    "smart_sortino",
    "skew",
    "split_stability",
    "storey_qvalues",
    "tail_ratio",
    "treynor_ratio",
    "ulcer_index",
    "ulcer_performance_index",
    "test_factor_effectiveness",
    "validate_bayesian_factor_family",
    "validate_factor_data",
    "validate_factor_family",
    "validate_hierarchical_bayesian_factor_family",
    "validate_series",
    "value_at_risk",
    "win_loss_ratio",
    "worst",
    "var",
]

_FACTOR_TEST_ATTRS: dict[str, tuple[str, str]] = {
    "MultipleTestingResult": ("tiger_factors.factor_test", "MultipleTestingResult"),
    "ValidationResult": ("tiger_factors.factor_test", "ValidationResult"),
    "BayesianMixtureResult": ("tiger_factors.factor_test", "BayesianMixtureResult"),
    "HierarchicalBayesianResult": ("tiger_factors.factor_test", "HierarchicalBayesianResult"),
    "RollingBayesianAlphaResult": ("tiger_factors.factor_test", "RollingBayesianAlphaResult"),
    "DynamicBayesianAlphaResult": ("tiger_factors.factor_test", "DynamicBayesianAlphaResult"),
    "AlphaHackingBayesianResult": ("tiger_factors.factor_test", "AlphaHackingBayesianResult"),
    "FactorConvexityResult": ("tiger_factors.factor_test", "FactorConvexityResult"),
    "FactorDecayResult": ("tiger_factors.factor_test", "FactorDecayResult"),
    "FactorRecentICResult": ("tiger_factors.factor_test", "FactorRecentICResult"),
    "FactorEffectivenessConfig": ("tiger_factors.factor_test", "FactorEffectivenessConfig"),
    "adjust_p_values": ("tiger_factors.factor_test", "adjust_p_values"),
    "alpha_hacking_bayesian_update": ("tiger_factors.factor_test", "alpha_hacking_bayesian_update"),
    "bayesian_fdr": ("tiger_factors.factor_test", "bayesian_fdr"),
    "bayesian_fwer": ("tiger_factors.factor_test", "bayesian_fwer"),
    "benjamini_hochberg": ("tiger_factors.factor_test", "benjamini_hochberg"),
    "benjamini_yekutieli": ("tiger_factors.factor_test", "benjamini_yekutieli"),
    "bonferroni_adjust": ("tiger_factors.factor_test", "bonferroni_adjust"),
    "bootstrap_confidence_interval": ("tiger_factors.factor_test", "bootstrap_confidence_interval"),
    "dynamic_bayesian_alpha": ("tiger_factors.factor_test", "dynamic_bayesian_alpha"),
    "estimate_pi0": ("tiger_factors.factor_test", "estimate_pi0"),
    "factor_convexity_test": ("tiger_factors.factor_test", "factor_convexity_test"),
    "factor_decay_test": ("tiger_factors.factor_test", "factor_decay_test"),
    "factor_effectiveness_test": ("tiger_factors.factor_test", "factor_effectiveness_test"),
    "factor_recent_ic_test": ("tiger_factors.factor_test", "factor_recent_ic_test"),
    "factor_stability_test": ("tiger_factors.factor_test", "factor_stability_test"),
    "fit_bayesian_mixture": ("tiger_factors.factor_test", "fit_bayesian_mixture"),
    "fit_hierarchical_bayesian_mixture": ("tiger_factors.factor_test", "fit_hierarchical_bayesian_mixture"),
    "hierarchical_bayesian_fdr": ("tiger_factors.factor_test", "hierarchical_bayesian_fdr"),
    "holm_adjust": ("tiger_factors.factor_test", "holm_adjust"),
    "permutation_test": ("tiger_factors.factor_test", "permutation_test"),
    "rolling_bayesian_alpha": ("tiger_factors.factor_test", "rolling_bayesian_alpha"),
    "split_stability": ("tiger_factors.factor_test", "split_stability"),
    "storey_qvalues": ("tiger_factors.factor_test", "storey_qvalues"),
    "test_factor_effectiveness": ("tiger_factors.factor_test", "test_factor_effectiveness"),
    "validate_bayesian_factor_family": ("tiger_factors.factor_test", "validate_bayesian_factor_family"),
    "validate_factor_data": ("tiger_factors.factor_test", "validate_factor_data"),
    "validate_factor_family": ("tiger_factors.factor_test", "validate_factor_family"),
    "validate_hierarchical_bayesian_factor_family": ("tiger_factors.factor_test", "validate_hierarchical_bayesian_factor_family"),
    "validate_series": ("tiger_factors.factor_test", "validate_series"),
}

__all__.extend(sorted(_FACTOR_TEST_ATTRS))


def __getattr__(name: str):
    target = _FACTOR_TEST_ATTRS.get(name)
    if target is not None:
        from importlib import import_module

        module = import_module(target[0])
        value = getattr(module, target[1])
        globals()[name] = value
        return value
    raise AttributeError(f"module 'tiger_factors.multifactor_evaluation' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_FACTOR_TEST_ATTRS))
