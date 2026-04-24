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
    "PositionAnalysisResult",
    "ReturnAnalysisResult",
    "TransactionAnalysisResult",
    "analyze_multifactor",
    "analyze_positions",
    "analyze_returns",
    "analyze_transactions",
    "best",
    "cagr",
    "calmar",
    "common_sense_ratio",
    "consecutive_losses",
    "consecutive_wins",
    "cpc_index",
    "cvar",
    "expected_shortfall",
    "create_full_tear_sheet",
    "create_analysis_report",
    "create_multifactor_evaluation",
    "gain_to_pain_ratio",
    "exposure",
    "information_ratio",
    "kelly_criterion",
    "omega",
    "payoff_ratio",
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
    "smart_sharpe",
    "smart_sortino",
    "skew",
    "tail_ratio",
    "treynor_ratio",
    "ulcer_index",
    "ulcer_performance_index",
    "value_at_risk",
    "win_loss_ratio",
    "worst",
    "var",
]
