from __future__ import annotations

import os
from importlib import import_module

if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)

_LAZY_MODULES: dict[str, str] = {
    "factor_algorithm": "tiger_factors.factor_algorithm",
    "factor_evaluation": "tiger_factors.factor_evaluation",
    "factor_frame": "tiger_factors.factor_frame",
    "factor_maker": "tiger_factors.factor_maker",
    "factor_ml": "tiger_factors.factor_ml",
    "factor_allocation": "tiger_factors.factor_allocation",
    "factor_backtest": "tiger_factors.factor_backtest",
    "factor_store": "tiger_factors.factor_store",
    "factor_screener": "tiger_factors.factor_screener",
    "multifactor_evaluation": "tiger_factors.multifactor_evaluation",
}

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "Alpha101Engine": ("tiger_factors.factor_algorithm.alpha101.engine", "Alpha101Engine"),
    "Alpha158FactorSet": ("tiger_factors.factor_algorithm.qlib_factors.alpha_sets", "Alpha158FactorSet"),
    "Alpha360FactorSet": ("tiger_factors.factor_algorithm.qlib_factors.alpha_sets", "Alpha360FactorSet"),
    "AnnualFinancialFactorEngine": ("tiger_factors.factor_algorithm.financial_factors.financial_factors", "AnnualFinancialFactorEngine"),
    "DataMiningEngine": ("tiger_factors.factor_algorithm.data_mining.factors", "DataMiningEngine"),
    "DataPreprocessingService": ("tiger_factors.factor_ml.data_mining", "DataPreprocessingService"),
    "FactorTimingPipelineEngine": ("tiger_factors.factor_algorithm.factor_timing.pipeline", "FactorTimingPipelineEngine"),
    "FactorTimingPipelineResult": ("tiger_factors.factor_algorithm.factor_timing.pipeline", "FactorTimingPipelineResult"),
    "FactorFrameContext": ("tiger_factors.factor_frame.engine", "FactorFrameContext"),
    "FactorFrameEngine": ("tiger_factors.factor_frame.engine", "FactorFrameEngine"),
    "FactorFrameFeed": ("tiger_factors.factor_frame.engine", "FactorFrameFeed"),
    "FactorFrameResult": ("tiger_factors.factor_frame.engine", "FactorFrameResult"),
    "FinancialFactorEngine": ("tiger_factors.factor_algorithm.financial_factors.financial_factors", "FinancialFactorEngine"),
    "FactorGeneratorService": ("tiger_factors.factor_ml.data_mining", "FactorGeneratorService"),
    "NeutralizationColumns": ("tiger_factors.factor_algorithm.alpha101.engine", "NeutralizationColumns"),
    "GeneticFactorMiningService": ("tiger_factors.factor_ml.data_mining", "GeneticFactorMiningService"),
    "PracticalFactorEngine": ("tiger_factors.factor_algorithm.data_mining.practical_factors.factors", "PracticalFactorEngine"),
    "QlibAlphaFactorSetEngine": ("tiger_factors.factor_algorithm.qlib_factors.alpha_sets", "QlibAlphaFactorSetEngine"),
    "QuarterlyFinancialFactorEngine": ("tiger_factors.factor_algorithm.financial_factors.financial_factors", "QuarterlyFinancialFactorEngine"),
    "StreamingFactorEngine": ("tiger_factors.factor_maker.streaming", "StreamingFactorEngine"),
    "TTMFinancialFactorEngine": ("tiger_factors.factor_algorithm.financial_factors.financial_factors", "TTMFinancialFactorEngine"),
    "TraditionalFactorPipelineEngine": ("tiger_factors.factor_algorithm.traditional_factors.pipeline", "TraditionalFactorPipelineEngine"),
    "TraditionalFactorPipelineResult": ("tiger_factors.factor_algorithm.traditional_factors.pipeline", "TraditionalFactorPipelineResult"),
    "ValuationFactorEngine": ("tiger_factors.factor_algorithm.valuation_factors.valuation_factors", "ValuationFactorEngine"),
    "FactorScreener": ("tiger_factors.factor_screener", "FactorScreener"),
    "FactorScreenerResult": ("tiger_factors.factor_screener", "FactorScreenerResult"),
    "FactorScreenerSpec": ("tiger_factors.factor_screener", "FactorScreenerSpec"),
    "MarginalScreener": ("tiger_factors.factor_screener", "MarginalScreener"),
    "MarginalScreenerResult": ("tiger_factors.factor_screener", "MarginalScreenerResult"),
    "MarginalScreenerSpec": ("tiger_factors.factor_screener", "MarginalScreenerSpec"),
    "BacktestMarginalScreener": ("tiger_factors.factor_screener", "BacktestMarginalScreener"),
    "BacktestMarginalScreenerResult": ("tiger_factors.factor_screener", "BacktestMarginalScreenerResult"),
    "BacktestMarginalScreenerSpec": ("tiger_factors.factor_screener", "BacktestMarginalScreenerSpec"),
    "ReturnAdapter": ("tiger_factors.factor_screener", "ReturnAdapter"),
    "ReturnAdapterResult": ("tiger_factors.factor_screener", "ReturnAdapterResult"),
    "ReturnAdapterSpec": ("tiger_factors.factor_screener", "ReturnAdapterSpec"),
    "Screener": ("tiger_factors.factor_screener", "Screener"),
    "ScreenerResult": ("tiger_factors.factor_screener", "ScreenerResult"),
    "MultifactorAnalysisReportResult": ("tiger_factors.multifactor_evaluation", "MultifactorAnalysisReportResult"),
    "MultifactorAnalysisReportSpec": ("tiger_factors.multifactor_evaluation", "MultifactorAnalysisReportSpec"),
    "create_analysis_report": ("tiger_factors.multifactor_evaluation", "create_analysis_report"),
    "build_single_factor_return_long_frame": ("tiger_factors.factor_evaluation.factor_screening", "build_single_factor_return_long_frame"),
    "allocate_from_return_panel": ("tiger_factors.factor_allocation", "allocate_from_return_panel"),
    "run_factor_screener": ("tiger_factors.factor_screener", "run_factor_screener"),
    "run_factor_screener_batch": ("tiger_factors.factor_screener", "run_factor_screener_batch"),
    "run_screener": ("tiger_factors.factor_screener", "run_screener"),
    "run_return_backtest": ("tiger_factors.factor_backtest", "run_return_backtest"),
    "alpha101_descriptions": ("tiger_factors.factor_algorithm.alpha101.descriptions", "alpha101_descriptions"),
    "alpha101_factor_names": ("tiger_factors.factor_algorithm.alpha101.engine", "alpha101_factor_names"),
    "alpha158_feature_config": ("tiger_factors.factor_algorithm.qlib_factors.alpha_sets", "alpha158_feature_config"),
    "alpha360_feature_config": ("tiger_factors.factor_algorithm.qlib_factors.alpha_sets", "alpha360_feature_config"),
    "available_qlib_factor_sets": ("tiger_factors.factor_algorithm.qlib_factors.alpha_sets", "available_qlib_factor_sets"),
    "available_traditional_factor_names": ("tiger_factors.factor_algorithm.traditional_factors.pipeline", "available_traditional_factor_names"),
    "data_mining_factor_names": ("tiger_factors.factor_algorithm.data_mining.factors", "available_factors"),
    "practical_factor_names": ("tiger_factors.factor_algorithm.data_mining.practical_factors.factors", "available_practical_factors"),
}

__all__ = [
    "factor_algorithm",
    "factor_evaluation",
    "factor_frame",
    "factor_maker",
    "factor_ml",
    "factor_allocation",
    "factor_backtest",
    "factor_store",
    "factor_screener",
    "multifactor_evaluation",
    "Alpha101Engine",
    "NeutralizationColumns",
    "alpha101_descriptions",
    "alpha101_factor_names",
    "DataMiningEngine",
    "data_mining_factor_names",
    "DataPreprocessingService",
    "FactorFrameContext",
    "FactorFrameEngine",
    "FactorFrameFeed",
    "FactorFrameResult",
    "FactorTimingPipelineEngine",
    "FactorGeneratorService",
    "PracticalFactorEngine",
    "GeneticFactorMiningService",
    "practical_factor_names",
    "StreamingFactorEngine",
    "TraditionalFactorPipelineEngine",
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
    "Screener",
    "ScreenerResult",
    "MultifactorAnalysisReportResult",
    "MultifactorAnalysisReportSpec",
    "create_analysis_report",
    "run_factor_screener",
    "run_factor_screener_batch",
    "run_screener",
    "build_single_factor_return_long_frame",
    "allocate_from_return_panel",
    "run_return_backtest",
]


def __getattr__(name: str):
    module_path = _LAZY_MODULES.get(name)
    if module_path is not None:
        module = import_module(module_path)
        globals()[name] = module
        return module

    target = _LAZY_ATTRS.get(name)
    if target is not None:
        module = import_module(target[0])
        value = getattr(module, target[1])
        globals()[name] = value
        return value

    raise AttributeError(f"module 'tiger_factors' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_LAZY_MODULES) | set(_LAZY_ATTRS))
