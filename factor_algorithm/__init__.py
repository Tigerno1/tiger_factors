"""Internal factor algorithm implementations.

The package exports are kept lazy so importing ``tiger_factors.factor_store``
or ``tiger_factors.factor_algorithm`` does not eagerly import every algorithm
family and accidentally create circular imports.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS: dict[str, tuple[str, str]] = {
    "Alpha101Engine": ("tiger_factors.factor_algorithm.alpha101.engine", "Alpha101Engine"),
    "NeutralizationColumns": ("tiger_factors.factor_algorithm.alpha101.engine", "NeutralizationColumns"),
    "alpha101_description": ("tiger_factors.factor_algorithm.alpha101.descriptions", "alpha101_description"),
    "alpha101_descriptions": ("tiger_factors.factor_algorithm.alpha101.descriptions", "alpha101_descriptions"),
    "alpha101_factor_names": ("tiger_factors.factor_algorithm.alpha101.engine", "alpha101_factor_names"),
    "build_code_industry_frame": ("tiger_factors.factor_algorithm.alpha101.engine", "build_code_industry_frame"),
    "GTJA191Engine": ("tiger_factors.factor_algorithm.gtja191.engine", "GTJA191Engine"),
    "gtja191_factor_names": ("tiger_factors.factor_algorithm.gtja191.engine", "gtja191_factor_names"),
    "Sunday100PlusEngine": ("tiger_factors.factor_algorithm.sunday100plus.engine", "Sunday100PlusEngine"),
    "sunday100plus_factor_names": ("tiger_factors.factor_algorithm.sunday100plus.engine", "sunday100plus_factor_names"),
    "DataMiningEngine": ("tiger_factors.factor_algorithm.data_mining.factors", "DataMiningEngine"),
    "data_mining_factor_names": ("tiger_factors.factor_algorithm.data_mining.factors", "available_factors"),
    "MarketBreathingColumns": ("tiger_factors.factor_algorithm.experimental.market_breathing", "MarketBreathingColumns"),
    "MarketBreathingEngine": ("tiger_factors.factor_algorithm.experimental.market_breathing", "MarketBreathingEngine"),
    "MarketBreathingResult": ("tiger_factors.factor_algorithm.experimental.market_breathing", "MarketBreathingResult"),
    "experimental_factor_catalog": ("tiger_factors.factor_algorithm.experimental.catalog", "experimental_factor_catalog"),
    "experimental_factor_spec": ("tiger_factors.factor_algorithm.experimental.catalog", "experimental_factor_spec"),
    "market_breathing_factor_names": (
        "tiger_factors.factor_algorithm.experimental.market_breathing",
        "market_breathing_factor_names",
    ),
    "NewsEntropyColumns": ("tiger_factors.factor_algorithm.experimental.news_entropy", "NewsEntropyColumns"),
    "NewsEntropyEngine": ("tiger_factors.factor_algorithm.experimental.news_entropy", "NewsEntropyEngine"),
    "NewsEntropyResult": ("tiger_factors.factor_algorithm.experimental.news_entropy", "NewsEntropyResult"),
    "news_entropy_factor_names": (
        "tiger_factors.factor_algorithm.experimental.news_entropy",
        "news_entropy_factor_names",
    ),
    "experimental_factor_names": ("tiger_factors.factor_algorithm.experimental.catalog", "experimental_factor_names"),
    "PracticalFactorEngine": (
        "tiger_factors.factor_algorithm.data_mining.practical_factors.factors",
        "PracticalFactorEngine",
    ),
    "practical_factor_names": (
        "tiger_factors.factor_algorithm.data_mining.practical_factors.factors",
        "available_practical_factors",
    ),
    "AnnualFinancialFactorEngine": (
        "tiger_factors.factor_algorithm.financial_factors.financial_factors",
        "AnnualFinancialFactorEngine",
    ),
    "FinancialFactorBundleResult": (
        "tiger_factors.factor_algorithm.financial_factors.financial_factors",
        "FinancialFactorBundleResult",
    ),
    "FinancialFactorEngine": (
        "tiger_factors.factor_algorithm.financial_factors.financial_factors",
        "FinancialFactorEngine",
    ),
    "QuarterlyFinancialFactorEngine": (
        "tiger_factors.factor_algorithm.financial_factors.financial_factors",
        "QuarterlyFinancialFactorEngine",
    ),
    "TTMFinancialFactorEngine": (
        "tiger_factors.factor_algorithm.financial_factors.financial_factors",
        "TTMFinancialFactorEngine",
    ),
    "Alpha158FactorSet": ("tiger_factors.factor_algorithm.qlib_factors.alpha_sets", "Alpha158FactorSet"),
    "Alpha360FactorSet": ("tiger_factors.factor_algorithm.qlib_factors.alpha_sets", "Alpha360FactorSet"),
    "QlibAlphaFactorSetEngine": ("tiger_factors.factor_algorithm.qlib_factors.alpha_sets", "QlibAlphaFactorSetEngine"),
    "alpha158_feature_config": ("tiger_factors.factor_algorithm.qlib_factors.alpha_sets", "alpha158_feature_config"),
    "alpha360_feature_config": ("tiger_factors.factor_algorithm.qlib_factors.alpha_sets", "alpha360_feature_config"),
    "available_qlib_factor_sets": ("tiger_factors.factor_algorithm.qlib_factors.alpha_sets", "available_qlib_factor_sets"),
    "TraditionalFactorPipelineEngine": (
        "tiger_factors.factor_algorithm.traditional_factors.pipeline",
        "TraditionalFactorPipelineEngine",
    ),
    "TraditionalFactorPipelineResult": (
        "tiger_factors.factor_algorithm.traditional_factors.pipeline",
        "TraditionalFactorPipelineResult",
    ),
    "TraditionalFactorGroup": (
        "tiger_factors.factor_algorithm.traditional_factors.index",
        "TraditionalFactorGroup",
    ),
    "CommonFactorSpec": (
        "tiger_factors.factor_algorithm.traditional_factors.common_factors",
        "CommonFactorSpec",
    ),
    "available_common_factors": (
        "tiger_factors.factor_algorithm.traditional_factors.common_factors",
        "available_common_factors",
    ),
    "common_factor_aliases": (
        "tiger_factors.factor_algorithm.traditional_factors.common_factors",
        "common_factor_aliases",
    ),
    "common_factor_family_markdown": (
        "tiger_factors.factor_algorithm.traditional_factors.common_factors",
        "common_factor_family_markdown",
    ),
    "common_factor_family_summary": (
        "tiger_factors.factor_algorithm.traditional_factors.common_factors",
        "common_factor_family_summary",
    ),
    "common_factor_display_names": (
        "tiger_factors.factor_algorithm.traditional_factors.common_factors",
        "common_factor_display_names",
    ),
    "common_factor_group_frame": (
        "tiger_factors.factor_algorithm.traditional_factors.common_factors",
        "common_factor_group_frame",
    ),
    "common_factor_group_markdown": (
        "tiger_factors.factor_algorithm.traditional_factors.common_factors",
        "common_factor_group_markdown",
    ),
    "common_factor_group_index": (
        "tiger_factors.factor_algorithm.traditional_factors.common_factors",
        "common_factor_group_index",
    ),
    "common_factor_group_names": (
        "tiger_factors.factor_algorithm.traditional_factors.common_factors",
        "common_factor_group_names",
    ),
    "common_factor_spec": (
        "tiger_factors.factor_algorithm.traditional_factors.common_factors",
        "common_factor_spec",
    ),
    "find_common_factor_group": (
        "tiger_factors.factor_algorithm.traditional_factors.common_factors",
        "find_common_factor_group",
    ),
    "TraditionalPortfolioEngine": (
        "tiger_factors.factor_algorithm.traditional_factors.portfolio",
        "TraditionalPortfolioEngine",
    ),
    "TraditionalPortfolioPaths": (
        "tiger_factors.factor_algorithm.traditional_factors.portfolio",
        "TraditionalPortfolioPaths",
    ),
    "check_signal_csvs": (
        "tiger_factors.factor_algorithm.traditional_factors.portfolio",
        "check_signal_csvs",
    ),
    "available_traditional_factor_names": (
        "tiger_factors.factor_algorithm.traditional_factors.pipeline",
        "available_factors",
    ),
    "create_crsp_predictors": (
        "tiger_factors.factor_algorithm.traditional_factors.portfolio",
        "create_crsp_predictors",
    ),
    "factor_metadata": (
        "tiger_factors.factor_algorithm.traditional_factors.factor_functions",
        "factor_metadata",
    ),
    "find_traditional_factor_group": (
        "tiger_factors.factor_algorithm.traditional_factors.index",
        "find_traditional_factor_group",
    ),
    "load_signal_doc": (
        "tiger_factors.factor_algorithm.traditional_factors.portfolio",
        "load_signal_doc",
    ),
    "run_original_factor": (
        "tiger_factors.factor_algorithm.traditional_factors.factor_functions",
        "run_original_factor",
    ),
    "traditional_factor_group_for_signal": (
        "tiger_factors.factor_algorithm.traditional_factors.index",
        "traditional_factor_group_for_signal",
    ),
    "traditional_factor_group_frame": (
        "tiger_factors.factor_algorithm.traditional_factors.index",
        "traditional_factor_group_frame",
    ),
    "traditional_factor_group_index": (
        "tiger_factors.factor_algorithm.traditional_factors.index",
        "traditional_factor_group_index",
    ),
    "traditional_factor_group_names": (
        "tiger_factors.factor_algorithm.traditional_factors.index",
        "traditional_factor_group_names",
    ),
    "traditional_factor_group_summary": (
        "tiger_factors.factor_algorithm.traditional_factors.index",
        "traditional_factor_group_summary",
    ),
    "traditional_factor_groups": (
        "tiger_factors.factor_algorithm.traditional_factors.index",
        "traditional_factor_groups",
    ),
    "run_common_factor": (
        "tiger_factors.factor_algorithm.traditional_factors.common_factors",
        "run_common_factor",
    ),
    "run_common_factors": (
        "tiger_factors.factor_algorithm.traditional_factors.common_factors",
        "run_common_factors",
    ),
    "run_value_quality_combo": (
        "tiger_factors.factor_algorithm.traditional_factors.common_factors",
        "run_value_quality_combo",
    ),
    "run_value_quality_combo_from_columns": (
        "tiger_factors.factor_algorithm.traditional_factors.common_factors",
        "run_value_quality_combo_from_columns",
    ),
    "run_value_quality_long_short_backtest": (
        "tiger_factors.factor_algorithm.traditional_factors.common_factors",
        "run_value_quality_long_short_backtest",
    ),
    "ValuationFactorEngine": (
        "tiger_factors.factor_algorithm.valuation_factors.valuation_factors",
        "ValuationFactorEngine",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(target[0])
    value = getattr(module, target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
