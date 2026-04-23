from tiger_factors import (
    FactorTimingPipelineEngine,
    TraditionalFactorPipelineEngine,
)
from tiger_factors import factor_maker as tiger_factor_maker_facade
from tiger_factors import factor_ml as tiger_factor_ml_package
from tiger_factors import factor_evaluation as tiger_factor_evaluation_facade
from tiger_factors.factor_maker import pipeline as tiger_factor_pipeline_facade
from tiger_factors.factor_maker import vectorization as tiger_factor_vectorization_facade
from tiger_factors import factor_store as tiger_factor_store_facade
from tiger_factors import multifactor_evaluation as tiger_multifactor_evaluation_facade


def test_factor_generation_facade_exports():
    assert hasattr(tiger_factor_maker_facade, "Alpha101Engine")
    assert hasattr(tiger_factor_maker_facade, "FinancialFactorEngine")
    assert hasattr(tiger_factor_maker_facade, "TraditionalFactorPipelineEngine")
    assert hasattr(tiger_factor_maker_facade, "FactorTimingPipelineEngine")


def test_evaluation_facades_exports():
    assert not hasattr(tiger_factor_evaluation_facade, "SingleFactorEvaluation")
    assert not hasattr(tiger_factor_evaluation_facade, "TigerFactorEvaluation")
    assert hasattr(tiger_multifactor_evaluation_facade, "screen_factor_metrics")


def test_data_store_facade_exports():
    assert hasattr(tiger_factor_store_facade, "FactorStore")
    assert hasattr(tiger_factor_store_facade, "MacroSpec")


def test_factor_ml_package_exports():
    assert hasattr(tiger_factor_ml_package, "AlphaFeatureEngineer")
    assert hasattr(tiger_factor_ml_package, "AlphaFormulaVM")
    assert hasattr(tiger_factor_ml_package, "FactorGeneratorService")
    assert hasattr(tiger_factor_ml_package, "DataPreprocessingService")
    assert hasattr(tiger_factor_ml_package, "GeneticFactorMiningService")
    assert hasattr(tiger_factor_ml_package, "__all__")
    assert "AlphaFeatureEngineer" in tiger_factor_ml_package.__all__


def test_pipeline_and_vectorization_facades():
    assert hasattr(tiger_factor_pipeline_facade, "PipelineEngine")
    assert hasattr(tiger_factor_pipeline_facade, "Pipeline")
    assert hasattr(tiger_factor_vectorization_facade, "FactorVectorizationTransformer")
    assert hasattr(tiger_factor_vectorization_facade, "Alpha101IndicatorTransformer")


def test_factor_maker_facade_groups_generation_stack():
    assert hasattr(tiger_factor_maker_facade, "Alpha101Engine")
    assert hasattr(tiger_factor_maker_facade, "PipelineEngine")
    assert hasattr(tiger_factor_maker_facade, "FactorVectorizationTransformer")
    assert hasattr(tiger_factor_maker_facade, "TraditionalFactorPipelineEngine")
    assert tiger_factor_maker_facade.__name__.endswith("factor_maker")


def test_top_level_pipeline_exports():
    assert TraditionalFactorPipelineEngine is not None
    assert FactorTimingPipelineEngine is not None
