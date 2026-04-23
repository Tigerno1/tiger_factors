from __future__ import annotations

from tiger_factors.factor_maker.pipeline import Pipeline
from tiger_factors.factor_maker.pipeline import PipelineEngine
from tiger_factors.factor_maker.pipeline import Returns
from tiger_factors.factor_maker.pipeline import USEquityPricing
from tiger_factors.factor_evaluation.evaluation import evaluate_from_pipeline
from tiger_factors.factor_store import FactorSpec
from tiger_factors.factor_store import TigerFactorLibrary


library = TigerFactorLibrary(
    region="us",
    sec_type="stock",
    price_provider="yahoo",
)
engine = PipelineEngine(
    library=library,
    region="us",
    sec_type="stock",
    price_provider="yahoo",
    as_ex=True,
    provider_overrides={"source_type:price": "yahoo"},
)

codes = library.resolve_universe_codes(
    provider="simfin",
    dataset="companies",
    code_column="symbol",
    limit=100,
    as_ex=True,
)

pipeline = Pipeline(
    columns={
        "close": USEquityPricing.column("close"),
        "volume": USEquityPricing.column("volume"),
        "momentum_20": Returns(window_length=20),
    },
    screen=USEquityPricing.column("close").notnull() & USEquityPricing.column("volume").notnull(),
)

result = evaluate_from_pipeline(
    spec=FactorSpec(
        provider="tiger",
        region="us",
        sec_type="stock",
        freq="1d",
        table_name="momentum_20",
    ),
    pipeline=pipeline,
    pipeline_engine=engine,
    codes=codes,
    start="2024-01-01",
    end="2024-03-31",
)
print(result.to_dict())
