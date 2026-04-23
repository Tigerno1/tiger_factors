from __future__ import annotations

from tiger_factors.factor_maker.pipeline import (
    Pipeline,
    PipelineEngine,
    Returns,
    RollingStdDev,
    SimFinBalanceSheet,
    SimFinIncomeStatement,
    USEquityPricing,
)
from tiger_factors.factor_store import TigerFactorLibrary


def run_example() -> None:
    library = TigerFactorLibrary(
        region="us",
        sec_type="stock",
        price_provider="yahoo",
    )

    # Example universe: resolve a stock pool from tiger_api static company master data.
    # Adjust provider / dataset / code_column to match your local data source.
    codes = library.resolve_universe_codes(
        provider="simfin",
        dataset="companies",
        code_column="symbol",
        limit=500,
        as_ex=True,
    )

    # Prices and fundamentals are both loaded through tiger_api providers.
    engine = PipelineEngine(
        library=library,
        region="us",
        sec_type="stock",
        price_provider="yahoo",
        calendar_source="auto",
        as_ex=True,
        provider_overrides={
            "source_type:price": "yahoo",
            "dataset:income_statement": "simfin",
            "dataset:balance_sheet": "simfin",
        },
        fundamental_use_point_in_time=True,
        fundamental_lag_sessions=1,
    )

    momentum_20 = Returns(window_length=20)
    volatility_20 = RollingStdDev(inputs=[USEquityPricing.column("close")], window_length=20)
    roe_like = SimFinIncomeStatement.column("net_income") / SimFinBalanceSheet.column("total_equity")
    size_like = USEquityPricing.column("close") * SimFinBalanceSheet.column("shares_basic")

    pipe = Pipeline(
        columns={
            "close": USEquityPricing.column("close"),
            "volume": USEquityPricing.column("volume"),
            "momentum_20": momentum_20,
            "volatility_20": volatility_20,
            "roe_like": roe_like,
            "size_like": size_like,
        },
        screen=(
            USEquityPricing.column("close").notnull()
            & USEquityPricing.column("volume").notnull()
            & USEquityPricing.column("volume").top(500)
        ),
    )

    result = engine.run_pipeline(
        pipe,
        codes=codes,
        start="2024-01-01",
        end="2024-03-31",
    )

    print(result.head(20))


if __name__ == "__main__":
    run_example()
