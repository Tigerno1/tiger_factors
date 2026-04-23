from __future__ import annotations

import numpy as np
import pandas as pd

from tiger_factors.factor_ml.data_mining import (
    DataPreprocessingService,
    FactorGeneratorService,
    GeneticFactorMiningService,
)


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date_": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "code": ["AAA", "AAA", "AAA"],
            "open": [1.0, 2.0, 1000.0],
            "close": [1.5, 2.5, 1001.0],
            "high": [1.6, 2.6, 1002.0],
            "low": [0.9, 1.9, 999.0],
            "volume": [10.0, 20.0, 30.0],
        }
    )


def test_data_preprocessing_standardize_and_quality() -> None:
    service = DataPreprocessingService()
    standardized = service.standardize_columns(_sample_frame())

    assert "date_" in standardized.columns
    assert "code" in standardized.columns

    valid, message = service.validate_data_quality(
        standardized,
        required_columns=["date_", "code", "open", "close", "high", "low", "volume"],
    )
    assert valid
    assert "passed" in message


def test_data_preprocessing_outlier_handling() -> None:
    service = DataPreprocessingService()
    frame = _sample_frame()
    clipped = service.handle_outliers(frame, "open", method="clip", n_sigma=1.0)
    assert clipped["open"].max() < 1000.0


def test_factor_generator_creates_candidate_expressions() -> None:
    service = FactorGeneratorService()
    binary = service.generate_binary_combinations(["close", "open"], max_combinations=8)
    hybrid = service.generate_hybrid_factors(["close", "open", "high"], n_factors=12)

    assert binary
    assert hybrid
    assert service.validate_expression("(close + open)")[0]
    parsed = service.parse_expression("ts_mean(close, 5)")
    assert parsed["depth"] >= 1


def test_genetic_factor_mining_runs_small_search() -> None:
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    frame = pd.DataFrame(
        {
            "close": np.linspace(1.0, 2.0, len(dates)),
            "open": np.linspace(0.9, 1.9, len(dates)),
        }
    )
    frame["return"] = frame["close"].pct_change().fillna(0.0)

    miner = GeneticFactorMiningService(
        base_factors=["close", "open"],
        data=frame,
        return_column="return",
        population_size=8,
        n_generations=1,
        random_state=0,
    )
    result = miner.run()

    assert result["best_expression"]
    assert result["best_fitness"] >= 0.0
    assert result["hall_of_fame"]
