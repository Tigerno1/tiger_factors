from __future__ import annotations

import pandas as pd

from tiger_factors.factor_algorithm.sunday100plus import Sunday100PlusEngine


def _sample_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    rows = []
    for code, base in [("AAA", 10.0), ("BBB", 12.0)]:
        for idx, date in enumerate(dates):
            price = base + idx
            rows.append(
                {
                    "date_": date,
                    "code": code,
                    "open": price,
                    "high": price + 1,
                    "low": price - 1,
                    "close": price + 0.5,
                    "volume": 1000 + idx * 10,
                    "vwap": price + 0.25,
                    "sector": "tech",
                    "industry": "software",
                    "subindustry": "saas",
                }
            )
    return pd.DataFrame(rows)


def test_sunday100plus_engine_can_register_and_compute() -> None:
    engine = Sunday100PlusEngine(_sample_frame())
    engine.register_formula(1, lambda eng: eng.data["close"] - eng.data["open"], description="close-open spread")
    result = engine.compute(1)
    assert list(result.columns) == ["date_", "code", "alpha_001"]
    assert result["alpha_001"].notna().any()
    assert engine.formula_description(1) == "close-open spread"

