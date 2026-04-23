from __future__ import annotations

import pandas as pd

from tiger_factors.factor_evaluation import build_alphalens_input


def test_build_alphalens_input_from_long_frames(tmp_path) -> None:
    factor_path = tmp_path / "factor.parquet"
    price_path = tmp_path / "price.parquet"

    factor_frame = pd.DataFrame(
        {
            "date_": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "code": ["AAA", "BBB", "AAA", "BBB"],
            "alpha_001": [1.0, 2.0, 1.5, 2.5],
        }
    )
    price_frame = pd.DataFrame(
        {
            "date_": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "code": ["AAA", "BBB", "AAA", "BBB"],
            "close": [10.0, 20.0, 11.0, 21.0],
        }
    )
    factor_frame.to_parquet(factor_path, index=False)
    price_frame.to_parquet(price_path, index=False)

    adapter = build_alphalens_input(
        factor_frame=factor_path,
        price_frame=price_path,
        factor_column="alpha_001",
    )

    assert adapter.factor_series.name == "alpha_001"
    assert adapter.prices.shape == (2, 2)
    assert float(adapter.prices.loc[pd.Timestamp("2024-01-01"), "AAA"]) == 10.0
    assert float(adapter.factor_series.loc[(pd.Timestamp("2024-01-01"), "AAA")]) == 1.0

    adapter2 = build_alphalens_input(
        factor_frame=factor_frame,
        price_frame=price_frame,
        factor_column="alpha_001",
    )
    assert adapter2.factor_series.equals(adapter.factor_series)
    assert adapter2.prices.equals(adapter.prices)
