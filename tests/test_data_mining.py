from __future__ import annotations

import numpy as np
import pandas as pd

from tiger_factors import DataMiningEngine, data_mining_factor_names
from tiger_factors.factor_algorithm.data_mining import factor_013_volume, factor_024_close_over_high


def _sample_ohlcv_panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    rows = []
    configs = [
        ("AAA", 100.0),
        ("BBB", 60.0),
        ("CCC", 80.0),
    ]
    for code, base in configs:
        for idx, date in enumerate(dates):
            close = base + 0.2 * idx + np.sin(idx / 6.0)
            rows.append(
                {
                    "date_": date,
                    "code": code,
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.2,
                    "close": close,
                    "volume": float(1_000_000 + idx * 1_000 + (abs(hash(code)) % 991)),
                }
            )
    return pd.DataFrame(rows)


def test_data_mining_registry_lists_cleaned_factors():
    names = data_mining_factor_names()
    assert len(names) == 15
    assert "factor_013_volume" in names
    assert "factor_040_volume_high_corr_strength" in names


def test_data_mining_engine_computes_cleaned_factors():
    engine = DataMiningEngine(_sample_ohlcv_panel())

    result = engine.compute_all(
        names=[
            "factor_013_volume",
            "factor_024_close_over_high",
            "factor_012_volume_sum5",
            "factor_032_low_pctchange_momentum",
        ]
    )

    assert {"date_", "code", "factor_013_volume", "factor_024_close_over_high"}.issubset(result.columns)
    assert result["factor_013_volume"].notna().all()
    assert result["factor_024_close_over_high"].notna().all()
    assert result["factor_012_volume_sum5"].notna().any()
    assert result["factor_032_low_pctchange_momentum"].notna().any()


def test_data_mining_factor_wrappers_match_manual_ratios():
    panel = _sample_ohlcv_panel()
    result = factor_013_volume(panel)
    expected = panel[["date_", "code", "volume"]].rename(columns={"volume": "factor_013_volume"})
    merged = result.merge(expected, on=["date_", "code"], how="inner", suffixes=("", "_expected"))
    assert np.allclose(merged["factor_013_volume"], merged["factor_013_volume_expected"])

    close_over_high = factor_024_close_over_high(panel)
    merged = close_over_high.merge(
        panel.assign(factor_024_close_over_high=panel["close"] / panel["high"])[
            ["date_", "code", "factor_024_close_over_high"]
        ],
        on=["date_", "code"],
        how="inner",
        suffixes=("", "_expected"),
    )
    assert np.allclose(merged["factor_024_close_over_high"], merged["factor_024_close_over_high_expected"])
