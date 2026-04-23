from __future__ import annotations

import numpy as np
import pandas as pd

from tiger_factors.factor_algorithm.gtja191 import GTJA191Engine


def _build_input() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    rows = []
    for symbol_idx, symbol in enumerate(["AAA", "BBB"]):
        base = 10.0 + symbol_idx * 2.0
        for i, date in enumerate(dates):
            close = base + np.sin(i / 2.0) + i * 0.3 + symbol_idx * 0.2
            open_ = close - 0.3 + ((-1) ** i) * 0.1
            high = close + 0.5 + (symbol_idx * 0.05)
            low = close - 0.7 - (symbol_idx * 0.03)
            rows.append(
                {
                    "date_": date,
                    "code": symbol,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": 1000 + i * 50 + symbol_idx * 75 + (i % 3) * 20,
                    "mkt": 0.0005 + 0.0001 * np.sin(i / 6.0),
                    "smb": 0.0003 + 0.0001 * np.cos(i / 5.0),
                    "hml": 0.0002 + 0.0001 * np.sin(i / 4.0),
                }
            )
    return pd.DataFrame(rows)


def test_gtja191_engine_computes_initial_factors() -> None:
    engine = GTJA191Engine(_build_input())

    result = engine.compute(2)
    assert not result.empty
    assert list(result.columns) == ["date_", "code", "alpha_002"]

    result_2 = engine.compute(13)
    assert not result_2.empty
    assert list(result_2.columns) == ["date_", "code", "alpha_013"]

    result_30 = engine.compute(30)
    assert not result_30.empty
    assert list(result_30.columns) == ["date_", "code", "alpha_030"]

    all_result = engine.compute_all([2, 13])
    assert {"date_", "code", "alpha_002", "alpha_013"}.issubset(all_result.columns)
    assert {2, 13}.issubset(set(engine.implemented_alpha_ids()))
    assert len(engine.implemented_alpha_ids()) == 191
    assert engine.implemented_alpha_ids()[0] == 1
    assert engine.implemented_alpha_ids()[-1] == 191
    assert engine.factor_names()[0] == "alpha_001"
    assert engine.factor_names()[-1] == "alpha_191"
