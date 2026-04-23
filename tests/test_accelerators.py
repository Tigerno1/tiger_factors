from __future__ import annotations

import pandas as pd

from tiger_factors.utils.accelerators import momentum_12m_1m, rolling_std


def test_rolling_std_fallback_shape():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rolling_std(series, 3)
    assert len(result) == len(series)
    assert result.index.equals(series.index)


def test_momentum_fallback_shape():
    series = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    result = momentum_12m_1m(series, 3, 1)
    assert len(result) == len(series)
    assert result.index.equals(series.index)
