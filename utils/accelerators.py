from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import tiger_rust_factors as _rust_factors
except Exception:
    _rust_factors = None


def rust_available() -> bool:
    return _rust_factors is not None


def _series_from_values(index: pd.Index, values) -> pd.Series:
    return pd.Series(np.asarray(values, dtype=float), index=index, dtype=float)


def rolling_std(series: pd.Series, window: int) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if _rust_factors is None:
        return numeric.rolling(window).std()
    values = _rust_factors.rolling_std_1d(numeric.fillna(np.nan).tolist(), int(window))
    return _series_from_values(numeric.index, values)


def momentum_12m_1m(series: pd.Series, long_window: int = 252, short_window: int = 21) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if _rust_factors is None:
        return numeric.pct_change(long_window) - numeric.pct_change(short_window)
    values = _rust_factors.momentum_12m_1m_1d(
        numeric.fillna(np.nan).tolist(),
        int(long_window),
        int(short_window),
    )
    return _series_from_values(numeric.index, values)
