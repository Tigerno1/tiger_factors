from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from tiger_factors.factor_preprocessing._core import coerce_factor_panel
from tiger_factors.utils.cross_sectional import winsorize_cross_section


def detect_outliers_factor_panel(
    values: pd.Series | pd.DataFrame,
    *,
    method: str = "iqr",
    axis: int = 1,
    threshold: float = 3.0,
    lower: float = 0.01,
    upper: float = 0.99,
    n_mad: float = 5.0,
) -> pd.Series | pd.DataFrame:
    """Return a boolean mask marking outliers."""

    normalized_method = str(method).strip().lower()

    if isinstance(values, pd.Series):
        series = pd.to_numeric(values, errors="coerce")
        valid = series.dropna()
        if valid.empty:
            return pd.Series(False, index=series.index)
        if normalized_method == "iqr":
            q1 = valid.quantile(0.25)
            q3 = valid.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return (series < lower_bound) | (series > upper_bound)
        if normalized_method == "zscore":
            std = valid.std(ddof=0)
            if std <= 0:
                return pd.Series(False, index=series.index)
            z = (series - valid.mean()) / std
            return z.abs() > threshold
        if normalized_method == "mad":
            med = valid.median()
            mad = (valid - med).abs().median() * 1.4826
            if mad <= 0:
                return pd.Series(False, index=series.index)
            return ((series - med).abs() / mad) > n_mad
        if normalized_method == "quantile":
            lo = valid.quantile(lower)
            hi = valid.quantile(upper)
            return (series < lo) | (series > hi)
        raise ValueError("Unsupported outlier detection method.")

    frame = coerce_factor_panel(values).apply(pd.to_numeric, errors="coerce")
    if axis == 0:
        frame = frame.T

    def _mask_row(row: pd.Series) -> pd.Series:
        valid = row.dropna()
        if valid.empty:
            return pd.Series(False, index=row.index)
        if normalized_method == "iqr":
            q1 = valid.quantile(0.25)
            q3 = valid.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return (row < lower_bound) | (row > upper_bound)
        if normalized_method == "zscore":
            std = valid.std(ddof=0)
            if std <= 0:
                return pd.Series(False, index=row.index)
            z = (row - valid.mean()) / std
            return z.abs() > threshold
        if normalized_method == "mad":
            med = valid.median()
            mad = (valid - med).abs().median() * 1.4826
            if mad <= 0:
                return pd.Series(False, index=row.index)
            return ((row - med).abs() / mad) > n_mad
        if normalized_method == "quantile":
            lo = valid.quantile(lower)
            hi = valid.quantile(upper)
            return (row < lo) | (row > hi)
        raise ValueError("Unsupported outlier detection method.")

    mask = frame.apply(_mask_row, axis=1)
    return mask.T if axis == 0 else mask


def replace_outliers_with_nan(
    values: pd.Series | pd.DataFrame,
    *,
    method: str = "iqr",
    axis: int = 1,
    threshold: float = 3.0,
    lower: float = 0.01,
    upper: float = 0.99,
    n_mad: float = 5.0,
) -> pd.Series | pd.DataFrame:
    mask = detect_outliers_factor_panel(
        values,
        method=method,
        axis=axis,
        threshold=threshold,
        lower=lower,
        upper=upper,
        n_mad=n_mad,
    )
    if isinstance(values, pd.Series):
        out = pd.to_numeric(values, errors="coerce").copy()
        out.loc[mask] = np.nan
        return out
    frame = coerce_factor_panel(values).apply(pd.to_numeric, errors="coerce")
    out = frame.copy()
    out = out.mask(mask)
    return out


def winsorize_factor_panel(
    values: pd.Series | pd.DataFrame,
    *,
    method: str = "mad",
    axis: int = 1,
    lower: float = 0.01,
    upper: float = 0.99,
    n_mad: float = 5.0,
) -> pd.Series | pd.DataFrame:
    return winsorize_cross_section(values, method=method, axis=axis, lower=lower, upper=upper, n_mad=n_mad)


def detect_anomalies_isolation_forest(
    values: pd.Series | pd.DataFrame,
    *,
    contamination: float = 0.05,
    random_state: int | None = 42,
) -> pd.Series | pd.DataFrame:
    """Flag anomalies using Isolation Forest."""

    if isinstance(values, pd.Series):
        frame = pd.to_numeric(values, errors="coerce").to_frame("value").dropna()
        if frame.empty:
            return pd.Series(False, index=values.index)
        model = IsolationForest(contamination=contamination, random_state=random_state)
        flags = model.fit_predict(frame.to_numpy(dtype=float)) == -1
        out = pd.Series(False, index=values.index)
        out.loc[frame.index] = flags
        return out

    frame = coerce_factor_panel(values).apply(pd.to_numeric, errors="coerce")
    clean = frame.fillna(frame.median())
    model = IsolationForest(contamination=contamination, random_state=random_state)
    flags = model.fit_predict(clean.to_numpy(dtype=float)) == -1
    return pd.DataFrame(np.repeat(flags.reshape(-1, 1), clean.shape[1], axis=1), index=clean.index, columns=clean.columns)


__all__ = [
    "detect_anomalies_isolation_forest",
    "detect_outliers_factor_panel",
    "replace_outliers_with_nan",
    "winsorize_factor_panel",
]
