from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.impute import KNNImputer

from tiger_factors.factor_preprocessing._core import _ensure_datetime_index
from tiger_factors.factor_preprocessing._core import coerce_factor_panel
from tiger_factors.factor_preprocessing._core import coerce_target_panel


def fill_missing_factor_panel(
    values: pd.Series | pd.DataFrame,
    *,
    method: str = "median",
    axis: int = 1,
    fill_value: float | int | str | None = None,
    n_neighbors: int = 5,
    max_iter: int = 20,
    random_state: int | None = 42,
) -> pd.Series | pd.DataFrame:
    """Fill missing values for factor data."""

    if isinstance(values, pd.Series):
        series = pd.to_numeric(values, errors="coerce")
        normalized_method = str(method).strip().lower()
        if normalized_method == "mean":
            return series.fillna(series.mean())
        if normalized_method == "median":
            return series.fillna(series.median())
        if normalized_method == "zero":
            return series.fillna(0.0)
        if normalized_method == "constant":
            if fill_value is None:
                raise ValueError("fill_value is required for constant imputation.")
            return series.fillna(fill_value)
        if normalized_method == "ffill":
            return series.ffill()
        if normalized_method == "bfill":
            return series.bfill()
        if normalized_method == "interpolate":
            return series.interpolate(limit_direction="both")
        raise ValueError("Unsupported missing value method.")

    frame = coerce_factor_panel(values)
    normalized_method = str(method).strip().lower()

    if normalized_method in {"mean", "median"}:
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        statistic = numeric.mean(axis=1) if normalized_method == "mean" else numeric.median(axis=1)
        return numeric.where(numeric.notna(), statistic, axis=0)

    if normalized_method == "zero":
        return frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if normalized_method == "constant":
        if fill_value is None:
            raise ValueError("fill_value is required for constant imputation.")
        return frame.apply(pd.to_numeric, errors="coerce").fillna(fill_value)

    if normalized_method == "ffill":
        return frame.ffill(axis=0 if axis == 0 else 1)

    if normalized_method == "bfill":
        return frame.bfill(axis=0 if axis == 0 else 1)

    if normalized_method == "interpolate":
        return frame.apply(pd.to_numeric, errors="coerce").interpolate(axis=0 if axis == 0 else 1, limit_direction="both")

    numeric = frame.apply(pd.to_numeric, errors="coerce")

    if normalized_method == "knn":
        imputer = KNNImputer(n_neighbors=int(n_neighbors), weights="distance")
        return pd.DataFrame(imputer.fit_transform(numeric), index=numeric.index, columns=numeric.columns)

    if normalized_method == "iterative":
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        from sklearn.linear_model import BayesianRidge

        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=int(max_iter),
            random_state=random_state,
        )
        return pd.DataFrame(imputer.fit_transform(numeric), index=numeric.index, columns=numeric.columns)

    raise ValueError("Unsupported missing value method.")


__all__ = [
    "coerce_factor_panel",
    "coerce_target_panel",
    "fill_missing_factor_panel",
]
