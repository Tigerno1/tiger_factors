from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from tiger_factors.factor_preprocessing._core import _frame_or_series
from tiger_factors.factor_preprocessing._core import coerce_factor_panel
from tiger_factors.factor_preprocessing._core import coerce_target_panel
from tiger_factors.factor_preprocessing.binning import bin_factor_panel
from tiger_factors.factor_preprocessing.missing import fill_missing_factor_panel
from tiger_factors.factor_preprocessing.neutralization import neutralize_factor_panel
from tiger_factors.factor_preprocessing.outliers import detect_anomalies_isolation_forest
from tiger_factors.factor_preprocessing.outliers import replace_outliers_with_nan
from tiger_factors.factor_preprocessing.outliers import winsorize_factor_panel
from tiger_factors.factor_preprocessing.scaling import scale_factor_panel


def preprocess_factor_panel(
    values: pd.Series | pd.DataFrame,
    *,
    missing_strategy: str | None = "median",
    missing_kwargs: Mapping[str, Any] | None = None,
    outlier_strategy: str | None = "winsorize",
    outlier_kwargs: Mapping[str, Any] | None = None,
    neutralize_strategy: str | None = None,
    groups: pd.Series | pd.DataFrame | None = None,
    exposures: pd.Series | pd.DataFrame | list[pd.Series | pd.DataFrame] | None = None,
    normalize_strategy: str | None = "zscore",
    normalize_kwargs: Mapping[str, Any] | None = None,
    bin_strategy: str | None = None,
    bin_kwargs: Mapping[str, Any] | None = None,
) -> pd.Series | pd.DataFrame:
    """Run a configurable preprocessing pipeline on factor data."""

    out: pd.Series | pd.DataFrame = _frame_or_series(values)

    if missing_strategy is not None:
        out = fill_missing_factor_panel(out, method=missing_strategy, **dict(missing_kwargs or {}))

    if outlier_strategy is not None:
        outlier_args = dict(outlier_kwargs or {})
        if outlier_strategy.lower() in {"winsorize", "cap"}:
            out = winsorize_factor_panel(out, method=outlier_args.pop("method", "mad"), **outlier_args)
        elif outlier_strategy.lower() in {"replace_nan", "nan", "mask"}:
            out = replace_outliers_with_nan(out, **outlier_args)
        elif outlier_strategy.lower() in {"isolation_forest", "iforest"}:
            out = out.mask(detect_anomalies_isolation_forest(out, **outlier_args))
        else:
            raise ValueError("Unsupported outlier_strategy.")

    if neutralize_strategy is not None:
        out = neutralize_factor_panel(out, groups=groups, exposures=exposures, method=neutralize_strategy)

    if normalize_strategy is not None:
        out = scale_factor_panel(out, method=normalize_strategy, **dict(normalize_kwargs or {}))

    if bin_strategy is not None:
        bin_args = dict(bin_kwargs or {})
        target_value = bin_args.pop("target", None)
        out = bin_factor_panel(out, method=bin_strategy, target=target_value, **bin_args)

    return out


@dataclass
class FactorPreprocessor:
    """Configurable factor preprocessing pipeline."""

    missing_strategy: str | None = "median"
    missing_kwargs: dict[str, Any] = field(default_factory=dict)
    outlier_strategy: str | None = "winsorize"
    outlier_kwargs: dict[str, Any] = field(default_factory=dict)
    neutralize_strategy: str | None = None
    normalize_strategy: str | None = "zscore"
    normalize_kwargs: dict[str, Any] = field(default_factory=dict)
    bin_strategy: str | None = None
    bin_kwargs: dict[str, Any] = field(default_factory=dict)

    def fit(self, values: pd.Series | pd.DataFrame, y: pd.Series | pd.DataFrame | None = None) -> FactorPreprocessor:
        self._fitted_shape = tuple(coerce_factor_panel(values).shape) if isinstance(values, pd.DataFrame) else None
        self._fitted_target_shape = tuple(coerce_target_panel(y).shape) if y is not None else None
        return self

    def transform(
        self,
        values: pd.Series | pd.DataFrame,
        *,
        target: pd.Series | pd.DataFrame | None = None,
        groups: pd.Series | pd.DataFrame | None = None,
        exposures: pd.Series | pd.DataFrame | list[pd.Series | pd.DataFrame] | None = None,
    ) -> pd.Series | pd.DataFrame:
        out = values
        if self.missing_strategy is not None:
            out = fill_missing_factor_panel(out, method=self.missing_strategy, **self.missing_kwargs)
        if self.outlier_strategy is not None:
            strategy = self.outlier_strategy.lower()
            if strategy in {"winsorize", "cap"}:
                out = winsorize_factor_panel(out, method=self.outlier_kwargs.get("method", "mad"), **{k: v for k, v in self.outlier_kwargs.items() if k != "method"})
            elif strategy in {"replace_nan", "nan", "mask"}:
                out = replace_outliers_with_nan(out, **self.outlier_kwargs)
            elif strategy in {"isolation_forest", "iforest"}:
                out = out.mask(detect_anomalies_isolation_forest(out, **self.outlier_kwargs))
            else:
                raise ValueError("Unsupported outlier_strategy.")
        if self.neutralize_strategy is not None:
            out = neutralize_factor_panel(out, groups=groups, exposures=exposures, method=self.neutralize_strategy)
        if self.normalize_strategy is not None:
            out = scale_factor_panel(out, method=self.normalize_strategy, **self.normalize_kwargs)
        if self.bin_strategy is not None:
            bin_kwargs = dict(self.bin_kwargs)
            if target is not None and "target" not in bin_kwargs:
                bin_kwargs["target"] = target
            out = bin_factor_panel(out, method=self.bin_strategy, **bin_kwargs)
        return out

    def fit_transform(
        self,
        values: pd.Series | pd.DataFrame,
        *,
        target: pd.Series | pd.DataFrame | None = None,
        groups: pd.Series | pd.DataFrame | None = None,
        exposures: pd.Series | pd.DataFrame | list[pd.Series | pd.DataFrame] | None = None,
    ) -> pd.Series | pd.DataFrame:
        return self.fit(values, target).transform(values, target=target, groups=groups, exposures=exposures)


__all__ = ["FactorPreprocessor", "preprocess_factor_panel"]
