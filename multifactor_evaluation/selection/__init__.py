from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from tiger_factors.multifactor_evaluation._inputs import coerce_factor_series
from tiger_factors.multifactor_evaluation._inputs import coerce_price_panel
from tiger_factors.multifactor_evaluation.redundancy import cluster_factors
from tiger_factors.multifactor_evaluation.redundancy import factor_correlation_matrix
from tiger_factors.multifactor_evaluation.redundancy import ic_correlation_matrix
from tiger_factors.multifactor_evaluation.redundancy import ic_time_series


def greedy_select_by_correlation(
    scores: dict[str, float],
    corr: pd.DataFrame,
    threshold: float,
) -> list[str]:
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    selected: list[str] = []

    for name, _ in ordered:
        if not selected:
            selected.append(name)
            continue
        too_correlated = any(abs(float(corr.loc[name, picked])) >= threshold for picked in selected)
        if not too_correlated:
            selected.append(name)

    return selected


def select_non_redundant_factors(
    factors: dict[str, pd.Series | pd.DataFrame],
    scores: dict[str, float],
    *,
    threshold: float = 0.75,
    standardize: bool = True,
) -> list[str]:
    corr = factor_correlation_matrix(factors, standardize=standardize)
    return greedy_select_by_correlation(scores, corr, threshold=threshold)


def select_ic_coherent_factors(
    factors: dict[str, pd.Series | pd.DataFrame],
    prices: pd.DataFrame,
    *,
    horizon: int = 1,
    min_names: int | None = 10,
    threshold: float = 0.75,
    scores: dict[str, float] | None = None,
) -> list[str]:
    ic_corr = ic_correlation_matrix(factors, coerce_price_panel(prices), horizon=horizon, min_names=min_names)
    if scores is None:
        scores = {name: float(np.nan_to_num(ic_corr.loc[name].abs().mean(), nan=0.0)) for name in ic_corr.columns}
    return greedy_select_by_correlation(scores, ic_corr, threshold=threshold)


__all__ = [
    "cluster_factors",
    "factor_correlation_matrix",
    "greedy_select_by_correlation",
    "ic_correlation_matrix",
    "ic_time_series",
    "select_ic_coherent_factors",
    "select_non_redundant_factors",
]
