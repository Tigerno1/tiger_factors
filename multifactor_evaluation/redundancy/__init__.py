from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from tiger_factors.factor_evaluation.horizon import HoldingPeriodAnalyzer
from tiger_factors.multifactor_evaluation._inputs import coerce_factor_series
from tiger_factors.multifactor_evaluation._inputs import coerce_price_panel


def factor_correlation_matrix(factor_dict: dict[str, pd.Series | pd.DataFrame]) -> pd.DataFrame:
    aligned: list[pd.Series] = []
    for name, factor in factor_dict.items():
        aligned.append(coerce_factor_series(factor).rename(name))
    combined = pd.concat(aligned, axis=1).dropna()
    return combined.corr()


def ic_time_series(
    factor: pd.Series | pd.DataFrame,
    prices: pd.DataFrame,
    *,
    horizon: int = 1,
    min_names: int | None = 10,
) -> pd.Series:
    analyzer = HoldingPeriodAnalyzer(
        coerce_factor_series(factor),
        coerce_price_panel(prices),
        min_names=min_names,
    )
    return analyzer._daily_ic_series(analyzer._forward_returns(horizon))


def ic_correlation_matrix(
    factor_dict: dict[str, pd.Series | pd.DataFrame],
    prices: pd.DataFrame,
    *,
    horizon: int = 1,
    min_names: int | None = 10,
) -> pd.DataFrame:
    ic_map: dict[str, pd.Series] = {}
    for name, factor in factor_dict.items():
        ic_map[name] = ic_time_series(factor, prices, horizon=horizon, min_names=min_names)
    return pd.DataFrame(ic_map).dropna().corr()


def cluster_factors(corr_matrix: pd.DataFrame, threshold: float = 0.7) -> dict[str, int]:
    if corr_matrix.empty:
        return {}
    if corr_matrix.shape[0] == 1:
        return {str(corr_matrix.index[0]): 1}

    matrix = corr_matrix.astype(float).clip(-1.0, 1.0)
    distance = 1.0 - matrix
    np.fill_diagonal(distance.values, 0.0)
    condensed = squareform(distance.values, checks=False)
    linkage_matrix = linkage(condensed, method="average")
    cluster_labels = fcluster(linkage_matrix, t=float(threshold), criterion="distance")
    return {str(name): int(label) for name, label in zip(corr_matrix.index, cluster_labels)}


__all__ = [
    "cluster_factors",
    "factor_correlation_matrix",
    "ic_correlation_matrix",
    "ic_time_series",
]
