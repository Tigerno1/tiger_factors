from __future__ import annotations

import pandas as pd

from tiger_factors.factor_screener.selection import FactorMarginalSelectionConfig
from tiger_factors.factor_screener.selection import cluster_factors
from tiger_factors.factor_screener.selection import factor_correlation_matrix
from tiger_factors.factor_screener.selection import greedy_select_by_correlation
from tiger_factors.factor_screener.selection import ic_correlation_matrix
from tiger_factors.factor_screener.selection import ic_time_series
from tiger_factors.factor_screener.selection import select_by_average_correlation
from tiger_factors.factor_screener.selection import select_by_graph_independent_set
from tiger_factors.factor_screener.selection import select_ic_by_average_correlation
from tiger_factors.factor_screener.selection import select_ic_by_graph_independent_set
from tiger_factors.factor_screener.selection import select_cluster_representatives
from tiger_factors.factor_screener.selection import select_cluster_representatives_from_correlation_matrix
from tiger_factors.factor_screener.selection import select_factors_by_marginal_gain
from tiger_factors.factor_screener.selection import select_ic_coherent_factors
from tiger_factors.factor_screener.selection import select_non_redundant_factors


def run_correlation_screening(
    factors: dict[str, pd.Series | pd.DataFrame],
    scores: dict[str, float],
    *,
    method: str = "greedy",
    threshold: float = 0.75,
    standardize: bool = True,
    metrics: pd.DataFrame | None = None,
    marginal_config: FactorMarginalSelectionConfig | None = None,
) -> list[str]:
    method_name = str(method).strip().lower()
    if method_name in {"greedy", "default", "correlation", "non_redundant"}:
        return select_non_redundant_factors(factors, scores, threshold=threshold, standardize=standardize)
    if method_name in {"average", "avg", "mean", "mean_corr", "average_corr"}:
        return select_by_average_correlation(factors, scores, threshold=threshold, standardize=standardize)
    if method_name in {"cluster", "clustered", "cluster_representative"}:
        return select_cluster_representatives(factors, scores, threshold=threshold, standardize=standardize)
    if method_name in {"graph", "independent_set", "max_independent_set", "graph_prune"}:
        return select_by_graph_independent_set(factors, scores, threshold=threshold, standardize=standardize)
    if method_name in {"marginal", "marginal_gain", "incremental", "stepwise"}:
        if metrics is None:
            raise ValueError("metrics is required for marginal_gain correlation screening")
        return select_factors_by_marginal_gain(factors, metrics, config=marginal_config)
    raise ValueError(f"unknown correlation screening method: {method!r}")


def run_ic_correlation_screening(
    factors: dict[str, pd.Series | pd.DataFrame],
    prices: pd.DataFrame,
    *,
    method: str = "greedy",
    horizon: int = 1,
    min_names: int | None = 10,
    threshold: float = 0.75,
    scores: dict[str, float] | None = None,
    metrics: pd.DataFrame | None = None,
    marginal_config: FactorMarginalSelectionConfig | None = None,
) -> list[str]:
    method_name = str(method).strip().lower()
    if method_name in {"greedy", "default", "ic", "ic_correlation", "non_redundant"}:
        return select_ic_coherent_factors(
            factors,
            prices,
            horizon=horizon,
            min_names=min_names,
            threshold=threshold,
            scores=scores,
        )
    if method_name in {"average", "avg", "mean", "mean_corr", "average_corr"}:
        return select_ic_by_average_correlation(
            factors,
            prices,
            horizon=horizon,
            min_names=min_names,
            threshold=threshold,
            scores=scores,
        )
    if method_name in {"cluster", "clustered", "cluster_representative"}:
        ic_corr = ic_correlation_matrix(factors, prices, horizon=horizon, min_names=min_names)
        if scores is None:
            scores = {name: float(ic_corr.loc[name].abs().mean()) for name in ic_corr.columns}
        return select_cluster_representatives_from_correlation_matrix(ic_corr, scores, threshold=threshold)
    if method_name in {"graph", "independent_set", "max_independent_set", "graph_prune"}:
        return select_ic_by_graph_independent_set(
            factors,
            prices,
            horizon=horizon,
            min_names=min_names,
            threshold=threshold,
            scores=scores,
        )
    if method_name in {"marginal", "marginal_gain", "incremental", "stepwise"}:
        if metrics is None:
            raise ValueError("metrics is required for marginal_gain IC screening")
        return select_factors_by_marginal_gain(factors, metrics, config=marginal_config)
    raise ValueError(f"unknown IC correlation screening method: {method!r}")


__all__ = [
    "FactorMarginalSelectionConfig",
    "cluster_factors",
    "factor_correlation_matrix",
    "greedy_select_by_correlation",
    "ic_correlation_matrix",
    "ic_time_series",
    "select_by_average_correlation",
    "select_by_graph_independent_set",
    "run_correlation_screening",
    "run_ic_correlation_screening",
    "select_cluster_representatives",
    "select_cluster_representatives_from_correlation_matrix",
    "select_factors_by_marginal_gain",
    "select_ic_by_average_correlation",
    "select_ic_by_graph_independent_set",
    "select_ic_coherent_factors",
    "select_non_redundant_factors",
]
