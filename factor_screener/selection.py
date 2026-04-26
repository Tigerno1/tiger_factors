from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from typing import Mapping

import numpy as np
import pandas as pd

from tiger_factors.factor_screener.redundancy import cluster_factors
from tiger_factors.factor_screener.redundancy import factor_correlation_matrix
from tiger_factors.factor_screener.redundancy import ic_correlation_matrix
from tiger_factors.factor_screener.redundancy import ic_time_series
from tiger_factors.multifactor_evaluation._inputs import coerce_price_panel


@dataclass(frozen=True)
class FactorMarginalSelectionConfig:
    score_fields: tuple[str, ...] = ("directional_fitness", "directional_ic_ir", "directional_sharpe")
    score_weights: tuple[float, ...] = (0.50, 0.25, 0.25)
    fallback_score_field: str = "selected_score"
    penalty_fields: tuple[str, ...] = ("turnover", "max_drawdown")
    penalty_weights: tuple[float, ...] = (0.20, 0.10)
    corr_threshold: float = 0.75
    corr_weight: float = 0.50
    min_improvement: float = 0.0
    min_base_score: float | None = None
    standardize: bool = True


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


def _select_from_correlation_matrix(
    corr: pd.DataFrame,
    scores: dict[str, float],
    *,
    threshold: float,
    comparator: str = "max",
) -> list[str]:
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    selected: list[str] = []

    for name, _ in ordered:
        if name not in corr.index or name not in corr.columns:
            continue
        if not selected:
            selected.append(name)
            continue

        selected_values = [abs(float(corr.loc[name, picked])) for picked in selected if picked in corr.columns]
        selected_values = [value for value in selected_values if pd.notna(value)]
        if not selected_values:
            selected.append(name)
            continue

        if comparator == "max":
            measure = max(selected_values)
        elif comparator == "mean":
            measure = float(np.mean(selected_values))
        elif comparator == "sum":
            measure = float(np.sum(selected_values))
        else:
            raise ValueError(f"unknown comparator: {comparator!r}")

        if measure < threshold:
            selected.append(name)

    return selected


def select_by_average_correlation(
    factors: dict[str, pd.Series | pd.DataFrame],
    scores: dict[str, float],
    *,
    threshold: float = 0.75,
    standardize: bool = True,
) -> list[str]:
    corr = factor_correlation_matrix(factors)
    return _select_from_correlation_matrix(corr, scores, threshold=threshold, comparator="mean")


def select_by_graph_independent_set(
    factors: dict[str, pd.Series | pd.DataFrame],
    scores: dict[str, float],
    *,
    threshold: float = 0.75,
    standardize: bool = True,
) -> list[str]:
    corr = factor_correlation_matrix(factors)
    return select_by_graph_independent_set_from_correlation_matrix(corr, scores, threshold=threshold)


def select_by_graph_independent_set_from_correlation_matrix(
    corr: pd.DataFrame,
    scores: dict[str, float],
    *,
    threshold: float = 0.75,
) -> list[str]:
    if corr.empty:
        return []
    degree: dict[str, int] = {}
    for name in corr.index:
        row = corr.loc[name].dropna().abs()
        degree[str(name)] = int((row >= threshold).sum() - 1)

    remaining = {str(name) for name in corr.index}
    selected: list[str] = []
    while remaining:
        def node_key(name: str) -> tuple[float, float, str]:
            score = float(scores.get(name, 0.0))
            penalty = float(degree.get(name, 0))
            return (score / (1.0 + penalty), score, name)

        best = max(remaining, key=node_key)
        selected.append(best)
        neighbors = set()
        if best in corr.index:
            row = corr.loc[best].dropna().abs()
            neighbors = {str(name) for name, value in row.items() if name in remaining and float(value) >= threshold}
        remaining.difference_update(neighbors | {best})
    return selected


def select_non_redundant_factors(
    factors: dict[str, pd.Series | pd.DataFrame],
    scores: dict[str, float],
    *,
    threshold: float = 0.75,
    standardize: bool = True,
) -> list[str]:
    corr = factor_correlation_matrix(factors)
    return greedy_select_by_correlation(scores, corr, threshold=threshold)


def select_cluster_representatives(
    factors: dict[str, pd.Series | pd.DataFrame],
    scores: dict[str, float],
    *,
    threshold: float = 0.75,
    standardize: bool = True,
) -> list[str]:
    corr = factor_correlation_matrix(factors)
    return select_cluster_representatives_from_correlation_matrix(corr, scores, threshold=threshold)


def select_cluster_representatives_from_correlation_matrix(
    corr: pd.DataFrame,
    scores: dict[str, float],
    *,
    threshold: float = 0.75,
) -> list[str]:
    clusters = cluster_factors(corr, threshold=threshold)
    if not clusters:
        return []

    cluster_members: dict[int, list[str]] = {}
    for name, cluster_id in clusters.items():
        cluster_members.setdefault(int(cluster_id), []).append(str(name))

    cluster_order: list[tuple[float, int]] = []
    for cluster_id, members in cluster_members.items():
        cluster_score = max(float(scores.get(member, 0.0)) for member in members) if members else 0.0
        cluster_order.append((cluster_score, cluster_id))
    cluster_order.sort(reverse=True)

    selected: list[str] = []
    for _, cluster_id in cluster_order:
        members = cluster_members.get(cluster_id, [])
        if not members:
            continue
        best_member = max(members, key=lambda name: float(scores.get(name, 0.0)))
        selected.append(best_member)
    return selected


def select_ic_by_average_correlation(
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
    return _select_from_correlation_matrix(ic_corr, scores, threshold=threshold, comparator="mean")


def select_ic_by_graph_independent_set(
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
    return select_by_graph_independent_set_from_correlation_matrix(ic_corr, scores, threshold=threshold)


def _weighted_metric_score(
    frame: pd.DataFrame,
    *,
    fields: tuple[str, ...],
    weights: tuple[float, ...],
    fallback_field: str,
) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    if len(fields) != len(weights):
        raise ValueError("fields and weights must have the same length")

    total = pd.Series(0.0, index=frame.index, dtype=float)
    total_weight = 0.0
    for field, weight in zip(fields, weights):
        if field not in frame.columns:
            continue
        values = pd.to_numeric(frame[field], errors="coerce")
        total = total.add(values.fillna(0.0) * float(weight), fill_value=0.0)
        total_weight += float(weight)

    if total_weight > 0:
        total = total / total_weight
        return total

    if fallback_field in frame.columns:
        return pd.to_numeric(frame[fallback_field], errors="coerce")
    if "fitness" in frame.columns:
        return pd.to_numeric(frame["fitness"], errors="coerce")
    return pd.Series(1.0, index=frame.index, dtype=float)


def _penalty_score(
    frame: pd.DataFrame,
    *,
    fields: tuple[str, ...],
    weights: tuple[float, ...],
) -> pd.Series:
    if frame.empty or not fields:
        return pd.Series(0.0, index=frame.index if not frame.empty else None, dtype=float)
    if len(fields) != len(weights):
        raise ValueError("penalty fields and weights must have the same length")

    penalty = pd.Series(0.0, index=frame.index, dtype=float)
    for field, weight in zip(fields, weights):
        if field not in frame.columns:
            continue
        values = pd.to_numeric(frame[field], errors="coerce").abs()
        penalty = penalty.add(values.fillna(0.0) * float(weight), fill_value=0.0)
    return penalty


def _mean_pairwise_abs_corr(corr: pd.DataFrame, names: list[str]) -> float:
    if len(names) < 2 or corr.empty:
        return 0.0
    values: list[float] = []
    for idx, left in enumerate(names):
        if left not in corr.index:
            continue
        for right in names[idx + 1 :]:
            if right not in corr.columns:
                continue
            value = corr.loc[left, right]
            if pd.notna(value):
                values.append(abs(float(value)))
    return float(np.mean(values)) if values else 0.0


def select_factors_by_marginal_gain(
    factors: dict[str, pd.Series | pd.DataFrame],
    metrics: pd.DataFrame | Mapping[str, object],
    *,
    config: FactorMarginalSelectionConfig | None = None,
    key_column: str = "factor_name",
) -> list[str]:
    cfg = config if config is not None else FactorMarginalSelectionConfig()
    metric_frame = metrics if isinstance(metrics, pd.DataFrame) else pd.DataFrame.from_dict(metrics, orient="index")
    if metric_frame.empty:
        return []
    if key_column not in metric_frame.columns:
        raise KeyError(f"metrics must contain a {key_column!r} column")

    metric_frame = metric_frame.copy().reset_index(drop=True)
    metric_frame[key_column] = metric_frame[key_column].astype(str)
    metric_frame = metric_frame.loc[metric_frame[key_column].isin(factors.keys())]
    if metric_frame.empty:
        return []

    scores = _weighted_metric_score(
        metric_frame,
        fields=cfg.score_fields,
        weights=cfg.score_weights,
        fallback_field=cfg.fallback_score_field,
    )
    penalties = _penalty_score(
        metric_frame,
        fields=cfg.penalty_fields,
        weights=cfg.penalty_weights,
    )
    scores.index = metric_frame[key_column].astype(str)
    penalties.index = metric_frame[key_column].astype(str)

    corr = factor_correlation_matrix(factors)
    ordered = scores.sort_values(ascending=False).index.astype(str).tolist()

    selected: list[str] = []
    current_objective = 0.0
    for name in ordered:
        if cfg.min_base_score is not None and pd.notna(scores.get(name)) and float(scores.get(name)) < cfg.min_base_score:
            continue

        trial = selected + [name]
        mean_score = float(pd.to_numeric(scores.reindex(trial), errors="coerce").dropna().mean()) if trial else 0.0
        mean_penalty = float(pd.to_numeric(penalties.reindex(trial), errors="coerce").dropna().mean()) if trial else 0.0
        corr_penalty = _mean_pairwise_abs_corr(corr, trial)
        trial_objective = mean_score - cfg.corr_weight * corr_penalty - mean_penalty

        if not selected:
            if trial_objective >= cfg.min_improvement:
                selected.append(name)
                current_objective = trial_objective
            continue

        if trial_objective >= current_objective + cfg.min_improvement:
            selected.append(name)
            current_objective = trial_objective

    return selected


__all__ = [
    "cluster_factors",
    "factor_correlation_matrix",
    "FactorMarginalSelectionConfig",
    "greedy_select_by_correlation",
    "ic_correlation_matrix",
    "ic_time_series",
    "select_by_average_correlation",
    "select_by_graph_independent_set",
    "select_by_graph_independent_set_from_correlation_matrix",
    "select_cluster_representatives",
    "select_cluster_representatives_from_correlation_matrix",
    "select_factors_by_marginal_gain",
    "select_ic_by_average_correlation",
    "select_ic_by_graph_independent_set",
    "select_non_redundant_factors",
]
