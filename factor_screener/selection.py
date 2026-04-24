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

    corr = factor_correlation_matrix(factors, standardize=cfg.standardize)
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
    "select_factors_by_marginal_gain",
    "select_ic_coherent_factors",
    "select_non_redundant_factors",
]
