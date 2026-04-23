from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tiger_factors.factor_evaluation.utils import _rowwise_cross_sectional_corr


@dataclass(frozen=True)
class FactorEvaluation:
    ic_mean: float
    ic_ir: float
    rank_ic_mean: float
    sharpe: float
    turnover: float
    decay_score: float
    capacity_score: float
    correlation_penalty: float
    regime_robustness: float
    out_of_sample_stability: float
    fitness: float
    factor_autocorr_mean: float = 0.0
    rank_factor_autocorr_mean: float = 0.0
    benchmark_alpha: float = 0.0
    benchmark_beta: float = 0.0
    benchmark_r2: float = 0.0
    benchmark_n_obs: int = 0


def _align_frames(factor: pd.DataFrame, forward_returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_index = factor.index.intersection(forward_returns.index)
    common_columns = factor.columns.intersection(forward_returns.columns)
    return factor.loc[common_index, common_columns], forward_returns.loc[common_index, common_columns]


def _cross_sectional_corr(a: pd.Series, b: pd.Series, *, rank: bool = False) -> float:
    joined = pd.concat([a, b], axis=1).dropna()
    if len(joined) < 3:
        return np.nan
    if rank:
        joined = joined.rank()
    x = pd.to_numeric(joined.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(joined.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    x = x[mask]
    y = y[mask]
    x_std = float(np.std(x, ddof=0))
    y_std = float(np.std(y, ddof=0))
    if x_std <= 1e-12 or y_std <= 1e-12:
        return np.nan
    x_centered = x - float(np.mean(x))
    y_centered = y - float(np.mean(y))
    denom = float(np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2)))
    if denom <= 1e-12:
        return np.nan
    return float(np.sum(x_centered * y_centered) / denom)


def _safe_mean(values: pd.Series) -> float:
    return float(values.dropna().mean()) if not values.dropna().empty else 0.0


def _annualized_sharpe(returns: pd.Series, annualization: int = 252) -> float:
    series = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return 0.0
    std = float(series.std(ddof=0))
    if std <= 1e-12:
        return 0.0
    return float(series.mean() / std * np.sqrt(annualization))


def _factor_autocorr_series(factor: pd.DataFrame, *, rank: bool = False) -> pd.Series:
    if factor.empty or len(factor.index) < 2:
        return pd.Series(dtype=float)
    ordered = factor.sort_index()
    if rank:
        ordered = ordered.rank(axis=1, pct=True)
    prev = ordered.shift(1)
    correlations = _rowwise_cross_sectional_corr(ordered.iloc[1:], prev.iloc[1:])
    correlations.index = pd.DatetimeIndex(correlations.index)
    return correlations


def _fit_linear_regression(y: pd.Series, x: pd.Series) -> tuple[float, float, float, int]:
    joined = pd.concat([pd.to_numeric(y, errors="coerce"), pd.to_numeric(x, errors="coerce")], axis=1).dropna()
    if len(joined) < 3:
        return 0.0, 0.0, 0.0, int(len(joined))
    y_values = joined.iloc[:, 0].to_numpy(dtype=float)
    x_values = joined.iloc[:, 1].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x_values)), x_values])
    coef, *_ = np.linalg.lstsq(design, y_values, rcond=None)
    fitted = design @ coef
    residual = y_values - fitted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_values - float(np.mean(y_values))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return float(coef[0]), float(coef[1]), float(r2), int(len(joined))


def _build_quantile_long_short_returns(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    *,
    quantiles: int = 5,
) -> pd.Series:
    factor, forward_returns = _align_frames(factor, forward_returns)
    if factor.empty or forward_returns.empty:
        return pd.Series(dtype=float)
    if factor.shape[1] < 2:
        return pd.Series(dtype=float)

    effective_quantiles = max(2, min(int(quantiles), int(factor.shape[1])))

    factor_ranks = factor.rank(axis=1, method="first", pct=True)
    quantile_membership = np.ceil(factor_ranks * effective_quantiles).clip(1, effective_quantiles)
    top = forward_returns.where(quantile_membership.eq(effective_quantiles)).mean(axis=1)
    bottom = forward_returns.where(quantile_membership.eq(1)).mean(axis=1)
    long_short = top - bottom
    long_short.name = "long_short"
    return long_short


def _normalize_group_labels(
    factor: pd.DataFrame,
    group_labels: pd.DataFrame | pd.Series,
) -> pd.DataFrame:
    if isinstance(group_labels, pd.Series):
        if group_labels.index.intersection(factor.columns).empty:
            raise ValueError("group_labels series must be indexed by factor columns.")
        labels = group_labels.reindex(factor.columns)
        return pd.DataFrame(
            np.repeat(labels.to_numpy()[None, :], len(factor.index), axis=0),
            index=factor.index,
            columns=factor.columns,
        )

    aligned_index = factor.index.intersection(group_labels.index)
    aligned_columns = factor.columns.intersection(group_labels.columns)
    return group_labels.loc[aligned_index, aligned_columns]


def _summarize_group(
    group_name: str,
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    *,
    benchmark_returns: pd.Series | None = None,
    split_index: int | None = None,
) -> dict[str, float | int | str]:
    evaluation = evaluate_factor_panel(
        factor,
        forward_returns,
        benchmark_returns=benchmark_returns,
        split_index=split_index,
    )
    observations = int(factor.notna().sum().sum())
    summary = {
        "group": group_name,
        "observations": observations,
        "ic_mean": evaluation.ic_mean,
        "ic_ir": evaluation.ic_ir,
        "rank_ic_mean": evaluation.rank_ic_mean,
        "sharpe": evaluation.sharpe,
        "turnover": evaluation.turnover,
        "decay_score": evaluation.decay_score,
        "capacity_score": evaluation.capacity_score,
        "correlation_penalty": evaluation.correlation_penalty,
        "regime_robustness": evaluation.regime_robustness,
        "out_of_sample_stability": evaluation.out_of_sample_stability,
        "fitness": evaluation.fitness,
        "factor_autocorr_mean": evaluation.factor_autocorr_mean,
        "rank_factor_autocorr_mean": evaluation.rank_factor_autocorr_mean,
        "benchmark_alpha": evaluation.benchmark_alpha,
        "benchmark_beta": evaluation.benchmark_beta,
        "benchmark_r2": evaluation.benchmark_r2,
        "benchmark_n_obs": evaluation.benchmark_n_obs,
    }
    return summary


def factor_autocorrelation(factor: pd.DataFrame) -> pd.Series:
    return _factor_autocorr_series(factor)


def rank_factor_autocorrelation(factor: pd.DataFrame) -> pd.Series:
    return _factor_autocorr_series(factor, rank=True)


def alpha_beta_regression(
    factor_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> dict[str, float | int]:
    alpha, beta, r2, n_obs = _fit_linear_regression(factor_returns, benchmark_returns)
    return {
        "alpha": alpha,
        "beta": beta,
        "r2": r2,
        "n_obs": n_obs,
    }


def evaluate_factor_groups(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    group_labels: pd.DataFrame | pd.Series,
    *,
    benchmark_returns: pd.Series | None = None,
    split_index: int | None = None,
) -> pd.DataFrame:
    factor, forward_returns = _align_frames(factor, forward_returns)
    normalized_groups = _normalize_group_labels(factor, group_labels)
    if normalized_groups.empty:
        return pd.DataFrame()

    records: list[dict[str, float | int | str]] = []
    unique_groups = pd.unique(normalized_groups.stack().dropna())
    for group_name in unique_groups:
        mask = normalized_groups.eq(group_name)
        group_factor = factor.where(mask)
        group_forward_returns = forward_returns.where(mask)
        records.append(
            _summarize_group(
                str(group_name),
                group_factor,
                group_forward_returns,
                benchmark_returns=benchmark_returns,
                split_index=split_index,
            )
        )
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values(["fitness", "ic_mean"], ascending=False).reset_index(drop=True)


def evaluate_factor_panel(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    *,
    existing_factors: list[pd.DataFrame] | None = None,
    liquidity_panel: pd.DataFrame | None = None,
    regime_labels: pd.Series | None = None,
    split_index: int | None = None,
    benchmark_returns: pd.Series | None = None,
) -> FactorEvaluation:
    factor, forward_returns = _align_frames(factor, forward_returns)
    ic_series = _rowwise_cross_sectional_corr(factor, forward_returns)
    rank_ic_series = _rowwise_cross_sectional_corr(factor, forward_returns, rank=True)
    ic_mean = _safe_mean(ic_series)
    ic_std = float(ic_series.dropna().std(ddof=0)) if not ic_series.dropna().empty else 0.0
    ic_ir = float(ic_mean / (ic_std + 1e-6))
    rank_ic_mean = _safe_mean(rank_ic_series)
    long_short_returns = _build_quantile_long_short_returns(factor, forward_returns)
    sharpe = _annualized_sharpe(long_short_returns)
    factor_autocorr_mean = _safe_mean(_factor_autocorr_series(factor))
    rank_factor_autocorr_mean = _safe_mean(_factor_autocorr_series(factor, rank=True))

    factor_rank = factor.rank(axis=1, pct=True)
    turnover = float(factor_rank.diff().abs().mean(axis=1).mean())
    decay_score = float(1.0 / (1.0 + turnover * 5.0))

    correlation_penalty = 0.0
    for existing in existing_factors or []:
        left, right = _align_frames(factor, existing)
        corr_series = _rowwise_cross_sectional_corr(left, right)
        correlation_penalty = max(correlation_penalty, float(abs(corr_series.mean())))

    capacity_score = 1.0
    if liquidity_panel is not None:
        left, liq = _align_frames(factor.abs(), liquidity_panel)
        demand = left.rank(axis=1, pct=True).sub(0.5).abs()
        usage = demand / (liq.abs() + 1e-6)
        capacity_score = float(1.0 / (1.0 + usage.mean(axis=1).mean()))

    regime_robustness = 1.0
    if regime_labels is not None:
        labels = regime_labels.reindex(ic_series.index)
        regime_means = ic_series.groupby(labels).mean().dropna()
        if len(regime_means) > 1:
            regime_robustness = float(1.0 / (1.0 + regime_means.std(ddof=0)))

    out_of_sample_stability = 1.0
    if split_index is None:
        split_index = len(ic_series) // 2
    if split_index and 0 < split_index < len(ic_series):
        train_ic = _safe_mean(ic_series.iloc[:split_index])
        test_ic = _safe_mean(ic_series.iloc[split_index:])
        out_of_sample_stability = float(1.0 / (1.0 + abs(train_ic - test_ic)))

    benchmark_alpha = 0.0
    benchmark_beta = 0.0
    benchmark_r2 = 0.0
    benchmark_n_obs = 0
    if benchmark_returns is not None and not long_short_returns.empty:
        benchmark_series = pd.to_numeric(benchmark_returns, errors="coerce")
        alpha, beta, r2, n_obs = _fit_linear_regression(long_short_returns, benchmark_series)
        benchmark_alpha = alpha
        benchmark_beta = beta
        benchmark_r2 = r2
        benchmark_n_obs = n_obs

    fitness = float(
        (ic_mean + rank_ic_mean + 0.5 * ic_ir)
        * decay_score
        * capacity_score
        * regime_robustness
        * out_of_sample_stability
        / (1.0 + turnover + correlation_penalty)
    )
    return FactorEvaluation(
        ic_mean=ic_mean,
        ic_ir=ic_ir,
        rank_ic_mean=rank_ic_mean,
        sharpe=sharpe,
        turnover=turnover,
        decay_score=decay_score,
        capacity_score=capacity_score,
        correlation_penalty=correlation_penalty,
        regime_robustness=regime_robustness,
        out_of_sample_stability=out_of_sample_stability,
        fitness=fitness,
        factor_autocorr_mean=factor_autocorr_mean,
        rank_factor_autocorr_mean=rank_factor_autocorr_mean,
        benchmark_alpha=benchmark_alpha,
        benchmark_beta=benchmark_beta,
        benchmark_r2=benchmark_r2,
        benchmark_n_obs=benchmark_n_obs,
    )
