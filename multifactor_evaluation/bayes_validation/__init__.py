from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm

from tiger_factors.multifactor_evaluation.redundancy import cluster_factors
from tiger_factors.multifactor_evaluation.redundancy import factor_correlation_matrix


@dataclass(frozen=True)
class BayesianMixtureResult:
    """Empirical-Bayes local-FDR result for factor families."""

    method: str
    alpha: float
    table: pd.DataFrame
    pi0: float
    alt_variance: float
    prior_variance: float
    discovered_count: int
    expected_false_discoveries: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "alpha": self.alpha,
            "pi0": self.pi0,
            "alt_variance": self.alt_variance,
            "prior_variance": self.prior_variance,
            "discovered_count": self.discovered_count,
            "expected_false_discoveries": self.expected_false_discoveries,
            "table": self.table.to_dict(orient="records"),
        }


@dataclass(frozen=True)
class HierarchicalBayesianResult:
    """Bayesian result that borrows evidence from factor clusters."""

    method: str
    alpha: float
    table: pd.DataFrame
    factor_pi0: float
    cluster_pi0: float | None
    factor_alt_variance: float
    cluster_alt_variance: float | None
    shared_evidence_weight: float
    discovered_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "alpha": self.alpha,
            "factor_pi0": self.factor_pi0,
            "cluster_pi0": self.cluster_pi0,
            "factor_alt_variance": self.factor_alt_variance,
            "cluster_alt_variance": self.cluster_alt_variance,
            "shared_evidence_weight": self.shared_evidence_weight,
            "discovered_count": self.discovered_count,
            "table": self.table.to_dict(orient="records"),
        }


@dataclass(frozen=True)
class RollingBayesianAlphaResult:
    """Rolling posterior-alpha summary for a time-evolving factor series."""

    method: str
    alpha: float
    window: int
    min_periods: int
    date_column: str
    alpha_column: str
    group_column: str | None
    table: pd.DataFrame
    average_ols_alpha: float
    average_posterior_alpha: float
    bayesian_fdr: float
    bayesian_fwer: float
    discovered_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "alpha": self.alpha,
            "window": self.window,
            "min_periods": self.min_periods,
            "date_column": self.date_column,
            "alpha_column": self.alpha_column,
            "group_column": self.group_column,
            "average_ols_alpha": self.average_ols_alpha,
            "average_posterior_alpha": self.average_posterior_alpha,
            "bayesian_fdr": self.bayesian_fdr,
            "bayesian_fwer": self.bayesian_fwer,
            "discovered_count": self.discovered_count,
            "table": self.table.to_dict(orient="records"),
        }


@dataclass(frozen=True)
class DynamicBayesianAlphaResult:
    """State-space Bayesian alpha path estimated with recursive updating."""

    method: str
    alpha: float
    date_column: str
    alpha_column: str
    group_column: str | None
    process_discount: float
    observation_variance: float
    process_variance: float
    table: pd.DataFrame
    average_ols_alpha: float
    average_posterior_alpha: float
    bayesian_fdr: float
    bayesian_fwer: float
    discovered_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "alpha": self.alpha,
            "date_column": self.date_column,
            "alpha_column": self.alpha_column,
            "group_column": self.group_column,
            "process_discount": self.process_discount,
            "observation_variance": self.observation_variance,
            "process_variance": self.process_variance,
            "average_ols_alpha": self.average_ols_alpha,
            "average_posterior_alpha": self.average_posterior_alpha,
            "bayesian_fdr": self.bayesian_fdr,
            "bayesian_fwer": self.bayesian_fwer,
            "discovered_count": self.discovered_count,
            "table": self.table.to_dict(orient="records"),
        }


@dataclass(frozen=True)
class AlphaHackingBayesianResult:
    """Bayesian posterior for paired in-sample / out-of-sample alpha paths."""

    method: str
    alpha: float
    date_column: str
    in_sample_column: str
    out_of_sample_column: str
    group_column: str | None
    bias_discount: float
    observation_discount: float
    table: pd.DataFrame
    average_in_sample_alpha: float
    average_out_of_sample_alpha: float
    average_posterior_alpha: float
    average_hacking_bias: float
    bayesian_fdr: float
    bayesian_fwer: float
    discovered_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "alpha": self.alpha,
            "date_column": self.date_column,
            "in_sample_column": self.in_sample_column,
            "out_of_sample_column": self.out_of_sample_column,
            "group_column": self.group_column,
            "bias_discount": self.bias_discount,
            "observation_discount": self.observation_discount,
            "average_in_sample_alpha": self.average_in_sample_alpha,
            "average_out_of_sample_alpha": self.average_out_of_sample_alpha,
            "average_posterior_alpha": self.average_posterior_alpha,
            "average_hacking_bias": self.average_hacking_bias,
            "bayesian_fdr": self.bayesian_fdr,
            "bayesian_fwer": self.bayesian_fwer,
            "discovered_count": self.discovered_count,
            "table": self.table.to_dict(orient="records"),
        }


def _coerce_factor_frame(
    factor_metrics: Mapping[str, Any] | pd.DataFrame,
    *,
    factor_name_column: str = "factor_name",
) -> pd.DataFrame:
    if isinstance(factor_metrics, pd.DataFrame):
        frame = factor_metrics.copy()
    else:
        values = list(factor_metrics.values())
        if values and all(np.isscalar(value) for value in values):
            frame = pd.DataFrame(
                {
                    factor_name_column: [str(key) for key in factor_metrics.keys()],
                    "p_value": [float(value) for value in values],
                }
            )
        else:
            frame = pd.DataFrame.from_dict(factor_metrics, orient="index").reset_index().rename(
                columns={"index": factor_name_column}
            )
    if factor_name_column not in frame.columns:
        frame = frame.reset_index().rename(columns={frame.index.name or "index": factor_name_column})
    if factor_name_column not in frame.columns:
        frame[factor_name_column] = frame.index.astype(str)
    return frame.copy()


def _infer_signed_z_scores(
    frame: pd.DataFrame,
    *,
    statistic_column: str = "statistic",
    p_value_column: str = "p_value",
    effect_column: str | None = "fitness",
) -> pd.Series:
    if statistic_column in frame.columns:
        return pd.to_numeric(frame[statistic_column], errors="coerce")

    if p_value_column not in frame.columns:
        raise KeyError(
            f"Need either {statistic_column!r} or {p_value_column!r} to infer Bayesian z-scores."
        )

    p_values = pd.to_numeric(frame[p_value_column], errors="coerce").clip(lower=1e-300, upper=1.0)
    z_abs = pd.Series(norm.isf(p_values / 2.0), index=frame.index, dtype="float64")
    if effect_column is not None and effect_column in frame.columns:
        signed_effect = pd.to_numeric(frame[effect_column], errors="coerce")
        sign = np.sign(signed_effect).replace(0.0, 1.0).fillna(1.0)
        return z_abs * sign
    return z_abs


def _fit_spike_slab_variance_model(
    z_scores: pd.Series,
    *,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    values = pd.to_numeric(z_scores, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if values.size == 0:
        return 1.0, 1.0, np.array([], dtype=float), np.array([], dtype=float)

    pi0 = 0.8
    alt_variance = max(float(np.var(values, ddof=0)), 1.05)
    if not np.isfinite(alt_variance):
        alt_variance = 1.5

    posterior_alt = np.zeros_like(values)
    posterior_null = np.ones_like(values)
    for _ in range(int(max_iter)):
        alt_scale = float(np.sqrt(max(alt_variance, 1.0 + 1e-9)))
        null_pdf = norm.pdf(values, loc=0.0, scale=1.0)
        alt_pdf = norm.pdf(values, loc=0.0, scale=alt_scale)
        mix_pdf = pi0 * null_pdf + (1.0 - pi0) * alt_pdf
        mix_pdf = np.clip(mix_pdf, 1e-300, np.inf)

        new_posterior_alt = ((1.0 - pi0) * alt_pdf) / mix_pdf
        new_posterior_null = 1.0 - new_posterior_alt

        effective_alt = float(new_posterior_alt.sum())
        if effective_alt <= 1e-12:
            break

        new_pi0 = float(np.clip(1.0 - new_posterior_alt.mean(), 0.0, 1.0))
        new_alt_variance = float(np.sum(new_posterior_alt * values**2) / effective_alt)
        new_alt_variance = max(new_alt_variance, 1.0 + 1e-9)

        if abs(new_pi0 - pi0) < tol and abs(new_alt_variance - alt_variance) < tol:
            pi0 = new_pi0
            alt_variance = new_alt_variance
            posterior_alt = new_posterior_alt
            posterior_null = new_posterior_null
            break

        pi0 = new_pi0
        alt_variance = new_alt_variance
        posterior_alt = new_posterior_alt
        posterior_null = new_posterior_null

    return float(pi0), float(alt_variance), posterior_null, posterior_alt


def _logit(prob: np.ndarray | pd.Series | float) -> np.ndarray:
    arr = np.asarray(prob, dtype=float)
    arr = np.clip(arr, 1e-12, 1.0 - 1e-12)
    return np.log(arr / (1.0 - arr))


def _sigmoid(score: np.ndarray | pd.Series | float) -> np.ndarray:
    arr = np.asarray(score, dtype=float)
    return 1.0 / (1.0 + np.exp(-arr))


def _cluster_evidence_table(
    frame: pd.DataFrame,
    *,
    cluster_column: str,
    z_scores: pd.Series,
    alpha: float,
    max_iter: int,
    tol: float,
) -> tuple[pd.DataFrame, float, float]:
    cluster_frame = frame.copy()
    cluster_frame["__z__"] = z_scores
    cluster_frame = cluster_frame.dropna(subset=[cluster_column, "__z__"])
    if cluster_frame.empty:
        empty = pd.DataFrame(columns=[cluster_column, "cluster_z_score", "cluster_posterior_null_prob", "cluster_posterior_signal_prob"])
        return empty, 1.0, 1.0

    cluster_summary = (
        cluster_frame.groupby(cluster_column, dropna=False)["__z__"]
        .mean()
        .rename("cluster_z_score")
        .reset_index()
        .sort_values(cluster_column)
        .reset_index(drop=True)
    )
    cluster_pi0 = 1.0
    cluster_alt_variance = 1.0
    cluster_posterior_null = np.ones(len(cluster_summary), dtype=float)
    cluster_posterior_alt = np.zeros(len(cluster_summary), dtype=float)
    cluster_posterior_mean = np.zeros(len(cluster_summary), dtype=float)
    cluster_posterior_sd = np.zeros(len(cluster_summary), dtype=float)
    cluster_pi0, cluster_alt_variance, cluster_posterior_null, cluster_posterior_alt = _fit_spike_slab_variance_model(
        cluster_summary["cluster_z_score"],
        max_iter=max_iter,
        tol=tol,
    )
    cluster_tau2 = max(cluster_alt_variance - 1.0, 0.0)
    cluster_shrinkage = 0.0 if cluster_alt_variance <= 0.0 else cluster_tau2 / cluster_alt_variance
    cluster_posterior_mean = cluster_shrinkage * cluster_summary["cluster_z_score"].to_numpy(dtype=float) * cluster_posterior_alt
    cluster_posterior_sd = np.sqrt(
        np.maximum(cluster_posterior_alt * (cluster_shrinkage + (cluster_shrinkage * cluster_summary["cluster_z_score"].to_numpy(dtype=float)) ** 2) - cluster_posterior_mean**2, 0.0)
    )
    cluster_summary["cluster_posterior_null_prob"] = cluster_posterior_null
    cluster_summary["cluster_posterior_signal_prob"] = cluster_posterior_alt
    cluster_summary["cluster_posterior_mean"] = cluster_posterior_mean
    cluster_summary["cluster_posterior_sd"] = cluster_posterior_sd
    cluster_summary["cluster_discovered"] = cluster_summary["cluster_posterior_null_prob"] <= float(alpha)
    return cluster_summary, float(cluster_pi0), float(cluster_alt_variance)


def fit_bayesian_mixture(
    factor_metrics: Mapping[str, Any] | pd.DataFrame,
    *,
    statistic_column: str = "statistic",
    p_value_column: str = "p_value",
    effect_column: str | None = "fitness",
    factor_name_column: str = "factor_name",
    alpha: float = 0.05,
    max_iter: int = 200,
    tol: float = 1e-8,
    cluster_threshold: float | None = None,
    factor_dict: Mapping[str, pd.Series | pd.DataFrame] | None = None,
) -> dict[str, Any]:
    frame = _coerce_factor_frame(factor_metrics, factor_name_column=factor_name_column)
    if factor_name_column not in frame.columns:
        raise KeyError(f"Missing factor name column {factor_name_column!r}")

    z_scores = _infer_signed_z_scores(
        frame,
        statistic_column=statistic_column,
        p_value_column=p_value_column,
        effect_column=effect_column,
    )
    pi0, alt_variance, posterior_null, posterior_alt = _fit_spike_slab_variance_model(
        z_scores,
        max_iter=max_iter,
        tol=tol,
    )

    tau2 = max(alt_variance - 1.0, 0.0)
    shrinkage = 0.0 if alt_variance <= 0.0 else tau2 / alt_variance
    posterior_alt_mean = shrinkage * z_scores.to_numpy(dtype=float)
    posterior_mean = posterior_alt * posterior_alt_mean
    posterior_alt_var = shrinkage
    posterior_variance = posterior_alt * (
        posterior_alt_var + posterior_alt_mean**2
    ) - posterior_mean**2
    posterior_variance = np.maximum(posterior_variance, 0.0)
    posterior_sd = np.sqrt(posterior_variance)

    table = frame.copy()
    table["z_score"] = z_scores
    table["posterior_null_prob"] = posterior_null
    table["posterior_signal_prob"] = posterior_alt
    table["posterior_mean"] = posterior_mean
    table["posterior_sd"] = posterior_sd
    table["discovered"] = table["posterior_null_prob"] <= float(alpha)
    table["local_fdr"] = table["posterior_null_prob"]

    cluster_map: dict[str, int] | None = None
    cluster_frame: pd.DataFrame | None = None
    if cluster_threshold is not None and factor_dict is not None:
        corr = factor_correlation_matrix(factor_dict)
        cluster_map = cluster_factors(corr, threshold=float(cluster_threshold))
        cluster_frame = pd.DataFrame(
            {
                factor_name_column: list(cluster_map.keys()),
                "cluster": list(cluster_map.values()),
            }
        ).sort_values(["cluster", factor_name_column]).reset_index(drop=True)
        table = table.merge(cluster_frame, on=factor_name_column, how="left")

    expected_false_discoveries = float(table.loc[table["discovered"], "posterior_null_prob"].sum())
    result_table = table.sort_values(
        ["discovered", "posterior_null_prob", p_value_column if p_value_column in table.columns else factor_name_column],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    result = BayesianMixtureResult(
        method="bayesian_mixture",
        alpha=float(alpha),
        table=result_table,
        pi0=float(pi0),
        alt_variance=float(alt_variance),
        prior_variance=float(tau2),
        discovered_count=int(table["discovered"].sum()),
        expected_false_discoveries=expected_false_discoveries,
    )
    return {
        "result": result,
        "table": result_table,
        "pi0": float(pi0),
        "alt_variance": float(alt_variance),
        "prior_variance": float(tau2),
        "clusters": cluster_frame.to_dict(orient="records") if cluster_frame is not None else None,
        "cluster_map": cluster_map,
    }


def fit_hierarchical_bayesian_mixture(
    factor_metrics: Mapping[str, Any] | pd.DataFrame,
    *,
    statistic_column: str = "statistic",
    p_value_column: str = "p_value",
    effect_column: str | None = "fitness",
    factor_name_column: str = "factor_name",
    alpha: float = 0.05,
    max_iter: int = 200,
    tol: float = 1e-8,
    cluster_threshold: float = 0.65,
    factor_dict: Mapping[str, pd.Series | pd.DataFrame] | None = None,
    shared_evidence_weight: float = 0.5,
) -> dict[str, Any]:
    fit = fit_bayesian_mixture(
        factor_metrics,
        statistic_column=statistic_column,
        p_value_column=p_value_column,
        effect_column=effect_column,
        factor_name_column=factor_name_column,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        cluster_threshold=cluster_threshold,
        factor_dict=factor_dict,
    )
    base_table = fit["table"].copy()
    cluster_map = fit["cluster_map"]
    if not cluster_map:
        result = HierarchicalBayesianResult(
            method="hierarchical_bayesian_mixture",
            alpha=float(alpha),
            table=base_table.copy(),
            factor_pi0=float(fit["pi0"]),
            cluster_pi0=None,
            factor_alt_variance=float(fit["alt_variance"]),
            cluster_alt_variance=None,
            shared_evidence_weight=float(shared_evidence_weight),
            discovered_count=int(base_table["discovered"].sum()),
        )
        return {
            "result": result,
            "table": result.table,
            "factor_fit": fit,
            "cluster_table": None,
            "factor_pi0": result.factor_pi0,
            "cluster_pi0": result.cluster_pi0,
        }

    cluster_column = "cluster"
    cluster_table, cluster_pi0, cluster_alt_variance = _cluster_evidence_table(
        base_table,
        cluster_column=cluster_column,
        z_scores=base_table["z_score"],
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
    )
    cluster_lookup = cluster_table[
        [
            cluster_column,
            "cluster_posterior_null_prob",
            "cluster_posterior_signal_prob",
            "cluster_posterior_mean",
            "cluster_posterior_sd",
        ]
    ].copy()
    cluster_lookup = cluster_lookup.rename(
        columns={
            "cluster_posterior_null_prob": "cluster_null_prob",
            "cluster_posterior_signal_prob": "cluster_signal_prob",
        }
    )

    merged = base_table.merge(cluster_lookup, on=cluster_column, how="left")
    merged["cluster_null_prob"] = merged["cluster_null_prob"].fillna(fit["pi0"])
    merged["cluster_signal_prob"] = merged["cluster_signal_prob"].fillna(1.0 - fit["pi0"])

    factor_odds = np.exp(_logit(merged["posterior_signal_prob"].to_numpy(dtype=float)))
    cluster_odds = np.exp(_logit(merged["cluster_signal_prob"].to_numpy(dtype=float)))
    combined_odds = factor_odds * np.power(cluster_odds, float(np.clip(shared_evidence_weight, 0.0, 1.0)))
    combined_signal_prob = _sigmoid(np.log(np.clip(combined_odds, 1e-12, np.inf)))
    combined_null_prob = 1.0 - combined_signal_prob

    merged["hierarchical_signal_prob"] = combined_signal_prob
    merged["hierarchical_null_prob"] = combined_null_prob
    merged["hierarchical_posterior_mean"] = (
        merged["posterior_mean"].to_numpy(dtype=float) * (1.0 - float(shared_evidence_weight))
        + merged["cluster_posterior_mean"].to_numpy(dtype=float) * float(shared_evidence_weight)
    )
    merged["hierarchical_posterior_sd"] = np.sqrt(
        (1.0 - float(shared_evidence_weight)) * np.square(merged["posterior_sd"].to_numpy(dtype=float))
        + float(shared_evidence_weight) * np.square(merged["cluster_posterior_sd"].to_numpy(dtype=float))
    )
    merged["hierarchical_discovered"] = merged["hierarchical_null_prob"] <= float(alpha)
    merged["hierarchical_local_fdr"] = merged["hierarchical_null_prob"]

    result = HierarchicalBayesianResult(
        method="hierarchical_bayesian_mixture",
        alpha=float(alpha),
        table=merged.sort_values(
            ["hierarchical_discovered", "hierarchical_null_prob", p_value_column if p_value_column in merged.columns else factor_name_column],
            ascending=[False, True, True],
        ).reset_index(drop=True),
        factor_pi0=float(fit["pi0"]),
        cluster_pi0=float(cluster_pi0),
        factor_alt_variance=float(fit["alt_variance"]),
        cluster_alt_variance=float(cluster_alt_variance),
        shared_evidence_weight=float(shared_evidence_weight),
        discovered_count=int(merged["hierarchical_discovered"].sum()),
    )
    return {
        "result": result,
        "table": result.table,
        "factor_fit": fit,
        "cluster_table": cluster_table,
        "factor_pi0": result.factor_pi0,
        "cluster_pi0": result.cluster_pi0,
    }


def bayesian_fdr(result: BayesianMixtureResult | pd.DataFrame) -> float:
    if isinstance(result, BayesianMixtureResult):
        table = result.table
    else:
        table = result
    if table.empty or "discovered" not in table.columns or "posterior_null_prob" not in table.columns:
        return 0.0
    discovered = table.loc[table["discovered"]]
    if discovered.empty:
        return 0.0
    return float(discovered["posterior_null_prob"].mean())


def bayesian_fwer(result: BayesianMixtureResult | HierarchicalBayesianResult | pd.DataFrame) -> float:
    if isinstance(result, BayesianMixtureResult):
        table = result.table
    elif isinstance(result, HierarchicalBayesianResult):
        table = result.table
    else:
        table = result
    if table.empty:
        return 0.0

    if "discovered" in table.columns and "posterior_null_prob" in table.columns:
        discovered = table.loc[table["discovered"], "posterior_null_prob"]
    elif "hierarchical_discovered" in table.columns and "hierarchical_null_prob" in table.columns:
        discovered = table.loc[table["hierarchical_discovered"], "hierarchical_null_prob"]
    else:
        null_column = "posterior_null_prob" if "posterior_null_prob" in table.columns else "hierarchical_null_prob" if "hierarchical_null_prob" in table.columns else None
        if null_column is None:
            return 0.0
        discovered = table[null_column]
    if discovered.empty:
        return 0.0
    discovered = pd.to_numeric(discovered, errors="coerce").dropna().clip(lower=0.0, upper=1.0)
    if discovered.empty:
        return 0.0
    return float(1.0 - np.prod(1.0 - discovered.to_numpy(dtype=float)))


def hierarchical_bayesian_fdr(result: HierarchicalBayesianResult | pd.DataFrame) -> float:
    if isinstance(result, HierarchicalBayesianResult):
        table = result.table
    else:
        table = result
    if table.empty or "hierarchical_discovered" not in table.columns or "hierarchical_null_prob" not in table.columns:
        return 0.0
    discovered = table.loc[table["hierarchical_discovered"]]
    if discovered.empty:
        return 0.0
    return float(discovered["hierarchical_null_prob"].mean())


def _rolling_spike_slab_summary(
    values: pd.Series,
    *,
    alpha: float,
    max_iter: int,
    tol: float,
) -> dict[str, float]:
    raw_values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if raw_values.size == 0:
        return {
            "pi0": 1.0,
            "alt_variance": 1.0,
            "prior_variance": 0.0,
            "scale": 1.0,
            "posterior_null_prob": 1.0,
            "posterior_signal_prob": 0.0,
            "posterior_alpha": 0.0,
            "posterior_alpha_sd": 0.0,
            "discovered_count": 0,
            "bayesian_fdr": 0.0,
            "bayesian_fwer": 0.0,
        }

    scale = float(np.nanstd(raw_values, ddof=0))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(max(np.nanmean(np.abs(raw_values)), 1e-6))
    scaled_values = pd.Series(raw_values / scale)

    pi0, alt_variance, posterior_null, posterior_alt = _fit_spike_slab_variance_model(
        scaled_values,
        max_iter=max_iter,
        tol=tol,
    )
    tau2 = max(alt_variance - 1.0, 0.0)
    shrinkage = 0.0 if alt_variance <= 0.0 else tau2 / alt_variance
    posterior_mean_scaled = posterior_alt * shrinkage * scaled_values.to_numpy(dtype=float)
    posterior_mean = posterior_mean_scaled * scale
    posterior_variance = posterior_alt * (
        shrinkage + (shrinkage * scaled_values.to_numpy(dtype=float)) ** 2
    ) - posterior_mean_scaled**2
    posterior_variance = np.maximum(posterior_variance, 0.0)
    posterior_sd = np.sqrt(posterior_variance) * scale
    discovered = posterior_null <= float(alpha)
    discovered_count = int(np.sum(discovered))
    bayes_fdr = float(np.mean(posterior_null[discovered])) if discovered_count else 0.0
    bayes_fwer = float(1.0 - np.prod(1.0 - posterior_null[discovered])) if discovered_count else 0.0
    return {
        "pi0": float(pi0),
        "alt_variance": float(alt_variance),
        "prior_variance": float(tau2),
        "scale": float(scale),
        "posterior_null_prob": float(np.mean(posterior_null)),
        "posterior_signal_prob": float(np.mean(posterior_alt)),
        "posterior_alpha": float(np.mean(posterior_mean)),
        "posterior_alpha_sd": float(np.mean(posterior_sd)),
        "discovered_count": discovered_count,
        "bayesian_fdr": bayes_fdr,
        "bayesian_fwer": bayes_fwer,
    }


def _dynamic_local_level_summary(
    values: pd.Series,
    *,
    alpha: float,
    process_discount: float,
    observation_variance: float | None,
) -> tuple[pd.DataFrame, float, float, float]:
    raw_values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if raw_values.size == 0:
        empty = pd.DataFrame(
            columns=[
                "time_index",
                "ols_alpha",
                "prior_alpha",
                "posterior_alpha",
                "posterior_alpha_sd",
                "posterior_positive_prob",
                "posterior_null_prob",
                "kalman_gain",
                "discovered",
            ]
        )
        return empty, 1.0, 1.0, 1.0

    obs_var = float(np.nanvar(raw_values, ddof=1)) if raw_values.size > 1 else float(np.nanvar(raw_values, ddof=0))
    if not np.isfinite(obs_var) or obs_var <= 1e-12:
        obs_var = float(max(np.nanmean(np.abs(raw_values)), 1e-6) ** 2)
    if observation_variance is not None and np.isfinite(float(observation_variance)) and float(observation_variance) > 0.0:
        obs_var = float(observation_variance)

    discount = float(np.clip(process_discount, 0.5, 0.999999))
    process_var = max(obs_var * (1.0 - discount) / discount, 1e-12)

    prior_mean = float(raw_values[0])
    prior_var = float(obs_var)
    rows: list[dict[str, float]] = []
    for idx, obs in enumerate(raw_values):
        prior_mean_t = prior_mean
        prior_var_t = prior_var + process_var
        kalman_gain = prior_var_t / (prior_var_t + obs_var)
        posterior_mean = prior_mean_t + kalman_gain * (obs - prior_mean_t)
        posterior_var = max((1.0 - kalman_gain) * prior_var_t, 1e-12)
        posterior_sd = float(np.sqrt(posterior_var))
        z_score = posterior_mean / posterior_sd if posterior_sd > 0 else 0.0
        posterior_positive_prob = float(norm.cdf(z_score))
        posterior_null_prob = float(np.clip(2.0 * norm.sf(abs(z_score)), 0.0, 1.0))
        rows.append(
            {
                "time_index": float(idx),
                "ols_alpha": float(obs),
                "prior_alpha": float(prior_mean_t),
                "posterior_alpha": float(posterior_mean),
                "posterior_alpha_sd": posterior_sd,
                "posterior_positive_prob": posterior_positive_prob,
                "posterior_null_prob": posterior_null_prob,
                "kalman_gain": float(kalman_gain),
                "discovered": posterior_null_prob <= float(alpha),
            }
        )
        prior_mean = posterior_mean
        prior_var = posterior_var

    table = pd.DataFrame(rows)
    discovered = table.loc[table["discovered"], "posterior_null_prob"]
    bayesian_fdr = float(discovered.mean()) if not discovered.empty else 0.0
    bayesian_fwer = float(1.0 - np.prod(1.0 - discovered.to_numpy(dtype=float))) if not discovered.empty else 0.0
    return table, float(obs_var), float(process_var), float(table["posterior_alpha"].mean())


def rolling_bayesian_alpha(
    alpha_series: Mapping[str, Any] | pd.DataFrame,
    *,
    date_column: str = "date_",
    alpha_column: str = "alpha",
    group_column: str | None = None,
    window: int = 60,
    min_periods: int | None = None,
    alpha: float = 0.05,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> dict[str, Any]:
    """Estimate a rolling posterior alpha series with empirical-Bayes shrinkage.

    Parameters
    ----------
    alpha_series:
        Long-form data with one alpha estimate per row. The frame must contain
        ``date_column`` and ``alpha_column``; an optional ``group_column`` lets
        you run separate rolling trajectories per factor / cluster / family.
    window:
        Number of trailing observations used in each posterior fit.
    min_periods:
        Minimum observations required before emitting a rolling estimate. If
        omitted, defaults to ``window``.
    """

    frame = _coerce_factor_frame(alpha_series, factor_name_column="factor_name")
    if date_column not in frame.columns:
        raise KeyError(f"Missing date column {date_column!r}")
    if alpha_column not in frame.columns:
        raise KeyError(f"Missing alpha column {alpha_column!r}")

    working = frame.copy()
    working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
    working[alpha_column] = pd.to_numeric(working[alpha_column], errors="coerce")
    working = working.dropna(subset=[date_column, alpha_column]).sort_values(date_column).reset_index(drop=True)
    if working.empty:
        empty = pd.DataFrame(columns=[
            date_column,
            "window_start",
            "window_end",
            "n_obs",
            "ols_alpha",
            "posterior_alpha",
            "posterior_alpha_sd",
            "posterior_null_prob",
            "posterior_signal_prob",
            "pi0",
            "alt_variance",
            "prior_variance",
            "scale",
            "discovered_count",
            "bayesian_fdr",
            "bayesian_fwer",
        ])
        return {
            "result": RollingBayesianAlphaResult(
                method="rolling_bayesian_alpha",
                alpha=float(alpha),
                window=int(window),
                min_periods=int(min_periods if min_periods is not None else window),
                date_column=date_column,
                alpha_column=alpha_column,
                group_column=group_column,
                table=empty,
                average_ols_alpha=0.0,
                average_posterior_alpha=0.0,
                bayesian_fdr=0.0,
                bayesian_fwer=0.0,
                discovered_count=0,
            ),
            "table": empty,
        }

    required = int(window) if min_periods is None else int(min_periods)
    if required <= 0:
        raise ValueError("min_periods/window must be positive")
    if int(window) <= 0:
        raise ValueError("window must be positive")

    rows: list[dict[str, Any]] = []
    group_items: list[tuple[Any, pd.DataFrame]]
    if group_column is None:
        group_items = [(None, working)]
    else:
        if group_column not in working.columns:
            raise KeyError(f"Missing group column {group_column!r}")
        group_items = list(working.groupby(group_column, dropna=False, sort=False))

    for group_value, group_frame in group_items:
        group_frame = group_frame.sort_values(date_column).reset_index(drop=True)
        if len(group_frame) < required:
            continue
        for end in range(required - 1, len(group_frame)):
            start = max(0, end - int(window) + 1)
            window_frame = group_frame.iloc[start : end + 1].copy()
            values = window_frame[alpha_column].dropna()
            if len(values) < required:
                continue
            summary = _rolling_spike_slab_summary(
                values,
                alpha=float(alpha),
                max_iter=max_iter,
                tol=tol,
            )
            row: dict[str, Any] = {
                date_column: window_frame[date_column].iloc[-1],
                "window_start": window_frame[date_column].iloc[0],
                "window_end": window_frame[date_column].iloc[-1],
                "n_obs": int(len(values)),
                "ols_alpha": float(values.mean()),
                **summary,
            }
            if group_column is not None:
                row[group_column] = group_value
            rows.append(row)

    result_table = pd.DataFrame(rows)
    if not result_table.empty:
        sort_columns = [date_column]
        if group_column is not None:
            sort_columns = [group_column, date_column]
        result_table = result_table.sort_values(sort_columns).reset_index(drop=True)

    if result_table.empty:
        average_ols_alpha = 0.0
        average_posterior_alpha = 0.0
        bayes_fdr = 0.0
        bayes_fwer = 0.0
        discovered_count = 0
    else:
        average_ols_alpha = float(result_table["ols_alpha"].mean())
        average_posterior_alpha = float(result_table["posterior_alpha"].mean())
        bayes_fdr = float(result_table["bayesian_fdr"].mean())
        bayes_fwer = float(result_table["bayesian_fwer"].mean())
        discovered_count = int(result_table["discovered_count"].sum())

    result = RollingBayesianAlphaResult(
        method="rolling_bayesian_alpha",
        alpha=float(alpha),
        window=int(window),
        min_periods=required,
        date_column=date_column,
        alpha_column=alpha_column,
        group_column=group_column,
        table=result_table,
        average_ols_alpha=average_ols_alpha,
        average_posterior_alpha=average_posterior_alpha,
        bayesian_fdr=bayes_fdr,
        bayesian_fwer=bayes_fwer,
        discovered_count=discovered_count,
    )
    return {
        "result": result,
        "table": result_table,
        "average_ols_alpha": average_ols_alpha,
        "average_posterior_alpha": average_posterior_alpha,
        "bayesian_fdr": bayes_fdr,
        "bayesian_fwer": bayes_fwer,
        "discovered_count": discovered_count,
    }


def dynamic_bayesian_alpha(
    alpha_series: Mapping[str, Any] | pd.DataFrame,
    *,
    date_column: str = "date_",
    alpha_column: str = "alpha",
    group_column: str | None = None,
    process_discount: float = 0.98,
    observation_variance: float | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Estimate a recursive Bayesian alpha path using a local-level state-space model."""

    frame = _coerce_factor_frame(alpha_series, factor_name_column="factor_name")
    if date_column not in frame.columns:
        raise KeyError(f"Missing date column {date_column!r}")
    if alpha_column not in frame.columns:
        raise KeyError(f"Missing alpha column {alpha_column!r}")

    working = frame.copy()
    working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
    working[alpha_column] = pd.to_numeric(working[alpha_column], errors="coerce")
    working = working.dropna(subset=[date_column, alpha_column]).sort_values(date_column).reset_index(drop=True)
    if working.empty:
        empty = pd.DataFrame(
            columns=[
                date_column,
                "ols_alpha",
                "prior_alpha",
                "posterior_alpha",
                "posterior_alpha_sd",
                "posterior_positive_prob",
                "posterior_null_prob",
                "kalman_gain",
                "discovered",
            ]
        )
        result = DynamicBayesianAlphaResult(
            method="dynamic_bayesian_alpha",
            alpha=float(alpha),
            date_column=date_column,
            alpha_column=alpha_column,
            group_column=group_column,
            process_discount=float(process_discount),
            observation_variance=float(observation_variance or 0.0),
            process_variance=0.0,
            table=empty,
            average_ols_alpha=0.0,
            average_posterior_alpha=0.0,
            bayesian_fdr=0.0,
            bayesian_fwer=0.0,
            discovered_count=0,
        )
        return {
            "result": result,
            "table": empty,
            "average_ols_alpha": 0.0,
            "average_posterior_alpha": 0.0,
            "bayesian_fdr": 0.0,
            "bayesian_fwer": 0.0,
            "discovered_count": 0,
        }

    if group_column is None:
        group_items = [(None, working)]
    else:
        if group_column not in working.columns:
            raise KeyError(f"Missing group column {group_column!r}")
        group_items = list(working.groupby(group_column, dropna=False, sort=False))

    rows: list[dict[str, Any]] = []
    observation_var_used = 0.0
    process_var_used = 0.0
    for group_value, group_frame in group_items:
        group_frame = group_frame.sort_values(date_column).reset_index(drop=True)
        group_table, obs_var_used, proc_var_used, _ = _dynamic_local_level_summary(
            group_frame[alpha_column],
            alpha=float(alpha),
            process_discount=float(process_discount),
            observation_variance=observation_variance,
        )
        if group_table.empty:
            continue
        group_table = group_table.copy()
        group_table[date_column] = group_frame[date_column].to_numpy()[: len(group_table)]
        if group_column is not None:
            group_table[group_column] = group_value
        rows.append(group_table)
        observation_var_used = obs_var_used
        process_var_used = proc_var_used

    if rows:
        result_table = pd.concat(rows, axis=0, ignore_index=True)
        sort_columns = [date_column]
        if group_column is not None:
            sort_columns = [group_column, date_column]
        result_table = result_table.sort_values(sort_columns).reset_index(drop=True)
    else:
        result_table = pd.DataFrame(
            columns=[
                date_column,
                "ols_alpha",
                "prior_alpha",
                "posterior_alpha",
                "posterior_alpha_sd",
                "posterior_positive_prob",
                "posterior_null_prob",
                "kalman_gain",
                "discovered",
            ]
        )

    if result_table.empty:
        average_ols_alpha = 0.0
        average_posterior_alpha = 0.0
        bayes_fdr = 0.0
        bayes_fwer = 0.0
        discovered_count = 0
    else:
        average_ols_alpha = float(result_table["ols_alpha"].mean())
        average_posterior_alpha = float(result_table["posterior_alpha"].mean())
        discovered = result_table.loc[result_table["discovered"], "posterior_null_prob"]
        bayes_fdr = float(discovered.mean()) if not discovered.empty else 0.0
        bayes_fwer = float(1.0 - np.prod(1.0 - discovered.to_numpy(dtype=float))) if not discovered.empty else 0.0
        discovered_count = int(result_table["discovered"].sum())

    result = DynamicBayesianAlphaResult(
        method="dynamic_bayesian_alpha",
        alpha=float(alpha),
        date_column=date_column,
        alpha_column=alpha_column,
        group_column=group_column,
        process_discount=float(process_discount),
        observation_variance=float(observation_var_used),
        process_variance=float(process_var_used),
        table=result_table,
        average_ols_alpha=average_ols_alpha,
        average_posterior_alpha=average_posterior_alpha,
        bayesian_fdr=bayes_fdr,
        bayesian_fwer=bayes_fwer,
        discovered_count=discovered_count,
    )
    return {
        "result": result,
        "table": result_table,
        "average_ols_alpha": average_ols_alpha,
        "average_posterior_alpha": average_posterior_alpha,
        "bayesian_fdr": bayes_fdr,
        "bayesian_fwer": bayes_fwer,
        "discovered_count": discovered_count,
    }


def _alpha_hacking_recursive_summary(
    frame: pd.DataFrame,
    *,
    date_column: str,
    in_sample_column: str,
    out_of_sample_column: str,
    alpha: float,
    bias_discount: float,
    observation_discount: float,
) -> pd.DataFrame:
    in_sample = pd.to_numeric(frame[in_sample_column], errors="coerce")
    out_of_sample = pd.to_numeric(frame[out_of_sample_column], errors="coerce")
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    valid_mask = dates.notna() & in_sample.notna() & out_of_sample.notna()
    frame = frame.loc[valid_mask].copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                date_column,
                "in_sample_alpha",
                "out_of_sample_alpha",
                "hacking_bias",
                "hacking_bias_sd",
                "prior_alpha",
                "posterior_alpha",
                "posterior_alpha_sd",
                "posterior_positive_prob",
                "posterior_null_prob",
                "in_sample_weight",
                "out_of_sample_weight",
                "discovered",
            ]
        )

    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    frame = frame.sort_values(date_column).reset_index(drop=True)

    prior_mean = float((frame[in_sample_column].iloc[0] + frame[out_of_sample_column].iloc[0]) / 2.0)
    prior_var = float(np.var(frame[[in_sample_column, out_of_sample_column]].iloc[0].to_numpy(dtype=float), ddof=0))
    if not np.isfinite(prior_var) or prior_var <= 1e-12:
        prior_var = 1e-4
    bias_mean = 0.0
    bias_var = prior_var
    in_obs_var = prior_var
    out_obs_var = prior_var

    rows: list[dict[str, Any]] = []
    obs_discount = float(np.clip(observation_discount, 0.5, 0.999999))
    bias_discount = float(np.clip(bias_discount, 0.5, 0.999999))
    for _, row in frame.iterrows():
        in_alpha = float(pd.to_numeric(row[in_sample_column], errors="coerce"))
        out_alpha = float(pd.to_numeric(row[out_of_sample_column], errors="coerce"))
        gap = in_alpha - out_alpha

        bias_mean = bias_discount * bias_mean + (1.0 - bias_discount) * gap
        bias_var = bias_discount * bias_var + (1.0 - bias_discount) * (gap - bias_mean) ** 2
        in_obs_var = obs_discount * in_obs_var + (1.0 - obs_discount) * (in_alpha - prior_mean) ** 2
        out_obs_var = obs_discount * out_obs_var + (1.0 - obs_discount) * (out_alpha - prior_mean) ** 2

        hacking_variance = max(bias_var - in_obs_var - out_obs_var, 0.0)
        adjusted_in_alpha = in_alpha - bias_mean

        in_precision = 1.0 / max(in_obs_var + hacking_variance, 1e-12)
        out_precision = 1.0 / max(out_obs_var, 1e-12)
        prior_precision = 1.0 / max(prior_var, 1e-12)
        posterior_precision = prior_precision + in_precision + out_precision
        posterior_mean = (
            prior_precision * prior_mean
            + in_precision * adjusted_in_alpha
            + out_precision * out_alpha
        ) / posterior_precision
        posterior_var = max(1.0 / posterior_precision, 1e-12)
        posterior_sd = float(np.sqrt(posterior_var))
        z_score = posterior_mean / posterior_sd if posterior_sd > 0 else 0.0
        posterior_positive_prob = float(norm.cdf(z_score))
        posterior_null_prob = float(np.clip(2.0 * norm.sf(abs(z_score)), 0.0, 1.0))
        oos_weight = out_precision / (in_precision + out_precision)
        in_weight = in_precision / (in_precision + out_precision)

        rows.append(
            {
                date_column: pd.Timestamp(row[date_column]),
                "in_sample_alpha": in_alpha,
                "out_of_sample_alpha": out_alpha,
                "hacking_bias": bias_mean,
                "hacking_bias_sd": float(np.sqrt(max(bias_var, 0.0))),
                "prior_alpha": prior_mean,
                "posterior_alpha": float(posterior_mean),
                "posterior_alpha_sd": posterior_sd,
                "posterior_positive_prob": posterior_positive_prob,
                "posterior_null_prob": posterior_null_prob,
                "in_sample_weight": float(in_weight),
                "out_of_sample_weight": float(oos_weight),
                "hacking_variance": float(hacking_variance),
                "discovered": posterior_null_prob <= float(alpha),
            }
        )
        prior_mean = posterior_mean
        prior_var = posterior_var

    return pd.DataFrame(rows)


def alpha_hacking_bayesian_update(
    alpha_series: Mapping[str, Any] | pd.DataFrame,
    *,
    date_column: str = "date_",
    in_sample_column: str = "in_sample_alpha",
    out_of_sample_column: str = "out_of_sample_alpha",
    group_column: str | None = None,
    alpha: float = 0.05,
    bias_discount: float = 0.97,
    observation_discount: float = 0.95,
) -> dict[str, Any]:
    """Bayesian posterior update for paired in-sample / OOS alpha series.

    The in-sample path is allowed to carry an additional hacking-bias term.
    As the estimated bias variance rises, the posterior naturally puts more
    weight on the OOS path.
    """

    frame = _coerce_factor_frame(alpha_series, factor_name_column="factor_name")
    if date_column not in frame.columns:
        raise KeyError(f"Missing date column {date_column!r}")
    if in_sample_column not in frame.columns:
        raise KeyError(f"Missing in-sample column {in_sample_column!r}")
    if out_of_sample_column not in frame.columns:
        raise KeyError(f"Missing out-of-sample column {out_of_sample_column!r}")

    working = frame.copy()
    working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
    working[in_sample_column] = pd.to_numeric(working[in_sample_column], errors="coerce")
    working[out_of_sample_column] = pd.to_numeric(working[out_of_sample_column], errors="coerce")
    working = working.dropna(subset=[date_column, in_sample_column, out_of_sample_column]).sort_values(date_column).reset_index(drop=True)

    if working.empty:
        empty = pd.DataFrame(
            columns=[
                date_column,
                "in_sample_alpha",
                "out_of_sample_alpha",
                "hacking_bias",
                "hacking_bias_sd",
                "prior_alpha",
                "posterior_alpha",
                "posterior_alpha_sd",
                "posterior_positive_prob",
                "posterior_null_prob",
                "in_sample_weight",
                "out_of_sample_weight",
                "hacking_variance",
                "discovered",
            ]
        )
        result = AlphaHackingBayesianResult(
            method="alpha_hacking_bayesian_update",
            alpha=float(alpha),
            date_column=date_column,
            in_sample_column=in_sample_column,
            out_of_sample_column=out_of_sample_column,
            group_column=group_column,
            bias_discount=float(bias_discount),
            observation_discount=float(observation_discount),
            table=empty,
            average_in_sample_alpha=0.0,
            average_out_of_sample_alpha=0.0,
            average_posterior_alpha=0.0,
            average_hacking_bias=0.0,
            bayesian_fdr=0.0,
            bayesian_fwer=0.0,
            discovered_count=0,
        )
        return {
            "result": result,
            "table": empty,
            "average_in_sample_alpha": 0.0,
            "average_out_of_sample_alpha": 0.0,
            "average_posterior_alpha": 0.0,
            "average_hacking_bias": 0.0,
            "bayesian_fdr": 0.0,
            "bayesian_fwer": 0.0,
            "discovered_count": 0,
        }

    if group_column is None:
        group_items = [(None, working)]
    else:
        if group_column not in working.columns:
            raise KeyError(f"Missing group column {group_column!r}")
        group_items = list(working.groupby(group_column, dropna=False, sort=False))

    tables: list[pd.DataFrame] = []
    for group_value, group_frame in group_items:
        group_frame = group_frame.sort_values(date_column).reset_index(drop=True)
        group_table = _alpha_hacking_recursive_summary(
            group_frame,
            date_column=date_column,
            in_sample_column=in_sample_column,
            out_of_sample_column=out_of_sample_column,
            alpha=float(alpha),
            bias_discount=float(bias_discount),
            observation_discount=float(observation_discount),
        )
        if group_table.empty:
            continue
        if group_column is not None:
            group_table[group_column] = group_value
        tables.append(group_table)

    result_table = pd.concat(tables, axis=0, ignore_index=True) if tables else pd.DataFrame(
        columns=[
            date_column,
            "in_sample_alpha",
            "out_of_sample_alpha",
            "hacking_bias",
            "hacking_bias_sd",
            "prior_alpha",
            "posterior_alpha",
            "posterior_alpha_sd",
            "posterior_positive_prob",
            "posterior_null_prob",
            "in_sample_weight",
            "out_of_sample_weight",
            "hacking_variance",
            "discovered",
        ]
    )
    if not result_table.empty:
        sort_columns = [date_column]
        if group_column is not None:
            sort_columns = [group_column, date_column]
        result_table = result_table.sort_values(sort_columns).reset_index(drop=True)

    if result_table.empty:
        average_in_sample_alpha = 0.0
        average_out_of_sample_alpha = 0.0
        average_posterior_alpha = 0.0
        average_hacking_bias = 0.0
        bayes_fdr = 0.0
        bayes_fwer = 0.0
        discovered_count = 0
    else:
        average_in_sample_alpha = float(result_table["in_sample_alpha"].mean())
        average_out_of_sample_alpha = float(result_table["out_of_sample_alpha"].mean())
        average_posterior_alpha = float(result_table["posterior_alpha"].mean())
        average_hacking_bias = float(result_table["hacking_bias"].mean())
        discovered = result_table.loc[result_table["discovered"], "posterior_null_prob"]
        bayes_fdr = float(discovered.mean()) if not discovered.empty else 0.0
        bayes_fwer = float(1.0 - np.prod(1.0 - discovered.to_numpy(dtype=float))) if not discovered.empty else 0.0
        discovered_count = int(result_table["discovered"].sum())

    result = AlphaHackingBayesianResult(
        method="alpha_hacking_bayesian_update",
        alpha=float(alpha),
        date_column=date_column,
        in_sample_column=in_sample_column,
        out_of_sample_column=out_of_sample_column,
        group_column=group_column,
        bias_discount=float(bias_discount),
        observation_discount=float(observation_discount),
        table=result_table,
        average_in_sample_alpha=average_in_sample_alpha,
        average_out_of_sample_alpha=average_out_of_sample_alpha,
        average_posterior_alpha=average_posterior_alpha,
        average_hacking_bias=average_hacking_bias,
        bayesian_fdr=bayes_fdr,
        bayesian_fwer=bayes_fwer,
        discovered_count=discovered_count,
    )
    return {
        "result": result,
        "table": result_table,
        "average_in_sample_alpha": average_in_sample_alpha,
        "average_out_of_sample_alpha": average_out_of_sample_alpha,
        "average_posterior_alpha": average_posterior_alpha,
        "average_hacking_bias": average_hacking_bias,
        "bayesian_fdr": bayes_fdr,
        "bayesian_fwer": bayes_fwer,
        "discovered_count": discovered_count,
    }


def validate_bayesian_factor_family(
    factor_metrics: Mapping[str, Any] | pd.DataFrame,
    *,
    statistic_column: str = "statistic",
    p_value_column: str = "p_value",
    effect_column: str | None = "fitness",
    factor_name_column: str = "factor_name",
    alpha: float = 0.05,
    max_iter: int = 200,
    tol: float = 1e-8,
    cluster_threshold: float | None = None,
    factor_dict: Mapping[str, pd.Series | pd.DataFrame] | None = None,
) -> dict[str, Any]:
    fit = fit_bayesian_mixture(
        factor_metrics,
        statistic_column=statistic_column,
        p_value_column=p_value_column,
        effect_column=effect_column,
        factor_name_column=factor_name_column,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        cluster_threshold=cluster_threshold,
        factor_dict=factor_dict,
    )
    result = fit["result"]
    return {
        "table": result.table.copy(),
        "result": result.to_dict(),
        "pi0": fit["pi0"],
        "alt_variance": fit["alt_variance"],
        "prior_variance": fit["prior_variance"],
        "bayesian_fdr": bayesian_fdr(result),
        "clusters": fit["clusters"],
        "cluster_map": fit["cluster_map"],
    }


def validate_hierarchical_bayesian_factor_family(
    factor_metrics: Mapping[str, Any] | pd.DataFrame,
    *,
    statistic_column: str = "statistic",
    p_value_column: str = "p_value",
    effect_column: str | None = "fitness",
    factor_name_column: str = "factor_name",
    alpha: float = 0.05,
    max_iter: int = 200,
    tol: float = 1e-8,
    cluster_threshold: float = 0.65,
    factor_dict: Mapping[str, pd.Series | pd.DataFrame] | None = None,
    shared_evidence_weight: float = 0.5,
) -> dict[str, Any]:
    fit = fit_hierarchical_bayesian_mixture(
        factor_metrics,
        statistic_column=statistic_column,
        p_value_column=p_value_column,
        effect_column=effect_column,
        factor_name_column=factor_name_column,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        cluster_threshold=cluster_threshold,
        factor_dict=factor_dict,
        shared_evidence_weight=shared_evidence_weight,
    )
    result = fit["result"]
    return {
        "table": result.table.copy(),
        "result": result.to_dict(),
        "factor_fit": fit["factor_fit"],
        "cluster_table": fit["cluster_table"],
        "factor_pi0": result.factor_pi0,
        "cluster_pi0": result.cluster_pi0,
        "hierarchical_fdr": hierarchical_bayesian_fdr(result),
    }


__all__ = [
    "BayesianMixtureResult",
    "HierarchicalBayesianResult",
    "AlphaHackingBayesianResult",
    "DynamicBayesianAlphaResult",
    "RollingBayesianAlphaResult",
    "bayesian_fdr",
    "bayesian_fwer",
    "hierarchical_bayesian_fdr",
    "fit_bayesian_mixture",
    "fit_hierarchical_bayesian_mixture",
    "alpha_hacking_bayesian_update",
    "dynamic_bayesian_alpha",
    "rolling_bayesian_alpha",
    "validate_bayesian_factor_family",
    "validate_hierarchical_bayesian_factor_family",
]
