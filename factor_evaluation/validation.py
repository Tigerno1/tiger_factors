from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from tiger_factors.factor_evaluation.core import evaluate_factor_panel
from tiger_factors.factor_evaluation.performance import factor_information_coefficient
from tiger_factors.factor_evaluation.performance import factor_returns
from tiger_factors.factor_evaluation.utils import TigerFactorData


@dataclass(frozen=True)
class ValidationResult:
    metric_name: str
    observed: float
    bootstrap_mean: float
    bootstrap_std: float
    ci_low: float
    ci_high: float
    p_value: float
    n_obs: int
    split_first: float | None = None
    split_second: float | None = None
    stability_gap: float | None = None
    stability_ratio: float | None = None
    notes: str = ""


def _coerce_numeric_series(values: pd.Series | pd.Index | list[float] | np.ndarray) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values.copy()
    else:
        series = pd.Series(values)
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def bootstrap_confidence_interval(
    values: pd.Series | list[float] | np.ndarray,
    *,
    statistic: Callable[[pd.Series], float] = pd.Series.mean,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int | np.random.Generator | None = None,
) -> dict[str, float | int]:
    series = _coerce_numeric_series(values)
    if series.empty:
        return {
            "observed": float("nan"),
            "bootstrap_mean": float("nan"),
            "bootstrap_std": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_obs": 0,
            "n_bootstrap": int(n_bootstrap),
        }

    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    observed = float(statistic(series))
    bootstrap_stats: list[float] = []
    values_array = series.to_numpy(dtype=float)
    n_obs = len(values_array)
    for _ in range(int(max(n_bootstrap, 1))):
        sample = rng.choice(values_array, size=n_obs, replace=True)
        bootstrap_stats.append(float(statistic(pd.Series(sample))))

    boot = np.asarray(bootstrap_stats, dtype=float)
    lower = float(np.nanquantile(boot, alpha / 2.0))
    upper = float(np.nanquantile(boot, 1.0 - alpha / 2.0))
    return {
        "observed": observed,
        "bootstrap_mean": float(np.nanmean(boot)),
        "bootstrap_std": float(np.nanstd(boot, ddof=0)),
        "ci_low": lower,
        "ci_high": upper,
        "n_obs": int(n_obs),
        "n_bootstrap": int(len(bootstrap_stats)),
    }


def permutation_test(
    values: pd.Series | list[float] | np.ndarray,
    *,
    statistic: Callable[[pd.Series], float] = pd.Series.mean,
    n_permutations: int = 1000,
    alternative: str = "two-sided",
    null_value: float = 0.0,
    random_state: int | np.random.Generator | None = None,
) -> dict[str, float | int]:
    series = _coerce_numeric_series(values)
    if series.empty:
        return {
            "observed": float("nan"),
            "p_value": float("nan"),
            "null_mean": float("nan"),
            "null_std": float("nan"),
            "n_obs": 0,
            "n_permutations": int(n_permutations),
        }

    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    centered = series - float(null_value)
    observed = float(statistic(series))
    null_stats: list[float] = []
    centered_values = centered.to_numpy(dtype=float)
    for _ in range(int(max(n_permutations, 1))):
        signs = rng.choice([-1.0, 1.0], size=len(centered_values))
        permuted = centered_values * signs + float(null_value)
        null_stats.append(float(statistic(pd.Series(permuted))))

    null = np.asarray(null_stats, dtype=float)
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be one of: two-sided, greater, less")
    if alternative == "greater":
        p_value = float((np.sum(null >= observed) + 1.0) / (len(null) + 1.0))
    elif alternative == "less":
        p_value = float((np.sum(null <= observed) + 1.0) / (len(null) + 1.0))
    else:
        p_value = float((np.sum(np.abs(null) >= abs(observed)) + 1.0) / (len(null) + 1.0))

    return {
        "observed": observed,
        "p_value": p_value,
        "null_mean": float(np.nanmean(null)),
        "null_std": float(np.nanstd(null, ddof=0)),
        "n_obs": int(len(series)),
        "n_permutations": int(len(null_stats)),
    }


def split_stability(
    values: pd.Series | list[float] | np.ndarray,
    *,
    split_index: int | None = None,
    statistic: Callable[[pd.Series], float] = pd.Series.mean,
) -> dict[str, float | int]:
    series = _coerce_numeric_series(values)
    if series.empty:
        return {
            "split_first": float("nan"),
            "split_second": float("nan"),
            "stability_gap": float("nan"),
            "stability_ratio": float("nan"),
            "split_index": 0,
            "n_obs": 0,
        }

    split = int(split_index) if split_index is not None else max(len(series) // 2, 1)
    split = min(max(split, 1), len(series) - 1 if len(series) > 1 else 1)
    first = float(statistic(series.iloc[:split]))
    second = float(statistic(series.iloc[split:]))
    gap = second - first
    ratio = second / first if abs(first) > 1e-12 else float("nan")
    return {
        "split_first": first,
        "split_second": second,
        "stability_gap": gap,
        "stability_ratio": ratio,
        "split_index": split,
        "n_obs": int(len(series)),
    }


def validate_series(
    values: pd.Series | list[float] | np.ndarray,
    *,
    metric_name: str = "metric",
    statistic: Callable[[pd.Series], float] = pd.Series.mean,
    n_bootstrap: int = 1000,
    n_permutations: int = 1000,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    random_state: int | np.random.Generator | None = None,
) -> ValidationResult:
    ci = bootstrap_confidence_interval(
        values,
        statistic=statistic,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        random_state=random_state,
    )
    permutation = permutation_test(
        values,
        statistic=statistic,
        n_permutations=n_permutations,
        alternative=alternative,
        null_value=0.0,
        random_state=random_state,
    )
    stability = split_stability(values, statistic=statistic)
    return ValidationResult(
        metric_name=metric_name,
        observed=float(ci["observed"]),
        bootstrap_mean=float(ci["bootstrap_mean"]),
        bootstrap_std=float(ci["bootstrap_std"]),
        ci_low=float(ci["ci_low"]),
        ci_high=float(ci["ci_high"]),
        p_value=float(permutation["p_value"]),
        n_obs=int(ci["n_obs"]),
        split_first=float(stability["split_first"]),
        split_second=float(stability["split_second"]),
        stability_gap=float(stability["stability_gap"]),
        stability_ratio=float(stability["stability_ratio"]),
    )


def validate_factor_data(
    factor_data: TigerFactorData,
    *,
    period: str | None = None,
    n_bootstrap: int = 1000,
    n_permutations: int = 1000,
    random_state: int | np.random.Generator | None = None,
) -> dict[str, Any]:
    ic_frame = factor_information_coefficient(factor_data.factor_data)
    periods = list(ic_frame.columns)
    if not periods:
        return {"error": "no forward return periods found in factor_data"}

    chosen_period = str(period) if period is not None else str(periods[0])
    if chosen_period not in ic_frame.columns:
        chosen_period = str(periods[0])

    ic_series = pd.to_numeric(ic_frame[chosen_period], errors="coerce").dropna()
    ls_frame = factor_returns(factor_data.factor_data)
    ls_series = pd.to_numeric(ls_frame[chosen_period], errors="coerce").dropna() if chosen_period in ls_frame.columns else pd.Series(dtype=float)

    core = evaluate_factor_panel(
        factor_data.factor_panel,
        factor_data.forward_returns,
    )
    ic_validation = validate_series(
        ic_series,
        metric_name=f"ic[{chosen_period}]",
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
        random_state=random_state,
    )
    ls_validation = validate_series(
        ls_series,
        metric_name=f"long_short[{chosen_period}]",
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
        random_state=random_state,
    ) if not ls_series.empty else None

    return {
        "period": chosen_period,
        "core": asdict(core),
        "ic_validation": asdict(ic_validation),
        "long_short_validation": None if ls_validation is None else asdict(ls_validation),
    }


__all__ = [
    "ValidationResult",
    "bootstrap_confidence_interval",
    "permutation_test",
    "split_stability",
    "validate_factor_data",
    "validate_series",
]
