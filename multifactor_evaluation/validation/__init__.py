from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from tiger_factors.multifactor_evaluation.redundancy import cluster_factors
from tiger_factors.multifactor_evaluation.redundancy import factor_correlation_matrix


@dataclass(frozen=True)
class MultipleTestingResult:
    method: str
    alpha: float
    table: pd.DataFrame
    rejected_count: int
    pi0: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "alpha": self.alpha,
            "rejected_count": self.rejected_count,
            "pi0": self.pi0,
            "table": self.table.to_dict(orient="records"),
        }


def _coerce_p_values(values: Mapping[str, float] | pd.Series | pd.DataFrame | list[float] | np.ndarray) -> pd.Series:
    if isinstance(values, pd.DataFrame):
        if "p_value" not in values.columns:
            raise KeyError("DataFrame input must include a 'p_value' column.")
        if "factor_name" in values.columns:
            series = values.set_index("factor_name")["p_value"]
        else:
            series = values["p_value"]
        return pd.to_numeric(series, errors="coerce")
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    if isinstance(values, Mapping):
        return pd.Series({str(key): value for key, value in values.items()}, dtype="float64")
    return pd.Series(values, dtype="float64")


def _build_adjustment_frame(p_values: pd.Series) -> pd.DataFrame:
    frame = p_values.rename("p_value").to_frame()
    frame["p_value"] = pd.to_numeric(frame["p_value"], errors="coerce")
    return frame.dropna(subset=["p_value"]).copy()


def estimate_pi0(p_values: Mapping[str, float] | pd.Series | pd.DataFrame | list[float] | np.ndarray, *, lambda_: float = 0.5) -> float:
    series = _coerce_p_values(p_values).dropna()
    if series.empty:
        return 1.0
    lambda_ = float(np.clip(lambda_, 0.0, 0.95))
    denominator = max(1.0 - lambda_, 1e-12)
    estimate = float((series > lambda_).mean() / denominator)
    return float(np.clip(estimate, 0.0, 1.0))


def adjust_p_values(
    p_values: Mapping[str, float] | pd.Series | pd.DataFrame | list[float] | np.ndarray,
    *,
    method: str = "holm",
    alpha: float = 0.05,
) -> MultipleTestingResult:
    series = _coerce_p_values(p_values).dropna()
    frame = _build_adjustment_frame(series)
    if frame.empty:
        empty = frame.copy()
        empty["adjusted_p_value"] = pd.Series(dtype="float64")
        empty["rejected"] = pd.Series(dtype=bool)
        return MultipleTestingResult(method=method, alpha=float(alpha), table=empty, rejected_count=0)

    method_name = str(method).strip().lower()
    ranked = frame.sort_values("p_value", ascending=True).copy()
    m = float(len(ranked))
    ranks = np.arange(1, len(ranked) + 1, dtype=float)
    pi0 = None

    if method_name in {"bonferroni", "bonf"}:
        ranked["adjusted_p_value"] = np.minimum(1.0, ranked["p_value"] * m)
    elif method_name == "holm":
        adjusted = np.minimum(1.0, (m - ranks + 1.0) * ranked["p_value"].to_numpy(dtype=float))
        adjusted = np.maximum.accumulate(adjusted)
        ranked["adjusted_p_value"] = adjusted
    elif method_name in {"bh", "benjamini-hochberg", "benjamini_hochberg"}:
        adjusted = ranked["p_value"].to_numpy(dtype=float) * m / ranks
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        ranked["adjusted_p_value"] = np.minimum(1.0, adjusted)
    elif method_name in {"by", "benjamini-yekutieli", "benjamini_yekutieli"}:
        c_m = float(np.sum(1.0 / np.arange(1, len(ranked) + 1, dtype=float)))
        adjusted = ranked["p_value"].to_numpy(dtype=float) * m * c_m / ranks
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        ranked["adjusted_p_value"] = np.minimum(1.0, adjusted)
    elif method_name in {"storey", "qvalue", "q-values"}:
        pi0 = estimate_pi0(ranked["p_value"], lambda_=0.5)
        q_values = ranked["p_value"].to_numpy(dtype=float) * pi0 * m / ranks
        q_values = np.minimum.accumulate(q_values[::-1])[::-1]
        ranked["adjusted_p_value"] = np.minimum(1.0, q_values)
    else:
        raise ValueError(
            "Unsupported adjustment method. Use bonferroni, holm, bh, by, or storey."
        )

    ranked["rejected"] = ranked["adjusted_p_value"] <= float(alpha)
    result = ranked.sort_index()
    return MultipleTestingResult(
        method=method_name,
        alpha=float(alpha),
        table=result.reset_index(names="factor_name"),
        rejected_count=int(result["rejected"].sum()),
        pi0=pi0,
    )


def bonferroni_adjust(
    p_values: Mapping[str, float] | pd.Series | pd.DataFrame | list[float] | np.ndarray,
    *,
    alpha: float = 0.05,
) -> MultipleTestingResult:
    return adjust_p_values(p_values, method="bonferroni", alpha=alpha)


def holm_adjust(
    p_values: Mapping[str, float] | pd.Series | pd.DataFrame | list[float] | np.ndarray,
    *,
    alpha: float = 0.05,
) -> MultipleTestingResult:
    return adjust_p_values(p_values, method="holm", alpha=alpha)


def benjamini_hochberg(
    p_values: Mapping[str, float] | pd.Series | pd.DataFrame | list[float] | np.ndarray,
    *,
    alpha: float = 0.05,
) -> MultipleTestingResult:
    return adjust_p_values(p_values, method="bh", alpha=alpha)


def benjamini_yekutieli(
    p_values: Mapping[str, float] | pd.Series | pd.DataFrame | list[float] | np.ndarray,
    *,
    alpha: float = 0.05,
) -> MultipleTestingResult:
    return adjust_p_values(p_values, method="by", alpha=alpha)


def storey_qvalues(
    p_values: Mapping[str, float] | pd.Series | pd.DataFrame | list[float] | np.ndarray,
    *,
    alpha: float = 0.05,
    lambda_: float = 0.5,
) -> MultipleTestingResult:
    series = _coerce_p_values(p_values).dropna()
    frame = _build_adjustment_frame(series)
    if frame.empty:
        return MultipleTestingResult(method="storey", alpha=float(alpha), table=frame, rejected_count=0, pi0=1.0)

    ranked = frame.sort_values("p_value", ascending=True).copy()
    m = float(len(ranked))
    pi0 = estimate_pi0(ranked["p_value"], lambda_=lambda_)
    ranks = np.arange(1, len(ranked) + 1, dtype=float)
    q_values = ranked["p_value"].to_numpy(dtype=float) * pi0 * m / ranks
    q_values = np.minimum.accumulate(q_values[::-1])[::-1]
    ranked["adjusted_p_value"] = np.minimum(1.0, q_values)
    ranked["rejected"] = ranked["adjusted_p_value"] <= float(alpha)
    result = ranked.sort_index()
    return MultipleTestingResult(
        method="storey",
        alpha=float(alpha),
        table=result.reset_index(names="factor_name"),
        rejected_count=int(result["rejected"].sum()),
        pi0=pi0,
    )


def validate_factor_family(
    factor_metrics: Mapping[str, Any] | pd.DataFrame,
    *,
    p_value_column: str = "p_value",
    factor_name_column: str = "factor_name",
    method: str = "holm",
    alpha: float = 0.05,
    cluster_threshold: float | None = None,
    factor_dict: Mapping[str, pd.Series | pd.DataFrame] | None = None,
) -> dict[str, Any]:
    if isinstance(factor_metrics, pd.DataFrame):
        frame = factor_metrics.copy()
    else:
        values = list(factor_metrics.values())
        if values and all(np.isscalar(value) for value in values):
            frame = pd.DataFrame(
                {
                    factor_name_column: [str(key) for key in factor_metrics.keys()],
                    p_value_column: [float(value) for value in values],
                }
            )
        else:
            frame = pd.DataFrame.from_dict(factor_metrics, orient="index").reset_index().rename(
                columns={"index": factor_name_column}
            )
    if p_value_column not in frame.columns:
        raise KeyError(f"Missing required p-value column {p_value_column!r}")

    if factor_name_column not in frame.columns:
        frame = frame.reset_index().rename(columns={frame.index.name or "index": factor_name_column})

    adjustment = adjust_p_values(frame[[factor_name_column, p_value_column]].set_index(factor_name_column), method=method, alpha=alpha)
    adjusted = adjustment.table.rename(
        columns={
            "p_value": f"{p_value_column}_adjusted",
            "adjusted_p_value": f"{method}_adjusted_p_value",
            "rejected": f"{method}_rejected",
        }
    )
    merged = frame.merge(adjusted, on="factor_name", how="left")

    cluster_map: dict[str, int] | None = None
    cluster_frame: pd.DataFrame | None = None
    if cluster_threshold is not None and factor_dict is not None:
        corr = factor_correlation_matrix(factor_dict)
        cluster_map = cluster_factors(corr, threshold=float(cluster_threshold))
        cluster_frame = pd.DataFrame(
            {
                "factor_name": list(cluster_map.keys()),
                "cluster": list(cluster_map.values()),
            }
        ).sort_values(["cluster", "factor_name"]).reset_index(drop=True)
        merged = merged.merge(cluster_frame, on="factor_name", how="left")

    return {
        "table": merged.sort_values(
            [f"{method}_rejected", f"{method}_adjusted_p_value", p_value_column],
            ascending=[False, True, True],
        ).reset_index(drop=True),
        "adjustment": adjustment.to_dict(),
        "clusters": cluster_frame.to_dict(orient="records") if cluster_frame is not None else None,
        "cluster_map": cluster_map,
    }


__all__ = [
    "MultipleTestingResult",
    "adjust_p_values",
    "benjamini_hochberg",
    "benjamini_yekutieli",
    "bonferroni_adjust",
    "estimate_pi0",
    "holm_adjust",
    "storey_qvalues",
    "validate_factor_family",
]
