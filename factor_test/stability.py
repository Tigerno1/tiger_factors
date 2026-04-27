from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

import numpy as np
import pandas as pd

from tiger_factors.factor_evaluation.analysis import FactorEffectivenessConfig
from tiger_factors.factor_evaluation.analysis import test_factor_effectiveness as _test_factor_effectiveness
from tiger_factors.factor_evaluation.core import _align_frames
from tiger_factors.factor_evaluation.performance import factor_information_coefficient
from tiger_factors.factor_evaluation.utils import TigerFactorData
from tiger_factors.factor_evaluation.utils import _rowwise_cross_sectional_corr
from tiger_factors.factor_evaluation.utils import get_forward_returns_columns


@dataclass(frozen=True)
class FactorDecayResult:
    method: str
    period_column: str
    table: pd.DataFrame
    n_periods: int
    ic_decay_slope: float
    rank_ic_decay_slope: float
    ic_decay_ratio: float
    rank_ic_decay_ratio: float
    first_horizon: int | None
    last_horizon: int | None
    best_horizon: int | None
    worst_horizon: int | None
    decay_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "period_column": self.period_column,
            "n_periods": self.n_periods,
            "ic_decay_slope": self.ic_decay_slope,
            "rank_ic_decay_slope": self.rank_ic_decay_slope,
            "ic_decay_ratio": self.ic_decay_ratio,
            "rank_ic_decay_ratio": self.rank_ic_decay_ratio,
            "first_horizon": self.first_horizon,
            "last_horizon": self.last_horizon,
            "best_horizon": self.best_horizon,
            "worst_horizon": self.worst_horizon,
            "decay_score": self.decay_score,
            "table": self.table.to_dict(orient="records"),
        }


@dataclass(frozen=True)
class FactorRecentICResult:
    method: str
    period_column: str
    recent_window: int
    historical_window: int
    table: pd.DataFrame
    historical_ic_mean: float
    recent_ic_mean: float
    historical_rank_ic_mean: float
    recent_rank_ic_mean: float
    historical_ic_ir: float
    recent_ic_ir: float
    historical_rank_ic_ir: float
    recent_rank_ic_ir: float
    ic_gap: float
    rank_ic_gap: float
    ic_ratio: float
    rank_ic_ratio: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "period_column": self.period_column,
            "recent_window": self.recent_window,
            "historical_window": self.historical_window,
            "historical_ic_mean": self.historical_ic_mean,
            "recent_ic_mean": self.recent_ic_mean,
            "historical_rank_ic_mean": self.historical_rank_ic_mean,
            "recent_rank_ic_mean": self.recent_rank_ic_mean,
            "historical_ic_ir": self.historical_ic_ir,
            "recent_ic_ir": self.recent_ic_ir,
            "historical_rank_ic_ir": self.historical_rank_ic_ir,
            "recent_rank_ic_ir": self.recent_rank_ic_ir,
            "ic_gap": self.ic_gap,
            "rank_ic_gap": self.rank_ic_gap,
            "ic_ratio": self.ic_ratio,
            "rank_ic_ratio": self.rank_ic_ratio,
            "passed": self.passed,
            "table": self.table.to_dict(orient="records"),
        }


def _horizon_from_label(label: str) -> int:
    text = str(label).strip()
    match = re.search(r"(\d+)", text)
    if match is None:
        return 1
    return max(int(match.group(1)), 1)


def _ic_frame(
    factor_data: pd.DataFrame,
    period: str,
) -> pd.DataFrame:
    ordered = factor_data.sort_index()
    factor_wide = ordered["factor"].unstack()
    forward_wide = ordered[period].unstack()
    ic = _rowwise_cross_sectional_corr(factor_wide, forward_wide)
    rank_ic = _rowwise_cross_sectional_corr(factor_wide, forward_wide, rank=True)
    return pd.DataFrame({"ic": ic, "rank_ic": rank_ic}).dropna(how="all")


def _safe_mean(values: pd.Series) -> float:
    series = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(series.mean()) if not series.empty else 0.0


def _safe_ir(values: pd.Series) -> float:
    series = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return 0.0
    std = float(series.std(ddof=0))
    if std <= 1e-12:
        return 0.0
    return float(series.mean() / std)


def factor_decay_test(
    factor_data: TigerFactorData,
    *,
    use_rank_ic: bool = False,
) -> FactorDecayResult:
    frame = factor_data.factor_data.copy()
    periods = get_forward_returns_columns(frame.columns)
    if not periods:
        empty = pd.DataFrame(columns=["period", "horizon", "ic_mean", "rank_ic_mean", "ic_ir", "rank_ic_ir", "n_obs"])
        return FactorDecayResult(
            method="horizon_decay",
            period_column="",
            table=empty,
            n_periods=0,
            ic_decay_slope=0.0,
            rank_ic_decay_slope=0.0,
            ic_decay_ratio=float("nan"),
            rank_ic_decay_ratio=float("nan"),
            first_horizon=None,
            last_horizon=None,
            best_horizon=None,
            worst_horizon=None,
            decay_score=0.0,
        )

    records: list[dict[str, float | int | str]] = []
    factor_wide = frame.sort_index()["factor"].unstack()
    for period in periods:
        period_wide = frame.sort_index()[period].unstack()
        ic_series = _rowwise_cross_sectional_corr(factor_wide, period_wide)
        rank_ic_series = _rowwise_cross_sectional_corr(factor_wide, period_wide, rank=True)
        period_label = str(period)
        horizon = _horizon_from_label(period_label)
        ic_mean = _safe_mean(ic_series)
        rank_ic_mean = _safe_mean(rank_ic_series)
        records.append(
            {
                "period": period_label,
                "horizon": horizon,
                "ic_mean": ic_mean,
                "rank_ic_mean": rank_ic_mean,
                "ic_ir": _safe_ir(ic_series),
                "rank_ic_ir": _safe_ir(rank_ic_series),
                "n_obs": int(ic_series.dropna().shape[0]),
            }
        )

    table = pd.DataFrame(records).sort_values("horizon").reset_index(drop=True)
    valid_ic = table[["horizon", "ic_mean"]].dropna()
    valid_rank_ic = table[["horizon", "rank_ic_mean"]].dropna()
    if len(valid_ic) >= 2:
        ic_slope, ic_intercept = np.polyfit(valid_ic["horizon"], valid_ic["ic_mean"], 1)
    else:
        ic_slope, ic_intercept = 0.0, float(valid_ic["ic_mean"].iloc[0]) if not valid_ic.empty else 0.0
    if len(valid_rank_ic) >= 2:
        rank_ic_slope, rank_ic_intercept = np.polyfit(valid_rank_ic["horizon"], valid_rank_ic["rank_ic_mean"], 1)
    else:
        rank_ic_slope, rank_ic_intercept = 0.0, float(valid_rank_ic["rank_ic_mean"].iloc[0]) if not valid_rank_ic.empty else 0.0

    first_row = table.iloc[0]
    last_row = table.iloc[-1]
    ic_decay_ratio = float(last_row["ic_mean"] / first_row["ic_mean"]) if abs(float(first_row["ic_mean"])) > 1e-12 else float("nan")
    rank_ic_decay_ratio = float(last_row["rank_ic_mean"] / first_row["rank_ic_mean"]) if abs(float(first_row["rank_ic_mean"])) > 1e-12 else float("nan")
    best_idx = table["ic_mean"].abs().idxmax()
    worst_idx = table["ic_mean"].abs().idxmin()
    best_horizon = int(table.loc[best_idx, "horizon"]) if pd.notna(best_idx) else None
    worst_horizon = int(table.loc[worst_idx, "horizon"]) if pd.notna(worst_idx) else None
    decay_score = float(1.0 / (1.0 + max(0.0, -float(ic_slope)) * max(float(table["horizon"].max() - table["horizon"].min()), 1.0)))
    return FactorDecayResult(
        method="horizon_decay",
        period_column="period",
        table=table,
        n_periods=int(len(table)),
        ic_decay_slope=float(ic_slope),
        rank_ic_decay_slope=float(rank_ic_slope),
        ic_decay_ratio=ic_decay_ratio,
        rank_ic_decay_ratio=rank_ic_decay_ratio,
        first_horizon=int(first_row["horizon"]),
        last_horizon=int(last_row["horizon"]),
        best_horizon=best_horizon,
        worst_horizon=worst_horizon,
        decay_score=decay_score,
    )


def factor_recent_ic_test(
    factor_data: TigerFactorData,
    *,
    period: str | None = None,
    recent_window: int = 60,
    min_recent_ic_mean: float = 0.0,
    min_recent_to_history_ratio: float = 0.5,
    min_recent_rank_ic_mean: float = 0.0,
    min_recent_rank_to_history_ratio: float = 0.5,
) -> FactorRecentICResult:
    frame = factor_data.factor_data.copy()
    periods = get_forward_returns_columns(frame.columns)
    if not periods:
        empty = pd.DataFrame(columns=["date_", "ic", "rank_ic", "segment"])
        return FactorRecentICResult(
            method="recent_ic_stability",
            period_column="",
            recent_window=0,
            historical_window=0,
            table=empty,
            historical_ic_mean=0.0,
            recent_ic_mean=0.0,
            historical_rank_ic_mean=0.0,
            recent_rank_ic_mean=0.0,
            historical_ic_ir=0.0,
            recent_ic_ir=0.0,
            historical_rank_ic_ir=0.0,
            recent_rank_ic_ir=0.0,
            ic_gap=0.0,
            rank_ic_gap=0.0,
            ic_ratio=float("nan"),
            rank_ic_ratio=float("nan"),
            passed=False,
        )

    chosen_period = str(period) if period is not None else str(periods[0])
    if chosen_period not in frame.columns:
        chosen_period = str(periods[0])

    ic_frame = _ic_frame(frame, chosen_period)
    if ic_frame.empty:
        empty = pd.DataFrame(columns=["date_", "ic", "rank_ic", "segment"])
        return FactorRecentICResult(
            method="recent_ic_stability",
            period_column=chosen_period,
            recent_window=0,
            historical_window=0,
            table=empty,
            historical_ic_mean=0.0,
            recent_ic_mean=0.0,
            historical_rank_ic_mean=0.0,
            recent_rank_ic_mean=0.0,
            historical_ic_ir=0.0,
            recent_ic_ir=0.0,
            historical_rank_ic_ir=0.0,
            recent_rank_ic_ir=0.0,
            ic_gap=0.0,
            rank_ic_gap=0.0,
            ic_ratio=float("nan"),
            rank_ic_ratio=float("nan"),
            passed=False,
        )

    recent_window = int(max(recent_window, 1))
    if len(ic_frame) <= 1:
        recent_window = len(ic_frame)
    if recent_window >= len(ic_frame):
        recent_window = max(len(ic_frame) // 2, 1)
    historical = ic_frame.iloc[:-recent_window].copy()
    recent = ic_frame.iloc[-recent_window:].copy()
    if historical.empty:
        historical = ic_frame.iloc[: len(ic_frame) - 1].copy()
        recent = ic_frame.iloc[len(ic_frame) - 1 :].copy()

    table = pd.concat(
        [
            historical.assign(segment="historical"),
            recent.assign(segment="recent"),
        ]
    ).reset_index(names="date_")

    historical_ic_mean = _safe_mean(historical["ic"])
    recent_ic_mean = _safe_mean(recent["ic"])
    historical_rank_ic_mean = _safe_mean(historical["rank_ic"])
    recent_rank_ic_mean = _safe_mean(recent["rank_ic"])
    historical_ic_ir = _safe_ir(historical["ic"])
    recent_ic_ir = _safe_ir(recent["ic"])
    historical_rank_ic_ir = _safe_ir(historical["rank_ic"])
    recent_rank_ic_ir = _safe_ir(recent["rank_ic"])

    ic_gap = float(recent_ic_mean - historical_ic_mean)
    rank_ic_gap = float(recent_rank_ic_mean - historical_rank_ic_mean)
    ic_ratio = float(recent_ic_mean / historical_ic_mean) if abs(historical_ic_mean) > 1e-12 else float("nan")
    rank_ic_ratio = float(recent_rank_ic_mean / historical_rank_ic_mean) if abs(historical_rank_ic_mean) > 1e-12 else float("nan")
    ic_sign_consistent = (
        abs(historical_ic_mean) <= 1e-12
        or abs(recent_ic_mean) <= 1e-12
        or np.sign(historical_ic_mean) == np.sign(recent_ic_mean)
    )
    rank_ic_sign_consistent = (
        abs(historical_rank_ic_mean) <= 1e-12
        or abs(recent_rank_ic_mean) <= 1e-12
        or np.sign(historical_rank_ic_mean) == np.sign(recent_rank_ic_mean)
    )

    passed = (
        ic_sign_consistent
        and
        abs(recent_ic_mean) >= float(min_recent_ic_mean)
        and (abs(historical_ic_mean) <= 1e-12 or abs(recent_ic_mean) >= abs(historical_ic_mean) * float(min_recent_to_history_ratio))
        and rank_ic_sign_consistent
        and abs(recent_rank_ic_mean) >= float(min_recent_rank_ic_mean)
        and (abs(historical_rank_ic_mean) <= 1e-12 or abs(recent_rank_ic_mean) >= abs(historical_rank_ic_mean) * float(min_recent_rank_to_history_ratio))
    )

    return FactorRecentICResult(
        method="recent_ic_stability",
        period_column=chosen_period,
        recent_window=int(len(recent)),
        historical_window=int(len(historical)),
        table=table,
        historical_ic_mean=historical_ic_mean,
        recent_ic_mean=recent_ic_mean,
        historical_rank_ic_mean=historical_rank_ic_mean,
        recent_rank_ic_mean=recent_rank_ic_mean,
        historical_ic_ir=historical_ic_ir,
        recent_ic_ir=recent_ic_ir,
        historical_rank_ic_ir=historical_rank_ic_ir,
        recent_rank_ic_ir=recent_rank_ic_ir,
        ic_gap=ic_gap,
        rank_ic_gap=rank_ic_gap,
        ic_ratio=ic_ratio,
        rank_ic_ratio=rank_ic_ratio,
        passed=passed,
    )


def factor_effectiveness_test(
    factor_data: TigerFactorData,
    *,
    existing_factors: list[pd.DataFrame] | None = None,
    config: FactorEffectivenessConfig | None = None,
    benchmark_returns: pd.Series | None = None,
) -> dict[str, Any]:
    return _test_factor_effectiveness(
        factor_data,
        existing_factors=existing_factors,
        config=config,
        benchmark_returns=benchmark_returns,
    )


def factor_stability_test(
    factor_data: TigerFactorData,
    *,
    existing_factors: list[pd.DataFrame] | None = None,
    config: FactorEffectivenessConfig | None = None,
    benchmark_returns: pd.Series | None = None,
    recent_window: int = 60,
    period: str | None = None,
    min_recent_ic_mean: float = 0.0,
    min_recent_to_history_ratio: float = 0.5,
    min_recent_rank_ic_mean: float = 0.0,
    min_recent_rank_to_history_ratio: float = 0.5,
) -> dict[str, Any]:
    effectiveness = factor_effectiveness_test(
        factor_data,
        existing_factors=existing_factors,
        config=config,
        benchmark_returns=benchmark_returns,
    )
    decay = factor_decay_test(factor_data)
    recent_ic = factor_recent_ic_test(
        factor_data,
        period=period,
        recent_window=recent_window,
        min_recent_ic_mean=min_recent_ic_mean,
        min_recent_to_history_ratio=min_recent_to_history_ratio,
        min_recent_rank_ic_mean=min_recent_rank_ic_mean,
        min_recent_rank_to_history_ratio=min_recent_rank_to_history_ratio,
    )

    return {
        "passed": bool(effectiveness.get("passed", False)) and bool(recent_ic.passed),
        "effectiveness": effectiveness,
        "decay": decay.to_dict(),
        "recent_ic": recent_ic.to_dict(),
    }


test_factor_effectiveness = factor_effectiveness_test

__all__ = [
    "FactorDecayResult",
    "FactorEffectivenessConfig",
    "FactorRecentICResult",
    "factor_decay_test",
    "factor_effectiveness_test",
    "factor_recent_ic_test",
    "factor_stability_test",
    "test_factor_effectiveness",
]
