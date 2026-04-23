from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from tiger_factors.factor_evaluation.core import FactorEvaluation
from tiger_factors.factor_evaluation.core import evaluate_factor_panel
from tiger_factors.factor_evaluation.performance import factor_alpha_beta
from tiger_factors.factor_evaluation.performance import factor_information_coefficient
from tiger_factors.factor_evaluation.performance import factor_returns
from tiger_factors.factor_evaluation.utils import TigerFactorData
from tiger_factors.factor_evaluation.utils import _rowwise_cross_sectional_corr
from tiger_factors.factor_evaluation.utils import get_forward_returns_columns


@dataclass(frozen=True)
class FactorEffectivenessConfig:
    min_abs_ic_mean: float = 0.005
    min_abs_ic_ir: float = 0.05
    min_sharpe: float = 0.30
    max_turnover: float = 0.70
    min_decay_score: float = 0.20
    min_capacity_score: float = 0.20
    max_correlation_penalty: float = 0.60
    min_regime_robustness: float = 0.60
    min_out_of_sample_stability: float = 0.60


def _safe_numeric_series(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _summary_row(evaluation: FactorEvaluation) -> dict[str, float]:
    return {
        "ic_mean": float(evaluation.ic_mean),
        "ic_ir": float(evaluation.ic_ir),
        "rank_ic_mean": float(evaluation.rank_ic_mean),
        "sharpe": float(evaluation.sharpe),
        "turnover": float(evaluation.turnover),
        "decay_score": float(evaluation.decay_score),
        "capacity_score": float(evaluation.capacity_score),
        "correlation_penalty": float(evaluation.correlation_penalty),
        "regime_robustness": float(evaluation.regime_robustness),
        "out_of_sample_stability": float(evaluation.out_of_sample_stability),
        "fitness": float(evaluation.fitness),
        "factor_autocorr_mean": float(evaluation.factor_autocorr_mean),
        "rank_factor_autocorr_mean": float(evaluation.rank_factor_autocorr_mean),
        "benchmark_alpha": float(evaluation.benchmark_alpha),
        "benchmark_beta": float(evaluation.benchmark_beta),
        "benchmark_r2": float(evaluation.benchmark_r2),
        "benchmark_n_obs": int(evaluation.benchmark_n_obs),
    }


def _rolling_mean_ir(series: pd.Series, window: int) -> pd.DataFrame:
    rolling_mean = series.rolling(window=window, min_periods=max(3, window // 3)).mean()
    rolling_std = series.rolling(window=window, min_periods=max(3, window // 3)).std(ddof=0)
    rolling_ir = rolling_mean / rolling_std.replace(0, np.nan)
    out = pd.DataFrame(
        {
            "rolling_mean": rolling_mean,
            "rolling_std": rolling_std,
            "rolling_ir": rolling_ir,
        }
    )
    return out


def analyze_ic_ir_by_horizon(
    factor_data: TigerFactorData,
) -> pd.DataFrame:
    frame = factor_data.factor_data.copy()
    periods = get_forward_returns_columns(frame.columns)
    if not periods:
        return pd.DataFrame(columns=["period", "ic_mean", "ic_std", "ic_ir", "rank_ic_mean", "rank_ic_std", "rank_ic_ir", "ic_positive_ratio", "rank_ic_positive_ratio", "n_obs"])

    records: list[dict[str, float | int | str]] = []
    factor = frame["factor"]
    for period in periods:
        if period not in frame.columns:
            continue
        numeric = pd.concat([factor.rename("factor"), _safe_numeric_series(frame[period]).rename("return")], axis=1).dropna()
        if numeric.empty:
            continue
        ic_daily = numeric.groupby(level=0).apply(lambda g: g["factor"].corr(g["return"]))
        rank_ic_daily = numeric.groupby(level=0).apply(lambda g: g["factor"].corr(g["return"], method="spearman"))
        ic_mean = float(ic_daily.mean()) if not ic_daily.empty else 0.0
        ic_std = float(ic_daily.std(ddof=0)) if not ic_daily.dropna().empty else 0.0
        rank_ic_mean = float(rank_ic_daily.mean()) if not rank_ic_daily.empty else 0.0
        rank_ic_std = float(rank_ic_daily.std(ddof=0)) if not rank_ic_daily.dropna().empty else 0.0
        records.append(
            {
                "period": str(period),
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "ic_ir": float(ic_mean / ic_std) if ic_std > 1e-12 else 0.0,
                "rank_ic_mean": rank_ic_mean,
                "rank_ic_std": rank_ic_std,
                "rank_ic_ir": float(rank_ic_mean / rank_ic_std) if rank_ic_std > 1e-12 else 0.0,
                "ic_positive_ratio": float((ic_daily > 0).mean()) if not ic_daily.empty else 0.0,
                "rank_ic_positive_ratio": float((rank_ic_daily > 0).mean()) if not rank_ic_daily.empty else 0.0,
                "n_obs": int(len(numeric)),
            }
        )

    return pd.DataFrame(records).sort_values("period").reset_index(drop=True)


def analyze_factor_exposure(
    factor_data: TigerFactorData,
    *,
    window: int = 20,
) -> dict[str, Any]:
    panel = factor_data.factor_panel.copy()
    if panel.empty:
        return {"error": "factor panel is empty"}

    panel = panel.sort_index()
    latest_date = panel.index.max()
    latest_row = panel.loc[latest_date].dropna()
    all_values = panel.stack(future_stack=True)
    all_values = _safe_numeric_series(all_values).dropna()
    if all_values.empty:
        return {"error": "no non-null factor values"}

    latest_rank = latest_row.rank(pct=True) if not latest_row.empty else pd.Series(dtype=float)
    latest_sorted = latest_row.sort_values(ascending=False)
    cross_section_mean = panel.mean(axis=1)
    cross_section_std = panel.std(axis=1, ddof=0)
    rolling_mean = cross_section_mean.rolling(window=window, min_periods=1).mean()
    rolling_std = cross_section_mean.rolling(window=window, min_periods=1).std(ddof=0)
    hist, bins = np.histogram(all_values.to_numpy(dtype=np.float64, copy=False), bins=min(50, max(10, int(np.sqrt(len(all_values))))))

    return {
        "latest_date": str(latest_date),
        "latest_cross_section": pd.DataFrame(
            {
                "factor_value": latest_row,
                "percentile": latest_rank.reindex(latest_row.index),
            }
        ).sort_values("factor_value", ascending=False),
        "latest_top_exposure": latest_sorted.head(10).to_dict(),
        "latest_bottom_exposure": latest_sorted.tail(10).to_dict(),
        "distribution": {
            "count": int(all_values.count()),
            "mean": float(all_values.mean()),
            "std": float(all_values.std(ddof=0)),
            "min": float(all_values.min()),
            "max": float(all_values.max()),
            "percentiles": {
                "p1": float(all_values.quantile(0.01)),
                "p5": float(all_values.quantile(0.05)),
                "p25": float(all_values.quantile(0.25)),
                "p50": float(all_values.quantile(0.50)),
                "p75": float(all_values.quantile(0.75)),
                "p95": float(all_values.quantile(0.95)),
                "p99": float(all_values.quantile(0.99)),
            },
        },
        "rolling": pd.DataFrame(
            {
                "cross_section_mean": cross_section_mean,
                "cross_section_std": cross_section_std,
                "rolling_mean": rolling_mean,
                "rolling_std": rolling_std,
            }
        ),
        "histogram": {
            "bins": [float(x) for x in bins],
            "counts": [int(x) for x in hist],
        },
    }


def test_factor_effectiveness(
    factor_data: TigerFactorData,
    *,
    existing_factors: list[pd.DataFrame] | None = None,
    config: FactorEffectivenessConfig | None = None,
    benchmark_returns: pd.Series | None = None,
) -> dict[str, Any]:
    cfg = config if config is not None else FactorEffectivenessConfig()
    evaluation = evaluate_factor_panel(
        factor_data.factor_panel,
        factor_data.forward_returns,
        existing_factors=existing_factors,
        benchmark_returns=benchmark_returns,
    )
    summary = _summary_row(evaluation)
    checks = {
        "ic_mean": abs(summary["ic_mean"]) >= cfg.min_abs_ic_mean,
        "ic_ir": abs(summary["ic_ir"]) >= cfg.min_abs_ic_ir,
        "sharpe": summary["sharpe"] >= cfg.min_sharpe,
        "turnover": summary["turnover"] <= cfg.max_turnover,
        "decay_score": summary["decay_score"] >= cfg.min_decay_score,
        "capacity_score": summary["capacity_score"] >= cfg.min_capacity_score,
        "correlation_penalty": summary["correlation_penalty"] <= cfg.max_correlation_penalty,
        "regime_robustness": summary["regime_robustness"] >= cfg.min_regime_robustness,
        "out_of_sample_stability": summary["out_of_sample_stability"] >= cfg.min_out_of_sample_stability,
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "summary": summary,
        "checks": checks,
        "score": float(summary["fitness"]),
        "grade": "pass" if passed else "reject",
    }


def analyze_factor_attribution(
    factor_data: TigerFactorData,
    *,
    benchmark_returns: pd.Series | None = None,
) -> dict[str, Any]:
    panel = factor_data.factor_data.copy()
    if panel.empty:
        return {"error": "factor data is empty"}

    factor_portfolio_returns = factor_returns(panel, demeaned=True, group_adjust=False, equal_weight=False)
    primary_period = factor_portfolio_returns.columns[0] if not factor_portfolio_returns.empty else None
    primary_series = factor_portfolio_returns[primary_period] if primary_period is not None else pd.Series(dtype=float)
    primary_series = _safe_numeric_series(primary_series).dropna()

    contribution = {
        "factor_portfolio": {
            "total_return": float((1.0 + primary_series).prod() - 1.0) if not primary_series.empty else 0.0,
            "annual_return": float(primary_series.mean() * 252) if not primary_series.empty else 0.0,
            "volatility": float(primary_series.std(ddof=0) * np.sqrt(252)) if not primary_series.empty else 0.0,
            "sharpe": float(primary_series.mean() / primary_series.std(ddof=0) * np.sqrt(252)) if primary_series.std(ddof=0) > 1e-12 else 0.0,
        }
    }

    alpha_beta = {}
    if benchmark_returns is not None and not primary_series.empty:
        aligned = pd.concat([primary_series.rename("portfolio"), _safe_numeric_series(pd.Series(benchmark_returns)).rename("benchmark")], axis=1).dropna()
        if not aligned.empty:
            alpha_beta = factor_alpha_beta(aligned["portfolio"], aligned["benchmark"])
    elif not primary_series.empty:
        alpha_beta = {
            "alpha": 0.0,
            "beta": 0.0,
            "r2": 0.0,
            "n_obs": int(len(primary_series)),
        }

    return_decomposition = pd.DataFrame(
        {
            "factor_cumulative": (1.0 + primary_series).cumprod() if not primary_series.empty else pd.Series(dtype=float),
        }
    )
    if benchmark_returns is not None:
        benchmark_series = _safe_numeric_series(pd.Series(benchmark_returns)).dropna()
        if not benchmark_series.empty:
            aligned = pd.concat([primary_series.rename("factor"), benchmark_series.rename("benchmark")], axis=1).dropna()
            if not aligned.empty:
                return_decomposition = pd.DataFrame(
                    {
                        "factor_cumulative": (1.0 + aligned["factor"]).cumprod(),
                        "benchmark_cumulative": (1.0 + aligned["benchmark"]).cumprod(),
                        "excess_cumulative": (1.0 + (aligned["factor"] - aligned["benchmark"])).cumprod(),
                    }
                )

    return {
        "factor_contribution": contribution,
        "alpha_beta": alpha_beta,
        "return_decomposition": return_decomposition,
        "factor_portfolio_returns": factor_portfolio_returns,
    }


def monitor_factor_dynamics(
    factor_data: TigerFactorData,
    *,
    window: int = 60,
) -> pd.DataFrame:
    frame = factor_data.factor_data.copy()
    if frame.empty:
        return pd.DataFrame(columns=["date_", "ic", "rank_ic", "rolling_ic_mean", "rolling_ic_ir", "coverage", "turnover"])

    factor = frame["factor"]
    periods = get_forward_returns_columns(frame.columns)
    if not periods:
        return pd.DataFrame()
    primary_period = periods[0]
    forward = _safe_numeric_series(frame[primary_period])
    aligned = pd.concat([factor.rename("factor"), forward.rename("forward")], axis=1).dropna()
    if aligned.empty:
        return pd.DataFrame()

    ic_series = aligned.groupby(level=0).apply(lambda g: g["factor"].corr(g["forward"]))
    rank_ic_series = aligned.groupby(level=0).apply(lambda g: g["factor"].corr(g["forward"], method="spearman"))
    factor_panel = factor_data.factor_panel.sort_index()
    cross_section_mean = factor_panel.mean(axis=1)
    coverage = factor_panel.notna().mean(axis=1)
    turnover = factor_panel.rank(axis=1, pct=True).diff().abs().mean(axis=1)
    monitor = pd.DataFrame(
        {
            "ic": ic_series,
            "rank_ic": rank_ic_series,
            "cross_section_mean": cross_section_mean.reindex(ic_series.index),
            "coverage": coverage.reindex(ic_series.index),
            "turnover": turnover.reindex(ic_series.index),
        }
    )
    rolling = _rolling_mean_ir(monitor["ic"], window=window)
    monitor["rolling_ic_mean"] = rolling["rolling_mean"]
    monitor["rolling_ic_ir"] = rolling["rolling_ir"]
    monitor["rolling_ic_std"] = rolling["rolling_std"]
    monitor["rolling_rank_ic_mean"] = monitor["rank_ic"].rolling(window=window, min_periods=max(3, window // 3)).mean()
    monitor["rolling_coverage"] = monitor["coverage"].rolling(window=window, min_periods=max(3, window // 3)).mean()
    monitor["rolling_turnover"] = monitor["turnover"].rolling(window=window, min_periods=max(3, window // 3)).mean()
    monitor.index.name = "date_"
    return monitor.reset_index()


__all__ = [
    "FactorEffectivenessConfig",
    "analyze_factor_attribution",
    "analyze_factor_exposure",
    "analyze_ic_ir_by_horizon",
    "monitor_factor_dynamics",
    "test_factor_effectiveness",
]
