from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Iterable
from typing import Mapping

import numpy as np
import pandas as pd

from tiger_factors.factor_backtest import run_return_backtest
from tiger_factors.factor_screener._evaluation_io import load_return_series
from tiger_factors.factor_screener._evaluation_io import load_summary_row
from tiger_factors.factor_store import FactorSpec
from tiger_factors.factor_store import FactorStore


def _metric_value(stats: Mapping[str, float], field: str) -> float:
    value = float(pd.to_numeric(pd.Series([stats.get(field, np.nan)]), errors="coerce").iloc[0])
    if pd.isna(value):
        return float("nan")
    if field in {"ann_volatility", "volatility"}:
        return -abs(value)
    return value


def _objective_from_stats(
    stats: Mapping[str, float],
    *,
    score_fields: tuple[str, ...],
    score_weights: tuple[float, ...],
    penalty_fields: tuple[str, ...],
    penalty_weights: tuple[float, ...],
) -> tuple[float, dict[str, float]]:
    if len(score_fields) != len(score_weights):
        raise ValueError("score_fields and score_weights must have the same length")
    if len(penalty_fields) != len(penalty_weights):
        raise ValueError("penalty_fields and penalty_weights must have the same length")

    score = 0.0
    for field, weight in zip(score_fields, score_weights):
        value = _metric_value(stats, field)
        if pd.notna(value):
            score += float(weight) * value

    penalty = 0.0
    for field, weight in zip(penalty_fields, penalty_weights):
        value = _metric_value(stats, field)
        if pd.notna(value):
            penalty += float(weight) * value

    objective = score - penalty
    return objective, {"score": score, "penalty": penalty, "objective": objective}


@dataclass(frozen=True)
class BacktestMarginalScreenerSpec:
    return_mode: str = "long_short"
    annual_trading_days: int = 252
    score_fields: tuple[str, ...] = ("ann_return", "sharpe", "calmar", "win_rate")
    score_weights: tuple[float, ...] = (0.35, 0.30, 0.20, 0.15)
    penalty_fields: tuple[str, ...] = ("ann_volatility",)
    penalty_weights: tuple[float, ...] = (0.10,)
    min_improvement: float = 0.0
    min_base_score: float = 0.0


@dataclass(frozen=True)
class BacktestMarginalScreenerResult:
    spec: BacktestMarginalScreenerSpec
    factor_specs: tuple[FactorSpec, ...]
    screened_at: pd.Timestamp
    summary: pd.DataFrame
    selection_summary: pd.DataFrame
    return_series: dict[str, pd.Series]
    return_panel: pd.DataFrame
    portfolio_returns: pd.Series
    backtest: pd.DataFrame
    stats: dict[str, Any]
    missing_return_factors: tuple[str, ...] = ()

    @property
    def selected_factor_names(self) -> list[str]:
        if self.selection_summary.empty or "selected" not in self.selection_summary.columns:
            return []
        if "factor_name" not in self.selection_summary.columns:
            return []
        return self.selection_summary.loc[self.selection_summary["selected"].fillna(False), "factor_name"].astype(str).tolist()

    @property
    def selected_factor_specs(self) -> list[FactorSpec]:
        spec_map = {spec.table_name: spec for spec in self.factor_specs}
        return [spec_map[name] for name in self.selected_factor_names if name in spec_map]

    @property
    def rejected_factor_names(self) -> list[str]:
        if self.selection_summary.empty or "selected" not in self.selection_summary.columns:
            return []
        frame = self.selection_summary.loc[~self.selection_summary["selected"].fillna(False)]
        if "factor_name" not in frame.columns:
            return []
        return frame["factor_name"].astype(str).tolist()

    @property
    def rejected_factor_specs(self) -> list[FactorSpec]:
        spec_map = {spec.table_name: spec for spec in self.factor_specs}
        return [spec_map[name] for name in self.rejected_factor_names if name in spec_map]

    @property
    def return_long(self) -> pd.DataFrame:
        if self.return_panel.empty:
            return pd.DataFrame(columns=["date_", "factor", "return", "return_mode"])
        long_frame = (
            self.return_panel.copy()
            .sort_index()
            .stack(future_stack=True)
            .rename("return")
            .reset_index()
        )
        if long_frame.empty:
            return pd.DataFrame(columns=["date_", "factor", "return", "return_mode"])
        long_frame.columns = ["date_", "factor", "return"]
        long_frame["return_mode"] = "backtest_marginal"
        long_frame = long_frame.dropna(subset=["date_", "return"])
        return long_frame.loc[:, ["date_", "factor", "return", "return_mode"]].sort_values(
            ["date_", "factor"],
            kind="stable",
        )

    def to_summary(self) -> dict[str, Any]:
        selected_names = self.selected_factor_names
        if self.portfolio_returns.empty:
            return_start = None
            return_end = None
        else:
            index = pd.DatetimeIndex(self.portfolio_returns.index).dropna().sort_values()
            return_start = None if index.empty else index[0].isoformat()
            return_end = None if index.empty else index[-1].isoformat()
        return {
            "screened_at": self.screened_at.isoformat(),
            "mode": "backtest_marginal",
            "factor_count": int(len(self.factor_specs)),
            "selected_factor_names": selected_names,
            "selected_count": int(len(selected_names)),
            "rejected_factor_names": self.rejected_factor_names,
            "missing_return_factors": list(self.missing_return_factors),
            "return_start": return_start,
            "return_end": return_end,
            "summary_rows": int(len(self.summary)),
            "selection_rows": int(len(self.selection_summary)),
            "portfolio_return_rows": int(len(self.portfolio_returns)),
            "portfolio_stats": self.stats.get("portfolio", {}),
        }


class BacktestMarginalScreener:
    def __init__(
        self,
        spec: BacktestMarginalScreenerSpec,
        *,
        factor_specs: Iterable[FactorSpec],
        store: FactorStore | None = None,
    ) -> None:
        self.spec = spec
        self.factor_specs = tuple(factor_specs)
        self.store = store or FactorStore()

    def run(self) -> BacktestMarginalScreenerResult:
        screened_at = pd.Timestamp.now(tz="UTC")
        if not self.factor_specs:
            empty = pd.DataFrame()
            return BacktestMarginalScreenerResult(
                spec=self.spec,
                factor_specs=tuple(),
                screened_at=screened_at,
                summary=empty,
                selection_summary=empty,
                return_series={},
                return_panel=empty,
                portfolio_returns=pd.Series(dtype=float),
                backtest=pd.DataFrame(),
                stats={"portfolio": {}, "benchmark": {}},
                missing_return_factors=(),
            )

        summary_rows = []
        return_series_map: dict[str, pd.Series] = {}
        missing_return_factors: list[str] = []
        for spec in self.factor_specs:
            row = load_summary_row(self.store, spec)
            if row is not None:
                summary_rows.append(row)
            series = load_return_series(self.store, spec, return_mode=self.spec.return_mode)
            if series is None or series.empty:
                missing_return_factors.append(spec.table_name)
                continue
            return_series_map[spec.table_name] = series

        summary = pd.DataFrame(summary_rows).reset_index(drop=True) if summary_rows else pd.DataFrame()
        selected_names: list[str] = []
        selection_records: list[dict[str, Any]] = []
        current_objective = float("-inf")
        current_backtest = pd.DataFrame()
        current_stats: dict[str, Any] = {"portfolio": {}}

        for spec in self.factor_specs:
            name = spec.table_name
            series = return_series_map.get(name)
            if series is None or series.empty:
                selection_records.append(
                    {
                        "factor_name": name,
                        "selected": False,
                        "reason": "missing_return",
                    }
                )
                continue

            trial_names = selected_names + [name]
            trial_series_map = {picked: return_series_map[picked] for picked in trial_names if picked in return_series_map}
            trial_panel = pd.concat(trial_series_map, axis=1).sort_index() if trial_series_map else pd.DataFrame()
            if trial_panel.empty:
                trial_portfolio = pd.Series(dtype=float)
                trial_backtest = pd.DataFrame()
                trial_stats = {"portfolio": {}, "benchmark": {}}
                objective = float("-inf")
                objective_breakdown = {"score": float("nan"), "penalty": float("nan"), "objective": float("-inf")}
            else:
                trial_portfolio = trial_panel.mean(axis=1, skipna=True).dropna()
                backtest_result = run_return_backtest(
                    trial_portfolio,
                    annual_trading_days=self.spec.annual_trading_days,
                )
                trial_backtest = backtest_result["backtest"]
                trial_stats = backtest_result["stats"]
                objective, objective_breakdown = _objective_from_stats(
                    trial_stats.get("portfolio", {}),
                    score_fields=self.spec.score_fields,
                    score_weights=self.spec.score_weights,
                    penalty_fields=self.spec.penalty_fields,
                    penalty_weights=self.spec.penalty_weights,
                )

            if not selected_names:
                keep = objective >= self.spec.min_base_score
            else:
                keep = objective >= current_objective + self.spec.min_improvement

            pre_portfolio_stats = current_stats.get("portfolio", {})
            post_portfolio_stats = trial_stats.get("portfolio", {})
            record = {
                "factor_name": name,
                "selected": bool(keep),
                "marginal_score": objective_breakdown["score"],
                "marginal_penalty": objective_breakdown["penalty"],
                "marginal_objective": objective_breakdown["objective"],
                "pre_objective": current_objective if selected_names else float("nan"),
                "post_objective": objective,
                "delta_objective": objective - current_objective if selected_names else objective,
                "pre_ann_return": float(pre_portfolio_stats.get("ann_return", np.nan)),
                "post_ann_return": float(post_portfolio_stats.get("ann_return", np.nan)),
                "delta_ann_return": float(post_portfolio_stats.get("ann_return", np.nan)) - float(pre_portfolio_stats.get("ann_return", np.nan)),
                "pre_sharpe": float(pre_portfolio_stats.get("sharpe", np.nan)),
                "post_sharpe": float(post_portfolio_stats.get("sharpe", np.nan)),
                "delta_sharpe": float(post_portfolio_stats.get("sharpe", np.nan)) - float(pre_portfolio_stats.get("sharpe", np.nan)),
                "pre_max_drawdown": float(pre_portfolio_stats.get("max_drawdown", np.nan)),
                "post_max_drawdown": float(post_portfolio_stats.get("max_drawdown", np.nan)),
                "delta_max_drawdown": float(post_portfolio_stats.get("max_drawdown", np.nan)) - float(pre_portfolio_stats.get("max_drawdown", np.nan)),
                "pre_calmar": float(pre_portfolio_stats.get("calmar", np.nan)),
                "post_calmar": float(post_portfolio_stats.get("calmar", np.nan)),
                "delta_calmar": float(post_portfolio_stats.get("calmar", np.nan)) - float(pre_portfolio_stats.get("calmar", np.nan)),
                "pre_win_rate": float(pre_portfolio_stats.get("win_rate", np.nan)),
                "post_win_rate": float(post_portfolio_stats.get("win_rate", np.nan)),
                "delta_win_rate": float(post_portfolio_stats.get("win_rate", np.nan)) - float(pre_portfolio_stats.get("win_rate", np.nan)),
                "pre_ann_volatility": float(pre_portfolio_stats.get("ann_volatility", np.nan)),
                "post_ann_volatility": float(post_portfolio_stats.get("ann_volatility", np.nan)),
                "delta_ann_volatility": float(post_portfolio_stats.get("ann_volatility", np.nan)) - float(pre_portfolio_stats.get("ann_volatility", np.nan)),
                "selected_count": len(trial_names),
            }
            selection_records.append(record)

            if keep:
                selected_names.append(name)
                current_objective = objective
                current_backtest = trial_backtest
                current_stats = trial_stats

        selected_return_series = {name: return_series_map[name] for name in selected_names if name in return_series_map}
        if selected_return_series:
            return_panel = pd.concat(selected_return_series, axis=1).sort_index()
            return_panel.index = pd.to_datetime(return_panel.index, errors="coerce")
            return_panel = return_panel.loc[~return_panel.index.isna()].sort_index()
            portfolio_returns = return_panel.mean(axis=1, skipna=True).dropna()
        else:
            return_panel = pd.DataFrame()
            portfolio_returns = pd.Series(dtype=float)

        selection_summary = pd.DataFrame(selection_records)
        return BacktestMarginalScreenerResult(
            spec=self.spec,
            factor_specs=self.factor_specs,
            screened_at=screened_at,
            summary=summary,
            selection_summary=selection_summary,
            return_series=selected_return_series,
            return_panel=return_panel,
            portfolio_returns=portfolio_returns,
            backtest=current_backtest,
            stats=current_stats,
            missing_return_factors=tuple(missing_return_factors),
        )


def run_backtest_marginal_screener(
    spec: BacktestMarginalScreenerSpec,
    factor_specs: Iterable[FactorSpec],
    *,
    store: FactorStore | None = None,
) -> BacktestMarginalScreenerResult:
    return BacktestMarginalScreener(
        spec,
        factor_specs=factor_specs,
        store=store,
    ).run()


__all__ = [
    "BacktestMarginalScreener",
    "BacktestMarginalScreenerResult",
    "BacktestMarginalScreenerSpec",
    "run_backtest_marginal_screener",
]
