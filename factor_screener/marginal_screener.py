from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from tiger_factors.factor_screener._evaluation_io import load_return_series
from tiger_factors.factor_screener._evaluation_io import load_summary_row
from tiger_factors.factor_store import FactorSpec
from tiger_factors.factor_store import FactorStore
from tiger_factors.factor_screener.selection import _penalty_score
from tiger_factors.factor_screener.selection import _weighted_metric_score
from tiger_factors.utils.returns_analysis import annualized_return
from tiger_factors.utils.returns_analysis import annualized_sortino
from tiger_factors.utils.returns_analysis import annualized_volatility_value
from tiger_factors.utils.returns_analysis import max_drawdown
from tiger_factors.utils.returns_analysis import sharpe_ratio
from tiger_factors.utils.returns_analysis import win_rate


def _summary_metrics_frame(summary_rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not summary_rows:
        return pd.DataFrame()
    frame = pd.DataFrame(summary_rows).copy()
    if "factor_name" not in frame.columns:
        return frame
    return frame.reset_index(drop=True)


def _aggregate_summary_metrics(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {}
    metrics: dict[str, float] = {}
    for column in frame.columns:
        if column == "factor_name":
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            metrics[column] = float(values.dropna().mean())
    return metrics


def _portfolio_metrics(returns: pd.Series, *, annualization: int) -> dict[str, float]:
    if returns.empty:
        return {
            "annual_return": float("nan"),
            "sharpe": float("nan"),
            "sortino": float("nan"),
            "volatility": float("nan"),
            "max_drawdown": float("nan"),
            "win_rate": float("nan"),
            "calmar": float("nan"),
        }
    ann_return = float(annualized_return(returns, annualization=annualization))
    dd = float(max_drawdown(returns))
    return {
        "annual_return": ann_return,
        "sharpe": float(sharpe_ratio(returns, annualization=annualization)),
        "sortino": float(annualized_sortino(returns, annualization=annualization)),
        "volatility": float(annualized_volatility_value(returns, annualization=annualization)),
        "max_drawdown": dd,
        "win_rate": float(win_rate(returns)),
        "calmar": float(ann_return / abs(dd)) if pd.notna(dd) and dd != 0 else float("nan"),
    }


def _objective_from_metrics(
    metrics: pd.DataFrame,
    *,
    score_fields: tuple[str, ...],
    score_weights: tuple[float, ...],
    penalty_fields: tuple[str, ...],
    penalty_weights: tuple[float, ...],
    fallback_field: str,
) -> tuple[float, dict[str, float]]:
    score = _weighted_metric_score(
        metrics,
        fields=score_fields,
        weights=score_weights,
        fallback_field=fallback_field,
    )
    penalty = _penalty_score(
        metrics,
        fields=penalty_fields,
        weights=penalty_weights,
    )
    score_value = float(score.iloc[0]) if not score.empty else 0.0
    penalty_value = float(penalty.iloc[0]) if not penalty.empty else 0.0
    objective = score_value - penalty_value
    return objective, {
        "score": score_value,
        "penalty": penalty_value,
        "objective": objective,
    }


@dataclass(frozen=True)
class MarginalScreenerSpec:
    score_fields: tuple[str, ...] = ("annual_return", "sharpe", "sortino", "ic_mean", "ic_ir")
    score_weights: tuple[float, ...] = (0.35, 0.25, 0.15, 0.15, 0.10)
    penalty_fields: tuple[str, ...] = ("max_drawdown", "turnover")
    penalty_weights: tuple[float, ...] = (0.35, 0.15)
    fallback_score_field: str = "selected_score"
    annualization: int = 252
    min_improvement: float = 0.0
    min_base_score: float | None = None


@dataclass(frozen=True)
class MarginalScreenerResult:
    spec: MarginalScreenerSpec
    factor_specs: tuple[FactorSpec, ...]
    screened_at: pd.Timestamp
    summary: pd.DataFrame
    selection_summary: pd.DataFrame
    return_series: dict[str, pd.Series]
    return_panel: pd.DataFrame
    portfolio_returns: pd.Series
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
        long_frame["return_mode"] = "marginal"
        long_frame = long_frame.dropna(subset=["date_", "return"])
        return long_frame.loc[:, ["date_", "factor", "return", "return_mode"]].sort_values(
            ["date_", "factor"],
            kind="stable",
        )

    def to_summary(self) -> dict[str, Any]:
        if self.portfolio_returns.empty:
            return_start = None
            return_end = None
        else:
            index = pd.DatetimeIndex(self.portfolio_returns.index).dropna().sort_values()
            return_start = None if index.empty else index[0].isoformat()
            return_end = None if index.empty else index[-1].isoformat()
        selected_names = self.selected_factor_names
        return {
            "screened_at": self.screened_at.isoformat(),
            "mode": "marginal",
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
        }


class MarginalScreener:
    def __init__(
        self,
        spec: MarginalScreenerSpec,
        *,
        factor_specs: Iterable[FactorSpec],
        store: FactorStore | None = None,
    ) -> None:
        self.spec = spec
        self.factor_specs = tuple(factor_specs)
        self.store = store or FactorStore()

    def run(self) -> MarginalScreenerResult:
        screened_at = pd.Timestamp.now(tz="UTC")
        if not self.factor_specs:
            empty = pd.DataFrame()
            return MarginalScreenerResult(
                spec=self.spec,
                factor_specs=tuple(),
                screened_at=screened_at,
                summary=empty,
                selection_summary=empty,
                return_series={},
                return_panel=empty,
                portfolio_returns=pd.Series(dtype=float),
                missing_return_factors=(),
            )

        summary_rows = []
        return_series_map: dict[str, pd.Series] = {}
        missing_return_factors: list[str] = []
        for spec in self.factor_specs:
            row = load_summary_row(self.store, spec)
            if row is not None:
                summary_rows.append(row)
            series = load_return_series(self.store, spec)
            if series is None or series.empty:
                missing_return_factors.append(spec.table_name)
                continue
            return_series_map[spec.table_name] = series

        summary = _summary_metrics_frame(summary_rows)
        selected_names: list[str] = []
        selection_records: list[dict[str, Any]] = []
        current_objective = float("-inf")

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
            if not trial_panel.empty:
                trial_portfolio = trial_panel.mean(axis=1, skipna=True).dropna()
            else:
                trial_portfolio = pd.Series(dtype=float)

            selected_summary = summary.loc[summary["factor_name"].astype(str).isin(trial_names)].copy() if not summary.empty and "factor_name" in summary.columns else pd.DataFrame()
            trial_metrics = _portfolio_metrics(trial_portfolio, annualization=self.spec.annualization)
            trial_metrics.update(_aggregate_summary_metrics(selected_summary))
            trial_metrics["factor_count"] = float(len(trial_names))

            metric_frame = pd.DataFrame([trial_metrics])
            objective, objective_breakdown = _objective_from_metrics(
                metric_frame,
                score_fields=self.spec.score_fields,
                score_weights=self.spec.score_weights,
                penalty_fields=self.spec.penalty_fields,
                penalty_weights=self.spec.penalty_weights,
                fallback_field=self.spec.fallback_score_field,
            )

            if not selected_names:
                keep = objective >= (self.spec.min_base_score if self.spec.min_base_score is not None else self.spec.min_improvement)
            else:
                keep = objective >= current_objective + self.spec.min_improvement

            record = {
                "factor_name": name,
                "selected": bool(keep),
                "marginal_score": objective_breakdown["score"],
                "marginal_penalty": objective_breakdown["penalty"],
                "marginal_objective": objective_breakdown["objective"],
                **trial_metrics,
            }
            selection_records.append(record)

            if keep:
                selected_names.append(name)
                current_objective = objective

        selected_return_series = {
            name: return_series_map[name]
            for name in selected_names
            if name in return_series_map
        }
        if selected_return_series:
            return_panel = pd.concat(selected_return_series, axis=1).sort_index()
            return_panel.index = pd.to_datetime(return_panel.index, errors="coerce")
            return_panel = return_panel.loc[~return_panel.index.isna()].sort_index()
            portfolio_returns = return_panel.mean(axis=1, skipna=True).dropna()
        else:
            return_panel = pd.DataFrame()
            portfolio_returns = pd.Series(dtype=float)

        selection_summary = pd.DataFrame(selection_records)
        return MarginalScreenerResult(
            spec=self.spec,
            factor_specs=self.factor_specs,
            screened_at=screened_at,
            summary=summary,
            selection_summary=selection_summary,
            return_series=selected_return_series,
            return_panel=return_panel,
            portfolio_returns=portfolio_returns,
            missing_return_factors=tuple(missing_return_factors),
        )


def run_marginal_screener(
    spec: MarginalScreenerSpec,
    factor_specs: Iterable[FactorSpec],
    *,
    store: FactorStore | None = None,
) -> MarginalScreenerResult:
    return MarginalScreener(
        spec,
        factor_specs=factor_specs,
        store=store,
    ).run()


__all__ = [
    "MarginalScreener",
    "MarginalScreenerResult",
    "MarginalScreenerSpec",
    "run_marginal_screener",
]
