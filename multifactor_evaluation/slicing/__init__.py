from __future__ import annotations

from typing import Any
from typing import Iterable

import pandas as pd

from tiger_factors.factor_evaluation.horizon import HoldingPeriodAnalyzer
from tiger_factors.multifactor_evaluation._inputs import coerce_factor_series
from tiger_factors.multifactor_evaluation._inputs import coerce_labels_frame
from tiger_factors.multifactor_evaluation._inputs import coerce_price_panel
from tiger_factors.multifactor_evaluation.screening import add_cost_analysis
from tiger_factors.multifactor_evaluation.screening import evaluate_factor_with_filter


class AutoSlicingAnalyzer:
    def __init__(
        self,
        factor: pd.Series | pd.DataFrame,
        prices: pd.DataFrame,
        labels: pd.DataFrame,
        *,
        horizons: Iterable[int] = (1, 3, 5, 10, 20),
        quantiles: int = 5,
        periods_per_year: int = 252,
        long_short_pct: float = 0.2,
        min_sample_dates: int = 30,
        min_names: int | None = 10,
        cost_rate: float = 0.001,
    ) -> None:
        self.factor = coerce_factor_series(factor)
        self.prices = coerce_price_panel(prices)
        self.labels = coerce_labels_frame(labels)
        self.horizons = list(horizons)
        self.quantiles = quantiles
        self.periods_per_year = periods_per_year
        self.long_short_pct = long_short_pct
        self.min_sample_dates = min_sample_dates
        self.min_names = min_names
        self.cost_rate = cost_rate

    def _subset_factor_by_mask(self, mask: pd.Series) -> pd.Series:
        aligned = mask.reindex(self.factor.index).fillna(False)
        return self.factor[aligned]

    @staticmethod
    def _count_dates(factor_subset: pd.Series) -> int:
        if factor_subset.empty:
            return 0
        return int(factor_subset.index.get_level_values(0).nunique())

    def analyze_slice(self, slice_mask: pd.Series, slice_name: dict[str, Any]) -> dict[str, Any] | None:
        factor_sub = self._subset_factor_by_mask(slice_mask)
        n_dates = self._count_dates(factor_sub)
        if factor_sub.empty or n_dates < self.min_sample_dates:
            return None

        analyzer = HoldingPeriodAnalyzer(
            factor_sub,
            self.prices,
            quantiles=self.quantiles,
            periods_per_year=self.periods_per_year,
            long_short_pct=self.long_short_pct,
            min_names=self.min_names,
        )
        result = add_cost_analysis(analyzer.run(self.horizons), cost_rate=self.cost_rate)
        decision = evaluate_factor_with_filter(result)

        summary_row: dict[str, Any] = {
            **slice_name,
            "n_dates": n_dates,
            "usable": decision.get("usable"),
            "direction": decision.get("direction"),
            "quality": decision.get("quality"),
            "best_horizon": decision.get("best_horizon"),
            "recommendation": decision.get("recommendation"),
            "score": decision.get("score"),
        }

        best_horizon = decision.get("best_horizon")
        if best_horizon is not None and not result.empty:
            best_row = result.loc[result["horizon"] == best_horizon]
            if not best_row.empty:
                row = best_row.iloc[0]
                summary_row.update(
                    {
                        "mean_ic": row.get("mean_ic"),
                        "ic_ir": row.get("ic_ir"),
                        "ann_return": row.get("ann_return"),
                        "net_sharpe": row.get("net_sharpe"),
                        "avg_turnover": row.get("avg_turnover"),
                        "max_drawdown": row.get("max_drawdown"),
                    }
                )

        return {"summary": summary_row, "detail": result}

    def run(
        self,
        slice_cols: list[str],
        *,
        dropna_labels: bool = True,
    ) -> tuple[pd.DataFrame, dict[tuple[Any, ...], pd.DataFrame]]:
        if not slice_cols:
            raise ValueError("slice_cols cannot be empty")

        labels = self.labels.copy()
        if dropna_labels:
            labels = labels.dropna(subset=slice_cols)

        summaries: list[dict[str, Any]] = []
        detail_map: dict[tuple[Any, ...], pd.DataFrame] = {}

        for keys, group_df in labels.groupby(slice_cols, dropna=False, observed=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            slice_name = dict(zip(slice_cols, keys))
            mask = pd.Series(False, index=self.labels.index)
            mask.loc[group_df.index] = True
            analyzed = self.analyze_slice(mask, slice_name)
            if analyzed is None:
                continue
            summaries.append(analyzed["summary"])
            detail_map[keys] = analyzed["detail"]

        summary_df = pd.DataFrame(summaries)
        if not summary_df.empty:
            sort_col = "score" if "score" in summary_df.columns else "mean_ic"
            summary_df = summary_df.sort_values(sort_col, ascending=False, na_position="last").reset_index(drop=True)
        return summary_df, detail_map


__all__ = ["AutoSlicingAnalyzer"]
