from __future__ import annotations

from typing import Any
from typing import Iterable

import numpy as np
import pandas as pd

from tiger_factors.factor_evaluation.horizon import HoldingPeriodAnalyzer
from tiger_factors.multifactor_evaluation._inputs import coerce_factor_series
from tiger_factors.multifactor_evaluation._inputs import coerce_price_panel
from tiger_factors.multifactor_evaluation.screening import FactorFilterConfig
from tiger_factors.multifactor_evaluation.screening import add_cost_analysis
from tiger_factors.multifactor_evaluation.screening import evaluate_factor_with_filter


class FactorScreeningEngine:
    def __init__(
        self,
        factor_dict: dict[str, pd.Series | pd.DataFrame],
        prices: pd.DataFrame,
        *,
        horizons: Iterable[int] = (1, 3, 5, 10, 20),
        quantiles: int = 5,
        periods_per_year: int = 252,
        long_short_pct: float = 0.2,
        min_names: int | None = 10,
        cost_rate: float = 0.001,
        filter_config: FactorFilterConfig | None = None,
    ) -> None:
        self.factor_dict = {name: coerce_factor_series(factor) for name, factor in factor_dict.items()}
        self.prices = coerce_price_panel(prices)
        self.horizons = list(horizons)
        self.quantiles = quantiles
        self.periods_per_year = periods_per_year
        self.long_short_pct = long_short_pct
        self.min_names = min_names
        self.cost_rate = cost_rate
        self.filter_config = filter_config or FactorFilterConfig()

    def analyze_one_factor(self, factor_name: str, factor: pd.Series) -> dict[str, Any]:
        analyzer = HoldingPeriodAnalyzer(
            factor,
            self.prices,
            quantiles=self.quantiles,
            periods_per_year=self.periods_per_year,
            long_short_pct=self.long_short_pct,
            min_names=self.min_names,
        )
        result = add_cost_analysis(analyzer.run(self.horizons), cost_rate=self.cost_rate)
        decision = evaluate_factor_with_filter(result, self.filter_config)
        return {
            "summary": {"factor_name": factor_name, **decision},
            "detail": result,
        }

    def run(self) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        summaries: list[dict[str, Any]] = []
        detail_map: dict[str, pd.DataFrame] = {}

        for factor_name, factor in self.factor_dict.items():
            try:
                analyzed = self.analyze_one_factor(factor_name, factor)
                summaries.append(analyzed["summary"])
                detail_map[factor_name] = analyzed["detail"]
            except Exception as exc:  # pragma: no cover - defensive
                summaries.append(
                    {
                        "factor_name": factor_name,
                        "usable": False,
                        "recommendation": "ERROR",
                        "failed_rules": [f"exception:{exc}"],
                        "score": np.nan,
                    }
                )

        summary_df = pd.DataFrame(summaries)
        if summary_df.empty:
            return summary_df, detail_map

        usable_rank = summary_df["usable"].fillna(False).astype(int) if "usable" in summary_df.columns else 0
        if "score" not in summary_df.columns:
            summary_df["score"] = np.nan

        summary_df = (
            summary_df.assign(_usable_rank=usable_rank)
            .sort_values(["_usable_rank", "score", "factor_name"], ascending=[False, False, True], na_position="last")
            .drop(columns=["_usable_rank"])
            .reset_index(drop=True)
        )
        return summary_df, detail_map


__all__ = ["FactorScreeningEngine"]
