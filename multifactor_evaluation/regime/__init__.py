from __future__ import annotations

from typing import Any
from typing import Iterable

import pandas as pd

from tiger_factors.factor_evaluation.horizon import HoldingPeriodAnalyzer
from tiger_factors.multifactor_evaluation._inputs import coerce_factor_series
from tiger_factors.multifactor_evaluation._inputs import coerce_price_panel
from tiger_factors.multifactor_evaluation.screening import add_cost_analysis
from tiger_factors.multifactor_evaluation.screening import evaluate_factor_with_filter


class SimpleRegimeDetector:
    def __init__(
        self,
        market_price: pd.Series,
        *,
        ma_window: int = 50,
        vol_window: int = 20,
    ) -> None:
        self.market_price = market_price.sort_index()
        self.ma_window = int(max(ma_window, 2))
        self.vol_window = int(max(vol_window, 2))

    def detect(self) -> pd.Series:
        px = self.market_price.dropna().sort_index()
        ret = px.pct_change(fill_method=None)
        ma = px.rolling(self.ma_window).mean()
        trend_up = px > ma
        vol = ret.rolling(self.vol_window).std()
        vol_med = float(vol.dropna().median()) if not vol.dropna().empty else float("nan")
        high_vol = vol > vol_med

        regime = pd.Series(index=px.index, dtype="object")
        regime[trend_up & ~high_vol] = "bull_low_vol"
        regime[trend_up & high_vol] = "bull_high_vol"
        regime[~trend_up & ~high_vol] = "bear_low_vol"
        regime[~trend_up & high_vol] = "bear_high_vol"
        return regime


class RegimeAwareAlphaEngine:
    def __init__(
        self,
        factor: pd.Series | pd.DataFrame,
        prices: pd.DataFrame,
        regime_series: pd.Series,
        *,
        horizons: Iterable[int] = (1, 3, 5, 10, 20),
        quantiles: int = 5,
        periods_per_year: int = 252,
        long_short_pct: float = 0.2,
        min_names: int | None = 10,
        min_dates_per_regime: int = 30,
        cost_rate: float = 0.001,
    ) -> None:
        self.factor = coerce_factor_series(factor)
        self.prices = coerce_price_panel(prices)
        self.regime_series = regime_series.sort_index()
        self.horizons = list(horizons)
        self.quantiles = quantiles
        self.periods_per_year = periods_per_year
        self.long_short_pct = long_short_pct
        self.min_names = min_names
        self.min_dates_per_regime = min_dates_per_regime
        self.cost_rate = cost_rate

    def _filter_factor_by_dates(self, dates: pd.Index) -> pd.Series:
        mask = self.factor.index.get_level_values(0).isin(dates)
        return self.factor[mask]

    def analyze_regime(self, regime_name: str) -> dict[str, Any] | None:
        regime_dates = self.regime_series.index[self.regime_series == regime_name]
        if len(regime_dates) < self.min_dates_per_regime:
            return None

        factor_sub = self._filter_factor_by_dates(regime_dates)
        if factor_sub.empty:
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
        return {
            "regime": regime_name,
            "n_dates": int(len(regime_dates)),
            "result": result,
            "decision": decision,
        }

    def run(self) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        summary_rows: list[dict[str, Any]] = []
        detail_map: dict[str, pd.DataFrame] = {}

        for regime_name in sorted(self.regime_series.dropna().astype(str).unique().tolist()):
            analyzed = self.analyze_regime(regime_name)
            if analyzed is None:
                continue
            decision = analyzed["decision"]
            result = analyzed["result"]
            best_horizon = decision.get("best_horizon")

            row: dict[str, Any] = {
                "regime": regime_name,
                "n_dates": analyzed["n_dates"],
                "usable": decision.get("usable"),
                "direction": decision.get("direction"),
                "quality": decision.get("quality"),
                "best_horizon": best_horizon,
                "recommendation": decision.get("recommendation"),
                "score": decision.get("score"),
            }
            if best_horizon is not None:
                best_row = result.loc[result["horizon"] == best_horizon]
                if not best_row.empty:
                    picked = best_row.iloc[0]
                    row.update(
                        {
                            "mean_ic": picked.get("mean_ic"),
                            "ic_ir": picked.get("ic_ir"),
                            "ann_return": picked.get("ann_return"),
                            "net_sharpe": picked.get("net_sharpe"),
                            "avg_turnover": picked.get("avg_turnover"),
                            "max_drawdown": picked.get("max_drawdown"),
                        }
                    )
            summary_rows.append(row)
            detail_map[regime_name] = result

        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty and "score" in summary_df.columns:
            summary_df = summary_df.sort_values("score", ascending=False, na_position="last").reset_index(drop=True)
        return summary_df, detail_map

    def recommend_for_current_regime(self) -> dict[str, Any]:
        current = self.regime_series.dropna()
        if current.empty:
            return {
                "current_regime": None,
                "usable": False,
                "recommendation": "No regime labels available",
            }

        current_regime = str(current.iloc[-1])
        analyzed = self.analyze_regime(current_regime)
        if analyzed is None:
            return {
                "current_regime": current_regime,
                "usable": False,
                "recommendation": "No sufficient data for current regime",
            }
        return {"current_regime": current_regime, **analyzed["decision"]}


__all__ = ["RegimeAwareAlphaEngine", "SimpleRegimeDetector"]
