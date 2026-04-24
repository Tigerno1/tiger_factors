from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from tiger_factors.factor_store import FactorSpec
from tiger_factors.factor_store import FactorStore
from tiger_factors.factor_screener.correlation_screener import CorrelationScreener
from tiger_factors.factor_screener.correlation_screener import CorrelationScreenerResult
from tiger_factors.factor_screener.correlation_screener import CorrelationScreenerSpec
from tiger_factors.factor_screener.factor_screener import FactorScreener
from tiger_factors.factor_screener.factor_screener import FactorScreenerResult
from tiger_factors.factor_screener.factor_screener import FactorScreenerSpec
from tiger_factors.factor_screener.backtest_marginal_screener import BacktestMarginalScreener
from tiger_factors.factor_screener.backtest_marginal_screener import BacktestMarginalScreenerResult
from tiger_factors.factor_screener.backtest_marginal_screener import BacktestMarginalScreenerSpec
from tiger_factors.factor_screener.return_adapter import ReturnAdapter
from tiger_factors.factor_screener.return_adapter import ReturnAdapterSpec
from tiger_factors.factor_screener.marginal_screener import MarginalScreener
from tiger_factors.factor_screener.marginal_screener import MarginalScreenerResult
from tiger_factors.factor_screener.marginal_screener import MarginalScreenerSpec


@dataclass(frozen=True)
class ScreenerResult:
    factor_result: FactorScreenerResult
    correlation_results: tuple[CorrelationScreenerResult, ...]
    screened_at: pd.Timestamp
    marginal_result: MarginalScreenerResult | None = None
    backtest_marginal_result: BacktestMarginalScreenerResult | None = None

    @property
    def screened_factor_names(self) -> list[str]:
        return self.factor_result.screened_factor_names

    @property
    def screened_factor_specs(self) -> list[FactorSpec]:
        return self.factor_result.screened_factor_specs

    @property
    def factor_selected_factor_names(self) -> list[str]:
        return self.factor_result.selected_factor_names

    @property
    def factor_selected_factor_specs(self) -> list[FactorSpec]:
        return self.factor_result.selected_factor_specs

    @property
    def selected_factor_names(self) -> list[str]:
        if self.backtest_marginal_result is not None:
            return self.backtest_marginal_result.selected_factor_names
        if self.marginal_result is not None:
            return self.marginal_result.selected_factor_names
        if not self.correlation_results:
            return self.factor_result.selected_factor_names
        return self.correlation_results[-1].selected_factor_names

    @property
    def selected_factor_specs(self) -> list[FactorSpec]:
        if self.backtest_marginal_result is not None:
            return self.backtest_marginal_result.selected_factor_specs
        if self.marginal_result is not None:
            return self.marginal_result.selected_factor_specs
        if not self.correlation_results:
            return self.factor_result.selected_factor_specs
        return self.correlation_results[-1].selected_factor_specs

    @property
    def correlation_result(self) -> CorrelationScreenerResult:
        if self.correlation_results:
            return self.correlation_results[-1]
        return CorrelationScreenerResult(
            spec=CorrelationScreenerSpec(),
            factor_specs=tuple(),
            screened_at=self.screened_at,
            summary=pd.DataFrame(),
            selection_summary=pd.DataFrame(),
            return_series={},
            return_panel=pd.DataFrame(),
            correlation_matrix=pd.DataFrame(),
        )

    @property
    def marginal_screener(self) -> MarginalScreenerResult:
        if self.marginal_result is not None:
            return self.marginal_result
        return MarginalScreenerResult(
            spec=MarginalScreenerSpec(),
            factor_specs=tuple(),
            screened_at=self.screened_at,
            summary=pd.DataFrame(),
            selection_summary=pd.DataFrame(),
            return_series={},
            return_panel=pd.DataFrame(),
            portfolio_returns=pd.Series(dtype=float),
            missing_return_factors=tuple(),
        )

    @property
    def backtest_marginal_screener(self) -> BacktestMarginalScreenerResult:
        if self.backtest_marginal_result is not None:
            return self.backtest_marginal_result
        return BacktestMarginalScreenerResult(
            spec=BacktestMarginalScreenerSpec(),
            factor_specs=tuple(),
            screened_at=self.screened_at,
            summary=pd.DataFrame(),
            selection_summary=pd.DataFrame(),
            return_series={},
            return_panel=pd.DataFrame(),
            portfolio_returns=pd.Series(dtype=float),
            backtest=pd.DataFrame(),
            stats={"portfolio": {}, "benchmark": {}},
            missing_return_factors=tuple(),
        )

    @property
    def return_long(self) -> pd.DataFrame:
        if self.backtest_marginal_result is not None:
            return self.backtest_marginal_result.return_long
        if self.marginal_result is not None:
            return self.marginal_result.return_long
        if not self.correlation_results or self.correlation_results[-1].return_panel.empty:
            return pd.DataFrame(columns=["date_", "factor", "return", "return_mode"])
        long_frame = (
            self.correlation_results[-1].return_panel.copy()
            .sort_index()
            .stack(dropna=False)
            .rename("return")
            .reset_index()
        )
        if long_frame.empty:
            return pd.DataFrame(columns=["date_", "factor", "return", "return_mode"])
        long_frame.columns = ["date_", "factor", "return"]
        long_frame["return_mode"] = str(self.correlation_results[-1].spec.evaluation_source)
        long_frame = long_frame.dropna(subset=["date_", "return"])
        return long_frame.loc[:, ["date_", "factor", "return", "return_mode"]].sort_values(
            ["date_", "factor"],
            kind="stable",
        )

    @property
    def return_panel(self) -> pd.DataFrame:
        if self.backtest_marginal_result is not None and not self.backtest_marginal_result.return_panel.empty:
            return self.backtest_marginal_result.return_panel
        if self.marginal_result is not None and not self.marginal_result.return_panel.empty:
            return self.marginal_result.return_panel
        if self.correlation_results and not self.correlation_results[-1].return_panel.empty:
            return self.correlation_results[-1].return_panel
        return self.factor_result.return_panel

    def build_return_adapter(
        self,
        *,
        store: FactorStore | None = None,
        spec: ReturnAdapterSpec | None = None,
    ) -> ReturnAdapter:
        return ReturnAdapter(
            spec or ReturnAdapterSpec(),
            factor_specs=self.selected_factor_specs,
            store=store,
        )

    def to_summary(self) -> dict[str, object]:
        factor_summary = self.factor_result.to_summary()
        correlation_summaries = [result.to_summary() for result in self.correlation_results]
        last_correlation_summary = correlation_summaries[-1] if correlation_summaries else {}
        marginal_summary = self.marginal_result.to_summary() if self.marginal_result is not None else {}
        backtest_marginal_summary = self.backtest_marginal_result.to_summary() if self.backtest_marginal_result is not None else {}
        return {
            "screened_at": self.screened_at.isoformat(),
            "factor_screener": factor_summary,
            "correlation_screener": last_correlation_summary,
            "correlation_screener_chain": correlation_summaries,
            "marginal_screener": marginal_summary,
            "backtest_marginal_screener": backtest_marginal_summary,
            "screened_factor_names": self.screened_factor_names,
            "factor_selected_factor_names": self.factor_selected_factor_names,
            "selected_factor_names": self.selected_factor_names,
            "selected_count": int(len(self.selected_factor_names)),
            "return_start": marginal_summary.get("return_start")
            or last_correlation_summary.get("return_start")
            or backtest_marginal_summary.get("return_start")
            or factor_summary.get("return_start"),
            "return_end": marginal_summary.get("return_end")
            or last_correlation_summary.get("return_end")
            or backtest_marginal_summary.get("return_end")
            or factor_summary.get("return_end"),
        }


class Screener:
    def __init__(
        self,
        factor_spec: FactorScreenerSpec,
        correlation_specs: Iterable[CorrelationScreenerSpec],
        *,
        factor_specs: Iterable[FactorSpec],
        store: FactorStore | None = None,
        mode: str = "correlation",
        marginal_spec: MarginalScreenerSpec | None = None,
        backtest_marginal_spec: BacktestMarginalScreenerSpec | None = None,
    ) -> None:
        self.factor_spec = factor_spec
        self.correlation_specs = tuple(correlation_specs)
        self.factor_specs = tuple(factor_specs)
        self.store = store or FactorStore()
        self.mode = str(mode).strip().lower()
        self.marginal_spec = marginal_spec
        self.backtest_marginal_spec = backtest_marginal_spec

    def run(self) -> ScreenerResult:
        factor_result = FactorScreener(
            self.factor_spec,
            factor_specs=self.factor_specs,
            store=self.store,
        ).run()
        factor_selected_specs = tuple(factor_result.selected_factor_specs)
        correlation_results: list[CorrelationScreenerResult] = []
        marginal_result: MarginalScreenerResult | None = None
        backtest_marginal_result: BacktestMarginalScreenerResult | None = None

        if self.mode == "correlation":
            current_specs = factor_selected_specs
            for correlation_spec in self.correlation_specs:
                if current_specs:
                    correlation_result = CorrelationScreener(
                        correlation_spec,
                        factor_specs=current_specs,
                        store=self.store,
                    ).run()
                else:
                    correlation_result = CorrelationScreenerResult(
                        spec=correlation_spec,
                        factor_specs=tuple(),
                        screened_at=factor_result.screened_at,
                        summary=pd.DataFrame(),
                        selection_summary=pd.DataFrame(),
                        return_series={},
                        return_panel=pd.DataFrame(),
                        correlation_matrix=pd.DataFrame(),
                    )
                correlation_results.append(correlation_result)
                current_specs = correlation_result.selected_factor_specs
        elif self.mode == "marginal":
            if self.marginal_spec is None:
                raise ValueError("marginal_spec is required when Screener.mode='marginal'")
            marginal_result = MarginalScreener(
                self.marginal_spec,
                factor_specs=factor_selected_specs,
                store=self.store,
            ).run()
        elif self.mode == "backtest_marginal":
            if self.backtest_marginal_spec is None:
                raise ValueError("backtest_marginal_spec is required when Screener.mode='backtest_marginal'")
            backtest_marginal_result = BacktestMarginalScreener(
                self.backtest_marginal_spec,
                factor_specs=factor_selected_specs,
                store=self.store,
            ).run()
        else:
            raise ValueError(f"unknown screener mode: {self.mode!r}")

        screened_at = pd.Timestamp.now(tz="UTC")
        return ScreenerResult(
            factor_result=factor_result,
            correlation_results=tuple(correlation_results),
            marginal_result=marginal_result,
            backtest_marginal_result=backtest_marginal_result,
            screened_at=screened_at,
        )


def run_screener(
    factor_spec: FactorScreenerSpec,
    correlation_specs: Iterable[CorrelationScreenerSpec],
    factor_specs: Iterable[FactorSpec],
    *,
    store: FactorStore | None = None,
    mode: str = "correlation",
    marginal_spec: MarginalScreenerSpec | None = None,
    backtest_marginal_spec: BacktestMarginalScreenerSpec | None = None,
) -> ScreenerResult:
    return Screener(
        factor_spec,
        correlation_specs,
        factor_specs=factor_specs,
        store=store,
        mode=mode,
        marginal_spec=marginal_spec,
        backtest_marginal_spec=backtest_marginal_spec,
    ).run()


__all__ = [
    "Screener",
    "ScreenerResult",
    "run_screener",
]
