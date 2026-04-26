from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from tiger_factors.factor_screener._evaluation_io import load_return_series
from tiger_factors.factor_store import FactorSpec
from tiger_factors.factor_store import FactorStore


@dataclass(frozen=True)
class ReturnAdapterSpec:
    return_mode: str = "long_short"
    return_table_name: str = "factor_portfolio_returns"


@dataclass(frozen=True)
class ReturnAdapterResult:
    spec: ReturnAdapterSpec
    factor_specs: tuple[FactorSpec, ...]
    screened_at: pd.Timestamp
    return_series: dict[str, pd.Series]
    return_long: pd.DataFrame
    return_panel: pd.DataFrame
    missing_return_factors: tuple[str, ...] = ()

    @property
    def factor_names(self) -> list[str]:
        return [spec.table_name for spec in self.factor_specs]

    def to_summary(self) -> dict[str, object]:
        if self.return_panel.empty:
            return_start = None
            return_end = None
        else:
            index = pd.DatetimeIndex(self.return_panel.index).dropna().sort_values()
            return_start = None if index.empty else index[0].isoformat()
            return_end = None if index.empty else index[-1].isoformat()
        return {
            "screened_at": self.screened_at.isoformat(),
            "return_mode": self.spec.return_mode,
            "return_table_name": self.spec.return_table_name,
            "factor_count": int(len(self.factor_specs)),
            "factor_names": self.factor_names,
            "selected_count": int(len(self.return_series)),
            "missing_return_factors": list(self.missing_return_factors),
            "return_start": return_start,
            "return_end": return_end,
            "return_rows": int(len(self.return_long)),
            "return_panel_rows": int(len(self.return_panel)),
            "return_panel_columns": int(len(self.return_panel.columns)),
        }


class ReturnAdapter:
    def __init__(
        self,
        spec: ReturnAdapterSpec,
        *,
        factor_specs: Iterable[FactorSpec],
        store: FactorStore | None = None,
    ) -> None:
        self.spec = spec
        self.factor_specs = tuple(factor_specs)
        self.store = store or FactorStore()

    def run(self) -> ReturnAdapterResult:
        screened_at = pd.Timestamp.now(tz="UTC")
        if not self.factor_specs:
            empty = pd.DataFrame()
            return ReturnAdapterResult(
                spec=self.spec,
                factor_specs=tuple(),
                screened_at=screened_at,
                return_series={},
                return_long=empty,
                return_panel=empty,
                missing_return_factors=(),
            )

        return_series_map: dict[str, pd.Series] = {}
        missing_return_factors: list[str] = []
        for factor_spec in self.factor_specs:
            series = load_return_series(
                self.store,
                factor_spec,
                return_mode=self.spec.return_mode,
                return_table_name=self.spec.return_table_name,
            )
            if series is None or series.empty:
                missing_return_factors.append(factor_spec.table_name)
                continue
            return_series_map[factor_spec.table_name] = series

        if return_series_map:
            return_panel = pd.concat(return_series_map, axis=1).sort_index()
            return_panel.index = pd.to_datetime(return_panel.index, errors="coerce")
            return_panel = return_panel.loc[~return_panel.index.isna()].sort_index()
        else:
            return_panel = pd.DataFrame()

        if return_series_map:
            return_long = (
                return_panel.copy()
                .sort_index()
                .stack(future_stack=True)
                .rename("return")
                .reset_index()
            )
            return_long.columns = ["date_", "factor", "return"]
            return_long["return_mode"] = self.spec.return_mode
            return_long = return_long.dropna(subset=["date_", "return"])
            return_long = return_long.loc[:, ["date_", "factor", "return", "return_mode"]].sort_values(
                ["date_", "factor"],
                kind="stable",
            )
        else:
            return_long = pd.DataFrame(columns=["date_", "factor", "return", "return_mode"])

        return ReturnAdapterResult(
            spec=self.spec,
            factor_specs=self.factor_specs,
            screened_at=screened_at,
            return_series=return_series_map,
            return_long=return_long,
            return_panel=return_panel,
            missing_return_factors=tuple(missing_return_factors),
        )


def run_return_adapter(
    spec: ReturnAdapterSpec,
    factor_specs: Iterable[FactorSpec],
    *,
    store: FactorStore | None = None,
) -> ReturnAdapterResult:
    return ReturnAdapter(
        spec,
        factor_specs=factor_specs,
        store=store,
    ).run()


__all__ = [
    "ReturnAdapter",
    "ReturnAdapterResult",
    "ReturnAdapterSpec",
    "run_return_adapter",
]
