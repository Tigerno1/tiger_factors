from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from tiger_factors.factor_store import FactorSpec
from tiger_factors.factor_store import FactorStore


def _normalize_return_series(series: pd.Series, *, factor_name: str) -> pd.Series:
    cleaned = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        return cleaned
    cleaned.index = pd.to_datetime(cleaned.index, errors="coerce")
    cleaned = cleaned[~cleaned.index.isna()].sort_index()
    cleaned.name = factor_name
    return cleaned


def _pick_return_column(frame: pd.DataFrame, *, return_mode: str) -> str:
    lookup = {str(column): column for column in frame.columns}
    preferred_mode = str(return_mode).strip().lower()
    if preferred_mode in lookup:
        return str(lookup[preferred_mode])
    for candidate in ("long_short", "long_short_returns", "factor_portfolio_returns", "returns", "return", "1D"):
        if candidate in lookup:
            return str(lookup[candidate])
    numeric_columns = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    if numeric_columns:
        return str(numeric_columns[0])
    raise KeyError(f"could not infer return column from: {list(frame.columns)!r}")


def _load_return_series(
    store: FactorStore,
    spec: FactorSpec,
    *,
    return_mode: str,
    return_table_name: str,
) -> pd.Series | None:
    try:
        frame = store.evaluation.returns(spec).get_table(return_table_name)
    except FileNotFoundError:
        return None
    if isinstance(frame, pd.Series):
        return _normalize_return_series(frame, factor_name=spec.table_name)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    working = frame.copy()
    if "date_" in working.columns:
        working["date_"] = pd.to_datetime(working["date_"], errors="coerce")
        working = working.loc[working["date_"].notna()]
    if "date_" in working.columns:
        numeric_columns = [column for column in working.columns if column != "date_" and pd.api.types.is_numeric_dtype(working[column])]
        if numeric_columns:
            column = _pick_return_column(working[numeric_columns], return_mode=return_mode)
            series = pd.Series(pd.to_numeric(working[column], errors="coerce").to_numpy(), index=working["date_"], name=spec.table_name)
            return _normalize_return_series(series, factor_name=spec.table_name)
    column = _pick_return_column(working, return_mode=return_mode)
    series = working[column]
    if isinstance(series, pd.DataFrame):
        series = series.squeeze(axis=1)
    return _normalize_return_series(series, factor_name=spec.table_name)


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
            series = _load_return_series(
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
