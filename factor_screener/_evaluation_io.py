from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tiger_factors.factor_evaluation.utils import period_to_label
from tiger_factors.factor_store import FactorSpec
from tiger_factors.factor_store import FactorStore


def normalize_time_series(series: pd.Series, *, name: str) -> pd.Series:
    cleaned = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        return cleaned
    cleaned.index = pd.to_datetime(cleaned.index, errors="coerce")
    cleaned = cleaned[~cleaned.index.isna()].sort_index()
    cleaned.name = name
    return cleaned


def pick_numeric_column(frame: pd.DataFrame) -> str:
    numeric_columns = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    if numeric_columns:
        return str(numeric_columns[0])
    raise KeyError(f"could not infer numeric column from: {list(frame.columns)!r}")


def pick_return_column(
    frame: pd.DataFrame,
    *,
    preferred_mode: str | None = None,
    preferred_period: str | int | pd.Timedelta | None = None,
) -> str:
    lookup = {str(column): column for column in frame.columns}
    if preferred_mode is not None:
        normalized_mode = str(preferred_mode).strip().lower()
        if normalized_mode in lookup:
            return str(lookup[normalized_mode])
    if preferred_period is not None:
        preferred_label = period_to_label(preferred_period)
        if preferred_label in lookup:
            return str(lookup[preferred_label])
        if str(preferred_period) in lookup:
            return str(lookup[str(preferred_period)])
    for candidate in ("long_short", "long_only", "long_short_returns", "factor_portfolio_returns", "returns", "return", "1D"):
        if candidate in lookup:
            return str(lookup[candidate])
    return pick_numeric_column(frame)


def load_summary_row(store: FactorStore, spec: FactorSpec) -> dict[str, Any] | None:
    try:
        frame = store.evaluation.summary(spec).get_table()
    except FileNotFoundError:
        return None
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    working = frame.copy().reset_index(drop=True)
    if "factor_name" not in working.columns:
        working.insert(0, "factor_name", spec.table_name)
    row = working.iloc[0].to_dict()
    row["factor_name"] = str(row.get("factor_name", spec.table_name))
    return row


def load_return_series(
    store: FactorStore,
    spec: FactorSpec,
    *,
    return_mode: str = "long_short",
    return_table_name: str = "factor_portfolio_returns",
    preferred_period: str | int | pd.Timedelta | None = 1,
) -> pd.Series | None:
    try:
        frame = store.evaluation.returns(spec).get_table(return_table_name)
    except FileNotFoundError:
        frame = _load_legacy_return_frame(store, spec, return_mode=return_mode)
        if frame is None:
            return None
    if isinstance(frame, pd.Series):
        return normalize_time_series(frame, name=spec.table_name)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    working = frame.copy()
    if "date_" in working.columns:
        working["date_"] = pd.to_datetime(working["date_"], errors="coerce")
        working = working.loc[working["date_"].notna()]
    if "date_" in working.columns:
        numeric_columns = [column for column in working.columns if column != "date_" and pd.api.types.is_numeric_dtype(working[column])]
        if numeric_columns:
            column = pick_return_column(
                working[numeric_columns],
                preferred_mode=return_mode,
                preferred_period=preferred_period,
            )
            series = pd.Series(pd.to_numeric(working[column], errors="coerce").to_numpy(), index=working["date_"], name=spec.table_name)
            return normalize_time_series(series, name=spec.table_name)
    column = _pick_top_quantile_column(working) if _is_long_only(return_mode) else None
    if column is None:
        column = pick_return_column(
            working,
            preferred_mode=return_mode,
            preferred_period=preferred_period,
        )
    series = working[column]
    if isinstance(series, pd.DataFrame):
        series = series.squeeze(axis=1)
    return normalize_time_series(series, name=spec.table_name)


def _is_long_only(return_mode: str) -> bool:
    return str(return_mode).strip().lower().replace("-", "_") == "long_only"


def _pick_top_quantile_column(frame: pd.DataFrame) -> object | None:
    if frame.empty:
        return None
    columns = pd.Index(frame.columns)
    numeric = pd.Series(pd.to_numeric(columns.astype(str), errors="coerce"))
    if numeric.notna().any():
        return columns[int(numeric.idxmax())]
    return None


def _load_legacy_return_frame(
    store: FactorStore,
    spec: FactorSpec,
    *,
    return_mode: str,
) -> pd.DataFrame | None:
    table_names = (
        ("primary_quantile_returns_by_date", "mean_return_by_quantile_by_date")
        if _is_long_only(return_mode)
        else ("mean_return_spread",)
    )
    for table_name in table_names:
        try:
            frame = store.evaluation.returns(spec).get_table(table_name)
        except FileNotFoundError:
            continue
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            return frame
    return None


def load_ic_series(
    store: FactorStore,
    spec: FactorSpec,
    *,
    period: str | int | pd.Timedelta | None = None,
    table_name: str = "information_coefficient",
) -> pd.Series | None:
    try:
        frame = store.evaluation.information(spec).get_table(table_name)
    except FileNotFoundError:
        return None
    if isinstance(frame, pd.Series):
        return normalize_time_series(frame, name=spec.table_name)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    column = pick_return_column(frame, preferred_period=period)
    series = frame[column]
    if isinstance(series, pd.DataFrame):
        series = series.squeeze(axis=1)
    return normalize_time_series(series, name=spec.table_name)


def load_factor_panel(store: FactorStore, spec: FactorSpec) -> pd.DataFrame:
    try:
        frame = store.get_factor(spec, engine="pandas")
    except Exception:
        return pd.DataFrame()
    return factor_frame_to_panel(frame, factor_name=spec.table_name)


def factor_frame_to_panel(frame: pd.DataFrame, *, factor_name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    normalized = frame.copy()
    if "date_" in normalized.columns:
        normalized["date_"] = pd.to_datetime(normalized["date_"], errors="coerce")
    if "code" in normalized.columns:
        normalized["code"] = normalized["code"].astype(str)
    if {"date_", "code"}.issubset(normalized.columns):
        value_columns = [
            column
            for column in normalized.columns
            if column not in {"date_", "code"} and pd.api.types.is_numeric_dtype(normalized[column])
        ]
        if not value_columns:
            return pd.DataFrame()
        panel = normalized.pivot_table(index="date_", columns="code", values=value_columns[0], aggfunc="last")
        panel.index = pd.to_datetime(panel.index, errors="coerce")
        panel = panel.loc[~panel.index.isna()].sort_index()
        panel.columns = panel.columns.astype(str)
        return panel
    if isinstance(normalized.index, pd.DatetimeIndex):
        panel = normalized.copy()
        panel.index = pd.to_datetime(panel.index, errors="coerce")
        panel = panel.loc[~panel.index.isna()].sort_index()
        panel.columns = panel.columns.astype(str)
        return panel
    return pd.DataFrame()
