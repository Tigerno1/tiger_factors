from __future__ import annotations

import pandas as pd


def _ensure_datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.index = pd.to_datetime(result.index, errors="coerce")
    result = result.loc[~result.index.isna()].sort_index()
    result.index.name = result.index.name or "date_"
    result.columns = result.columns.astype(str)
    return result


def coerce_factor_panel(factor: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Convert factor input into a date x code panel."""

    if isinstance(factor, pd.Series):
        if isinstance(factor.index, pd.MultiIndex) and factor.index.nlevels == 2:
            panel = factor.sort_index().unstack("code")
            return _ensure_datetime_index(panel)
        raise ValueError("factor series must use a (date_, code) MultiIndex.")

    if not isinstance(factor, pd.DataFrame):
        raise TypeError("factor must be a pandas Series or DataFrame.")

    if {"date_", "code"}.issubset(factor.columns):
        value_columns = [col for col in factor.columns if col not in {"date_", "code"}]
        if len(value_columns) != 1:
            raise ValueError("long factor frame must contain exactly one value column besides date_ and code.")
        frame = (
            factor.copy()
            .assign(date_=pd.to_datetime(factor["date_"], errors="coerce"))
            .dropna(subset=["date_"])
            .set_index(["date_", "code"])[value_columns[0]]
            .sort_index()
            .unstack("code")
        )
        return _ensure_datetime_index(frame)

    return _ensure_datetime_index(factor)


def coerce_target_panel(target: pd.Series | pd.DataFrame | None) -> pd.DataFrame | None:
    if target is None:
        return None
    if isinstance(target, pd.Series):
        if isinstance(target.index, pd.MultiIndex) and target.index.nlevels == 2:
            return _ensure_datetime_index(target.sort_index().unstack("code"))
        raise ValueError("target series must use a (date_, code) MultiIndex.")
    if not isinstance(target, pd.DataFrame):
        raise TypeError("target must be a pandas Series or DataFrame.")
    return coerce_factor_panel(target)


def _frame_or_series(values: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return values.copy()
