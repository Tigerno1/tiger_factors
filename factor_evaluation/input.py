from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


TableLike = pd.DataFrame | str | Path
SeriesLike = pd.Series | pd.DataFrame | Mapping[Any, Any] | str | Path

_GROUP_LABELS_CACHE: dict[str, pd.DataFrame | pd.Series] = {}


@dataclass(frozen=True)
class TigerEvaluationInput:
    factor_frame: pd.DataFrame
    price_frame: pd.DataFrame
    factor_series: pd.Series
    factor_panel: pd.DataFrame
    price_panel: pd.DataFrame
    forward_returns: pd.DataFrame
    factor_column: str
    date_column: str = "date_"
    code_column: str = "code"
    price_column: str = "close"
    forward_days: int = 1


def _resolve_table_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def _read_table(source: TableLike | SeriesLike) -> pd.DataFrame | pd.Series:
    if isinstance(source, pd.DataFrame) or isinstance(source, pd.Series):
        return source.copy()
    if isinstance(source, Mapping):
        return pd.Series(source, dtype=object)

    path = _resolve_table_path(source)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path.suffix}")


def _normalize_date_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce")


def _normalize_code_series(values: pd.Series) -> pd.Series:
    return values.astype(str)


def _normalize_numeric_series(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def _prepare_long_frame(
    frame: pd.DataFrame,
    *,
    date_column: str,
    code_column: str,
    value_column: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[date_column, code_column, value_column])

    out = frame.copy()
    if {date_column, code_column, value_column}.issubset(out.columns):
        out = out[[date_column, code_column, value_column]].copy()
    elif value_column in out.columns and isinstance(out.index, pd.MultiIndex):
        out = out.reset_index()
    elif value_column in out.columns and date_column not in out.columns and code_column not in out.columns:
        out = out.reset_index()
    elif value_column not in out.columns:
        candidate_columns = [column for column in out.columns if column not in {date_column, code_column}]
        if len(candidate_columns) == 1:
            out = out.rename(columns={candidate_columns[0]: value_column})
        else:
            missing = [column for column in (date_column, code_column, value_column) if column not in out.columns]
            raise ValueError(f"frame is missing required columns: {missing}")
    else:
        missing = [column for column in (date_column, code_column, value_column) if column not in out.columns]
        raise ValueError(f"frame is missing required columns: {missing}")

    out[date_column] = _normalize_date_series(out[date_column])
    out[code_column] = _normalize_code_series(out[code_column])
    out[value_column] = _normalize_numeric_series(out[value_column])
    out = out.dropna(subset=[date_column, code_column, value_column]).copy()
    if out.empty:
        return pd.DataFrame(columns=[date_column, code_column, value_column])

    out = out.sort_values([date_column, code_column], kind="stable")
    out = out.drop_duplicates(subset=[date_column, code_column], keep="last")
    return out.reset_index(drop=True)


def to_panel(
    frame: pd.DataFrame,
    *,
    value_column: str,
    date_column: str = "date_",
    code_column: str = "code",
) -> pd.DataFrame:
    long_frame = _prepare_long_frame(
        frame,
        date_column=date_column,
        code_column=code_column,
        value_column=value_column,
    )
    if long_frame.empty:
        return pd.DataFrame()

    panel = long_frame.pivot(index=date_column, columns=code_column, values=value_column).sort_index()
    panel.columns = panel.columns.astype(str)
    return panel


def load_factor_frame(
    source: TableLike,
    *,
    factor_column: str,
    date_column: str = "date_",
    code_column: str = "code",
) -> pd.DataFrame:
    frame = _read_table(source)
    if isinstance(frame, pd.Series):
        frame = frame.to_frame(name=factor_column)

    prepared = _prepare_long_frame(
        frame,
        date_column=date_column,
        code_column=code_column,
        value_column=factor_column,
    )
    return prepared


def load_price_frame(
    source: TableLike,
    *,
    date_column: str = "date_",
    code_column: str = "code",
    price_column: str = "close",
) -> pd.DataFrame:
    frame = _read_table(source)
    if isinstance(frame, pd.Series):
        frame = frame.to_frame(name=price_column)

    prepared = _prepare_long_frame(
        frame,
        date_column=date_column,
        code_column=code_column,
        value_column=price_column,
    )
    return prepared


def _normalize_series(
    source: pd.Series | pd.DataFrame,
    *,
    date_column: str,
    value_column: str,
) -> pd.Series:
    if isinstance(source, pd.Series):
        series = source.copy()
        if isinstance(series.index, pd.MultiIndex):
            series = series.sort_index()
        else:
            series.index = pd.to_datetime(series.index, errors="coerce")
            series = series[series.index.notna()].sort_index()
        series = _normalize_numeric_series(series)
        return series.dropna()

    frame = source.copy()
    if {date_column, value_column}.issubset(frame.columns):
        out = frame[[date_column, value_column]].copy()
        out[date_column] = _normalize_date_series(out[date_column])
        out[value_column] = _normalize_numeric_series(out[value_column])
        out = out.dropna(subset=[date_column, value_column]).copy()
        out = out.drop_duplicates(subset=[date_column], keep="last").sort_values(date_column, kind="stable")
        series = out.set_index(date_column)[value_column]
        series.name = value_column
        return series

    if value_column in frame.columns and isinstance(frame.index, pd.DatetimeIndex):
        series = pd.to_numeric(frame[value_column], errors="coerce")
        series.index = pd.to_datetime(series.index, errors="coerce")
        return series.dropna().sort_index()

    raise ValueError(f"frame is missing required columns: {sorted({date_column, value_column} - set(frame.columns))}")


def load_series(
    source: SeriesLike,
    *,
    date_column: str = "date_",
    value_column: str = "returns",
) -> pd.Series:
    loaded = _read_table(source)
    if isinstance(loaded, pd.Series):
        return _normalize_series(loaded, date_column=date_column, value_column=value_column)
    return _normalize_series(loaded, date_column=date_column, value_column=value_column)


def _normalize_group_series(series: pd.Series) -> pd.Series:
    if isinstance(series.index, pd.MultiIndex):
        index = series.index
        if index.nlevels >= 2 and isinstance(index.get_level_values(0), pd.DatetimeIndex):
            normalized = series.copy()
            normalized.index = pd.MultiIndex.from_arrays(
                [
                    pd.to_datetime(index.get_level_values(0), errors="coerce"),
                    index.get_level_values(1).astype(str),
                ],
                names=index.names[:2],
            )
            return normalized.sort_index()
        return series.sort_index()

    normalized = series.copy()
    normalized.index = normalized.index.astype(str)
    return normalized.sort_index()


def _group_frame_to_wide(
    frame: pd.DataFrame,
    *,
    date_column: str,
    code_column: str,
    value_column: str,
) -> pd.DataFrame | pd.Series:
    if {date_column, code_column, value_column}.issubset(frame.columns):
        long_frame = frame[[date_column, code_column, value_column]].copy()
        long_frame[date_column] = _normalize_date_series(long_frame[date_column])
        long_frame[code_column] = _normalize_code_series(long_frame[code_column])
        long_frame = long_frame.dropna(subset=[date_column, code_column]).copy()
        long_frame = long_frame.sort_values([date_column, code_column], kind="stable")
        long_frame = long_frame.drop_duplicates(subset=[date_column, code_column], keep="last")
        if long_frame.empty:
            return pd.DataFrame()
        wide = long_frame.pivot(index=date_column, columns=code_column, values=value_column).sort_index()
        wide.columns = wide.columns.astype(str)
        return wide

    if {code_column, value_column}.issubset(frame.columns):
        series = frame[[code_column, value_column]].copy()
        series[code_column] = _normalize_code_series(series[code_column])
        series[value_column] = series[value_column].astype(object)
        series = series.dropna(subset=[code_column]).drop_duplicates(subset=[code_column], keep="last")
        result = series.set_index(code_column)[value_column]
        result.index = result.index.astype(str)
        return result.sort_index()

    if date_column not in frame.columns and code_column not in frame.columns:
        wide = frame.copy()
        if not wide.empty:
            wide.index = pd.to_datetime(wide.index, errors="coerce")
            valid_index = pd.notna(wide.index)
            wide = wide.loc[valid_index].sort_index()
            wide.columns = wide.columns.astype(str)
        return wide

    raise ValueError(f"frame is missing required columns for group labels: {sorted(frame.columns)}")


def load_group_labels(
    source: SeriesLike,
    *,
    date_column: str = "date_",
    code_column: str = "code",
    value_column: str = "group",
    use_cache: bool = True,
) -> pd.DataFrame | pd.Series:
    if isinstance(source, (str, Path)):
        path = _resolve_table_path(source)
        cache_key = str(path)
        if use_cache and cache_key in _GROUP_LABELS_CACHE:
            cached = _GROUP_LABELS_CACHE[cache_key]
            return cached.copy(deep=True)

        loaded = _read_table(path)
        result = load_group_labels(
            loaded,
            date_column=date_column,
            code_column=code_column,
            value_column=value_column,
            use_cache=use_cache,
        )
        if use_cache:
            _GROUP_LABELS_CACHE[cache_key] = result.copy(deep=True)
        return result

    if isinstance(source, Mapping):
        return _normalize_group_series(pd.Series(source, dtype=object))

    if isinstance(source, pd.Series):
        result = _normalize_group_series(source.astype(object))
        return result.copy(deep=True)

    if not isinstance(source, pd.DataFrame):
        source = pd.DataFrame(source)

    result = _group_frame_to_wide(
        source,
        date_column=date_column,
        code_column=code_column,
        value_column=value_column,
    )
    return result.copy(deep=True)


def clear_group_labels_cache() -> None:
    _GROUP_LABELS_CACHE.clear()


def prewarm_group_labels_cache(
    paths: Sequence[str | Path],
    *,
    date_column: str = "date_",
    code_column: str = "code",
    value_column: str = "group",
) -> None:
    for path in paths:
        load_group_labels(
            path,
            date_column=date_column,
            code_column=code_column,
            value_column=value_column,
            use_cache=True,
        )


def build_tiger_evaluation_input(
    *,
    factor_frame: TableLike,
    price_frame: TableLike,
    factor_column: str,
    date_column: str = "date_",
    code_column: str = "code",
    price_column: str = "close",
    forward_days: int = 1,
) -> TigerEvaluationInput:
    factor_frame_df = load_factor_frame(
        factor_frame,
        factor_column=factor_column,
        date_column=date_column,
        code_column=code_column,
    )
    price_frame_df = load_price_frame(
        price_frame,
        date_column=date_column,
        code_column=code_column,
        price_column=price_column,
    )

    factor_series = factor_frame_df.set_index([date_column, code_column])[factor_column].sort_index()
    factor_series.name = factor_column
    factor_panel = factor_series.unstack(code_column).sort_index()
    price_panel = to_panel(
        price_frame_df,
        value_column=price_column,
        date_column=date_column,
        code_column=code_column,
    ).sort_index()

    common_dates = factor_panel.index.intersection(price_panel.index)
    common_codes = factor_panel.columns.intersection(price_panel.columns)
    factor_panel = factor_panel.loc[common_dates, common_codes]
    price_panel = price_panel.loc[common_dates, common_codes]
    forward_returns = price_panel.shift(-int(max(forward_days, 1))).div(price_panel).sub(1.0)

    return TigerEvaluationInput(
        factor_frame=factor_frame_df,
        price_frame=price_frame_df,
        factor_series=factor_series,
        factor_panel=factor_panel,
        price_panel=price_panel,
        forward_returns=forward_returns,
        factor_column=factor_column,
        date_column=date_column,
        code_column=code_column,
        price_column=price_column,
        forward_days=int(max(forward_days, 1)),
    )


__all__ = [
    "TableLike",
    "SeriesLike",
    "TigerEvaluationInput",
    "build_tiger_evaluation_input",
    "clear_group_labels_cache",
    "load_factor_frame",
    "load_group_labels",
    "load_price_frame",
    "load_series",
    "prewarm_group_labels_cache",
    "to_panel",
]
