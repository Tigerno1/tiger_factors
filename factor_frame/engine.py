from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from typing import Iterable

import numpy as np
import pandas as pd

from tiger_factors.factor_frame.factors import FactorFrameFactor
from tiger_factors.factor_frame.time_layer import FactorFrameTimeLayer


TableLike = pd.DataFrame | pd.Series


def normalize_dates(values) -> pd.Series:
    converted = pd.to_datetime(values, utc=True, errors="coerce")
    if isinstance(converted, pd.Series):
        return converted.dt.tz_localize(None)
    if isinstance(converted, pd.DatetimeIndex):
        return converted.tz_localize(None)
    return pd.Series(pd.DatetimeIndex(converted).tz_localize(None))


def to_long_factor(frame: pd.DataFrame, factor_name: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date_", "code", factor_name])
    long_df = (
        frame.rename_axis(index="date_")
        .reset_index()
        .melt(id_vars="date_", var_name="code", value_name=factor_name)
    )
    long_df["date_"] = normalize_dates(long_df["date_"])
    long_df["code"] = long_df["code"].astype(str)
    return long_df.dropna(subset=[factor_name]).sort_values(["date_", "code"]).reset_index(drop=True)


def _dedupe_columns(frame: pd.DataFrame, key_columns: set[str]) -> pd.DataFrame:
    columns = [column for column in frame.columns if column in key_columns]
    columns.extend(column for column in frame.columns if column not in key_columns)
    return frame.loc[:, columns]


def _prefixed_columns(frame: pd.DataFrame, prefix: str, key_columns: set[str]) -> pd.DataFrame:
    renamed = frame.copy()
    rename_map = {column: f"{prefix}__{column}" for column in renamed.columns if column not in key_columns}
    if rename_map:
        renamed = renamed.rename(columns=rename_map)
    return renamed


def _normalize_multiindex_series(
    series: pd.Series,
    *,
    name: str,
    date_column: str,
    code_column: str | None,
) -> pd.DataFrame:
    frame = series.rename(name).reset_index()
    if code_column is not None and frame.shape[1] >= 3:
        frame.columns = [date_column, code_column, name, *frame.columns[3:]]
        frame = frame.loc[:, [date_column, code_column, name]]
        frame[date_column] = normalize_dates(frame[date_column])
        frame[code_column] = frame[code_column].astype(str)
        return frame.dropna(subset=[date_column, code_column, name]).drop_duplicates(
            subset=[date_column, code_column], keep="last"
        )
    if frame.shape[1] >= 2:
        frame.columns = [date_column, name, *frame.columns[2:]]
        frame = frame.loc[:, [date_column, name]]
        frame[date_column] = normalize_dates(frame[date_column])
        return frame.dropna(subset=[date_column, name]).drop_duplicates(subset=[date_column], keep="last")
    raise ValueError("Unable to normalize multi-index series input.")


def _normalize_code_date_feed_frame(
    data: TableLike,
    *,
    name: str,
    date_column: str,
    code_column: str | None,
    value_column: str | None,
    availability_column: str | None,
    use_point_in_time: bool,
) -> pd.DataFrame:
    if isinstance(data, pd.Series):
        series_name = value_column or data.name or name
        if isinstance(data.index, pd.MultiIndex):
            if code_column is None:
                raise ValueError(f"{name} series has a MultiIndex but code_column is disabled.")
            return _normalize_multiindex_series(
                data,
                name=series_name,
                date_column=date_column,
                code_column=code_column,
            )

        frame = data.rename(series_name).to_frame().reset_index()
        if frame.shape[1] < 2:
            raise ValueError(f"{name} series must be indexed by date.")
        frame.columns = [date_column, series_name, *frame.columns[2:]]
        frame = frame.loc[:, [date_column, series_name]]
        frame[date_column] = normalize_dates(frame[date_column])
        return frame.dropna(subset=[date_column, series_name]).drop_duplicates(subset=[date_column], keep="last")

    frame = data.copy()
    if date_column in frame.columns:
        if availability_column and use_point_in_time and availability_column in frame.columns:
            source_date_column = f"source_{date_column}"
            if source_date_column not in frame.columns:
                frame[source_date_column] = normalize_dates(frame[date_column])
            frame[date_column] = normalize_dates(frame[availability_column])
        else:
            frame[date_column] = normalize_dates(frame[date_column])
        if code_column is not None and code_column in frame.columns:
            frame[code_column] = frame[code_column].astype(str)
            frame = frame.dropna(subset=[date_column, code_column]).copy()
            return frame.drop_duplicates(subset=[date_column, code_column], keep="last")
        return frame.dropna(subset=[date_column]).copy().drop_duplicates(subset=[date_column], keep="last")

    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.reset_index()
        if frame.empty:
            if code_column is not None:
                return pd.DataFrame(columns=[date_column, code_column, value_column or "value"])
            return pd.DataFrame(columns=[date_column, value_column or "value"])
        first = frame.columns[0]
        frame = frame.rename(columns={first: date_column})
        if availability_column and use_point_in_time and availability_column in frame.columns:
            source_date_column = f"source_{date_column}"
            if source_date_column not in frame.columns:
                frame[source_date_column] = normalize_dates(frame[date_column])
            frame[date_column] = normalize_dates(frame[availability_column])
        if code_column is not None and frame.shape[1] >= 3:
            second = frame.columns[1]
            frame = frame.rename(columns={second: code_column})
            frame[code_column] = frame[code_column].astype(str)
            return frame.dropna(subset=[date_column, code_column]).drop_duplicates(
                subset=[date_column, code_column], keep="last"
            )
        return frame.dropna(subset=[date_column]).drop_duplicates(subset=[date_column], keep="last")

    if isinstance(frame.index, pd.DatetimeIndex):
        if code_column is not None:
            value_name = value_column or "value"
            wide = frame.copy()
            wide.index = normalize_dates(wide.index)
            long = to_long_factor(wide, value_name)
            if value_name != "value" and value_name in long.columns:
                return long
            return long
        frame = frame.reset_index()
        first = frame.columns[0]
        frame = frame.rename(columns={first: date_column})
        frame[date_column] = normalize_dates(frame[date_column])
        return frame.dropna(subset=[date_column]).drop_duplicates(subset=[date_column], keep="last")

    raise ValueError(
        f"Unable to normalize feed {name!r}; expected a DataFrame/Series with a date column or a DatetimeIndex."
    )


def _normalize_date_feed_frame(
    data: TableLike,
    *,
    name: str,
    date_column: str,
    value_column: str | None,
    availability_column: str | None,
    use_point_in_time: bool,
) -> pd.DataFrame:
    if isinstance(data, pd.Series):
        series_name = value_column or data.name or name
        if isinstance(data.index, pd.MultiIndex):
            frame = data.rename(series_name).reset_index()
            if frame.shape[1] < 2:
                raise ValueError(f"{name} series must be indexed by date.")
            first = frame.columns[0]
            frame = frame.rename(columns={first: date_column})
            frame = frame.loc[:, [date_column, series_name]]
            frame[date_column] = normalize_dates(frame[date_column])
            return frame.dropna(subset=[date_column, series_name]).drop_duplicates(subset=[date_column], keep="last")

        frame = data.rename(series_name).to_frame().reset_index()
        if frame.shape[1] < 2:
            raise ValueError(f"{name} series must be indexed by date.")
        frame.columns = [date_column, series_name, *frame.columns[2:]]
        frame = frame.loc[:, [date_column, series_name]]
        frame[date_column] = normalize_dates(frame[date_column])
        return frame.dropna(subset=[date_column, series_name]).drop_duplicates(subset=[date_column], keep="last")

    frame = data.copy()
    if date_column in frame.columns:
        if availability_column and use_point_in_time and availability_column in frame.columns:
            source_date_column = f"source_{date_column}"
            if source_date_column not in frame.columns:
                frame[source_date_column] = normalize_dates(frame[date_column])
            frame[date_column] = normalize_dates(frame[availability_column])
        else:
            frame[date_column] = normalize_dates(frame[date_column])
        return frame.dropna(subset=[date_column]).copy().drop_duplicates(subset=[date_column], keep="last")

    if isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.reset_index()
        first = frame.columns[0]
        frame = frame.rename(columns={first: date_column})
        frame[date_column] = normalize_dates(frame[date_column])
        return frame.dropna(subset=[date_column]).drop_duplicates(subset=[date_column], keep="last")

    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.reset_index()
        if frame.empty:
            return pd.DataFrame(columns=[date_column, value_column or "value"])
        first = frame.columns[0]
        frame = frame.rename(columns={first: date_column})
        frame[date_column] = normalize_dates(frame[date_column])
        return frame.dropna(subset=[date_column]).drop_duplicates(subset=[date_column], keep="last")

    raise ValueError(
        f"Unable to normalize date feed {name!r}; expected a date column or a DatetimeIndex."
    )


def _normalize_code_feed_frame(
    data: TableLike,
    *,
    name: str,
    code_column: str | None,
    value_column: str | None,
) -> pd.DataFrame:
    if code_column is None:
        raise ValueError(f"{name} feed is code-aligned but code_column is disabled.")

    if isinstance(data, pd.Series):
        series_name = value_column or data.name or name
        frame = data.rename(series_name).to_frame().reset_index()
        if frame.shape[1] < 2:
            raise ValueError(f"{name} series must be indexed by code.")
        first = frame.columns[0]
        frame = frame.rename(columns={first: code_column})
        frame[code_column] = frame[code_column].astype(str)
        return frame.dropna(subset=[code_column]).drop_duplicates(subset=[code_column], keep="last")

    frame = data.copy()
    if code_column in frame.columns:
        frame[code_column] = frame[code_column].astype(str)
        return frame.dropna(subset=[code_column]).drop_duplicates(subset=[code_column], keep="last")

    if isinstance(frame.index, pd.Index) and not isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.reset_index()
        first = frame.columns[0]
        frame = frame.rename(columns={first: code_column})
        frame[code_column] = frame[code_column].astype(str)
        return frame.dropna(subset=[code_column]).drop_duplicates(subset=[code_column], keep="last")

    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.reset_index()
        if frame.empty:
            return pd.DataFrame(columns=[code_column, value_column or "value"])
        first = frame.columns[0]
        frame = frame.rename(columns={first: code_column})
        frame[code_column] = frame[code_column].astype(str)
        return frame.dropna(subset=[code_column]).drop_duplicates(subset=[code_column], keep="last")

    raise ValueError(
        f"Unable to normalize code feed {name!r}; expected a code column or a non-date Index."
    )


def _normalize_feed_frame(
    data: TableLike,
    *,
    name: str,
    align_mode: str,
    date_column: str,
    code_column: str | None,
    value_column: str | None,
    availability_column: str | None,
    use_point_in_time: bool,
) -> pd.DataFrame:
    align_mode = _normalize_feed_align_mode(align_mode)
    if align_mode == "date":
        return _normalize_date_feed_frame(
            data,
            name=name,
            date_column=date_column,
            value_column=value_column,
            availability_column=availability_column,
            use_point_in_time=use_point_in_time,
        )
    if align_mode == "code":
        return _normalize_code_feed_frame(
            data,
            name=name,
            code_column=code_column,
            value_column=value_column,
        )
    return _normalize_code_date_feed_frame(
        data,
        name=name,
        date_column=date_column,
        code_column=code_column,
        value_column=value_column,
        availability_column=availability_column,
        use_point_in_time=use_point_in_time,
    )


def _resolve_feed_date_column(frame: pd.DataFrame, *, requested_date_column: str, as_ex: bool) -> str:
    if not as_ex:
        return requested_date_column
    if requested_date_column != "date_":
        return requested_date_column
    if "ex_date" in frame.columns:
        return "ex_date"
    return requested_date_column


def _shift_feed_dates_with_calendar(
    frame: pd.DataFrame,
    *,
    date_column: str,
    lag_sessions: int,
    calendar_name: str,
) -> pd.DataFrame:
    if lag_sessions == 0 or frame.empty or date_column not in frame.columns:
        return frame

    shifted = frame.copy()
    dates = pd.to_datetime(shifted[date_column], errors="coerce")
    valid = dates.notna()
    if not valid.any():
        return shifted

    min_date = pd.Timestamp(dates.loc[valid].min()).normalize()
    max_date = pd.Timestamp(dates.loc[valid].max()).normalize()
    sessions = build_trading_sessions(
        start=min_date - pd.Timedelta(days=max(5, int(abs(lag_sessions)) * 2 + 5)),
        end=max_date + pd.Timedelta(days=max(5, int(abs(lag_sessions)) * 2 + 5)),
        calendar_name=calendar_name,
        fallback_dates=dates.loc[valid],
    )
    if len(sessions) == 0:
        return _shift_feed_dates(
            shifted,
            date_column=date_column,
            lag_sessions=lag_sessions,
            bday_lag=True,
        )

    valid_dates = pd.Series(dates.loc[valid])
    positions = session_index_on_or_after(valid_dates, sessions)
    lagged = apply_session_lag(positions, lag_sessions=lag_sessions, session_count=len(sessions))
    shifted.loc[valid, date_column] = pd.NaT

    valid_index = dates.loc[valid].index
    safe_mask = lagged >= 0
    if safe_mask.any():
        session_values = pd.Series(sessions[lagged[safe_mask]].values, index=valid_index[safe_mask])
        time_offsets = valid_dates.loc[safe_mask] - valid_dates.loc[safe_mask].dt.normalize()
        shifted.loc[valid_index[safe_mask], date_column] = session_values.to_numpy() + time_offsets.to_numpy()
    return shifted


def _shift_feed_dates(
    frame: pd.DataFrame,
    *,
    date_column: str,
    lag_sessions: int,
    bday_lag: bool,
) -> pd.DataFrame:
    if lag_sessions == 0 or frame.empty:
        return frame
    offset = pd.offsets.BDay(lag_sessions) if bday_lag else pd.Timedelta(days=lag_sessions)
    shifted = frame.copy()
    shifted[date_column] = normalize_dates(shifted[date_column]) + offset
    return shifted


def _normalize_fill_method(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"ffill", "pad"}:
        return "ffill"
    if normalized in {"bfill", "backfill"}:
        return "bfill"
    if normalized in {"both", "ffill_bfill", "bfill_ffill"}:
        return "both"
    raise ValueError("fill_method must be one of: ffill, bfill, both.")


def _fill_merged_columns(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    key_column: str,
    code_column: str | None,
    fill_method: str | None,
    fill_limit: int | None,
) -> pd.DataFrame:
    if not columns:
        return frame
    method = _normalize_fill_method(fill_method)
    if method is None:
        return frame

    # Fill each value column independently over time. Columns never borrow
    # values from one another; only missing points within the same column are
    # propagated forward/backward.
    sort_keys = [key_column]
    if code_column is not None and code_column in frame.columns:
        sort_keys = [code_column, key_column]

    ordered = frame.sort_values(sort_keys, kind="stable").copy()
    fill_columns = list(columns)
    if code_column is not None and code_column in ordered.columns:
        grouped = ordered.groupby(code_column, sort=False)
        if method == "ffill":
            ordered.loc[:, fill_columns] = grouped[fill_columns].ffill(limit=fill_limit)
        elif method == "bfill":
            ordered.loc[:, fill_columns] = grouped[fill_columns].bfill(limit=fill_limit)
        else:
            ordered.loc[:, fill_columns] = grouped[fill_columns].ffill(limit=fill_limit)
            ordered.loc[:, fill_columns] = grouped[fill_columns].bfill(limit=fill_limit)
    else:
        if method == "ffill":
            ordered.loc[:, fill_columns] = ordered.loc[:, fill_columns].ffill(limit=fill_limit)
        elif method == "bfill":
            ordered.loc[:, fill_columns] = ordered.loc[:, fill_columns].bfill(limit=fill_limit)
        else:
            ordered.loc[:, fill_columns] = ordered.loc[:, fill_columns].ffill(limit=fill_limit)
            ordered.loc[:, fill_columns] = ordered.loc[:, fill_columns].bfill(limit=fill_limit)

    return ordered.sort_values(sort_keys, kind="stable").reset_index(drop=True)


def _candidate_value_columns(frame: pd.DataFrame, key_columns: set[str]) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in key_columns and not str(column).startswith("source_")
    ]


def _normalize_align_mode(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"outer", "intersection"}:
        raise ValueError(f"Unsupported align_mode={value!r}; expected 'outer' or 'intersection'.")
    return normalized


def _normalize_feed_align_mode(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"code_date", "date", "code", "asof"}:
        raise ValueError(
            f"Unsupported feed align_mode={value!r}; expected 'code_date', 'date', 'code', or 'asof'."
        )
    return normalized


def _normalize_window_bound(value: Any | None) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts


def _normalize_freq(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized or None


def _resolve_time_kind_from_freq(freq: str | None) -> str | None:
    normalized = _normalize_freq(freq)
    if normalized is None:
        return None

    if normalized == "1d":
        return "daily"

    if normalized in {"1min", "15min", "20min", "30min", "1h", "2h"}:
        return "intraday"
    return None


def _infer_feed_time_kind(feed: "FactorFrameFeed") -> str:
    if not feed.has_date_axis:
        return "static"

    dates = pd.Series(pd.to_datetime(feed.frame[feed.date_column], errors="coerce")).dropna()
    if dates.empty:
        return "daily"

    normalized = dates.dt.normalize()
    return "intraday" if (dates != normalized).any() else "daily"


def _resolve_time_kind(feeds: Iterable["FactorFrameFeed"]) -> str:
    time_kinds: set[str] = set()
    for feed in feeds:
        kind = _infer_feed_time_kind(feed)
        if kind != "static":
            time_kinds.add(kind)

    if not time_kinds:
        return "static"
    if time_kinds == {"daily"}:
        return "daily"
    if time_kinds == {"intraday"}:
        return "intraday"
    raise ValueError("Cannot mix daily and intraday feeds in one FactorFrameEngine run.")


def _expand_window_end(value: pd.Timestamp | None, *, time_kind: str) -> pd.Timestamp | None:
    if value is None or time_kind != "intraday":
        return value
    if value != value.normalize():
        return value
    return value + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)


def _filter_frame_by_window(
    frame: pd.DataFrame,
    *,
    date_column: str,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> pd.DataFrame:
    if frame.empty or date_column not in frame.columns or (start is None and end is None):
        return frame
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    mask = dates.notna()
    if start is not None:
        mask &= dates >= start
    if end is not None:
        mask &= dates <= end
    return frame.loc[mask].copy()


def _feed_key_columns(feed: "FactorFrameFeed") -> tuple[str, ...]:
    key_columns: list[str] = []
    if feed.align_mode in {"code_date", "date", "asof"} and feed.date_column in feed.frame.columns:
        key_columns.append(feed.date_column)
    if (
        feed.align_mode in {"code_date", "code", "asof"}
        and feed.code_column is not None
        and feed.code_column in feed.frame.columns
    ):
        key_columns.append(feed.code_column)
    return tuple(key_columns)


@dataclass(frozen=True)
class FactorFrameFeed:
    kind: str
    name: str
    frame: pd.DataFrame
    align_mode: str = "code_date"
    adjusted: bool | None = None
    save: bool = False
    lag_sessions: int = 0
    bday_lag: bool = True
    availability_column: str | None = None
    use_point_in_time: bool = True
    fill_method: str | None = None
    fill_limit: int | None = None
    date_column: str = "date_"
    code_column: str | None = "code"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def value_columns(self) -> tuple[str, ...]:
        return tuple(_candidate_value_columns(self.frame, set(_feed_key_columns(self))))

    @property
    def has_date_axis(self) -> bool:
        return self.date_column in _feed_key_columns(self)

    @property
    def has_code_axis(self) -> bool:
        return self.code_column is not None and self.code_column in _feed_key_columns(self)

    @property
    def code_level(self) -> bool:
        return self.has_code_axis


@dataclass(frozen=True)
class FactorFrameStrategySpec:
    name: str
    fn: Callable[["FactorFrameContext"], Any]
    save: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FactorFrameScreenSpec:
    name: str
    fn: Callable[["FactorFrameContext"], Any]
    save: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FactorFrameClassifierSpec:
    name: str
    fn: Callable[["FactorFrameContext"], Any]
    save: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FactorFrameBuildConfig:
    bday_lag: bool = True
    as_ex: bool = False
    calendar: str | None = None
    label_side: str = "right"
    freq: str | None = None
    time_kind: str = "daily"
    use_point_in_time: bool = True
    availability_column: str | None = None
    align_mode: str = "outer"
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None


@dataclass(frozen=True)
class FactorFrameResult:
    factor_frame: pd.DataFrame
    combined_frame: pd.DataFrame
    feeds: tuple[FactorFrameFeed, ...]
    screen_frames: dict[str, pd.DataFrame]
    classifier_frames: dict[str, pd.DataFrame]
    screen_mask: pd.DataFrame | None
    strategy_frames: dict[str, pd.DataFrame]
    output_dir: Path | None
    saved_paths: dict[str, Path]
    manifest: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_frame": self.factor_frame.to_dict(orient="records"),
            "combined_frame": self.combined_frame.to_dict(orient="records"),
            "feeds": [
                {
                    "kind": feed.kind,
                    "name": feed.name,
                    "align_mode": feed.align_mode,
                    "adjusted": feed.adjusted,
                    "date_column": feed.date_column,
                    "code_column": feed.code_column,
                    "fill_method": feed.fill_method,
                    "fill_limit": feed.fill_limit,
                    "code_level": feed.code_level,
                    "rows": int(len(feed.frame)),
                    "columns": list(feed.frame.columns),
                    "metadata": feed.metadata,
                }
                for feed in self.feeds
            ],
            "screen_frames": {name: frame.to_dict(orient="records") for name, frame in self.screen_frames.items()},
            "classifier_frames": {
                name: frame.to_dict(orient="records") for name, frame in self.classifier_frames.items()
            },
            "screen_mask": None if self.screen_mask is None else self.screen_mask.to_dict(orient="records"),
            "strategy_frames": {name: frame.to_dict(orient="records") for name, frame in self.strategy_frames.items()},
            "output_dir": None if self.output_dir is None else str(self.output_dir),
            "saved_paths": {key: str(path) for key, path in self.saved_paths.items()},
            "manifest": self.manifest,
        }


class FactorFrameContext:
    def __init__(
        self,
        *,
        feeds: dict[str, FactorFrameFeed],
        combined_frame: pd.DataFrame,
        config: FactorFrameBuildConfig,
        screen_frames: dict[str, pd.DataFrame] | None = None,
        classifier_frames: dict[str, pd.DataFrame] | None = None,
        screen_mask: pd.DataFrame | None = None,
    ) -> None:
        self._feeds = dict(feeds)
        self._combined_frame = combined_frame.copy()
        self._config = config
        self._screen_frames = {name: frame.copy() for name, frame in (screen_frames or {}).items()}
        self._classifier_frames = {name: frame.copy() for name, frame in (classifier_frames or {}).items()}
        self._screen_mask = None if screen_mask is None else screen_mask.copy()
        self._feeds_by_kind: dict[str, list[str]] = {}
        for name, feed in self._feeds.items():
            self._feeds_by_kind.setdefault(feed.kind, []).append(name)

    @property
    def combined_frame(self) -> pd.DataFrame:
        return self._combined_frame.copy()

    @property
    def feeds(self) -> tuple[FactorFrameFeed, ...]:
        return tuple(self._feeds.values())

    @property
    def data(self) -> pd.DataFrame:
        return self.combined_frame

    @property
    def screen_frames(self) -> dict[str, pd.DataFrame]:
        return {name: frame.copy() for name, frame in self._screen_frames.items()}

    @property
    def classifier_frames(self) -> dict[str, pd.DataFrame]:
        return {name: frame.copy() for name, frame in self._classifier_frames.items()}

    @property
    def screen_mask(self) -> pd.DataFrame | None:
        return None if self._screen_mask is None else self._screen_mask.copy()

    @property
    def build_config(self) -> FactorFrameBuildConfig:
        return self._config

    @property
    def as_ex(self) -> bool:
        return self._config.as_ex

    @property
    def calendar(self) -> str | None:
        return self._config.calendar

    @property
    def label_side(self) -> str:
        return self._config.label_side

    @property
    def bday_lag(self) -> bool:
        return self._config.bday_lag

    @property
    def freq(self) -> str | None:
        return self._config.freq

    @property
    def time_kind(self) -> str:
        return self._config.time_kind

    @property
    def use_point_in_time(self) -> bool:
        return self._config.use_point_in_time

    @property
    def availability_column(self) -> str | None:
        return self._config.availability_column

    @property
    def align_mode(self) -> str:
        return self._config.align_mode

    @property
    def start(self) -> pd.Timestamp | None:
        return self._config.start

    @property
    def end(self) -> pd.Timestamp | None:
        return self._config.end

    def feed(self, name: str) -> FactorFrameFeed:
        try:
            return self._feeds[name]
        except KeyError as exc:
            raise KeyError(f"Unknown feed {name!r}. Available feeds: {sorted(self._feeds)}") from exc

    def feed_frame(self, name: str) -> pd.DataFrame:
        return self.feed(name).frame.copy()

    def screen(self, name: str) -> pd.DataFrame:
        try:
            return self._screen_frames[name].copy()
        except KeyError as exc:
            raise KeyError(f"Unknown screen {name!r}. Available screens: {sorted(self._screen_frames)}") from exc

    def classifier(self, name: str) -> pd.DataFrame:
        try:
            return self._classifier_frames[name].copy()
        except KeyError as exc:
            raise KeyError(
                f"Unknown classifier {name!r}. Available classifiers: {sorted(self._classifier_frames)}"
            ) from exc

    def feeds_of_kind(self, kind: str) -> tuple[FactorFrameFeed, ...]:
        names = self._feeds_by_kind.get(kind, [])
        return tuple(self._feeds[name] for name in names)

    def primary_feed(self, kind: str) -> FactorFrameFeed:
        feeds = self.feeds_of_kind(kind)
        if not feeds:
            raise KeyError(f"No feed registered for kind={kind!r}")
        return feeds[0]

    @property
    def price(self) -> pd.DataFrame:
        return self.primary_feed("price").frame.copy()

    @property
    def financial(self) -> pd.DataFrame:
        return self.primary_feed("financial").frame.copy()

    @property
    def valuation(self) -> pd.DataFrame:
        return self.primary_feed("valuation").frame.copy()

    @property
    def macro(self) -> pd.DataFrame:
        return self.primary_feed("macro").frame.copy()

    @property
    def news(self) -> pd.DataFrame:
        return self.primary_feed("news").frame.copy()

    def feed_wide(self, name: str, value_column: str | None = None) -> pd.DataFrame:
        feed = self.feed(name)
        if not (feed.has_date_axis and feed.has_code_axis):
            raise ValueError(f"Feed {name!r} is not code-date aligned and cannot be reshaped to wide codes.")
        value_columns = feed.value_columns
        if value_column is None:
            if len(value_columns) != 1:
                raise ValueError(
                    f"Feed {name!r} has multiple value columns {list(value_columns)!r}; "
                    "provide value_column explicitly."
                )
            value_column = value_columns[0]
        prefixed_column = f"{feed.name}__{value_column}"
        if prefixed_column in self._combined_frame.columns and {"date_", "code"}.issubset(self._combined_frame.columns):
            frame = self._combined_frame.loc[:, ["date_", "code", prefixed_column]].copy()
            wide = frame.pivot_table(index="date_", columns="code", values=prefixed_column, aggfunc="last").sort_index()
        else:
            frame = feed.frame.copy()
            if value_column not in frame.columns:
                raise KeyError(f"Feed {name!r} has no value column {value_column!r}")
            wide = frame.pivot_table(
                index=feed.date_column,
                columns=feed.code_column,
                values=value_column,
                aggfunc="last",
            ).sort_index()
        wide.columns = wide.columns.astype(str)
        return wide

    def feed_series(self, name: str, value_column: str | None = None) -> pd.Series:
        feed = self.feed(name)
        if not feed.has_date_axis or feed.has_code_axis:
            raise ValueError(f"Feed {name!r} is not date-aligned and cannot be represented as a single series.")
        value_columns = feed.value_columns
        if value_column is None:
            if len(value_columns) != 1:
                raise ValueError(
                    f"Feed {name!r} has multiple value columns {list(value_columns)!r}; "
                    "provide value_column explicitly."
                )
            value_column = value_columns[0]
        prefixed_column = f"{feed.name}__{value_column}"
        if prefixed_column in self._combined_frame.columns and "date_" in self._combined_frame.columns:
            frame = self._combined_frame.loc[:, ["date_", prefixed_column]].copy()
            series = frame.groupby("date_", sort=True)[prefixed_column].last()
        else:
            frame = feed.frame.copy()
            if value_column not in frame.columns:
                raise KeyError(f"Feed {name!r} has no value column {value_column!r}")
            series = frame.set_index(feed.date_column)[value_column].sort_index()
        series.index = pd.DatetimeIndex(pd.to_datetime(series.index, errors="coerce"))
        return series


def _normalize_strategy_output(name: str, output: Any) -> pd.DataFrame:
    if output is None:
        return pd.DataFrame(columns=["date_", "code", name])

    if isinstance(output, pd.Series):
        if isinstance(output.index, pd.MultiIndex):
            frame = output.rename(name).reset_index()
            if frame.shape[1] < 3:
                raise ValueError(f"Strategy {name!r} returned a MultiIndex series that could not be normalized.")
            frame.columns = ["date_", "code", name, *frame.columns[3:]]
            frame = frame.loc[:, ["date_", "code", name]]
            frame["date_"] = normalize_dates(frame["date_"])
            frame["code"] = frame["code"].astype(str)
            return frame.dropna(subset=[name]).drop_duplicates(subset=["date_", "code"], keep="last")
        frame = output.rename(name).to_frame().reset_index()
        if frame.shape[1] < 2:
            raise ValueError(f"Strategy {name!r} returned a series without an index.")
        frame.columns = ["code", name, *frame.columns[2:]]
        frame["code"] = frame["code"].astype(str)
        return frame.dropna(subset=[name]).drop_duplicates(subset=["code"], keep="last")

    if isinstance(output, pd.DataFrame):
        if {"date_", "code"}.issubset(output.columns):
            value_columns = [column for column in output.columns if column not in {"date_", "code"}]
            if len(value_columns) != 1:
                raise ValueError(
                    f"Strategy {name!r} must return exactly one value column when using long format; "
                    f"got columns={list(output.columns)!r}"
                )
            value_column = value_columns[0]
            frame = output.loc[:, ["date_", "code", value_column]].copy()
            frame = frame.rename(columns={value_column: name})
            frame["date_"] = normalize_dates(frame["date_"])
            frame["code"] = frame["code"].astype(str)
            return frame.dropna(subset=[name]).drop_duplicates(subset=["date_", "code"], keep="last")

        if not isinstance(output.index, pd.DatetimeIndex) and not isinstance(output.index, pd.MultiIndex):
            frame = output.copy().reset_index()
            if frame.shape[1] < 2:
                raise ValueError(f"Strategy {name!r} returned a DataFrame without an index.")
            first = frame.columns[0]
            frame = frame.rename(columns={first: "code"})
            frame["code"] = frame["code"].astype(str)
            return frame.drop_duplicates(subset=["code"], keep="last")

        if isinstance(output.index, pd.DatetimeIndex):
            long_df = to_long_factor(output, name)
            return long_df.rename(columns={name: name})

        if isinstance(output.index, pd.MultiIndex):
            frame = output.copy().reset_index()
            if frame.shape[1] < 3:
                raise ValueError(f"Strategy {name!r} returned a MultiIndex DataFrame that could not be normalized.")
            value_columns = [column for column in frame.columns if column not in {"date_", "code"}]
            if len(value_columns) == 1:
                value_column = value_columns[0]
                frame = frame.rename(columns={value_column: name})
                frame["date_"] = normalize_dates(frame["date_"])
                frame["code"] = frame["code"].astype(str)
                return frame.loc[:, ["date_", "code", name]].dropna(subset=[name]).drop_duplicates(
                    subset=["date_", "code"], keep="last"
                )

        if output.shape[1] == 1:
            frame = output.copy()
            frame.columns = [name]
            return _normalize_strategy_output(name, frame)

    raise ValueError(f"Strategy {name!r} must return a DataFrame or Series that can be normalized.")


def _panel_codes(context: FactorFrameContext) -> list[str]:
    frame = context.combined_frame
    if "code" not in frame.columns:
        return []
    return frame["code"].dropna().astype(str).drop_duplicates().tolist()


def _panel_dates(context: FactorFrameContext) -> list[pd.Timestamp]:
    frame = context.combined_frame
    if "date_" not in frame.columns:
        return []
    dates = pd.to_datetime(frame["date_"], errors="coerce").dropna().drop_duplicates().sort_values()
    return [pd.Timestamp(date) for date in dates]


def _broadcast_term_frame(frame: pd.DataFrame, context: FactorFrameContext) -> pd.DataFrame:
    if frame.empty:
        return frame
    has_date = "date_" in frame.columns
    has_code = "code" in frame.columns
    if has_date and has_code:
        return frame

    broadcast = frame.copy()
    if has_date and not has_code:
        codes = _panel_codes(context)
        if not codes:
            raise ValueError("Cannot broadcast a date-aligned frame without any codes in the research panel.")
        broadcast["_broadcast_key"] = 1
        code_frame = pd.DataFrame({"code": codes, "_broadcast_key": 1})
        broadcast = broadcast.merge(code_frame, on="_broadcast_key", how="left").drop(columns="_broadcast_key")
    elif has_code and not has_date:
        dates = _panel_dates(context)
        if not dates:
            raise ValueError("Cannot broadcast a code-aligned frame without any dates in the research panel.")
        broadcast["_broadcast_key"] = 1
        date_frame = pd.DataFrame({"date_": dates, "_broadcast_key": 1})
        broadcast = broadcast.merge(date_frame, on="_broadcast_key", how="left").drop(columns="_broadcast_key")
    else:
        raise ValueError("Unable to broadcast a frame without a date or code axis.")

    ordered_columns = [column for column in ["date_", "code"] if column in broadcast.columns]
    ordered_columns.extend(column for column in broadcast.columns if column not in ordered_columns)
    return broadcast.loc[:, ordered_columns]


def _normalize_screen_output(name: str, output: Any, context: FactorFrameContext) -> pd.DataFrame:
    frame = _normalize_strategy_output(name, output)
    frame = _broadcast_term_frame(frame, context)
    value_columns = [column for column in frame.columns if column not in {"date_", "code"}]
    if len(value_columns) != 1:
        raise ValueError(f"Screen {name!r} must resolve to exactly one boolean column; got {list(frame.columns)!r}")
    value_column = value_columns[0]
    frame = frame.loc[:, ["date_", "code", value_column]].copy()
    frame = frame.rename(columns={value_column: name})
    frame["date_"] = pd.to_datetime(frame["date_"], errors="coerce")
    frame["code"] = frame["code"].astype(str)
    frame[name] = frame[name].astype("boolean").fillna(False).astype(bool)
    return frame.drop_duplicates(subset=["date_", "code"], keep="last").sort_values(["date_", "code"]).reset_index(
        drop=True
    )


def _normalize_classifier_output(name: str, output: Any) -> pd.DataFrame:
    frame = _normalize_strategy_output(name, output)
    if "date_" in frame.columns:
        frame["date_"] = pd.to_datetime(frame["date_"], errors="coerce")
    if "code" in frame.columns:
        frame["code"] = frame["code"].astype(str)
    return frame


def _combine_screen_frames(screen_frames: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    if not screen_frames:
        return None
    merged: pd.DataFrame | None = None
    for name, frame in screen_frames.items():
        value_columns = [column for column in frame.columns if column not in {"date_", "code"}]
        if len(value_columns) != 1:
            raise ValueError(
                f"Screen {name!r} must resolve to exactly one boolean column; got {list(frame.columns)!r}"
            )
        current = frame.loc[:, ["date_", "code", value_columns[0]]].copy()
        current = current.rename(columns={value_columns[0]: name})
        current[name] = current[name].astype("boolean").fillna(False).astype(bool)
        merged = current if merged is None else merged.merge(current, on=["date_", "code"], how="outer")

    assert merged is not None
    screen_names = list(screen_frames.keys())
    merged[screen_names] = merged[screen_names].astype("boolean").fillna(False).astype(bool)
    merged["__screen__"] = merged[screen_names].all(axis=1)
    return merged.loc[:, ["date_", "code", "__screen__"]].sort_values(["date_", "code"]).reset_index(drop=True)


def _filter_feed_frame_by_screen_mask(feed: FactorFrameFeed, screen_mask: pd.DataFrame | None) -> pd.DataFrame:
    if screen_mask is None or screen_mask.empty or feed.frame.empty:
        return feed.frame.copy()

    frame = feed.frame.copy()
    mask = screen_mask.loc[:, ["date_", "code", "__screen__"]].copy()
    mask["date_"] = pd.to_datetime(mask["date_"], errors="coerce")
    mask["code"] = mask["code"].astype(str)
    mask = mask.rename(columns={"date_": "__mask_date__", "code": "__mask_code__"})
    allowed_codes = (
        mask.loc[mask["__screen__"].astype("boolean").fillna(False).astype(bool), "__mask_code__"]
        .dropna()
        .astype(str)
        .drop_duplicates()
    )

    has_date = feed.date_column in frame.columns
    has_code = feed.code_column is not None and feed.code_column in frame.columns

    if has_date and has_code:
        frame[feed.date_column] = pd.to_datetime(frame[feed.date_column], errors="coerce")
        frame[feed.code_column] = frame[feed.code_column].astype(str)
        merged = frame.merge(
            mask,
            left_on=[feed.date_column, feed.code_column],
            right_on=["__mask_date__", "__mask_code__"],
            how="left",
        )
        merged["__screen__"] = merged["__screen__"].astype("boolean").fillna(False).astype(bool)
        drop_columns = [column for column in ["__mask_date__", "__mask_code__", "__screen__"] if column in merged.columns]
        return merged.loc[merged["__screen__"]].drop(columns=drop_columns).reset_index(drop=True)

    if has_code:
        frame[feed.code_column] = frame[feed.code_column].astype(str)
        return frame.loc[frame[feed.code_column].isin(set(allowed_codes))].reset_index(drop=True)

    return frame


def _filter_combined_frame_by_screen_mask(
    frame: pd.DataFrame,
    screen_mask: pd.DataFrame | None,
) -> pd.DataFrame:
    if frame.empty or screen_mask is None or screen_mask.empty:
        return frame
    if "code" not in frame.columns:
        return frame

    masked = frame.copy()
    mask = screen_mask.loc[:, ["date_", "code", "__screen__"]].copy()
    mask["date_"] = pd.to_datetime(mask["date_"], errors="coerce")
    mask["code"] = mask["code"].astype(str)
    mask = mask.rename(columns={"date_": "__mask_date__", "code": "__mask_code__"})
    allowed_codes = (
        mask.loc[mask["__screen__"].astype("boolean").fillna(False).astype(bool), "__mask_code__"]
        .dropna()
        .astype(str)
        .drop_duplicates()
    )

    if "date_" in masked.columns:
        masked["date_"] = pd.to_datetime(masked["date_"], errors="coerce")
        merged = masked.merge(
            mask,
            left_on=["date_", "code"],
            right_on=["__mask_date__", "__mask_code__"],
            how="left",
        )
        merged["__screen__"] = merged["__screen__"].astype("boolean").fillna(False).astype(bool)
        drop_columns = [column for column in ["__mask_date__", "__mask_code__", "__screen__"] if column in merged.columns]
        return merged.loc[merged["__screen__"]].drop(columns=drop_columns).reset_index(drop=True)

    return masked.loc[masked["code"].astype(str).isin(set(allowed_codes))].reset_index(drop=True)


def _screen_feeds(feeds: Iterable[FactorFrameFeed], screen_mask: pd.DataFrame | None) -> tuple[FactorFrameFeed, ...]:
    screened: list[FactorFrameFeed] = []
    for feed in feeds:
        screened.append(
            FactorFrameFeed(
                kind=feed.kind,
                name=feed.name,
                frame=_filter_feed_frame_by_screen_mask(feed, screen_mask),
                align_mode=feed.align_mode,
                adjusted=feed.adjusted,
                save=feed.save,
                lag_sessions=feed.lag_sessions,
                bday_lag=feed.bday_lag,
                availability_column=feed.availability_column,
                use_point_in_time=feed.use_point_in_time,
                fill_method=feed.fill_method,
                fill_limit=feed.fill_limit,
                date_column=feed.date_column,
                code_column=feed.code_column,
                metadata=feed.metadata,
            )
        )
    return tuple(screened)


def _normalize_combined_feeds(
    feeds: Iterable[FactorFrameFeed],
    *,
    align_mode: str = "outer",
    time_kind: str = "daily",
) -> pd.DataFrame:
    feeds = tuple(feeds)
    if not feeds:
        return pd.DataFrame(columns=["date_", "code"])

    align_mode = _normalize_align_mode(align_mode)
    date_feeds = [feed for feed in feeds if feed.has_date_axis]
    code_feeds = [feed for feed in feeds if feed.has_code_axis]
    date_sets: list[set[pd.Timestamp]] = []
    for feed in date_feeds:
        dates = pd.to_datetime(feed.frame[feed.date_column], errors="coerce").dropna()
        if time_kind == "intraday":
            date_sets.append({pd.Timestamp(date) for date in dates})
        else:
            normalized_dates = {pd.Timestamp(date).normalize() for date in dates}
            date_sets.append(normalized_dates)

    if align_mode == "intersection" and date_sets:
        all_dates = sorted(set.intersection(*date_sets)) if len(date_sets) > 1 else sorted(date_sets[0])
    else:
        all_dates = sorted(set.union(*date_sets)) if date_sets else []
    all_dates = list(pd.DatetimeIndex(pd.to_datetime(all_dates)).sort_values())

    code_sets: list[set[str]] = []
    if code_feeds:
        code_sets = [
            {str(code) for code in feed.frame[feed.code_column].dropna().astype(str).tolist()}
            for feed in code_feeds
        ]
        if align_mode == "intersection" and code_sets:
            all_codes = sorted(set.intersection(*code_sets)) if len(code_sets) > 1 else sorted(code_sets[0])
        else:
            all_codes = sorted(set.union(*code_sets)) if code_sets else []
    else:
        all_codes = []

    if all_dates and all_codes:
        base = pd.MultiIndex.from_product([all_dates, all_codes], names=["date_", "code"]).to_frame(index=False)
    elif all_dates:
        base = pd.DataFrame({"date_": all_dates})
    elif all_codes:
        base = pd.DataFrame({"code": all_codes})
    else:
        base = pd.DataFrame(columns=["date_"])

    merged = base.copy()
    for feed in feeds:
        key_columns = set(_feed_key_columns(feed))
        if not key_columns:
            raise ValueError(
                f"Feed {feed.name!r} does not expose any keys for align_mode={feed.align_mode!r}."
            )

        rename_map: dict[str, str] = {}
        if feed.date_column in key_columns:
            rename_map[feed.date_column] = "date_"
        if feed.code_column is not None and feed.code_column in key_columns:
            rename_map[feed.code_column] = "code"
        prefixed = _prefixed_columns(feed.frame, feed.name, key_columns)
        prefixed = prefixed.rename(columns=rename_map)
        merged = merged.merge(prefixed, on=sorted(rename_map.values()), how="left")
        joined_columns = [
            column
            for column in prefixed.columns
            if column not in set(rename_map.values()) and not str(column).startswith("source_")
        ]
        if feed.has_date_axis and (feed.fill_method is not None or feed.align_mode == "asof"):
            merged = _fill_merged_columns(
                merged,
                columns=joined_columns,
                key_column="date_",
                code_column="code" if "code" in merged.columns else None,
                fill_method="ffill" if feed.align_mode == "asof" else feed.fill_method,
                fill_limit=feed.fill_limit,
            )

    if "date_" in merged.columns:
        merged["date_"] = pd.to_datetime(merged["date_"], errors="coerce")
    if "date_" in merged.columns and "code" in merged.columns:
        merged["code"] = merged["code"].astype(str)
        merged = merged.sort_values(["date_", "code"]).reset_index(drop=True)
    elif "date_" in merged.columns:
        merged = merged.sort_values(["date_"]).reset_index(drop=True)
    elif "code" in merged.columns:
        merged["code"] = merged["code"].astype(str)
        merged = merged.sort_values(["code"]).reset_index(drop=True)
    else:
        merged = merged.reset_index(drop=True)
    return merged


class FactorFrameEngine:
    """Vectorized research engine for Tiger factor research.

    The engine deliberately stays one layer above evaluation:

    - feed materialized DataFrames directly
    - declare the primary research data frequency once with ``freq`` such as
      ``"1d"``, ``"1h"``, ``"1min"``, ``"15min"``, ``"20min"``,
      ``"30min"``, or ``"2h"``
    - choose the intraday timestamp convention with ``label_side``:
      ``"right"`` for end-labeled bars, ``"left"`` for start-labeled bars,
      or ``"auto"`` when the engine should try to infer the source
    - align and broadcast date-level feeds over the equity universe
    - hand strategy callables a unified research context
    - return a factor_frame that can be passed to FactorEvaluationEngine
    """

    def __init__(
        self,
        *,
        output_root_dir: str | Path | None = None,
        save: bool = False,
        bday_lag: bool = True,
        as_ex: bool = False,
        calendar: str | None = None,
        label_side: str = "right",
        freq: str | None = None,
        use_point_in_time: bool = True,
        availability_column: str | None = None,
        align_mode: str = "outer",
        start: Any | None = None,
        end: Any | None = None,
        definition_registry: Any | None = None,
    ) -> None:
        if definition_registry is None:
            from tiger_factors.factor_frame.definition_registry import FactorDefinitionRegistry

            definition_registry = FactorDefinitionRegistry()
        self.output_root_dir = None if output_root_dir is None else Path(output_root_dir)
        self.save = bool(save)
        self._time_layer = FactorFrameTimeLayer(
            freq=freq,
            bday_lag=bool(bday_lag),
            as_ex=bool(as_ex),
            calendar=calendar,
            label_side=label_side,
        )
        self.bday_lag = self._time_layer.bday_lag
        self.calendar = self._time_layer.calendar
        self.label_side = self._time_layer.normalized_label_side
        self.as_ex = self._time_layer.as_ex
        self.freq = self._time_layer.normalized_freq
        self.use_point_in_time = bool(use_point_in_time)
        self.availability_column = availability_column
        self.align_mode = _normalize_align_mode(align_mode)
        self.start = self._time_layer.normalize_window_bound(start)
        self.end = self._time_layer.normalize_window_bound(end)
        self._definition_registry = definition_registry
        self._feeds: list[FactorFrameFeed] = []
        self._screens: list[FactorFrameScreenSpec] = []
        self._classifiers: list[FactorFrameClassifierSpec] = []
        self._strategies: list[FactorFrameStrategySpec] = []
        self._result: FactorFrameResult | None = None

    def feed(
        self,
        kind: str,
        data: TableLike,
        *,
        name: str | None = None,
        adjusted: bool | None = None,
        date_column: str = "date_",
        code_column: str | None = "code",
        value_column: str | None = None,
        align_mode: str | None = None,
        availability_column: str | None = None,
        lag_sessions: int = 0,
        fill_method: str | None = None,
        fill_limit: int | None = None,
        metadata: dict[str, Any] | None = None,
        save: bool = False,
    ) -> "FactorFrameEngine":
        feed_name = name or kind
        effective_availability_column = availability_column if availability_column is not None else self.availability_column
        effective_align_mode = _normalize_feed_align_mode(
            align_mode or ("date" if code_column is None else "code_date")
        )
        normalized_fill_method = _normalize_fill_method(fill_method)
        normalized_fill_limit = None if fill_limit is None else int(fill_limit)
        resolved_date_column = self._time_layer.resolve_feed_date_column(
            data if isinstance(data, pd.DataFrame) else pd.DataFrame(),
            requested_date_column=date_column,
        )
        if effective_align_mode == "code" and int(lag_sessions) != 0:
            raise ValueError("lag_sessions is not supported for code-only feeds.")
        if effective_align_mode == "code" and effective_availability_column is not None:
            raise ValueError("availability_column is not supported for code-only feeds.")
        frame = _normalize_feed_frame(
            data,
            name=feed_name,
            align_mode=effective_align_mode,
            date_column=resolved_date_column,
            code_column=code_column,
            value_column=value_column,
            availability_column=effective_availability_column,
            use_point_in_time=self.use_point_in_time,
        )
        frame = self._time_layer.normalize_feed_labels(frame, date_column=resolved_date_column)
        if effective_align_mode != "code" and int(lag_sessions) != 0:
            frame = self._time_layer.shift_feed_dates(
                frame,
                date_column=resolved_date_column,
                lag_sessions=int(lag_sessions),
            )
        self._feeds.append(
            FactorFrameFeed(
                kind=str(kind),
                name=str(feed_name),
                frame=frame,
                align_mode=effective_align_mode,
                adjusted=adjusted,
                save=save,
                lag_sessions=int(lag_sessions),
                bday_lag=self.bday_lag,
                availability_column=effective_availability_column,
                use_point_in_time=self.use_point_in_time,
                fill_method=normalized_fill_method,
                fill_limit=normalized_fill_limit,
                date_column=resolved_date_column,
                code_column=code_column,
                metadata=dict(metadata or {}),
            )
        )
        return self

    def feed_price(
        self,
        data: TableLike,
        *,
        name: str = "price",
        date_column: str = "date_",
        code_column: str | None = "code",
        align_mode: str | None = "code_date",
        availability_column: str | None = None,
        lag_sessions: int = 0,
        fill_method: str | None = None,
        fill_limit: int | None = None,
        metadata: dict[str, Any] | None = None,
        save: bool = False,
    ) -> "FactorFrameEngine":
        return self.feed(
            "price",
            data,
            name=name,
            date_column=date_column,
            code_column=code_column,
            value_column="close",
            align_mode=align_mode,
            availability_column=availability_column,
            lag_sessions=lag_sessions,
            fill_method=fill_method,
            fill_limit=fill_limit,
            metadata=metadata,
            save=save,
        )

    def feed_financial(
        self,
        data: TableLike,
        *,
        name: str = "financial",
        adjusted: bool | None = None,
        date_column: str = "date_",
        code_column: str | None = "code",
        value_column: str | None = None,
        align_mode: str | None = "code_date",
        availability_column: str | None = None,
        lag_sessions: int = 1,
        fill_method: str | None = "ffill",
        fill_limit: int | None = None,
        metadata: dict[str, Any] | None = None,
        save: bool = False,
    ) -> "FactorFrameEngine":
        return self.feed(
            "financial",
            data,
            name=name,
            adjusted=adjusted,
            date_column=date_column,
            code_column=code_column,
            value_column=value_column,
            align_mode=align_mode,
            availability_column=availability_column,
            lag_sessions=lag_sessions,
            fill_method=fill_method,
            fill_limit=fill_limit,
            metadata=metadata,
            save=save,
        )

    def feed_valuation(
        self,
        data: TableLike,
        *,
        name: str = "valuation",
        adjusted: bool | None = None,
        date_column: str = "date_",
        code_column: str | None = "code",
        value_column: str | None = None,
        align_mode: str | None = "code_date",
        availability_column: str | None = None,
        lag_sessions: int = 1,
        fill_method: str | None = "ffill",
        fill_limit: int | None = None,
        metadata: dict[str, Any] | None = None,
        save: bool = False,
    ) -> "FactorFrameEngine":
        return self.feed(
            "valuation",
            data,
            name=name,
            adjusted=adjusted,
            date_column=date_column,
            code_column=code_column,
            value_column=value_column,
            align_mode=align_mode,
            availability_column=availability_column,
            lag_sessions=lag_sessions,
            fill_method=fill_method,
            fill_limit=fill_limit,
            metadata=metadata,
            save=save,
        )

    def feed_macro(
        self,
        data: TableLike,
        *,
        name: str = "macro",
        adjusted: bool | None = None,
        date_column: str = "date_",
        code_column: str | None = None,
        value_column: str | None = None,
        align_mode: str | None = "date",
        availability_column: str | None = None,
        lag_sessions: int = 0,
        fill_method: str | None = None,
        fill_limit: int | None = None,
        metadata: dict[str, Any] | None = None,
        save: bool = False,
    ) -> "FactorFrameEngine":
        return self.feed(
            "macro",
            data,
            name=name,
            adjusted=adjusted,
            date_column=date_column,
            code_column=code_column,
            value_column=value_column,
            align_mode=align_mode,
            availability_column=availability_column,
            lag_sessions=lag_sessions,
            fill_method=fill_method,
            fill_limit=fill_limit,
            metadata=metadata,
            save=save,
        )

    def feed_news(
        self,
        data: TableLike,
        *,
        name: str = "news",
        adjusted: bool | None = None,
        date_column: str = "date_",
        code_column: str | None = "code",
        value_column: str | None = None,
        align_mode: str | None = "code_date",
        availability_column: str | None = None,
        lag_sessions: int = 0,
        fill_method: str | None = None,
        fill_limit: int | None = None,
        metadata: dict[str, Any] | None = None,
        save: bool = False,
    ) -> "FactorFrameEngine":
        return self.feed(
            "news",
            data,
            name=name,
            adjusted=adjusted,
            date_column=date_column,
            code_column=code_column,
            value_column=value_column,
            align_mode=align_mode,
            availability_column=availability_column,
            lag_sessions=lag_sessions,
            fill_method=fill_method,
            fill_limit=fill_limit,
            metadata=metadata,
            save=save,
        )

    def add_strategy(
        self,
        name: str,
        fn: Callable[[FactorFrameContext], Any],
        *,
        save: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> "FactorFrameEngine":
        self._strategies.append(
            FactorFrameStrategySpec(
                name=str(name),
                fn=fn,
                save=save,
                metadata=dict(metadata or {}),
            )
        )
        return self

    def add_screen(
        self,
        name: str,
        fn: Callable[[FactorFrameContext], Any],
        *,
        save: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> "FactorFrameEngine":
        self._screens.append(
            FactorFrameScreenSpec(
                name=str(name),
                fn=fn,
                save=save,
                metadata=dict(metadata or {}),
            )
        )
        return self

    def add_classifier(
        self,
        name: str,
        fn: Callable[[FactorFrameContext], Any],
        *,
        save: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> "FactorFrameEngine":
        self._classifiers.append(
            FactorFrameClassifierSpec(
                name=str(name),
                fn=fn,
                save=save,
                metadata=dict(metadata or {}),
            )
        )
        return self

    def add_factor(
        self,
        name: str | FactorFrameFactor | "FactorFrameTemplate" | "FactorDefinition",
        fn: Callable[[FactorFrameContext], Any] | None = None,
        *,
        save: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> "FactorFrameEngine":
        from tiger_factors.factor_frame.definition import FactorDefinition
        from tiger_factors.factor_frame.factors import FactorFrameTemplate

        if isinstance(name, FactorFrameFactor):
            component = name
            return self.add_strategy(
                component.name,
                component.fn,
                save=save or component.save,
                metadata={**component.metadata, **dict(metadata or {})},
            )
        if isinstance(name, FactorFrameTemplate):
            return self.add_factor(name.build(), save=save, metadata=metadata)
        if isinstance(name, FactorDefinition):
            return self.add_definition(name, save=save, metadata=metadata)
        if fn is None:
            raise TypeError("add_factor() missing required callable fn for factor name.")
        return self.add_strategy(name, fn, save=save, metadata=metadata)

    def add_factors(
        self,
        *factors: str | FactorFrameFactor | "FactorFrameTemplate" | "FactorDefinition",
    ) -> "FactorFrameEngine":
        for factor in factors:
            self.add_factor(factor)
        return self

    def add_definition(
        self,
        definition: "FactorDefinition | str",
        *,
        save: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> "FactorFrameEngine":
        from tiger_factors.factor_frame.definition import FactorDefinition

        if isinstance(definition, str):
            definition = self._definition_registry.get(definition)
        if not isinstance(definition, FactorDefinition):
            raise TypeError("add_definition() expects a FactorDefinition or registered definition name.")
        factor = definition.to_factor(save=save, metadata=metadata)
        return self.add_factor(factor)

    @property
    def definition_registry(self) -> Any:
        return self._definition_registry

    def _windowed_feeds(self, *, time_kind: str) -> tuple[FactorFrameFeed, ...]:
        if self.start is None and self.end is None:
            return tuple(self._feeds)

        window_end = self._time_layer.expand_window_end(self.end, time_kind=time_kind)
        windowed: list[FactorFrameFeed] = []
        for feed in self._feeds:
            frame = feed.frame
            if feed.has_date_axis:
                frame = self._time_layer.filter_frame_by_window(
                    frame,
                    date_column=feed.date_column,
                    start=self.start,
                    end=window_end,
                )
            windowed.append(
                FactorFrameFeed(
                    kind=feed.kind,
                    name=feed.name,
                    frame=frame,
                    align_mode=feed.align_mode,
                    adjusted=feed.adjusted,
                    save=feed.save,
                    lag_sessions=feed.lag_sessions,
                    bday_lag=feed.bday_lag,
                    availability_column=feed.availability_column,
                    use_point_in_time=feed.use_point_in_time,
                    fill_method=feed.fill_method,
                    fill_limit=feed.fill_limit,
                    date_column=feed.date_column,
                    code_column=feed.code_column,
                    metadata=dict(feed.metadata),
                )
            )
        return tuple(windowed)

    @property
    def result(self) -> FactorFrameResult:
        if self._result is None:
            raise RuntimeError("FactorFrameEngine has not been run yet.")
        return self._result

    def build_context(self) -> FactorFrameContext:
        time_kind = self._time_layer.resolve_time_kind(self._feeds)
        feeds = self._windowed_feeds(time_kind=time_kind)
        combined_frame = _normalize_combined_feeds(feeds, align_mode=self.align_mode, time_kind=time_kind)
        feed_map = {feed.name: feed for feed in feeds}
        return FactorFrameContext(
            feeds=feed_map,
            combined_frame=combined_frame,
            config=FactorFrameBuildConfig(
                bday_lag=self.bday_lag,
                as_ex=self.as_ex,
                calendar=self.calendar,
                label_side=self.label_side,
                freq=self.freq,
                time_kind=time_kind,
                use_point_in_time=self.use_point_in_time,
                availability_column=self.availability_column,
                align_mode=self.align_mode,
                start=self.start,
                end=self.end,
            ),
        )

    def _output_dir(self, save: bool) -> Path | None:
        if not save:
            return None
        if self.output_root_dir is not None:
            return self.output_root_dir
        return Path.cwd() / "out" / "factor_frame"

    def _save_parquet(self, path: Path, frame: pd.DataFrame) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        return path

    def run(self, *, save: bool | None = None) -> FactorFrameResult:
        effective_save = self.save if save is None else bool(save)
        base_context = self.build_context()
        feeds = base_context.feeds
        combined_frame = base_context.combined_frame

        classifier_frames: dict[str, pd.DataFrame] = {}
        for spec in self._classifiers:
            output = spec.fn(base_context)
            normalized = _normalize_classifier_output(spec.name, output)
            classifier_frames[spec.name] = normalized

        context_with_classifiers = FactorFrameContext(
            feeds={feed.name: feed for feed in feeds},
            combined_frame=combined_frame,
            config=base_context.build_config,
            classifier_frames=classifier_frames,
        )

        screen_frames: dict[str, pd.DataFrame] = {}
        for spec in self._screens:
            output = spec.fn(context_with_classifiers)
            normalized = _normalize_screen_output(spec.name, output, context_with_classifiers)
            screen_frames[spec.name] = normalized
        screen_mask = _combine_screen_frames(screen_frames)

        screened_feeds = _screen_feeds(feeds, screen_mask)
        screened_combined_frame = _filter_combined_frame_by_screen_mask(combined_frame, screen_mask)

        context = FactorFrameContext(
            feeds={feed.name: feed for feed in screened_feeds},
            combined_frame=screened_combined_frame,
            config=base_context.build_config,
            screen_frames=screen_frames,
            classifier_frames=classifier_frames,
            screen_mask=screen_mask,
        )

        strategy_frames: dict[str, pd.DataFrame] = {}
        factor_frame = pd.DataFrame(columns=["date_", "code"])
        for spec in self._strategies:
            output = spec.fn(context)
            normalized = _normalize_strategy_output(spec.name, output)
            strategy_frames[spec.name] = normalized
            factor_frame = normalized if factor_frame.empty else factor_frame.merge(normalized, on=["date_", "code"], how="outer")

        if not factor_frame.empty:
            factor_frame["date_"] = pd.to_datetime(factor_frame["date_"], errors="coerce")
            factor_frame["code"] = factor_frame["code"].astype(str)
            factor_frame = factor_frame.sort_values(["date_", "code"]).reset_index(drop=True)

        should_save = (
            effective_save
            or any(feed.save for feed in feeds)
            or any(spec.save for spec in self._strategies)
            or any(spec.save for spec in self._screens)
            or any(spec.save for spec in self._classifiers)
        )
        output_dir = self._output_dir(should_save)
        saved_paths: dict[str, Path] = {}
        manifest: dict[str, Any] = {
            "feeds": [
                {
                    "kind": feed.kind,
                    "name": feed.name,
                    "align_mode": feed.align_mode,
                    "rows": int(len(feed.frame)),
                    "columns": list(feed.frame.columns),
                    "adjusted": feed.adjusted,
                    "save": feed.save,
                    "lag_sessions": feed.lag_sessions,
                    "bday_lag": feed.bday_lag,
                    "availability_column": feed.availability_column,
                    "use_point_in_time": feed.use_point_in_time,
                    "fill_method": feed.fill_method,
                    "fill_limit": feed.fill_limit,
                    "code_level": feed.code_level,
                    "metadata": feed.metadata,
                }
                for feed in feeds
            ],
            "screens": [
                {
                    "name": spec.name,
                    "rows": int(len(screen_frames.get(spec.name, pd.DataFrame()))),
                    "save": spec.save,
                    "metadata": spec.metadata,
                }
                for spec in self._screens
            ],
            "classifiers": [
                {
                    "name": spec.name,
                    "rows": int(len(classifier_frames.get(spec.name, pd.DataFrame()))),
                    "save": spec.save,
                    "metadata": spec.metadata,
                }
                for spec in self._classifiers
            ],
            "strategies": [spec.name for spec in self._strategies],
            "factor_rows": int(len(factor_frame)),
            "combined_rows": int(len(combined_frame)),
            "build_config": {
                "calendar": self.calendar,
                "bday_lag": self.bday_lag,
                "label_side": self.label_side,
                "freq": base_context.freq,
                "time_kind": base_context.time_kind,
                "use_point_in_time": self.use_point_in_time,
                "availability_column": self.availability_column,
                "align_mode": self.align_mode,
                "start": None if self.start is None else str(self.start),
                "end": None if self.end is None else str(self.end),
            },
        }

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            if should_save:
                saved_paths["combined_frame"] = self._save_parquet(output_dir / "combined_frame.parquet", combined_frame)
                saved_paths["factor_frame"] = self._save_parquet(output_dir / "factor_frame.parquet", factor_frame)
                if screen_mask is not None:
                    saved_paths["screen_mask"] = self._save_parquet(output_dir / "screen_mask.parquet", screen_mask)
                for name, frame in screen_frames.items():
                    spec = next((spec for spec in self._screens if spec.name == name), None)
                    if spec is not None and (spec.save or effective_save):
                        saved_paths[f"screen:{name}"] = self._save_parquet(output_dir / f"screen__{name}.parquet", frame)
                for name, frame in classifier_frames.items():
                    spec = next((spec for spec in self._classifiers if spec.name == name), None)
                    if spec is not None and (spec.save or effective_save):
                        saved_paths[f"classifier:{name}"] = self._save_parquet(
                            output_dir / f"classifier__{name}.parquet",
                            frame,
                        )
                for feed in feeds:
                    if feed.save or effective_save:
                        feed_path = output_dir / f"feed__{feed.name}.parquet"
                        saved_paths[f"feed:{feed.name}"] = self._save_parquet(feed_path, feed.frame)
                for spec in self._strategies:
                    if spec.save or effective_save:
                        strategy_path = output_dir / f"strategy__{spec.name}.parquet"
                        saved_paths[f"strategy:{spec.name}"] = self._save_parquet(strategy_path, strategy_frames[spec.name])
                manifest["output_dir"] = str(output_dir)
                manifest_path = output_dir / "manifest.json"
                manifest["saved_paths"] = {key: str(path) for key, path in {**saved_paths, "manifest": manifest_path}.items()}
                manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
                saved_paths["manifest"] = manifest_path

        result = FactorFrameResult(
            factor_frame=factor_frame,
            combined_frame=combined_frame,
            feeds=feeds,
            screen_frames=screen_frames,
            classifier_frames=classifier_frames,
            screen_mask=screen_mask,
            strategy_frames=strategy_frames,
            output_dir=output_dir,
            saved_paths=saved_paths,
            manifest=manifest,
        )
        self._result = result
        return result


__all__ = [
    "FactorFrameContext",
    "FactorFrameEngine",
    "FactorFrameFeed",
    "FactorFrameResult",
    "FactorFrameStrategySpec",
    "FactorFrameScreenSpec",
    "FactorFrameClassifierSpec",
]
