from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Iterable

import pandas as pd

from tiger_factors.utils.calculation.time_engine import FactorTimeEngine
from tiger_factors.utils.calculation.types import Interval
from tiger_reference.calendar import apply_session_lag
from tiger_reference.calendar import build_trading_sessions
from tiger_reference.calendar import session_index_on_or_after


def _normalize_freq(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized or None


def _normalize_label_side(value: str | None) -> str:
    normalized = "right" if value is None else str(value).strip().lower()
    if normalized not in {"left", "right", "auto"}:
        raise ValueError("label_side must be one of: left, right, auto.")
    return normalized


_SUPPORTED_FREQS = {
    "1d",
    "1min",
    "15min",
    "20min",
    "30min",
    "1h",
    "2h",
}


def _resolve_time_kind_from_freq(freq: str | None) -> str | None:
    normalized = _normalize_freq(freq)
    if normalized is None:
        return None

    if normalized in {"1d"}:
        return "daily"

    if normalized in {"1min", "15min", "20min", "30min", "1h", "2h"}:
        return "intraday"
    return None


def _freq_to_timedelta(freq: str | None) -> pd.Timedelta | None:
    normalized = _normalize_freq(freq)
    if normalized is None:
        return None
    if normalized == "1d":
        return pd.Timedelta(days=1)
    if normalized.endswith("min"):
        try:
            minutes = int(normalized[:-3])
        except ValueError:
            return None
        return pd.Timedelta(minutes=minutes)
    if normalized.endswith("h"):
        try:
            hours = int(normalized[:-1])
        except ValueError:
            return None
        return pd.Timedelta(hours=hours)
    return None


def _freq_to_interval(freq: str | None) -> Interval | None:
    normalized = _normalize_freq(freq)
    if normalized is None:
        return None
    if normalized == "1d":
        return Interval(day=1)
    if normalized.endswith("min"):
        try:
            minutes = int(normalized[:-3])
        except ValueError:
            return None
        return Interval(day=0, hour=0, min=minutes, sec=0)
    if normalized.endswith("h"):
        try:
            hours = int(normalized[:-1])
        except ValueError:
            return None
        return Interval(day=0, hour=hours, min=0, sec=0)
    return None


def _naive_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        return ts.tz_convert(None)
    return ts


def _normalize_datetime_series(values: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(values, errors="coerce")
    if isinstance(timestamps, pd.Series):
        if getattr(timestamps.dt, "tz", None) is not None:
            return timestamps.dt.tz_convert(None)
        return timestamps
    if isinstance(timestamps, pd.DatetimeIndex):
        if timestamps.tz is not None:
            return pd.Series(timestamps.tz_convert(None), index=getattr(values, "index", None))
        return pd.Series(timestamps, index=getattr(values, "index", None))
    return pd.Series(pd.DatetimeIndex(timestamps), index=getattr(values, "index", None))


def _normalize_window_bound(value: Any | None) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts


def _infer_feed_time_kind(feed: Any) -> str:
    if not getattr(feed, "has_date_axis", False):
        return "static"

    date_column = getattr(feed, "date_column", "date_")
    frame = getattr(feed, "frame", pd.DataFrame())
    dates = pd.Series(pd.to_datetime(frame[date_column], errors="coerce")).dropna()
    if dates.empty:
        return "daily"

    normalized = dates.dt.normalize()
    return "intraday" if (dates != normalized).any() else "daily"


def _resolve_time_kind(feeds: Iterable[Any]) -> str:
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


def _infer_label_side_from_calendar(
    frame: pd.DataFrame,
    *,
    date_column: str,
    freq: str | None,
    calendar_name: str | None,
) -> str:
    if calendar_name is None:
        return "right"

    time_delta = _freq_to_timedelta(freq)
    interval = _freq_to_interval(freq)
    if time_delta is None or interval is None:
        return "right"
    if frame.empty or date_column not in frame.columns:
        return "right"

    source_times = pd.to_datetime(frame[date_column], errors="coerce").dropna()
    if source_times.empty:
        return "right"

    source_times = pd.DatetimeIndex(sorted({_naive_timestamp(timestamp) for timestamp in source_times}))
    if len(source_times) < 2:
        return "right"

    days = pd.DatetimeIndex(source_times.normalize()).dropna().unique()
    if len(days) == 0:
        return "right"

    try:
        engine = FactorTimeEngine(calendar=calendar_name, start=days[0], end=days[-1], interval=interval)
    except Exception:
        return "right"

    left_score = 0
    right_score = 0
    inspected_days = 0
    for day in days[:3]:
        day_mask = source_times.normalize() == day
        day_times = source_times[day_mask]
        if len(day_times) < 2:
            continue
        try:
            schedule = engine.resolve_schedule_points(start=day, end=day, trading_days=[day])
        except Exception:
            continue
        session_times = pd.DatetimeIndex(
            [
                _naive_timestamp(point.at)
                for point in schedule
                if _naive_timestamp(point.at).normalize() == day
            ]
        )
        if len(session_times) < len(day_times):
            continue
        left_candidate = session_times[: len(day_times)]
        right_candidate = session_times[1 : len(day_times) + 1]
        if len(left_candidate) == len(day_times):
            left_score += int((day_times == left_candidate).sum())
        if len(right_candidate) == len(day_times):
            right_score += int((day_times == right_candidate).sum())
        inspected_days += 1

    if inspected_days == 0:
        return "right"
    if left_score > right_score:
        return "left"
    if right_score > left_score:
        return "right"
    return "right"


def _shift_intraday_label(
    frame: pd.DataFrame,
    *,
    date_column: str,
    freq: str | None,
    label_side: str,
    calendar_name: str | None,
) -> pd.DataFrame:
    if frame.empty or date_column not in frame.columns:
        return frame

    side = _normalize_label_side(label_side)
    if side == "auto":
        side = _infer_label_side_from_calendar(
            frame,
            date_column=date_column,
            freq=freq,
            calendar_name=calendar_name,
        )
    if side != "left":
        return frame

    delta = _freq_to_timedelta(freq)
    if delta is None or delta <= pd.Timedelta(0):
        return frame

    shifted = frame.copy()
    shifted[date_column] = _normalize_datetime_series(shifted[date_column]) + delta
    return shifted


def _resolve_feed_date_column(frame: pd.DataFrame, *, requested_date_column: str, as_ex: bool) -> str:
    if not as_ex:
        return requested_date_column
    if requested_date_column != "date_":
        return requested_date_column
    if "ex_date" in frame.columns:
        return "ex_date"
    return requested_date_column


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
    shifted[date_column] = pd.to_datetime(shifted[date_column], errors="coerce").dt.tz_localize(None) + offset
    return shifted


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


@dataclass
class FactorFrameTimeLayer:
    freq: str | None = None
    bday_lag: bool = True
    as_ex: bool = False
    calendar: str | None = None
    label_side: str = "right"

    @property
    def normalized_freq(self) -> str | None:
        normalized = _normalize_freq(self.freq)
        if normalized is None:
            return None
        if normalized not in _SUPPORTED_FREQS:
            raise ValueError(
                f"Unsupported freq={self.freq!r}; expected one of {sorted(_SUPPORTED_FREQS)!r}."
            )
        return normalized

    @property
    def time_kind(self) -> str | None:
        return _resolve_time_kind_from_freq(self.freq)

    @property
    def normalized_label_side(self) -> str:
        return _normalize_label_side(self.label_side)

    def resolve_time_kind(self, feeds: Iterable[Any]) -> str:
        explicit = self.time_kind
        inferred = _resolve_time_kind(feeds)
        if explicit is None:
            return inferred
        if inferred not in {"static", explicit}:
            raise ValueError(
                f"freq={self.freq!r} resolves to time_kind={explicit!r}, but the loaded feeds resolve to {inferred!r}."
            )
        return explicit

    def normalize_window_bound(self, value: Any | None) -> pd.Timestamp | None:
        return _normalize_window_bound(value)

    def expand_window_end(self, value: pd.Timestamp | None, *, time_kind: str) -> pd.Timestamp | None:
        return _expand_window_end(value, time_kind=time_kind)

    def resolve_feed_date_column(self, frame: pd.DataFrame, *, requested_date_column: str) -> str:
        return _resolve_feed_date_column(frame, requested_date_column=requested_date_column, as_ex=self.as_ex)

    def filter_frame_by_window(
        self,
        frame: pd.DataFrame,
        *,
        date_column: str,
        start: pd.Timestamp | None,
        end: pd.Timestamp | None,
    ) -> pd.DataFrame:
        return _filter_frame_by_window(frame, date_column=date_column, start=start, end=end)

    def shift_feed_dates(
        self,
        frame: pd.DataFrame,
        *,
        date_column: str,
        lag_sessions: int,
    ) -> pd.DataFrame:
        if self.calendar:
            return _shift_feed_dates_with_calendar(
                frame,
                date_column=date_column,
                lag_sessions=lag_sessions,
                calendar_name=self.calendar,
            )
        return _shift_feed_dates(
            frame,
            date_column=date_column,
            lag_sessions=lag_sessions,
            bday_lag=self.bday_lag,
        )

    def normalize_feed_labels(self, frame: pd.DataFrame, *, date_column: str) -> pd.DataFrame:
        if self.time_kind != "intraday":
            return frame
        return _shift_intraday_label(
            frame,
            date_column=date_column,
            freq=self.freq,
            label_side=self.label_side,
            calendar_name=self.calendar,
        )


__all__ = [
    "FactorFrameTimeLayer",
]
