from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Sequence

import pandas as pd

from tiger_factors.utils.calculation.strategy.template import FactorStrategyTemplate
from tiger_factors.utils.calculation.types import CalculationResult
from tiger_factors.utils.calculation.types import CalculationStep
from tiger_factors.utils.calculation.types import Interval
from tiger_reference.calendar import TradingCalendarProtocol
from tiger_reference.calendar import load_exchange_calendar


EXCHANGE_CALENDAR_ALIASES: dict[str, str] = {
    "UTC": "XNYS",
    "US": "XNYS",
    "USA": "XNYS",
    "NYSE": "XNYS",
    "NASDAQ": "XNAS",
    "US_STOCK": "XNYS",
    "AU": "XASX",
    "ASX": "XASX",
    "UK": "XLON",
    "LSE": "XLON",
    "GB": "XLON",
    "GBR": "XLON",
    "DE": "XFRA",
    "FRA": "XFRA",
    "FR": "XPAR",
    "CAN": "XTSE",
    "CA": "XTSE",
    "JP": "XTKS",
    "JPN": "XTKS",
    "HK": "XHKG",
    "HKG": "XHKG",
    "CN": "XSHG",
    "CHINA": "XSHG",
}


@dataclass(frozen=True)
class _StepPoint:
    trading_day: date
    at: date | pd.Timestamp
    step_index: int
    step_kind: str
    session_open: pd.Timestamp | None = None
    session_close: pd.Timestamp | None = None
    is_session_open: bool = False
    is_session_close: bool = False


class FactorTimeEngine:
    """
    A very small time-driven engine.

    It only resolves trading days and advances them with a configurable interval.
    """

    def __init__(
        self,
        *,
        calendar: str | TradingCalendarProtocol = "XNYS",
        start: date | str | pd.Timestamp,
        end: date | str | pd.Timestamp | None = None,
        interval: Interval | None = None,
    ) -> None:
        self.calendar_name, self.calendar = self._resolve_calendar(calendar)
        self.start = self._coerce_day(start)
        self.end = self._coerce_day(end) if end is not None else pd.Timestamp.now(tz="UTC").date()
        self.interval = interval.copy() if interval is not None else Interval(day=1)

    @staticmethod
    def _resolve_calendar(calendar: str | TradingCalendarProtocol) -> tuple[str, Any]:
        if isinstance(calendar, str):
            normalized = str(calendar).strip().upper() or "XNYS"
            resolved_calendar_name = EXCHANGE_CALENDAR_ALIASES.get(normalized, normalized)
            return resolved_calendar_name, load_exchange_calendar(resolved_calendar_name)

        resolved_calendar_name = getattr(calendar, "name", calendar.__class__.__name__)
        return str(resolved_calendar_name), calendar

    @staticmethod
    def _coerce_day(value: date | str | pd.Timestamp) -> date:
        return pd.Timestamp(value).date()

    @staticmethod
    def _to_utc_timestamp(value: date | str | pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    def _is_trading_day(self, trading_day: date) -> bool:
        is_trading_day = getattr(self.calendar, "is_trading_day", None)
        if callable(is_trading_day):
            return bool(is_trading_day(trading_day))
        is_session = getattr(self.calendar, "is_session", None)
        if callable(is_session):
            return bool(is_session(pd.Timestamp(trading_day)))
        raise TypeError("calendar must implement is_trading_day(trading_day).")

    def _session_open_close(self, trading_day: date) -> tuple[pd.Timestamp, pd.Timestamp]:
        session_label = pd.Timestamp(trading_day)

        direct = getattr(self.calendar, "session_open_close", None)
        if callable(direct):
            session_open, session_close = direct(session_label)
            return self._to_utc_timestamp(session_open), self._to_utc_timestamp(session_close)

        candidate_calendars = [self.calendar, getattr(self.calendar, "calendar", None)]
        for candidate in candidate_calendars:
            if candidate is None:
                continue
            session_open = getattr(candidate, "session_open", None)
            session_close = getattr(candidate, "session_close", None)
            if callable(session_open) and callable(session_close):
                open_ts = session_open(session_label)
                close_ts = session_close(session_label)
                return self._to_utc_timestamp(open_ts), self._to_utc_timestamp(close_ts)

        raise TypeError(
            "calendar must implement session_open_close(session_label) or expose session_open/session_close."
        )

    def _sessions_in_range(self, start_day: date, end_day: date) -> tuple[date, ...] | None:
        sessions_in_range = getattr(self.calendar, "sessions_in_range", None)
        if not callable(sessions_in_range):
            return None

        sessions = sessions_in_range(start_day, end_day)
        normalized = pd.to_datetime(pd.Index(sessions), errors="coerce")
        normalized = pd.DatetimeIndex(normalized.dropna().unique()).sort_values()
        return tuple(pd.Timestamp(session).date() for session in normalized)

    def set_interval(self, interval: Interval | None = None, **components: int) -> Interval:
        if interval is not None:
            if components:
                raise ValueError("set_interval accepts either an Interval or keyword components, not both.")
            self.interval = interval.copy()
            return self.interval

        if not components:
            return self.interval

        day_keys = {"day", "days"} & set(components)
        time_keys = {"hour", "hours", "min", "minutes", "sec", "seconds"} & set(components)
        if day_keys and time_keys:
            raise ValueError("interval cannot mix day and intraday components.")

        if day_keys:
            day_value = components.get("day", components.get("days", 0))
            self.interval = Interval(day=int(day_value), hour=0, min=0, sec=0)
            return self.interval

        if not time_keys:
            return self.interval

        hour_value = components.get("hour", components.get("hours", 0))
        min_value = components.get("min", components.get("minutes", 0))
        sec_value = components.get("sec", components.get("seconds", 0))
        self.interval = Interval(day=0, hour=int(hour_value), min=int(min_value), sec=int(sec_value))
        return self.interval

    def resolve_trading_days(
        self,
        *,
        start: date | str | pd.Timestamp | None = None,
        end: date | str | pd.Timestamp | None = None,
        trading_days: Sequence[date | str | pd.Timestamp] | None = None,
    ) -> tuple[date, ...]:
        if trading_days is not None:
            normalized: list[date] = []
            seen: set[date] = set()
            for value in trading_days:
                day = self._coerce_day(value)
                if day in seen:
                    continue
                seen.add(day)
                normalized.append(day)
            if not normalized:
                return tuple()
            session_days = self._sessions_in_range(min(normalized), max(normalized))
            if session_days is not None:
                session_set = set(session_days)
                return tuple(day for day in normalized if day in session_set)
            return tuple(day for day in normalized if self._is_trading_day(day))

        start = self.start if start is None else start
        end = self.end if end is None else end

        start_day = self._coerce_day(start)
        end_day = self._coerce_day(end)
        if end_day < start_day:
            raise ValueError("end must be on or after start.")

        session_days = self._sessions_in_range(start_day, end_day)
        if session_days is not None:
            return session_days

        trading_days_list: list[date] = []
        current_day = start_day
        while current_day <= end_day:
            if self._is_trading_day(current_day):
                trading_days_list.append(current_day)
            current_day += timedelta(days=1)
        return tuple(trading_days_list)

    def _resolve_step_points(
        self,
        *,
        start: date | str | pd.Timestamp | None = None,
        end: date | str | pd.Timestamp | None = None,
        trading_days: Sequence[date | str | pd.Timestamp] | None = None,
    ) -> tuple[_StepPoint, ...]:
        days = self.resolve_trading_days(start=start, end=end, trading_days=trading_days)
        if not days:
            return tuple()

        if self.interval.day > 0 and (self.interval.hour or self.interval.min or self.interval.sec):
            raise ValueError("interval cannot mix day and intraday components.")

        points: list[_StepPoint] = []
        step_index = 0

        if self.interval.is_daily:
            stride = max(int(self.interval.day), 1)
            for trading_day in days[::stride]:
                points.append(
                    _StepPoint(
                        trading_day=trading_day,
                        at=trading_day,
                        step_index=step_index,
                        step_kind="daily",
                    )
                )
                step_index += 1
            return tuple(points)

        step_delta = self.interval.to_timedelta()
        if step_delta <= timedelta(0):
            raise ValueError("interval must be positive.")

        for trading_day in days:
            session_open, session_close = self._session_open_close(trading_day)
            timestamps: list[pd.Timestamp] = [session_open]
            current = session_open
            while True:
                next_ts = current + step_delta
                if next_ts >= session_close:
                    break
                timestamps.append(next_ts)
                current = next_ts
            if timestamps[-1] != session_close:
                timestamps.append(session_close)

            for idx, ts in enumerate(timestamps):
                points.append(
                    _StepPoint(
                        trading_day=trading_day,
                        at=self._to_utc_timestamp(ts),
                        step_index=step_index,
                        step_kind="intraday",
                        session_open=session_open,
                        session_close=session_close,
                        is_session_open=idx == 0,
                        is_session_close=idx == len(timestamps) - 1,
                    )
                )
                step_index += 1

        return tuple(points)

    def _build_step(
        self,
        strategy,
        point: _StepPoint,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> CalculationStep:
        if not isinstance(strategy, FactorStrategyTemplate):
            raise TypeError("strategy must inherit from FactorStrategyTemplate.")

        return CalculationStep(
            trading_day=point.trading_day,
            at=point.at,
            step_index=point.step_index,
            metadata=dict(metadata or {}),
            session_open=point.session_open,
            session_close=point.session_close,
            is_session_open=point.is_session_open,
            is_session_close=point.is_session_close,
            step_kind=point.step_kind,
        )

    @staticmethod
    def _call_hook(strategy, hook_name: str, step: CalculationStep) -> Any:
        hook = getattr(strategy, hook_name, None)
        if callable(hook):
            return hook(step)
        return None

    def _run_day_points(
        self,
        strategy,
        points: Sequence[_StepPoint],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> list[CalculationResult]:
        if not points:
            return []

        results: list[CalculationResult] = []
        first_step = self._build_step(strategy, points[0], metadata=metadata)
        self._call_hook(strategy, "on_pre_open", first_step)

        for point in points:
            step = self._build_step(strategy, point, metadata=metadata)
            output = strategy.on_step(step)
            results.append(CalculationResult(step=step, output=output))

        self._call_hook(strategy, "on_post_close", results[-1].step)
        return results

    def advance_day(
        self,
        strategy,
        trading_day: date | str | pd.Timestamp,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> CalculationResult:
        point = _StepPoint(
            trading_day=self._coerce_day(trading_day),
            at=self._coerce_day(trading_day),
            step_index=0,
            step_kind="daily",
        )
        return self._run_day_points(strategy, [point], metadata=metadata)[0]

    def advance_point(
        self,
        strategy,
        point: _StepPoint,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> CalculationResult:
        if not isinstance(strategy, FactorStrategyTemplate):
            raise TypeError("strategy must inherit from FactorStrategyTemplate.")

        step = self._build_step(strategy, point, metadata=metadata)
        output = strategy.on_step(step)
        return CalculationResult(step=step, output=output)

    def resolve_schedule_points(
        self,
        *,
        start: date | str | pd.Timestamp | None = None,
        end: date | str | pd.Timestamp | None = None,
        trading_days: Sequence[date | str | pd.Timestamp] | None = None,
    ) -> tuple[CalculationStep, ...]:
        return tuple(
            CalculationStep(
                trading_day=point.trading_day,
                at=point.at,
                step_index=point.step_index,
                metadata={},
                session_open=point.session_open,
                session_close=point.session_close,
                is_session_open=point.is_session_open,
                is_session_close=point.is_session_close,
                step_kind=point.step_kind,
            )
            for point in self._resolve_step_points(start=start, end=end, trading_days=trading_days)
        )

    def run(
        self,
        strategy,
        *,
        start: date | str | pd.Timestamp | None = None,
        end: date | str | pd.Timestamp | None = None,
        trading_days: Sequence[date | str | pd.Timestamp] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[CalculationResult]:
        if not isinstance(strategy, FactorStrategyTemplate):
            raise TypeError("strategy must inherit from FactorStrategyTemplate.")

        start = self.start if start is None else start
        end = self.end if end is None else end
        days = self.resolve_trading_days(start=start, end=end, trading_days=trading_days)
        strategy.on_start(self, days)
        results: list[CalculationResult] = []
        try:
            points = self._resolve_step_points(trading_days=days)
            current_day: date | None = None
            day_points: list[_StepPoint] = []
            for point in points:
                if current_day is None:
                    current_day = point.trading_day
                if point.trading_day != current_day:
                    results.extend(self._run_day_points(strategy, day_points, metadata=metadata))
                    current_day = point.trading_day
                    day_points = [point]
                    continue
                day_points.append(point)
            if day_points:
                results.extend(self._run_day_points(strategy, day_points, metadata=metadata))
        finally:
            strategy.on_finish(self, tuple(results))
        return results

__all__ = ["FactorTimeEngine"]
