from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from tiger_factors.utils.calculation import CallbackCalculationStrategy
from tiger_factors.utils.calculation import FactorStrategyTemplate
from tiger_factors.utils.calculation import FactorTimeEngine
from tiger_factors.utils.calculation import Interval


@dataclass
class FakeCalendar:
    trading_days: set[date]

    def is_trading_day(self, trading_day: date) -> bool:
        return trading_day in self.trading_days

    def sessions_in_range(self, start_day: date, end_day: date) -> pd.DatetimeIndex:
        sessions = [
            pd.Timestamp(day)
            for day in sorted(self.trading_days)
            if start_day <= day <= end_day
        ]
        return pd.DatetimeIndex(sessions)


@dataclass
class FakeIntradayCalendar(FakeCalendar):
    open_time: str = "09:30"
    close_time: str = "16:00"

    def session_open_close(self, session_label):
        day = pd.Timestamp(session_label).date()
        open_ts = pd.Timestamp(f"{day.isoformat()} {self.open_time}", tz="UTC")
        close_ts = pd.Timestamp(f"{day.isoformat()} {self.close_time}", tz="UTC")
        return open_ts, close_ts


class DemoStrategy(FactorStrategyTemplate):
    def __init__(self) -> None:
        self.started = False
        self.finished = False
        self.seen_days: list[date] = []

    def on_start(self, engine, trading_steps):
        self.started = True
        self.trading_steps = tuple(trading_steps)

    def on_day(self, step):
        self.seen_days.append(step.trading_day)
        return {
            "day": step.trading_day.isoformat(),
            "time": step.at.isoformat(),
            "step_index": step.step_index,
            "step_kind": step.step_kind,
            "open": step.is_session_open,
            "close": step.is_session_close,
        }

    def on_finish(self, engine, results):
        self.finished = True
        self.result_count = len(results)


def test_interval_supports_aliases_and_copy():
    interval = Interval()
    interval.days = 2
    interval.hours = 1
    interval.minutes = 15
    interval.seconds = 3

    copied = interval.copy()

    assert repr(interval) == "Interval(day=2, hour=1, min=15, sec=3)"
    assert str(interval) == "Interval(day=2, hour=1, min=15, sec=3)"
    assert interval.as_dict() == {"day": 2, "hour": 1, "min": 15, "sec": 3}
    assert interval.to_timedelta() == pd.Timedelta(days=2, hours=1, minutes=15, seconds=3)
    assert interval.is_daily is False
    assert interval.is_intraday is True
    assert copied is not interval
    assert copied.as_dict() == interval.as_dict()


def test_engine_runs_over_explicit_days():
    calendar = FakeCalendar(trading_days={date(2024, 1, 2), date(2024, 1, 4)})
    engine = FactorTimeEngine(calendar=calendar, start="2024-01-01")
    strategy = DemoStrategy()

    results = engine.run(
        strategy,
        trading_days=[date(2024, 1, 2), date(2024, 1, 4)],
    )

    assert strategy.started is True
    assert strategy.finished is True
    assert strategy.seen_days == [date(2024, 1, 2), date(2024, 1, 4)]
    assert strategy.trading_steps == (date(2024, 1, 2), date(2024, 1, 4))
    assert strategy.result_count == 2
    assert [item.output["time"] for item in results] == [
        "2024-01-02",
        "2024-01-04",
    ]
    assert [item.step.step_kind for item in results] == ["daily", "daily"]


def test_engine_can_use_init_start_and_end_for_run():
    calendar = FakeCalendar(trading_days={date(2024, 1, 2), date(2024, 1, 4), date(2024, 1, 5)})
    engine = FactorTimeEngine(calendar=calendar, start="2024-01-01", end="2024-01-05")
    strategy = DemoStrategy()

    results = engine.run(strategy)

    assert strategy.seen_days == [date(2024, 1, 2), date(2024, 1, 4), date(2024, 1, 5)]
    assert [item.step.trading_day for item in results] == [
        date(2024, 1, 2),
        date(2024, 1, 4),
        date(2024, 1, 5),
    ]


def test_engine_triggers_daily_hooks_in_order():
    calendar = FakeCalendar(trading_days={date(2024, 1, 2)})
    engine = FactorTimeEngine(calendar=calendar, start="2024-01-01")
    events: list[tuple[str, str, bool, bool]] = []

    strategy = CallbackCalculationStrategy(
        on_day=lambda step: events.append(("day", step.step_kind, step.is_session_open, step.is_session_close))
        or step.step_index,
        on_pre_open=lambda step: events.append(("pre_open", step.step_kind, step.is_session_open, step.is_session_close)),
        on_post_close=lambda step: events.append(("post_close", step.step_kind, step.is_session_open, step.is_session_close)),
    )

    engine.run(strategy, trading_days=[date(2024, 1, 2)])

    assert events == [
        ("pre_open", "daily", False, False),
        ("day", "daily", False, False),
        ("post_close", "daily", False, False),
    ]


def test_engine_resolves_days_from_calendar():
    calendar = FakeCalendar(trading_days={date(2024, 1, 2), date(2024, 1, 4), date(2024, 1, 5)})
    engine = FactorTimeEngine(calendar=calendar, start="2024-01-01", end="2024-01-05")

    days = engine.resolve_trading_days()

    assert days == (date(2024, 1, 2), date(2024, 1, 4), date(2024, 1, 5))


def test_engine_advance_day_returns_utc_timestamp():
    calendar = FakeCalendar(trading_days={date(2024, 1, 2)})
    engine = FactorTimeEngine(calendar=calendar, start="2024-01-01")
    strategy = DemoStrategy()

    result = engine.advance_day(strategy, date(2024, 1, 2))

    assert result.step.trading_day == date(2024, 1, 2)
    assert result.step.at == date(2024, 1, 2)
    assert result.step.timestamp == result.step.at
    assert result.step.step_kind == "daily"
    assert result.output["step_index"] == 0


def test_engine_can_resolve_calendar_from_alias():
    engine = FactorTimeEngine(calendar="NASDAQ", start="2024-01-01")

    assert engine.calendar_name == "XNAS"


def test_engine_set_interval_switches_modes_cleanly():
    engine = FactorTimeEngine(calendar="XNYS", start="2024-01-01")

    engine.set_interval(hour=1)
    assert engine.interval.day == 0
    assert engine.interval.hour == 1
    assert engine.interval.min == 0
    assert engine.interval.sec == 0

    engine.set_interval(day=2)
    assert engine.interval.day == 2
    assert engine.interval.hour == 0
    assert engine.interval.min == 0
    assert engine.interval.sec == 0


def test_callback_strategy_runs_through_engine():
    calendar = FakeCalendar(trading_days={date(2024, 1, 2)})
    engine = FactorTimeEngine(calendar=calendar, start="2024-01-01")

    seen: list[date] = []
    strategy = CallbackCalculationStrategy(
        on_day=lambda step: seen.append(step.trading_day) or step.step_index,
    )

    results = engine.run(strategy, trading_days=[date(2024, 1, 2)])

    assert seen == [date(2024, 1, 2)]
    assert results[0].output == 0


def test_run_filters_non_trading_days_from_explicit_input():
    calendar = FakeCalendar(trading_days={date(2024, 1, 2)})
    engine = FactorTimeEngine(calendar=calendar, start="2024-01-01")
    strategy = DemoStrategy()

    results = engine.run(
        strategy,
        trading_days=[date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 2)],
    )

    assert strategy.seen_days == [date(2024, 1, 2)]
    assert [item.step.trading_day for item in results] == [date(2024, 1, 2)]


def test_engine_can_resolve_intraday_points_with_open_and_close():
    calendar = FakeIntradayCalendar(trading_days={date(2024, 1, 2)})
    engine = FactorTimeEngine(calendar=calendar, start="2024-01-01")
    engine.set_interval(hour=1)

    steps = engine.resolve_schedule_points(trading_days=[date(2024, 1, 2)])

    assert len(steps) == 8
    assert steps[0].at == pd.Timestamp("2024-01-02 09:30:00+00:00")
    assert steps[-1].at == pd.Timestamp("2024-01-02 16:00:00+00:00")
    assert steps[0].is_session_open is True
    assert steps[-1].is_session_close is True
    assert all(step.step_kind == "intraday" for step in steps)


def test_engine_can_run_intraday_strategy():
    calendar = FakeIntradayCalendar(trading_days={date(2024, 1, 2)})
    engine = FactorTimeEngine(calendar=calendar, start="2024-01-01")
    engine.set_interval(min=30)
    strategy = DemoStrategy()

    results = engine.run(strategy, trading_days=[date(2024, 1, 2)])

    assert len(results) == 14
    assert results[0].step.at == pd.Timestamp("2024-01-02 09:30:00+00:00")
    assert results[-1].step.at == pd.Timestamp("2024-01-02 16:00:00+00:00")
    assert results[0].step.is_session_open is True
    assert results[-1].step.is_session_close is True
    assert results[0].step.session_open == pd.Timestamp("2024-01-02 09:30:00+00:00")
    assert results[0].step.session_close == pd.Timestamp("2024-01-02 16:00:00+00:00")


def test_engine_defaults_end_to_utc_today_when_omitted():
    calendar = FakeCalendar(trading_days={date(2024, 1, 2)})
    engine = FactorTimeEngine(calendar=calendar, start="2024-01-01")

    assert engine.end == pd.Timestamp.now(tz="UTC").date()
