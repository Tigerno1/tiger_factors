from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from tiger_factors.utils.calculation import FactorMainEngine
from tiger_factors.utils.calculation import FactorStrategyTemplate
from tiger_factors.utils.calculation import FactorTimeEngine
from tiger_factors.utils.calculation import Interval


@dataclass
class FakeCalendar:
    trading_days: set[date]

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
        self.events: list[tuple[str, object]] = []

    def on_pre_open(self, step):
        self.events.append(("pre_open", step.trading_day))

    def on_day(self, step):
        self.events.append(("day", step.at))
        return {"day": step.trading_day.isoformat(), "at": step.at}

    def on_post_close(self, step):
        self.events.append(("post_close", step.trading_day))

    def on_finish(self, engine, results):
        self.events.append(("finish", len(results)))


def test_calculation_main_engine_runs_strategy_through_time_engine():
    calendar = FakeCalendar(trading_days={date(2024, 1, 2)})
    time_engine = FactorTimeEngine(calendar=calendar, start="2024-01-01", interval=Interval(day=1))
    main_engine = FactorMainEngine(time_engine=time_engine)
    strategy = DemoStrategy()

    results = main_engine.run(strategy, trading_days=[date(2024, 1, 2)])

    assert len(results) == 1
    assert strategy.events == [
        ("pre_open", date(2024, 1, 2)),
        ("day", date(2024, 1, 2)),
        ("post_close", date(2024, 1, 2)),
        ("finish", 1),
    ]


def test_calculation_main_engine_supports_intraday_steps():
    calendar = FakeIntradayCalendar(trading_days={date(2024, 1, 2)})
    time_engine = FactorTimeEngine(calendar=calendar, start="2024-01-01")
    time_engine.set_interval(min=30)
    main_engine = FactorMainEngine(time_engine=time_engine)
    strategy = DemoStrategy()

    results = main_engine.run(strategy, trading_days=[date(2024, 1, 2)])

    assert len(results) == 14
    assert results[0].step.at == pd.Timestamp("2024-01-02 09:30:00+00:00")
    assert results[-1].step.at == pd.Timestamp("2024-01-02 16:00:00+00:00")
