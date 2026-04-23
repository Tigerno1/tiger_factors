from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import pandas as pd


@dataclass
class Interval:
    day: int = 0
    hour: int = 0
    min: int = 0
    sec: int = 0


    @property
    def days(self) -> int:
        return self.day

    @days.setter
    def days(self, value: int) -> None:
        self.day = int(value)

    @property
    def hours(self) -> int:
        return self.hour

    @hours.setter
    def hours(self, value: int) -> None:
        self.hour = int(value)

    @property
    def minutes(self) -> int:
        return self.min

    @minutes.setter
    def minutes(self, value: int) -> None:
        self.min = int(value)

    @property
    def seconds(self) -> int:
        return self.sec

    @seconds.setter
    def seconds(self, value: int) -> None:
        self.sec = int(value)

    def copy(self) -> Interval:
        return Interval(day=int(self.day), hour=int(self.hour), min=int(self.min), sec=int(self.sec))

    def as_dict(self) -> dict[str, int]:
        return {"day": int(self.day), "hour": int(self.hour), "min": int(self.min), "sec": int(self.sec)}

    def to_timedelta(self) -> timedelta:
        return timedelta(days=int(self.day), hours=int(self.hour), minutes=int(self.min), seconds=int(self.sec))
    
    @property
    def is_daily(self) -> bool:
        return int(self.day) > 0 and int(self.hour) == 0 and int(self.min) == 0 and int(self.sec) == 0

    @property
    def is_intraday(self) -> bool:
        return not self.is_daily
    
    def get_pandas_freq(self) -> str:
        if self.is_daily:
            return "B"
        else:
            total_seconds = self.to_timedelta().total_seconds()
            if total_seconds % 3600 == 0:
                hours = int(total_seconds // 3600)
                return f"{hours}H"
            elif total_seconds % 60 == 0:
                minutes = int(total_seconds // 60)
                return f"{minutes}T"
            else:
                return f"{int(total_seconds)}S"
    

    def __str__(self) -> str:
        return f"Interval(day={self.day}, hour={self.hour}, min={self.min}, sec={self.sec})"
    
   

@dataclass
class CalculationStep:
    trading_day: date
    at: date | pd.Timestamp
    step_index: int
    metadata: dict[str, Any] = field(default_factory=dict)
    session_open: pd.Timestamp | None = None
    session_close: pd.Timestamp | None = None
    is_session_open: bool = False
    is_session_close: bool = False
    step_kind: str = "daily"

    @property
    def timestamp(self) -> date | pd.Timestamp:
        return self.at


@dataclass(frozen=True)
class CalculationResult:
    step: CalculationStep
    output: Any


__all__ = [
    "CalculationResult",
    "CalculationStep",
    "Interval",
]
