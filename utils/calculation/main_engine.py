from __future__ import annotations

from itertools import groupby

from tiger_factors.utils.calculation.time_engine import FactorTimeEngine
from tiger_factors.utils.calculation.strategy.template import FactorStrategyTemplate
from tiger_factors.utils.calculation.types import CalculationResult
from tiger_factors.utils.calculation.types import Interval


class FactorMainEngine:
    """
    Lightweight coordinator for calculation strategies and the time engine.

    The time engine builds trading steps and the strategy consumes them.
    """

    def __init__(
        self,
        time_engine: FactorTimeEngine | None = None,
        calendar: str = "XNYS",
        start=None,
        end=None,
        interval: Interval | None = Interval(day=1),
    ) -> None:
        if time_engine is not None:
            self.time_engine = time_engine
        else:
            if start is None:
                raise ValueError("start is required when time_engine is not provided.")
            self.time_engine = FactorTimeEngine(
                calendar=calendar,
                start=start,
                end=end,
                interval=interval,
            )

    def run(
        self,
        strategy: FactorStrategyTemplate,
        *,
        start=None,
        end=None,
        trading_days=None,
    ) -> list[CalculationResult]:
        if not isinstance(strategy, FactorStrategyTemplate):
            raise TypeError("strategy must inherit from FactorStrategyTemplate.")

        days = self.time_engine.resolve_trading_days(start=start, end=end, trading_days=trading_days)
        steps = self.time_engine.resolve_schedule_points(start=start, end=end, trading_days=trading_days)
        strategy.on_start(self, days)

        results: list[CalculationResult] = []
        try:
            for _, day_steps_iter in groupby(steps, key=lambda item: item.trading_day):
                day_steps = list(day_steps_iter)
                if not day_steps:
                    continue

                strategy.on_pre_open(day_steps[0])

                for step in day_steps:
                    output = strategy.on_step(step)
                    results.append(CalculationResult(step=step, output=output))

                strategy.on_post_close(day_steps[-1])
        finally:
            strategy.on_finish(self, tuple(results))

        return results

__all__ = ["FactorMainEngine"]
