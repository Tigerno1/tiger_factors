from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Sequence

from tiger_factors.utils.calculation.types import CalculationResult
from tiger_factors.utils.calculation.types import CalculationStep


class FactorStrategyTemplate(ABC):
    """
    Base template for step-driven calculation strategies.

    The engine only advances time and hands each step to the strategy.
    """

    name: str = "calculation_strategy"

    def on_start(self, engine, trading_steps: Sequence[Any]) -> None:
        return None

    def on_finish(self, engine, results: Sequence[CalculationResult]) -> None:
        return None

    def on_pre_open(self, step: CalculationStep) -> Any:
        return None

    def on_post_close(self, step: CalculationStep) -> Any:
        return None

    def on_step(self, step: CalculationStep) -> Any:
        return self.on_day(step)

    @abstractmethod
    def on_day(self, step: CalculationStep) -> Any:
        raise NotImplementedError


class CallbackCalculationStrategy(FactorStrategyTemplate):
    """
    A concrete strategy wrapper for quick experiments.

    Pass callbacks instead of subclassing when you want to run logic directly.
    """

    def __init__(
        self,
        *,
        on_day: Callable[[CalculationStep], Any],
        name: str = "callback_strategy",
        on_start: Callable[[Any, Sequence[Any]], Any] | None = None,
        on_finish: Callable[[Any, Sequence[CalculationResult]], Any] | None = None,
        on_pre_open: Callable[[CalculationStep], Any] | None = None,
        on_post_close: Callable[[CalculationStep], Any] | None = None,
    ) -> None:
        self.name = name
        self._on_day = on_day
        self._on_start = on_start
        self._on_finish = on_finish
        self._on_pre_open = on_pre_open
        self._on_post_close = on_post_close

    def on_start(self, engine, trading_steps: Sequence[Any]) -> None:
        if self._on_start is not None:
            self._on_start(engine, trading_steps)

    def on_finish(self, engine, results: Sequence[CalculationResult]) -> None:
        if self._on_finish is not None:
            self._on_finish(engine, results)

    def on_pre_open(self, step: CalculationStep) -> Any:
        if self._on_pre_open is not None:
            return self._on_pre_open(step)
        return None

    def on_post_close(self, step: CalculationStep) -> Any:
        if self._on_post_close is not None:
            return self._on_post_close(step)
        return None

    def on_day(self, step: CalculationStep) -> Any:
        return self._on_day(step)


__all__ = ["FactorStrategyTemplate", "CallbackCalculationStrategy"]
