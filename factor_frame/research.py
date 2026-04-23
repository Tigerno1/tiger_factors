from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from tiger_factors.factor_frame.engine import FactorFrameContext
from tiger_factors.factor_frame.engine import FactorFrameEngine
from tiger_factors.factor_frame.engine import FactorFrameResult
from tiger_factors.factor_frame.definition import FactorDefinition
from tiger_factors.factor_frame.definition_registry import FactorDefinitionRegistry
from tiger_factors.factor_frame.factors import FactorFrameFactor
from tiger_factors.factor_frame.factors import FactorFrameTemplate


@dataclass
class FactorResearchEngine:
    """Thin research façade on top of :class:`FactorFrameEngine`.

    The façade keeps the data-input and template construction story compact:

    - bind the research window once with ``start`` / ``end``
    - set the primary data frequency once with ``freq`` such as ``"1d"``,
      ``"1h"``, ``"1min"``, ``"15min"``, ``"20min"``, ``"30min"``, or ``"2h"``
    - choose the intraday timestamp convention with ``label_side``:
      ``"right"`` for end-labeled bars, ``"left"`` for start-labeled bars,
      or ``"auto"`` when the engine should try to infer the source
    - feed materialized frames
    - add reusable factor templates or plain factor callables
    - run once and read ``result``
    """

    start: Any | None = None
    end: Any | None = None
    freq: str | None = None
    bday_lag: bool = True
    as_ex: bool = False
    calendar: str | None = None
    label_side: str = "right"
    use_point_in_time: bool = True
    availability_column: str | None = None
    align_mode: str = "outer"
    output_root_dir: str | None = None
    save: bool = False
    definition_registry: FactorDefinitionRegistry | None = None

    def __post_init__(self) -> None:
        self._engine = FactorFrameEngine(
            start=self.start,
            end=self.end,
            freq=self.freq,
            bday_lag=self.bday_lag,
            as_ex=self.as_ex,
            calendar=self.calendar,
            label_side=self.label_side,
            use_point_in_time=self.use_point_in_time,
            availability_column=self.availability_column,
            align_mode=self.align_mode,
            output_root_dir=self.output_root_dir,
            save=self.save,
            definition_registry=self.definition_registry,
        )

    @property
    def engine(self) -> FactorFrameEngine:
        return self._engine

    @property
    def result(self) -> FactorFrameResult:
        return self._engine.result

    def feed(self, *args: Any, **kwargs: Any) -> "FactorResearchEngine":
        self._engine.feed(*args, **kwargs)
        return self

    def feed_price(self, *args: Any, **kwargs: Any) -> "FactorResearchEngine":
        self._engine.feed_price(*args, **kwargs)
        return self

    def feed_financial(self, *args: Any, **kwargs: Any) -> "FactorResearchEngine":
        self._engine.feed_financial(*args, **kwargs)
        return self

    def feed_valuation(self, *args: Any, **kwargs: Any) -> "FactorResearchEngine":
        self._engine.feed_valuation(*args, **kwargs)
        return self

    def feed_macro(self, *args: Any, **kwargs: Any) -> "FactorResearchEngine":
        self._engine.feed_macro(*args, **kwargs)
        return self

    def feed_news(self, *args: Any, **kwargs: Any) -> "FactorResearchEngine":
        self._engine.feed_news(*args, **kwargs)
        return self

    def add_strategy(
        self,
        name: str,
        fn: Callable[[FactorFrameContext], Any],
        *,
        save: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> "FactorResearchEngine":
        self._engine.add_strategy(name, fn, save=save, metadata=metadata)
        return self

    def add_screen(
        self,
        name: str,
        fn: Callable[[FactorFrameContext], Any],
        *,
        save: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> "FactorResearchEngine":
        self._engine.add_screen(name, fn, save=save, metadata=metadata)
        return self

    def add_classifier(
        self,
        name: str,
        fn: Callable[[FactorFrameContext], Any],
        *,
        save: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> "FactorResearchEngine":
        self._engine.add_classifier(name, fn, save=save, metadata=metadata)
        return self

    def add_factor(
        self,
        name: str | FactorFrameFactor | FactorFrameTemplate | FactorDefinition,
        fn: Callable[[FactorFrameContext], Any] | None = None,
        *,
        save: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> "FactorResearchEngine":
        self._engine.add_factor(name, fn, save=save, metadata=metadata)
        return self

    def add_definition(
        self,
        definition: FactorDefinition,
        *,
        save: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> "FactorResearchEngine":
        self._engine.add_definition(definition, save=save, metadata=metadata)
        return self

    def add_template(
        self,
        template: FactorFrameTemplate,
        *,
        factor_name: str | None = None,
        save: bool = False,
        metadata: dict[str, Any] | None = None,
        **template_kwargs: Any,
    ) -> "FactorResearchEngine":
        factor = template.build(factor_name=factor_name, **template_kwargs)
        self._engine.add_factor(factor, save=save, metadata=metadata)
        return self

    def add_factor_template(
        self,
        template: FactorFrameTemplate,
        *,
        factor_name: str | None = None,
        save: bool = False,
        metadata: dict[str, Any] | None = None,
        **template_kwargs: Any,
    ) -> "FactorResearchEngine":
        return self.add_template(
            template,
            factor_name=factor_name,
            save=save,
            metadata=metadata,
            **template_kwargs,
        )

    def add_factors(
        self,
        *factors: FactorFrameFactor | FactorFrameTemplate | FactorDefinition | str,
    ) -> "FactorResearchEngine":
        self._engine.add_factors(*factors)
        return self

    def build_context(self) -> FactorFrameContext:
        return self._engine.build_context()

    def run(self, *, save: bool | None = None) -> FactorFrameResult:
        return self._engine.run(save=save)
