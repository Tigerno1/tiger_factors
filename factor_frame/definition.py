from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from tiger_factors.factor_frame.engine import FactorFrameContext
from tiger_factors.factor_frame.factors import FactorFrameFactor
from tiger_factors.factor_frame.factors import factor as make_factor
from tiger_factors.factor_frame.factors import group_neutralize

FeedColumnSpec = str | tuple[str, str]


def _parse_feed_column_spec(spec: FeedColumnSpec) -> tuple[str, str]:
    if isinstance(spec, tuple):
        if len(spec) != 2:
            raise ValueError(f"Expected a (feed, column) tuple, got {spec!r}")
        return str(spec[0]), str(spec[1])
    if not isinstance(spec, str):
        raise TypeError(f"Unsupported feed-column spec type: {type(spec)!r}")
    for separator in ("::", ":", ".", "__"):
        if separator in spec:
            feed_name, column_name = spec.split(separator, 1)
            if feed_name and column_name:
                return feed_name, column_name
    raise ValueError(
        "Feed-column specs must be '(feed, column)' tuples or strings like 'feed:column'."
    )


def _evaluate_component(component: Any, ctx: FactorFrameContext) -> pd.DataFrame | pd.Series:
    if isinstance(component, FactorDefinition):
        return component.run(ctx)
    if isinstance(component, FactorFrameFactor):
        return component(ctx)
    if callable(component):
        return component(ctx)
    raise TypeError(f"Unsupported component type: {type(component)!r}")


def cross_sectional_residual(
    target: pd.DataFrame,
    exposures: Sequence[pd.DataFrame] | None = None,
    *,
    add_intercept: bool = True,
) -> pd.DataFrame:
    numeric_target = target.apply(pd.to_numeric, errors="coerce")
    if not exposures:
        return numeric_target.sub(numeric_target.mean(axis=1), axis=0)

    prepared_exposures = [
        exposure.reindex(index=numeric_target.index, columns=numeric_target.columns).apply(pd.to_numeric, errors="coerce")
        for exposure in exposures
    ]
    residuals = pd.DataFrame(index=numeric_target.index, columns=numeric_target.columns, dtype="float64")

    for date in numeric_target.index:
        y_row = numeric_target.loc[date]
        mask = y_row.notna()
        if int(mask.sum()) < 5:
            continue

        matrices: list[np.ndarray] = []
        for exposure in prepared_exposures:
            exposure_row = exposure.loc[date, mask]
            matrices.append(exposure_row.to_numpy(dtype=float).reshape(-1, 1))

        if not matrices:
            continue

        design = np.concatenate(matrices, axis=1)
        y_values = y_row[mask].to_numpy(dtype=float)
        finite_mask = np.isfinite(design).all(axis=1) & np.isfinite(y_values)
        if int(finite_mask.sum()) < max(2, design.shape[1] + int(add_intercept)):
            continue

        design = design[finite_mask]
        y_values = y_values[finite_mask]
        asset_index = y_row[mask].index[finite_mask]

        if add_intercept:
            design = np.c_[np.ones((design.shape[0], 1), dtype=float), design]

        beta, *_ = np.linalg.lstsq(design, y_values, rcond=None)
        fitted = design @ beta
        residuals.loc[date, asset_index] = y_values - fitted

    return residuals


def event_window_abnormal_return(
    events: pd.DataFrame,
    close: pd.DataFrame,
    *,
    window: int,
) -> pd.DataFrame:
    event_mask = events.apply(pd.to_numeric, errors="coerce").fillna(0.0) > 0.0
    returns = pd.to_numeric(close, errors="coerce").pct_change(int(window), fill_method=None)
    return returns.where(event_mask)


def weighted_sum(
    frames: dict[str, pd.DataFrame | pd.Series],
    weights: dict[str, float],
    *,
    normalize: bool = True,
) -> pd.DataFrame | pd.Series:
    if not frames:
        return pd.DataFrame()

    total_weight = sum(abs(float(weight)) for weight in weights.values()) if normalize else 1.0
    result: pd.DataFrame | pd.Series | None = None
    for name, frame in frames.items():
        weight = float(weights.get(name, 0.0))
        if normalize and total_weight not in {0.0, 0}:
            weight = weight / total_weight
        if isinstance(frame, pd.Series):
            contribution = pd.to_numeric(frame, errors="coerce") * weight
        else:
            contribution = frame.apply(pd.to_numeric, errors="coerce") * weight
        result = contribution if result is None else result.add(contribution, fill_value=0.0)
    assert result is not None
    return result


@dataclass
class FactorDefinition:
    """Structured factor recipe for complex research workflows.

    The definition layer sits above plain ``template`` and ``strategy`` helpers:

    - ``template`` is a lightweight parameterized builder
    - ``strategy`` is a free-form callable
    - ``FactorDefinition`` is a structured recipe with optional preparation,
      explicit dependencies, and deterministic post-processing
    """

    name: str
    inputs: list[str] = field(default_factory=list)
    classifiers: list[str] = field(default_factory=list)
    screens: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    save: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def prepare(self, ctx: FactorFrameContext) -> dict[str, Any]:
        return {}

    def compute(self, ctx: FactorFrameContext, state: dict[str, Any]) -> pd.DataFrame | pd.Series:
        raise NotImplementedError

    def postprocess(self, frame: pd.DataFrame | pd.Series, ctx: FactorFrameContext) -> pd.DataFrame | pd.Series:
        return frame

    def run(self, ctx: FactorFrameContext) -> pd.DataFrame | pd.Series:
        state = self.prepare(ctx)
        frame = self.compute(ctx, state)
        return self.postprocess(frame, ctx)

    def to_factor(
        self,
        *,
        save: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FactorFrameFactor:
        merged_metadata = {**self.metadata, **dict(metadata or {})}
        effective_save = self.save if save is None else bool(save)
        return make_factor(self.name, self.run, save=effective_save, metadata=merged_metadata)

    def __call__(self, ctx: FactorFrameContext) -> pd.DataFrame | pd.Series:
        return self.run(ctx)


@dataclass
class IndustryNeutralMomentumDefinition(FactorDefinition):
    window: int = 20
    price_feed: str = "price"
    price_column: str = "close"
    classifier_name: str = "sector"
    classifier_column: str = "sector"
    neutralize_method: str = "demean"

    def prepare(self, ctx: FactorFrameContext) -> dict[str, Any]:
        return {"window": int(self.window)}

    def compute(self, ctx: FactorFrameContext, state: dict[str, Any]) -> pd.DataFrame:
        close = ctx.feed_wide(self.price_feed, self.price_column)
        raw = close.pct_change(int(state["window"]), fill_method=None)
        classifier_frame = ctx.classifier(self.classifier_name)
        if self.classifier_column not in classifier_frame.columns:
            raise KeyError(
                f"Classifier {self.classifier_name!r} has no column {self.classifier_column!r}."
            )
        groups = classifier_frame.set_index("code")[self.classifier_column].reindex(raw.columns)
        return group_neutralize(raw, groups, method=self.neutralize_method)


@dataclass
class CrossSectionalResidualDefinition(FactorDefinition):
    target_feed: str = "price"
    target_column: str = "close"
    window: int = 20
    regressors: list[FeedColumnSpec] = field(default_factory=list)
    add_intercept: bool = True

    def prepare(self, ctx: FactorFrameContext) -> dict[str, Any]:
        return {"window": int(self.window)}

    def compute(self, ctx: FactorFrameContext, state: dict[str, Any]) -> pd.DataFrame:
        target = ctx.feed_wide(self.target_feed, self.target_column).pct_change(int(state["window"]), fill_method=None)
        exposures = [ctx.feed_wide(feed_name, column_name) for feed_name, column_name in (_parse_feed_column_spec(spec) for spec in self.regressors)]
        return cross_sectional_residual(target, exposures, add_intercept=self.add_intercept)


@dataclass
class EventDrivenFactorDefinition(FactorDefinition):
    event_feed: str = "events"
    event_column: str = "event_flag"
    price_feed: str = "price"
    price_column: str = "close"
    window: int = 5
    threshold: float = 0.5

    def prepare(self, ctx: FactorFrameContext) -> dict[str, Any]:
        return {"window": int(self.window), "threshold": float(self.threshold)}

    def compute(self, ctx: FactorFrameContext, state: dict[str, Any]) -> pd.DataFrame:
        events = ctx.feed_wide(self.event_feed, self.event_column)
        close = ctx.feed_wide(self.price_feed, self.price_column)
        event_mask = events.apply(pd.to_numeric, errors="coerce").fillna(0.0) >= float(state["threshold"])
        signal = close.pct_change(int(state["window"]), fill_method=None)
        return signal.where(event_mask)


@dataclass
class WeightedSumFactorDefinition(FactorDefinition):
    components: dict[str, float] = field(default_factory=dict)
    component_fns: dict[str, FactorDefinition | FactorFrameFactor | Callable[[FactorFrameContext], Any]] = field(
        default_factory=dict
    )
    normalize_weights: bool = True

    def compute(self, ctx: FactorFrameContext, state: dict[str, Any]) -> pd.DataFrame | pd.Series:
        frames: dict[str, pd.DataFrame | pd.Series] = {}
        for name, component in self.component_fns.items():
            frames[name] = _evaluate_component(component, ctx)
        return weighted_sum(frames, self.components, normalize=self.normalize_weights)


__all__ = [
    "CrossSectionalResidualDefinition",
    "EventDrivenFactorDefinition",
    "FactorDefinition",
    "IndustryNeutralMomentumDefinition",
    "WeightedSumFactorDefinition",
    "cross_sectional_residual",
    "event_window_abnormal_return",
    "weighted_sum",
]
