from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from tiger_factors.factor_frame.definition import FactorDefinition
from tiger_factors.factor_frame.engine import FactorFrameContext
from tiger_factors.factor_frame.engine import to_long_factor
from tiger_factors.factor_frame.factors import FactorFrameFactor
from tiger_factors.factor_frame.definition import weighted_sum
from tiger_factors.factor_frame.definition_registry import FactorDefinitionRegistry


def _coerce_numeric_panel(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _normalize_index(index: pd.Index) -> pd.Index:
    try:
        converted = pd.DatetimeIndex(pd.to_datetime(index, errors="coerce"))
        if converted.notna().any():
            return converted
    except Exception:  # pragma: no cover - defensive fallback
        pass
    return index


def _coerce_output_to_panel(output: Any, *, member_name: str) -> pd.DataFrame:
    if output is None:
        return pd.DataFrame()

    if isinstance(output, pd.Series):
        series = pd.to_numeric(output, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if isinstance(series.index, pd.MultiIndex):
            if series.index.nlevels < 2:
                raise ValueError(f"Member {member_name!r} returned a MultiIndex series that cannot be reshaped.")
            panel = series.unstack(level=-1)
            panel.index = _normalize_index(panel.index)
            panel.columns = panel.columns.astype(str)
            return panel.sort_index().sort_index(axis=1)
        frame = series.to_frame(name=member_name)
        frame.index = _normalize_index(frame.index)
        return frame.sort_index()

    if isinstance(output, pd.DataFrame):
        frame = output.copy()
        if {"date_", "code"}.issubset(frame.columns):
            value_columns = [column for column in frame.columns if column not in {"date_", "code"}]
            if len(value_columns) != 1:
                raise ValueError(
                    f"Member {member_name!r} returned long-form data with {len(value_columns)} value columns; "
                    "expected exactly one."
                )
            value_column = value_columns[0]
            panel = frame.pivot_table(index="date_", columns="code", values=value_column, aggfunc="last").sort_index()
            panel.index = _normalize_index(panel.index)
            panel.columns = panel.columns.astype(str)
            return _coerce_numeric_panel(panel)

        if isinstance(frame.index, pd.MultiIndex) and frame.shape[1] == 1:
            panel = frame.iloc[:, 0].unstack(level=-1)
            panel.index = _normalize_index(panel.index)
            panel.columns = panel.columns.astype(str)
            return _coerce_numeric_panel(panel.sort_index().sort_index(axis=1))

        frame.index = _normalize_index(frame.index)
        frame.columns = frame.columns.astype(str)
        return _coerce_numeric_panel(frame.sort_index().sort_index(axis=1))

    raise TypeError(f"Unsupported member output type for {member_name!r}: {type(output)!r}")


def _align_panels(panels: list[pd.DataFrame]) -> list[pd.DataFrame]:
    if not panels:
        return []
    index = panels[0].index
    columns = panels[0].columns
    for panel in panels[1:]:
        index = index.union(panel.index)
        columns = columns.union(panel.columns)
    index = index.sort_values()
    columns = columns.sort_values()
    return [panel.reindex(index=index, columns=columns) for panel in panels]


def _panel_mean(panels: list[pd.DataFrame]) -> pd.DataFrame:
    if not panels:
        return pd.DataFrame()
    aligned = _align_panels([_coerce_numeric_panel(panel) for panel in panels])
    total = pd.DataFrame(0.0, index=aligned[0].index, columns=aligned[0].columns)
    count = pd.DataFrame(0.0, index=aligned[0].index, columns=aligned[0].columns)
    for panel in aligned:
        valid = panel.notna().astype(float)
        total = total.add(panel.fillna(0.0), fill_value=0.0)
        count = count.add(valid, fill_value=0.0)
    return total.where(count > 0).divide(count.where(count > 0))


def _panel_median(panels: list[pd.DataFrame]) -> pd.DataFrame:
    if not panels:
        return pd.DataFrame()
    aligned = _align_panels([_coerce_numeric_panel(panel) for panel in panels])
    stacked = np.stack([panel.to_numpy(dtype=float) for panel in aligned], axis=0)
    with np.errstate(all="ignore"):
        median = np.nanmedian(stacked, axis=0)
    median[np.sum(np.isfinite(stacked), axis=0) == 0] = np.nan
    return pd.DataFrame(median, index=aligned[0].index, columns=aligned[0].columns)


def _panel_rank_mean(panels: list[pd.DataFrame]) -> pd.DataFrame:
    ranked = [panel.rank(axis=1, pct=True) for panel in _align_panels([_coerce_numeric_panel(panel) for panel in panels])]
    return _panel_mean(ranked)


def _panel_zscore_mean(panels: list[pd.DataFrame]) -> pd.DataFrame:
    normalized: list[pd.DataFrame] = []
    for panel in _align_panels([_coerce_numeric_panel(panel) for panel in panels]):
        mean = panel.mean(axis=1)
        std = panel.std(axis=1, ddof=0).replace(0.0, np.nan)
        normalized.append(panel.sub(mean, axis=0).div(std, axis=0))
    return _panel_mean(normalized)


def _panel_weighted_sum(
    panels: Mapping[str, pd.DataFrame],
    weights: Mapping[str, float],
    *,
    normalize: bool = True,
) -> pd.DataFrame:
    if not panels:
        return pd.DataFrame()
    aligned = _align_panels([_coerce_numeric_panel(panel) for panel in panels.values()])
    panel_names = list(panels.keys())
    total_weight = sum(abs(float(weights.get(name, 0.0))) for name in panel_names) if normalize else 1.0
    result = pd.DataFrame(0.0, index=aligned[0].index, columns=aligned[0].columns)
    support = pd.DataFrame(0.0, index=aligned[0].index, columns=aligned[0].columns)
    for name, panel in zip(panel_names, aligned):
        weight = float(weights.get(name, 0.0))
        if normalize and total_weight not in {0.0, 0}:
            weight = weight / total_weight
        contribution = panel.fillna(0.0) * weight
        result = result.add(contribution, fill_value=0.0)
        support = support.add(panel.notna().astype(float), fill_value=0.0)
    return result.where(support > 0)


@dataclass(frozen=True)
class FactorGroupSpec:
    name: str
    members: tuple[str, ...]
    weights: dict[str, float] = field(default_factory=dict)
    combine_method: str = "weighted_sum"
    normalize_weights: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FactorGroupResult:
    group_panels: dict[str, pd.DataFrame]
    member_panels: dict[str, pd.DataFrame]
    summary: pd.DataFrame
    long_frame: pd.DataFrame
    manifest: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_panels": {name: panel.to_dict(orient="index") for name, panel in self.group_panels.items()},
            "member_panels": {name: panel.to_dict(orient="index") for name, panel in self.member_panels.items()},
            "summary": self.summary.to_dict(orient="records"),
            "long_frame": self.long_frame.to_dict(orient="records"),
            "manifest": self.manifest,
        }


class FactorGroupEngine:
    """Combine registered factors into family/group scores.

    The engine is intentionally lightweight and research-oriented:

    - member definitions can be resolved from a :class:`FactorDefinitionRegistry`
      or from explicit callables
    - each member is expected to produce a panel-like object
      (date x code preferred)
    - group-level scores can be combined with weighted sums, means, medians,
      or cross-sectional normalizations
    """

    def __init__(
        self,
        *,
        definition_registry: FactorDefinitionRegistry | None = None,
        member_sources: Mapping[str, FactorDefinition | FactorFrameFactor | Callable[[Any], Any]] | None = None,
        group_specs: Mapping[str, FactorGroupSpec] | None = None,
    ) -> None:
        self.definition_registry = definition_registry
        self.member_sources = dict(member_sources or {})
        self.group_specs = dict(group_specs or {})

    def register_member(
        self,
        name: str,
        source: FactorDefinition | FactorFrameFactor | Callable[[Any], Any],
    ) -> "FactorGroupEngine":
        self.member_sources[str(name)] = source
        return self

    def register_group(self, spec: FactorGroupSpec) -> "FactorGroupEngine":
        self.group_specs[str(spec.name)] = spec
        return self

    def register_groups(self, *specs: FactorGroupSpec) -> "FactorGroupEngine":
        for spec in specs:
            self.register_group(spec)
        return self

    def _resolve_member(self, name: str, ctx: Any) -> Any:
        if name in self.member_sources:
            source = self.member_sources[name]
        elif self.definition_registry is not None and name in self.definition_registry:
            source = self.definition_registry.get(name)
        else:
            available = sorted({*self.member_sources.keys(), *(self.definition_registry.names() if self.definition_registry else ())})
            raise KeyError(f"Unknown group member {name!r}. Available members: {available}")

        if isinstance(source, FactorDefinition):
            return source.run(ctx)
        if isinstance(source, FactorFrameFactor):
            return source(ctx)
        if callable(source):
            return source(ctx)
        raise TypeError(f"Unsupported source type for member {name!r}: {type(source)!r}")

    def _combine_group(self, member_panels: dict[str, pd.DataFrame], spec: FactorGroupSpec) -> pd.DataFrame:
        if not member_panels:
            return pd.DataFrame()
        method = str(spec.combine_method).strip().lower()
        ordered_members = [member for member in spec.members if member in member_panels]
        if not ordered_members:
            return pd.DataFrame()

        panels = [member_panels[member] for member in ordered_members]
        if method == "weighted_sum":
            weights = {member: float(spec.weights.get(member, 1.0)) for member in ordered_members}
            return _panel_weighted_sum({member: member_panels[member] for member in ordered_members}, weights, normalize=spec.normalize_weights)
        if method == "mean":
            return _panel_mean(panels)
        if method == "median":
            return _panel_median(panels)
        if method == "rank_mean":
            return _panel_rank_mean(panels)
        if method == "zscore_mean":
            return _panel_zscore_mean(panels)
        raise ValueError(
            f"Unsupported combine_method={spec.combine_method!r}; expected weighted_sum, mean, median, rank_mean, or zscore_mean."
        )

    @staticmethod
    def _panel_summary(name: str, panel: pd.DataFrame, spec: FactorGroupSpec) -> dict[str, Any]:
        numeric = _coerce_numeric_panel(panel)
        try:
            values = numeric.stack(future_stack=True)
        except TypeError:  # pragma: no cover - older pandas fallback
            values = numeric.stack()
        values = pd.to_numeric(values, errors="coerce").dropna()
        rows, cols = numeric.shape
        observations = int(values.size)
        return {
            "group": name,
            "combine_method": spec.combine_method,
            "n_members": len(spec.members),
            "n_rows": int(rows),
            "n_columns": int(cols),
            "observations": observations,
            "coverage": float(observations / max(rows * cols, 1)),
            "mean": float(values.mean()) if not values.empty else np.nan,
            "std": float(values.std(ddof=0)) if not values.empty else np.nan,
            "min": float(values.min()) if not values.empty else np.nan,
            "max": float(values.max()) if not values.empty else np.nan,
        }

    @staticmethod
    def _panel_to_long(group_name: str, panel: pd.DataFrame) -> pd.DataFrame:
        if panel.empty:
            return pd.DataFrame(columns=["date_", "code", "group", "value"])
        long_df = to_long_factor(panel, "value")
        long_df["group"] = group_name
        return long_df.loc[:, ["date_", "code", "group", "value"]].sort_values(["date_", "code"]).reset_index(drop=True)

    def run(self, ctx: FactorFrameContext | Any) -> FactorGroupResult:
        member_panels: dict[str, pd.DataFrame] = {}
        for name in sorted({member for spec in self.group_specs.values() for member in spec.members}):
            member_panels[name] = _coerce_output_to_panel(self._resolve_member(name, ctx), member_name=name)

        group_panels: dict[str, pd.DataFrame] = {}
        summaries: list[dict[str, Any]] = []
        long_frames: list[pd.DataFrame] = []
        for group_name, spec in self.group_specs.items():
            combined = self._combine_group(member_panels, spec)
            group_panels[group_name] = combined
            summaries.append(self._panel_summary(group_name, combined, spec))
            if not combined.empty:
                long_frames.append(self._panel_to_long(group_name, combined))

        summary_frame = pd.DataFrame(summaries).sort_values(["coverage", "mean"], ascending=[False, False]).reset_index(drop=True)
        long_frame = pd.concat(long_frames, axis=0, ignore_index=True) if long_frames else pd.DataFrame(columns=["date_", "code", "group", "value"])
        manifest = {
            "group_names": list(self.group_specs.keys()),
            "member_names": sorted(member_panels.keys()),
            "groups": {name: asdict(spec) for name, spec in self.group_specs.items()},
        }
        return FactorGroupResult(
            group_panels=group_panels,
            member_panels=member_panels,
            summary=summary_frame,
            long_frame=long_frame,
            manifest=manifest,
        )


def build_factor_group_engine(
    *,
    definition_registry: FactorDefinitionRegistry | None = None,
    member_sources: Mapping[str, FactorDefinition | FactorFrameFactor | Callable[[Any], Any]] | None = None,
    group_specs: Mapping[str, FactorGroupSpec] | None = None,
) -> FactorGroupEngine:
    return FactorGroupEngine(
        definition_registry=definition_registry,
        member_sources=member_sources,
        group_specs=group_specs,
    )


__all__ = [
    "FactorGroupEngine",
    "FactorGroupResult",
    "FactorGroupSpec",
    "build_factor_group_engine",
]
