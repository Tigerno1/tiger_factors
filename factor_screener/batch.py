from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Iterable
from typing import Sequence

import pandas as pd

from tiger_factors.factor_screener.batch_persistence import FactorScreenerDetailManifest
from tiger_factors.factor_screener.batch_persistence import save_batch_detail
from tiger_factors.factor_screener.batch_persistence import save_batch_summary
from tiger_factors.factor_screener.batch_selection import FactorReturnGainSelectionConfig
from tiger_factors.factor_screener.batch_selection import FactorSelectionMode
from tiger_factors.factor_screener.batch_selection import combined_return_panel
from tiger_factors.factor_screener.batch_selection import item_label
from tiger_factors.factor_screener.batch_selection import resolve_return_gain_config
from tiger_factors.factor_screener.batch_selection import resolve_selection_mode
from tiger_factors.factor_screener.batch_selection import spec_signature
from tiger_factors.factor_screener.batch_selection import tag_frame
from tiger_factors.factor_screener.batch_selection import build_result_return_artifacts
from tiger_factors.factor_screener.batch_global_selection import select_global_factor_keys
from tiger_factors.factor_screener.factor_screener import FactorScreener
from tiger_factors.factor_screener.factor_screener import FactorScreenerResult
from tiger_factors.factor_screener.factor_screener import FactorScreenerSpec
from tiger_factors.factor_screener.selection import FactorMarginalSelectionConfig
from tiger_factors.factor_store import FactorSpec
from tiger_factors.factor_store import FactorStore


@dataclass(frozen=True)
class FactorScreenerBatchItem:
    spec: FactorScreenerSpec
    label: str | None = None


@dataclass(frozen=True)
class FactorScreenerBatchSpec:
    items: tuple[FactorScreenerBatchItem, ...]
    cross_spec_selection_threshold: float | None = 0.75
    cross_spec_selection_score_field: str = "selected_score"
    selection_mode: str = FactorSelectionMode.CORRELATION
    marginal_selection_config: FactorMarginalSelectionConfig = field(default_factory=FactorMarginalSelectionConfig)
    return_gain_config: FactorReturnGainSelectionConfig = field(default_factory=FactorReturnGainSelectionConfig)
    return_gain_preset: str | None = None

    def normalized_items(self) -> tuple[FactorScreenerBatchItem, ...]:
        if not self.items:
            raise ValueError("items must not be empty")
        return self.items


@dataclass(frozen=True)
class FactorScreenerBatchResult:
    spec: FactorScreenerBatchSpec
    results: tuple[FactorScreenerResult, ...]
    summary: pd.DataFrame
    selection_summary: pd.DataFrame
    return_long: pd.DataFrame
    return_panel: pd.DataFrame
    global_selected_factor_keys: tuple[str, ...] = ()
    global_selected_factor_names: tuple[str, ...] = ()
    detail_manifest: FactorScreenerDetailManifest | None = None

    @property
    def selected_factor_names(self) -> list[str]:
        if self.selection_summary.empty:
            return []
        if "global_selected" in self.selection_summary.columns:
            frame = self.selection_summary.loc[self.selection_summary["global_selected"].fillna(False)]
        elif "selected" in self.selection_summary.columns:
            frame = self.selection_summary.loc[self.selection_summary["selected"].fillna(False)]
        else:
            return []
        if "factor_name" not in frame.columns:
            return []
        return frame["factor_name"].astype(str).tolist()

    @property
    def global_selected_factor_names_list(self) -> list[str]:
        return list(self.global_selected_factor_names)

    @property
    def selected_factor_specs(self) -> list[FactorSpec]:
        if self.global_selected_factor_specs:
            return list(self.global_selected_factor_specs)
        specs: list[FactorSpec] = []
        for result in self.results:
            specs.extend(result.selected_factor_specs)
        return specs

    @property
    def global_selected_factor_specs(self) -> list[FactorSpec]:
        if not self.global_selected_factor_keys:
            return []
        specs_by_label: dict[str, FactorScreenerResult] = {}
        for index, (item, result) in enumerate(zip(self.spec.items, self.results)):
            label = item_label(item, index)
            specs_by_label[label] = result
        selected_specs: list[FactorSpec] = []
        for key in self.global_selected_factor_keys:
            label, factor_name = key.split("::", 1) if "::" in key else ("", key)
            result = specs_by_label.get(label)
            if result is None:
                continue
            match = next((spec for spec in result.factor_specs if spec.table_name == factor_name), None)
            if match is not None:
                selected_specs.append(match)
        return selected_specs

    def spec_summaries(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for index, (item, result) in enumerate(zip(self.spec.items, self.results)):
            label = item_label(item, index)
            selected_names = result.selected_factor_names
            rows.append(
                {
                    "batch_index": index,
                    "batch_label": label,
                    "group": result.spec.group,
                    "factor_count": int(len(result.spec.factor_names)),
                    "screened_factor_count": int(len(result.screened_factor_names)),
                    "selected_factor_count": int(len(selected_names)),
                    "missing_return_factor_count": int(len(result.missing_return_factors)),
                    "return_long_rows": int(len(result.return_long)),
                    "return_panel_columns": int(len(result.return_panel.columns)),
                    "correlation_method": result.spec.correlation_method,
                    "ic_correlation_method": result.spec.ic_correlation_method,
                    "return_modes": sorted({str(key.split(":", 1)[1]) for key in result.return_series if ":" in key}),
                    "selected_factor_names": list(selected_names),
                    "screened_factor_names": list(result.screened_factor_names),
                    "rejected_factor_names": list(result.rejected_factor_names),
                }
            )
        return pd.DataFrame(rows)

    def to_summary(self) -> dict[str, object]:
        spec_summary = self.spec_summaries()
        selection_mode = resolve_selection_mode(self.spec.selection_mode)
        groups = [
            str(item.spec.group)
            for item in self.spec.items
            if getattr(item.spec, "group", None) is not None
        ]
        unique_groups = sorted(set(groups))
        return {
            "selection_mode": selection_mode,
            "group": unique_groups[0] if len(unique_groups) == 1 else None,
            "groups": unique_groups,
            "return_gain_preset": self.spec.return_gain_preset,
            "correlation_method": self.spec.items[0].spec.correlation_method if self.spec.items else None,
            "ic_correlation_method": self.spec.items[0].spec.ic_correlation_method if self.spec.items else None,
            "spec_count": int(len(self.spec.items)),
            "screened_rows": int(len(self.summary)),
            "selection_rows": int(len(self.selection_summary)),
            "selected_factor_count": int(len(self.selected_factor_names)),
            "global_selected_factor_count": int(len(self.global_selected_factor_names)),
            "selected_factor_names": list(self.selected_factor_names),
            "global_selected_factor_names": list(self.global_selected_factor_names),
            "return_long_rows": int(len(self.return_long)),
            "return_panel_columns": int(len(self.return_panel.columns)),
            "spec_summaries": spec_summary.to_dict(orient="records"),
        }

    def to_summary_frame(self) -> pd.DataFrame:
        overall = pd.DataFrame([self.to_summary()])
        if overall.empty:
            return overall
        overall = overall.assign(record_type="overall")
        spec_summary = self.spec_summaries()
        if not spec_summary.empty:
            spec_summary = spec_summary.assign(
                record_type="spec",
                selection_mode=resolve_selection_mode(self.spec.selection_mode),
                return_gain_preset=self.spec.return_gain_preset,
                spec_count=pd.NA,
                screened_rows=pd.NA,
                selection_rows=pd.NA,
                selected_factor_count=pd.NA,
                global_selected_factor_count=pd.NA,
                return_long_rows=pd.NA,
                return_panel_columns=pd.NA,
                selected_factor_names=pd.NA,
                global_selected_factor_names=pd.NA,
            )
        combined = pd.concat([overall, spec_summary], ignore_index=True, sort=False) if not spec_summary.empty else overall
        preferred_columns = [
            "record_type",
            "selection_mode",
            "group",
            "return_gain_preset",
            "correlation_method",
            "ic_correlation_method",
            "batch_index",
            "batch_label",
            "spec_count",
            "factor_count",
            "screened_factor_count",
            "selected_factor_count",
            "missing_return_factor_count",
            "screened_rows",
            "selection_rows",
            "global_selected_factor_count",
            "return_long_rows",
            "return_panel_columns",
            "selected_factor_names",
            "global_selected_factor_names",
        ]
        existing = [column for column in preferred_columns if column in combined.columns]
        remaining = [column for column in combined.columns if column not in existing]
        return combined.loc[:, existing + remaining]

    def save_summary(self, path: str | Path) -> Path:
        return save_batch_summary(self, path)

    def save_detail(self, path: str | Path) -> FactorScreenerDetailManifest:
        manifest = save_batch_detail(self, path)
        object.__setattr__(self, "detail_manifest", manifest)
        return manifest


class FactorScreenerBatch:
    def __init__(
        self,
        spec: FactorScreenerBatchSpec,
        *,
        store: FactorStore | None = None,
    ) -> None:
        self.spec = spec
        self.store = store or FactorStore()

    def _run_item(self, item: FactorScreenerBatchItem, *, index: int) -> tuple[str, FactorScreener, FactorScreenerResult]:
        label = item_label(item, index)
        screener = FactorScreener(item.spec, store=self.store)
        result = screener.run()
        return label, screener, result

    def run(self) -> FactorScreenerBatchResult:
        items = self.spec.normalized_items()
        selection_mode = resolve_selection_mode(self.spec.selection_mode)
        results: list[tuple[str, FactorScreenerResult]] = []
        screeners: list[tuple[str, FactorScreener]] = []
        return_panel_frames: list[tuple[str, pd.DataFrame]] = []

        for index, item in enumerate(items):
            label, screener, result = self._run_item(item, index=index)
            results.append((label, result))
            screeners.append((label, screener))

        return_long_frames: list[pd.DataFrame] = []
        for index, ((label, screener), (_, result)) in enumerate(zip(screeners, results)):
            _, return_long, return_panel, _ = build_result_return_artifacts(result)
            if not return_long.empty:
                return_long_frames.append(
                    tag_frame(
                        return_long,
                        batch_index=index,
                        batch_label=label,
                        batch_signature=spec_signature(screener.spec),
                    )
                )
            if not return_panel.empty:
                tagged_panel = return_panel.copy()
                tagged_panel.index = pd.to_datetime(tagged_panel.index, errors="coerce")
                tagged_panel = tagged_panel.loc[~tagged_panel.index.isna()].sort_index()
                return_panel_frames.append((label, tagged_panel))

        summary_frames = [
            tag_frame(
                result.summary,
                batch_index=index,
                batch_label=label,
                batch_signature=spec_signature(result.spec),
            )
            for index, (label, result) in enumerate(results)
        ]
        selection_frames = [
            tag_frame(
                result.selection_summary,
                batch_index=index,
                batch_label=label,
                batch_signature=spec_signature(result.spec),
            )
            for index, (label, result) in enumerate(results)
        ]

        summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
        selection_summary = pd.concat(selection_frames, ignore_index=True) if selection_frames else pd.DataFrame()
        return_long = pd.concat(return_long_frames, ignore_index=True) if return_long_frames else pd.DataFrame()
        if not return_panel_frames:
            return_panel = combined_return_panel(results)
        elif len(return_panel_frames) == 1:
            return_panel = return_panel_frames[0][1]
        else:
            panels: list[pd.DataFrame] = []
            for label, panel in return_panel_frames:
                prefixed = panel.copy()
                prefixed.columns = [f"{label}::{column}" for column in prefixed.columns]
                panels.append(prefixed)
            return_panel = pd.concat(panels, axis=1).sort_index()

        global_selected_keys, global_selected_names, selection_summary = select_global_factor_keys(
            screeners,
            results,
            selection_summary,
            selection_mode=selection_mode,
            cross_spec_selection_threshold=self.spec.cross_spec_selection_threshold,
            cross_spec_selection_score_field=self.spec.cross_spec_selection_score_field,
            marginal_selection_config=self.spec.marginal_selection_config,
            return_gain_config=self.spec.return_gain_config,
        )

        return FactorScreenerBatchResult(
            spec=self.spec,
            results=tuple(result for _, result in results),
            summary=summary.reset_index(drop=True) if not summary.empty else summary,
            selection_summary=selection_summary.reset_index(drop=True) if not selection_summary.empty else selection_summary,
            return_long=return_long.reset_index(drop=True) if not return_long.empty else return_long,
            return_panel=return_panel,
            global_selected_factor_keys=global_selected_keys,
            global_selected_factor_names=global_selected_names,
        )


def run_factor_screener_batch(
    items: Iterable[FactorScreenerBatchItem | FactorScreenerSpec],
    *,
    store: FactorStore | None = None,
    cross_spec_selection_threshold: float | None = 0.75,
    cross_spec_selection_score_field: str = "selected_score",
    selection_mode: str = "correlation",
    marginal_selection_config: FactorMarginalSelectionConfig | None = None,
    return_gain_config: FactorReturnGainSelectionConfig | None = None,
    return_gain_preset: str | None = None,
) -> FactorScreenerBatchResult:
    coerced_items = tuple(
        item if isinstance(item, FactorScreenerBatchItem) else FactorScreenerBatchItem(spec=item)
        for item in items
    )
    spec = FactorScreenerBatchSpec(
        items=coerced_items,
        cross_spec_selection_threshold=cross_spec_selection_threshold,
        cross_spec_selection_score_field=cross_spec_selection_score_field,
        selection_mode=resolve_selection_mode(selection_mode),
        marginal_selection_config=marginal_selection_config or FactorMarginalSelectionConfig(),
        return_gain_config=resolve_return_gain_config(return_gain_config, return_gain_preset),
        return_gain_preset=return_gain_preset,
    )
    return FactorScreenerBatch(spec, store=store).run()


def run_factor_screener_flow(
    specs: Iterable[FactorScreenerSpec],
    *,
    labels: Sequence[str | None] | None = None,
    store: FactorStore | None = None,
    cross_spec_selection_threshold: float | None = 0.75,
    cross_spec_selection_score_field: str = "selected_score",
    selection_mode: str = "correlation",
    marginal_selection_config: FactorMarginalSelectionConfig | None = None,
    return_gain_config: FactorReturnGainSelectionConfig | None = None,
    return_gain_preset: str | None = None,
    save_dir: str | Path | None = None,
) -> FactorScreenerBatchResult:
    specs_tuple = tuple(specs)
    if labels is not None and len(labels) != len(specs_tuple):
        raise ValueError("labels must have the same length as specs")
    items: list[FactorScreenerBatchItem] = []
    for idx, spec in enumerate(specs_tuple):
        label = None if labels is None else labels[idx]
        items.append(FactorScreenerBatchItem(spec=spec, label=label))
    result = run_factor_screener_batch(
        items,
        store=store,
        cross_spec_selection_threshold=cross_spec_selection_threshold,
        cross_spec_selection_score_field=cross_spec_selection_score_field,
        selection_mode=resolve_selection_mode(selection_mode),
        marginal_selection_config=marginal_selection_config,
        return_gain_config=return_gain_config,
        return_gain_preset=return_gain_preset,
    )
    if save_dir is not None:
        manifest = result.save_detail(save_dir)
        object.__setattr__(result, "detail_manifest", manifest)
    return result


__all__ = [
    "FactorScreenerBatch",
    "FactorScreenerBatchItem",
    "FactorScreenerDetailManifest",
    "FactorScreenerBatchResult",
    "FactorScreenerBatchSpec",
    "FactorReturnGainSelectionConfig",
    "FactorSelectionMode",
    "run_factor_screener_batch",
    "run_factor_screener_flow",
]
