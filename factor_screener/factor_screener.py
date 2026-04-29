from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from tiger_factors.factor_screener._evaluation_io import factor_frame_to_panel
from tiger_factors.factor_screener._evaluation_io import load_return_series
from tiger_factors.factor_screener._evaluation_io import normalize_time_series
from tiger_factors.factor_screener._evaluation_io import pick_return_column
from tiger_factors.factor_evaluation.utils import period_to_label
from tiger_factors.factor_store import FactorSpec
from tiger_factors.factor_store import FactorStore
from tiger_factors.factor_screener.screening import FactorMetricFilterConfig
from tiger_factors.factor_screener.screening import screen_factor_metrics
from tiger_factors.factor_screener.validation import ScreeningEffectivenessResult
from tiger_factors.factor_screener.validation import ScreeningEffectivenessSpec
from tiger_factors.factor_screener.validation import validate_screening_effectiveness


def _series_to_long_frame(
    series: pd.Series,
    *,
    factor_name: str,
    return_mode: str,
) -> pd.DataFrame:
    if series.empty:
        return pd.DataFrame(columns=["date_", "factor", "return", "return_mode"])
    frame = pd.DataFrame(
        {
            "date_": pd.to_datetime(pd.Index(series.index), errors="coerce").to_numpy(),
            "return": pd.to_numeric(series, errors="coerce").to_numpy(),
        }
    ).reset_index(drop=True)
    frame["factor"] = factor_name
    frame["return_mode"] = return_mode
    frame = frame.dropna(subset=["date_", "return"])
    return frame.loc[:, ["date_", "factor", "return", "return_mode"]].sort_values(
        ["date_", "factor", "return_mode"],
        kind="stable",
    )


def _factor_panel_data_profile(panel: pd.DataFrame) -> dict[str, float | int]:
    if panel.empty:
        return {
            "data_rows": 0,
            "data_dates": 0,
            "data_codes": 0,
            "data_non_na": 0,
            "data_coverage": 0.0,
        }

    normalized = panel.copy()
    normalized.index = pd.to_datetime(normalized.index, errors="coerce")
    normalized = normalized.loc[~normalized.index.isna()]
    non_na = int(normalized.notna().sum().sum())
    total = int(normalized.size)
    coverage = float(non_na / total) if total > 0 else 0.0
    return {
        "data_rows": total,
        "data_dates": int(normalized.index.nunique()),
        "data_codes": int(normalized.columns.nunique()),
        "data_non_na": non_na,
        "data_coverage": coverage,
    }
def _load_factor_panel_from_store(
    store: FactorStore,
    spec: FactorSpec,
) -> pd.DataFrame:
    try:
        frame = store.get_factor(spec, engine="pandas")
    except Exception:
        return pd.DataFrame()
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    panel = factor_frame_to_panel(frame, factor_name=spec.table_name)
    return panel if isinstance(panel, pd.DataFrame) else pd.DataFrame()


def _load_factor_panels_from_store(
    store: FactorStore,
    specs: Iterable[FactorSpec],
) -> dict[str, pd.DataFrame]:
    panels: dict[str, pd.DataFrame] = {}
    for spec in specs:
        panel = _load_factor_panel_from_store(store, spec)
        if not panel.empty:
            panels[spec.table_name] = panel
    return panels


def _factor_panel_is_sufficient(
    panel: pd.DataFrame,
    *,
    min_observations: int | None,
    min_dates: int | None,
    min_codes: int | None,
    min_coverage: float | None,
) -> tuple[bool, list[str], dict[str, float | int]]:
    profile = _factor_panel_data_profile(panel)
    failed_rules: list[str] = []

    data_non_na = int(profile["data_non_na"])
    data_dates = int(profile["data_dates"])
    data_codes = int(profile["data_codes"])
    data_coverage = float(profile["data_coverage"])

    if min_observations is not None and data_non_na < int(min_observations):
        failed_rules.append(f"data_non_na<{int(min_observations)}")
    if min_dates is not None and data_dates < int(min_dates):
        failed_rules.append(f"data_dates<{int(min_dates)}")
    if min_codes is not None and data_codes < int(min_codes):
        failed_rules.append(f"data_codes<{int(min_codes)}")
    if min_coverage is not None and data_coverage < float(min_coverage):
        failed_rules.append(f"data_coverage<{float(min_coverage)}")

    return not failed_rules, failed_rules, profile


@dataclass(frozen=True)
class FactorScreenerSpec:
    min_factor_observations: int | None = 5
    min_factor_dates: int | None = 3
    min_factor_codes: int | None = 3
    min_factor_coverage: float | None = 0.01
    screening_config: FactorMetricFilterConfig = field(default_factory=FactorMetricFilterConfig)


@dataclass(frozen=True)
class FactorScreenerResult:
    spec: FactorScreenerSpec
    factor_specs: tuple[FactorSpec, ...]
    screened_at: pd.Timestamp
    summary: pd.DataFrame
    selection_summary: pd.DataFrame
    return_long: pd.DataFrame
    return_panel: pd.DataFrame
    return_series: dict[str, pd.Series]
    missing_return_factors: tuple[str, ...] = ()

    @property
    def selected_factor_names(self) -> list[str]:
        frame = self.selection_summary if not self.selection_summary.empty else self.summary
        if frame.empty or "selected" not in frame.columns and "usable" not in frame.columns:
            return []
        if "selected" in frame.columns:
            frame = frame.loc[frame["selected"].fillna(False)]
        else:
            frame = frame.loc[frame["usable"].fillna(False)]
        if "factor_name" not in frame.columns:
            return []
        return frame["factor_name"].astype(str).tolist()

    @property
    def selected_factor_specs(self) -> list[FactorSpec]:
        spec_map = {spec.table_name: spec for spec in self.factor_specs}
        return [spec_map[name] for name in self.selected_factor_names if name in spec_map]

    @property
    def screened_factor_names(self) -> list[str]:
        if self.summary.empty or "factor_name" not in self.summary.columns:
            return []
        return self.summary["factor_name"].astype(str).tolist()

    @property
    def screened_factor_specs(self) -> list[FactorSpec]:
        spec_map = {spec.table_name: spec for spec in self.factor_specs}
        return [spec_map[name] for name in self.screened_factor_names if name in spec_map]

    @property
    def rejected_factor_names(self) -> list[str]:
        if self.summary.empty or "usable" not in self.summary.columns:
            return []
        frame = self.summary.loc[~self.summary["usable"].fillna(False)]
        if "factor_name" not in frame.columns:
            return []
        return frame["factor_name"].astype(str).tolist()

    @property
    def rejected_factor_specs(self) -> list[FactorSpec]:
        spec_map = {spec.table_name: spec for spec in self.factor_specs}
        return [spec_map[name] for name in self.rejected_factor_names if name in spec_map]

    def to_summary(self) -> dict[str, Any]:
        if self.return_panel.empty:
            return_start = None
            return_end = None
        else:
            index = pd.DatetimeIndex(self.return_panel.index).dropna().sort_values()
            if index.empty:
                return_start = None
                return_end = None
            else:
                return_start = index[0].isoformat()
                return_end = index[-1].isoformat()
        return {
            "screened_at": self.screened_at.isoformat(),
            "factor_count": int(len(self.factor_specs)),
            "screened_factor_names": self.screened_factor_names,
            "selected_factor_names": self.selected_factor_names,
            "selected_count": int(len(self.selected_factor_names)),
            "rejected_factor_names": self.rejected_factor_names,
            "missing_return_factors": list(self.missing_return_factors),
            "return_start": return_start,
            "return_end": return_end,
            "summary_rows": int(len(self.summary)),
            "return_long_rows": int(len(self.return_long)),
            "return_modes": sorted({str(key.split(":", 1)[1]) for key in self.return_series if ":" in key}),
        }

    def validate_effectiveness(
        self,
        *,
        spec: ScreeningEffectivenessSpec | None = None,
    ) -> ScreeningEffectivenessResult:
        return validate_screening_effectiveness(
            self.summary,
            self.selection_summary,
            spec=spec,
        )


class FactorScreener:
    def __init__(
        self,
        spec: FactorScreenerSpec,
        *,
        factor_specs: tuple[FactorSpec, ...],
        store: FactorStore | None = None,
    ) -> None:
        if not factor_specs:
            raise ValueError("factor_specs must not be empty")
        self.spec = spec
        self.factor_specs = tuple(factor_specs)
        self.store = store or FactorStore()

    def _summary_frame(self, factor_spec: FactorSpec) -> pd.DataFrame:
        section = self.store.evaluation.section(factor_spec, "summary")
        frame = section.get_table()

        if frame.empty:
            raise ValueError(f"summary table is empty for factor {factor_spec.table_name!r}")
        if len(frame) != 1:
            raise ValueError(
                "summary table must contain exactly one row per factor; "
                f"got {len(frame)} rows for {factor_spec.table_name!r}"
            )

        normalized = frame.copy().reset_index(drop=True)
        if "factor_name" not in normalized.columns:
            normalized.insert(0, "factor_name", factor_spec.table_name)
        else:
            normalized["factor_name"] = normalized["factor_name"].fillna(factor_spec.table_name).astype(str)
        if "factor_name" not in normalized.columns:
            normalized.insert(0, "factor_name", factor_spec.table_name)
        return normalized

    def _stored_long_short_series(self, factor_spec: FactorSpec) -> pd.Series | None:
        return load_return_series(self.store, factor_spec, return_mode="long_short")

    def _stored_long_only_series(self, factor_spec: FactorSpec) -> pd.Series | None:
        section = self.store.evaluation.section(factor_spec, "returns")
        candidates = ("mean_return_by_quantile_by_date", "mean_return_by_quantile")
        period_label = period_to_label(1)
        for table_name in candidates:
            try:
                frame = section.get_table(table_name)
            except FileNotFoundError:
                continue
            if frame.empty:
                continue
            if isinstance(frame.index, pd.MultiIndex) and "factor_quantile" in frame.index.names and period_label in frame.columns:
                quantile_frame = frame[period_label].unstack("factor_quantile").sort_index()
                if quantile_frame.empty:
                    continue
                quantile_cols = pd.Index(quantile_frame.columns)
                numeric_quantiles = pd.to_numeric(quantile_cols, errors="coerce")
                top_quantile = numeric_quantiles.max() if numeric_quantiles.notna().any() else quantile_cols[-1]
                return normalize_time_series(quantile_frame[top_quantile], name=factor_spec.table_name)
            if isinstance(frame.columns, pd.MultiIndex) and "factor_quantile" in frame.columns.names:
                try:
                    level = frame.columns.names.index("factor_quantile")
                    columns = frame.columns.get_level_values(level)
                    if period_label in frame.columns:
                        quantile_frame = frame[period_label].copy()
                        if isinstance(quantile_frame, pd.Series):
                            return normalize_time_series(quantile_frame, name=factor_spec.table_name)
                except Exception:
                    continue
        return None

    def _return_modes(self, factor_spec: FactorSpec) -> dict[str, pd.Series]:
        return_modes: dict[str, pd.Series] = {}
        stored = self._stored_long_short_series(factor_spec)
        if stored is not None and not stored.empty:
            return_modes["long_short"] = stored
        stored = self._stored_long_only_series(factor_spec)
        if stored is not None and not stored.empty:
            return_modes["long_only"] = stored
        return return_modes

    def _build_return_long_table(self, selected_specs: list[FactorSpec]) -> tuple[dict[str, pd.Series], pd.DataFrame, list[str]]:
        return_series: dict[str, pd.Series] = {}
        missing_return_factors: list[str] = []
        return_long_frames: list[pd.DataFrame] = []

        for factor_spec in selected_specs:
            factor_name = factor_spec.table_name
            modes = self._return_modes(factor_spec)
            if not modes:
                missing_return_factors.append(factor_name)
                continue
            for mode, series in modes.items():
                if series is None or series.empty:
                    continue
                key = f"{factor_name}:{mode}"
                return_series[key] = series
                return_long_frames.append(_series_to_long_frame(series, factor_name=factor_name, return_mode=mode))
            if factor_name not in {key.split(":", 1)[0] for key in return_series}:
                missing_return_factors.append(factor_name)

        return_long = (
            pd.concat(return_long_frames, ignore_index=True)
            if return_long_frames
            else pd.DataFrame(columns=["date_", "factor", "return", "return_mode"])
        )
        if not return_long.empty:
            return_long["date_"] = pd.to_datetime(return_long["date_"], errors="coerce")
            return_long = (
                return_long.dropna(subset=["date_", "factor", "return"])
                .sort_values(["date_", "factor", "return_mode"], kind="stable")
                .reset_index(drop=True)
            )

        return return_series, return_long, missing_return_factors

    def run(self) -> FactorScreenerResult:
        summary_frames: list[pd.DataFrame] = []
        for factor_spec in self.factor_specs:
            summary_frames.append(self._summary_frame(factor_spec))

        combined_summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
        screened_summary = (
            screen_factor_metrics(combined_summary, config=self.spec.screening_config)
            if not combined_summary.empty
            else combined_summary
        )

        screened_at = pd.Timestamp.now(tz="UTC")
        if not screened_summary.empty and "screened_at" not in screened_summary.columns:
            screened_summary = screened_summary.copy()
            screened_summary["screened_at"] = screened_at.isoformat()

        selection_summary = screened_summary.copy() if not screened_summary.empty else screened_summary
        selected_names: list[str] = []
        screened_names = (
            screened_summary.loc[screened_summary["usable"].fillna(False), "factor_name"].astype(str).tolist()
            if not screened_summary.empty and "usable" in screened_summary.columns and "factor_name" in screened_summary.columns
            else []
        )
        factor_panels: dict[str, pd.DataFrame] = {}
        if screened_names:
            selected_specs = [spec for spec in self.factor_specs if spec.table_name in set(screened_names)]
            factor_panels = _load_factor_panels_from_store(self.store, selected_specs)

        if not screened_summary.empty and "factor_name" in screened_summary.columns:
            data_profile_rows: list[dict[str, float | int]] = []
            data_usable_flags: list[bool] = []
            data_failed_rules: list[list[str]] = []
            for _, row in screened_summary.iterrows():
                factor_name = str(row.get("factor_name", "")).strip()
                panel = factor_panels.get(factor_name)
                if panel is None or panel.empty:
                    data_profile_rows.append(
                        {
                            "data_rows": 0,
                            "data_dates": 0,
                            "data_codes": 0,
                            "data_non_na": 0,
                            "data_coverage": 0.0,
                        }
                    )
                    data_usable_flags.append(False)
                    data_failed_rules.append(["missing_factor_panel"])
                    continue

                data_usable, failed_rules, profile = _factor_panel_is_sufficient(
                    panel,
                    min_observations=self.spec.min_factor_observations,
                    min_dates=self.spec.min_factor_dates,
                    min_codes=self.spec.min_factor_codes,
                    min_coverage=self.spec.min_factor_coverage,
                )
                data_profile_rows.append(profile)
                data_usable_flags.append(data_usable)
                data_failed_rules.append(failed_rules)

            profile_frame = pd.DataFrame(data_profile_rows)
            for column in profile_frame.columns:
                screened_summary[column] = profile_frame[column].to_numpy()
            screened_summary["data_usable"] = data_usable_flags
            screened_summary["data_failed_rules"] = data_failed_rules
            if "usable" in screened_summary.columns:
                screened_summary["usable"] = screened_summary["usable"].fillna(False) & screened_summary["data_usable"].fillna(False)
            else:
                screened_summary["usable"] = screened_summary["data_usable"].fillna(False)

        screened_names = (
            screened_summary.loc[screened_summary["usable"].fillna(False), "factor_name"].astype(str).tolist()
            if not screened_summary.empty and "usable" in screened_summary.columns and "factor_name" in screened_summary.columns
            else []
        )
        factor_panels = {name: panel for name, panel in factor_panels.items() if name in screened_names}
        selected_names = list(factor_panels.keys()) if factor_panels else screened_names

        if not selection_summary.empty and "factor_name" in selection_summary.columns:
            selection_summary = selection_summary.copy()
            selection_summary["selected"] = selection_summary["factor_name"].astype(str).isin(selected_names)
            if "selected_score" not in selection_summary.columns:
                score_field = "fitness" if "fitness" in selection_summary.columns else None
                if score_field is not None:
                    selection_summary["selected_score"] = selection_summary[score_field]

        selected_factor_specs = [spec for spec in self.factor_specs if spec.table_name in set(selected_names)]
        return_series, return_long, missing_return_factors = self._build_return_long_table(selected_factor_specs)
        return_panel = pd.DataFrame()
        if not return_long.empty and "return_mode" in return_long.columns:
            available_modes = [str(mode) for mode in return_long["return_mode"].dropna().astype(str).unique().tolist()]
            primary_mode = "long_short" if "long_short" in available_modes else ("long_only" if "long_only" in available_modes else available_modes[0])
            primary_frame = return_long.loc[return_long["return_mode"] == primary_mode].copy()
            if not primary_frame.empty:
                return_panel = (
                    primary_frame.pivot_table(index="date_", columns="factor", values="return", aggfunc="last")
                    .sort_index()
                )
                return_panel.index = pd.to_datetime(return_panel.index, errors="coerce")
                return_panel = return_panel.loc[~return_panel.index.isna()].sort_index()

        return FactorScreenerResult(
            spec=self.spec,
            factor_specs=self.factor_specs,
            screened_at=screened_at,
            summary=screened_summary.reset_index(drop=True) if not screened_summary.empty else screened_summary,
            selection_summary=selection_summary.reset_index(drop=True) if not selection_summary.empty else selection_summary,
            return_long=return_long,
            return_panel=return_panel,
            return_series=return_series,
            missing_return_factors=tuple(missing_return_factors),
        )


def run_factor_screener(
    spec: FactorScreenerSpec,
    factor_specs: Iterable[FactorSpec],
    *,
    store: FactorStore | None = None,
) -> FactorScreenerResult:
    resolved_specs = tuple(factor_specs)
    return FactorScreener(
        spec,
        factor_specs=resolved_specs,
        store=store,
    ).run()


__all__ = [
    "FactorScreener",
    "FactorScreenerResult",
    "FactorScreenerSpec",
    "run_factor_screener",
]
