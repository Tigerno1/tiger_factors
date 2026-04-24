from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from tiger_factors.factor_evaluation.utils import period_to_label
from tiger_factors.factor_screener.selection import select_by_graph_independent_set_from_correlation_matrix
from tiger_factors.factor_screener.selection import select_cluster_representatives_from_correlation_matrix
from tiger_factors.factor_store import FactorSpec
from tiger_factors.factor_store import FactorStore
from tiger_factors.multifactor_evaluation._inputs import coerce_factor_series


def _normalize_return_series(series: pd.Series, *, factor_name: str) -> pd.Series:
    cleaned = pd.to_numeric(series, errors="coerce").replace([pd.NA, pd.NaT], pd.NA).dropna()
    if cleaned.empty:
        return cleaned
    cleaned.index = pd.to_datetime(cleaned.index, errors="coerce")
    cleaned = cleaned[~cleaned.index.isna()].sort_index()
    cleaned.name = factor_name
    return cleaned


def _pick_numeric_column(frame: pd.DataFrame) -> str:
    numeric_columns = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    if numeric_columns:
        return str(numeric_columns[0])
    raise KeyError(f"could not infer numeric column from: {list(frame.columns)!r}")


def _pick_return_column(frame: pd.DataFrame, *, preferred_period: str | int | pd.Timedelta | None = None) -> str:
    lookup = {str(column): column for column in frame.columns}
    if preferred_period is not None:
        preferred_label = period_to_label(preferred_period)
        if preferred_label in lookup:
            return str(lookup[preferred_label])
        if str(preferred_period) in lookup:
            return str(lookup[str(preferred_period)])
    for candidate in ("long_short", "long_short_returns", "factor_portfolio_returns", "returns", "return", "1D"):
        if candidate in lookup:
            return str(lookup[candidate])
    return _pick_numeric_column(frame)


def _load_factor_return_series(store: FactorStore, spec: FactorSpec) -> pd.Series | None:
    section = store.evaluation.returns(spec)
    try:
        frame = section.get_table("factor_portfolio_returns")
    except FileNotFoundError:
        return None
    if isinstance(frame, pd.Series):
        return _normalize_return_series(frame, factor_name=spec.table_name)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    working = frame.copy()
    if "date_" in working.columns:
        working["date_"] = pd.to_datetime(working["date_"], errors="coerce")
        working = working.loc[working["date_"].notna()]
    if "date_" in working.columns:
        numeric_columns = [column for column in working.columns if column != "date_" and pd.api.types.is_numeric_dtype(working[column])]
        if numeric_columns:
            column = _pick_return_column(working[numeric_columns])
            series = pd.Series(pd.to_numeric(working[column], errors="coerce").to_numpy(), index=working["date_"], name=spec.table_name)
            return _normalize_return_series(series, factor_name=spec.table_name)
    column = _pick_return_column(working)
    series = working[column]
    if isinstance(series, pd.DataFrame):
        series = series.squeeze(axis=1)
    return _normalize_return_series(series, factor_name=spec.table_name)


def _load_factor_ic_series(
    store: FactorStore,
    spec: FactorSpec,
    *,
    period: str | int | pd.Timedelta | None = None,
) -> pd.Series | None:
    section = store.evaluation.information(spec)
    try:
        frame = section.get_table("information_coefficient")
    except FileNotFoundError:
        return None
    if isinstance(frame, pd.Series):
        return _normalize_return_series(frame, factor_name=spec.table_name)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    column = _pick_return_column(frame, preferred_period=period)
    series = frame[column]
    if isinstance(series, pd.DataFrame):
        series = series.squeeze(axis=1)
    return _normalize_return_series(series, factor_name=spec.table_name)


def _load_factor_panel(store: FactorStore, spec: FactorSpec) -> pd.DataFrame:
    try:
        frame = store.get_factor(spec, engine="pandas")
    except Exception:
        return pd.DataFrame()
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    normalized = frame.copy()
    if "date_" in normalized.columns:
        normalized["date_"] = pd.to_datetime(normalized["date_"], errors="coerce")
    if "code" in normalized.columns:
        normalized["code"] = normalized["code"].astype(str)
    if {"date_", "code"}.issubset(normalized.columns):
        value_columns = [
            column
            for column in normalized.columns
            if column not in {"date_", "code"} and pd.api.types.is_numeric_dtype(normalized[column])
        ]
        if not value_columns:
            return pd.DataFrame()
        panel = normalized.pivot_table(index="date_", columns="code", values=value_columns[0], aggfunc="last")
        panel.index = pd.to_datetime(panel.index, errors="coerce")
        panel = panel.loc[~panel.index.isna()].sort_index()
        panel.columns = panel.columns.astype(str)
        return panel
    if isinstance(normalized.index, pd.DatetimeIndex):
        panel = normalized.copy()
        panel.index = pd.to_datetime(panel.index, errors="coerce")
        panel = panel.loc[~panel.index.isna()].sort_index()
        panel.columns = panel.columns.astype(str)
        return panel
    return pd.DataFrame()


def _build_series_map(
    store: FactorStore,
    factor_specs: tuple[FactorSpec, ...],
    *,
    source: str,
    ic_period: str | int | pd.Timedelta | None = None,
) -> dict[str, pd.Series | pd.DataFrame]:
    series_map: dict[str, pd.Series | pd.DataFrame] = {}
    for spec in factor_specs:
        if source == "ic":
            series = _load_factor_ic_series(store, spec, period=ic_period)
        elif source == "factor":
            series = _load_factor_panel(store, spec)
        else:
            series = _load_factor_return_series(store, spec)
        if series is not None and not series.empty:
            series_map[spec.table_name] = series
    return series_map


def _normalize_series_value(value: pd.Series | pd.DataFrame, *, name: str) -> pd.Series:
    if isinstance(value, pd.DataFrame):
        try:
            series = coerce_factor_series(value)
        except Exception:
            return pd.Series(dtype=float, name=name)
        series = series.sort_index()
        series.name = name
        return series
    if isinstance(value, pd.Series):
        if isinstance(value.index, pd.MultiIndex) and value.index.nlevels == 2:
            series = value.sort_index()
            series.name = name
            return series
        series = pd.to_numeric(value, errors="coerce").replace([pd.NA, pd.NaT], pd.NA).dropna()
        if series.empty:
            return series
        series.index = pd.to_datetime(series.index, errors="coerce")
        series = series[~series.index.isna()].sort_index()
        series.name = name
        return series
    return pd.Series(dtype=float, name=name)


def _build_correlation_matrix(series_map: dict[str, pd.Series | pd.DataFrame]) -> pd.DataFrame:
    aligned: list[pd.Series] = []
    for name, value in series_map.items():
        series = _normalize_series_value(value, name=name)
        if series is not None and not series.empty:
            aligned.append(series.rename(name))
    if not aligned:
        return pd.DataFrame()
    frame = pd.concat(aligned, axis=1).dropna()
    return frame.corr() if not frame.empty else pd.DataFrame()


def _select_from_correlation_matrix(
    corr: pd.DataFrame,
    scores: dict[str, float],
    *,
    threshold: float,
    comparator: str = "max",
) -> list[str]:
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    selected: list[str] = []

    for name, _ in ordered:
        if name not in corr.index or name not in corr.columns:
            continue
        if not selected:
            selected.append(name)
            continue

        selected_values = [abs(float(corr.loc[name, picked])) for picked in selected if picked in corr.columns]
        selected_values = [value for value in selected_values if pd.notna(value)]
        if not selected_values:
            selected.append(name)
            continue

        if comparator == "max":
            measure = max(selected_values)
        elif comparator == "mean":
            measure = float(np.mean(selected_values))
        elif comparator == "sum":
            measure = float(np.sum(selected_values))
        else:
            raise ValueError(f"unknown comparator: {comparator!r}")

        if measure < threshold:
            selected.append(name)

    return selected


def _build_summary_map(
    store: FactorStore,
    factor_specs: tuple[FactorSpec, ...],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for spec in factor_specs:
        frame = store.evaluation.summary(spec).get_table()
        if frame.empty:
            continue
        normalized = frame.copy().reset_index(drop=True)
        if "factor_name" not in normalized.columns:
            normalized.insert(0, "factor_name", spec.table_name)
        else:
            normalized["factor_name"] = normalized["factor_name"].fillna(spec.table_name).astype(str)
        frames.append(normalized)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@dataclass(frozen=True)
class CorrelationScreenerSpec:
    evaluation_source: str = "factor"
    method: str = "greedy"
    threshold: float = 0.75
    score_field: str = "fitness"
    ic_period: str | int | pd.Timedelta | None = None


@dataclass(frozen=True)
class CorrelationScreenerResult:
    spec: CorrelationScreenerSpec
    factor_specs: tuple[FactorSpec, ...]
    screened_at: pd.Timestamp
    summary: pd.DataFrame
    selection_summary: pd.DataFrame
    return_series: dict[str, pd.Series]
    return_panel: pd.DataFrame
    correlation_matrix: pd.DataFrame

    @property
    def selected_factor_names(self) -> list[str]:
        frame = self.selection_summary
        if frame.empty or "selected" not in frame.columns or "factor_name" not in frame.columns:
            return []
        return frame.loc[frame["selected"].fillna(False), "factor_name"].astype(str).tolist()

    @property
    def selected_factor_specs(self) -> list[FactorSpec]:
        spec_map = {spec.table_name: spec for spec in self.factor_specs}
        return [spec_map[name] for name in self.selected_factor_names if name in spec_map]

    @property
    def rejected_factor_names(self) -> list[str]:
        if self.selection_summary.empty or "selected" not in self.selection_summary.columns:
            return []
        frame = self.selection_summary.loc[~self.selection_summary["selected"].fillna(False)]
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
            return_start = None if index.empty else index[0].isoformat()
            return_end = None if index.empty else index[-1].isoformat()
        return {
            "screened_at": self.screened_at.isoformat(),
            "factor_count": int(len(self.factor_specs)),
            "evaluation_source": self.spec.evaluation_source,
            "method": self.spec.method,
            "threshold": self.spec.threshold,
            "score_field": self.spec.score_field,
            "ic_period": None if self.spec.ic_period is None else str(self.spec.ic_period),
            "selected_factor_names": self.selected_factor_names,
            "selected_count": int(len(self.selected_factor_names)),
            "rejected_factor_names": self.rejected_factor_names,
            "return_start": return_start,
            "return_end": return_end,
            "summary_rows": int(len(self.summary)),
            "correlation_rows": int(len(self.correlation_matrix)),
        }


class CorrelationScreener:
    def __init__(
        self,
        spec: CorrelationScreenerSpec,
        *,
        factor_specs: tuple[FactorSpec, ...],
        store: FactorStore | None = None,
    ) -> None:
        self.spec = spec
        self.factor_specs = tuple(factor_specs)
        self.store = store or FactorStore()

    def run(self) -> CorrelationScreenerResult:
        if not self.factor_specs:
            screened_at = pd.Timestamp.now(tz="UTC")
            empty = pd.DataFrame()
            return CorrelationScreenerResult(
                spec=self.spec,
                factor_specs=tuple(),
                screened_at=screened_at,
                summary=empty,
                selection_summary=empty,
                return_series={},
                return_panel=empty,
                correlation_matrix=empty,
            )
        source = str(self.spec.evaluation_source).strip().lower()
        if source not in {"factor", "ic", "return", "returns"}:
            raise ValueError(f"unknown correlation screening evaluation_source: {self.spec.evaluation_source!r}")
        summary = _build_summary_map(self.store, self.factor_specs)
        selection_series_map = _build_series_map(
            self.store,
            self.factor_specs,
            source="ic" if source == "ic" else ("factor" if source == "factor" else "return"),
            ic_period=self.spec.ic_period,
        )
        return_series_map = _build_series_map(
            self.store,
            self.factor_specs,
            source="returns",
        )
        screened_at = pd.Timestamp.now(tz="UTC")
        if summary.empty or not selection_series_map:
            empty = pd.DataFrame()
            return CorrelationScreenerResult(
                spec=self.spec,
                factor_specs=self.factor_specs,
                screened_at=screened_at,
                summary=summary,
                selection_summary=empty,
                return_series=return_series_map,
                return_panel=empty,
                correlation_matrix=empty,
            )

        data_map = selection_series_map
        correlation_matrix = _build_correlation_matrix(data_map)

        metric_frame = summary.copy()
        if "factor_name" not in metric_frame.columns:
            metric_frame["factor_name"] = [spec.table_name for spec in self.factor_specs[: len(metric_frame)]]

        score_field = self.spec.score_field
        if score_field not in metric_frame.columns:
            score_field = "fitness" if "fitness" in metric_frame.columns else metric_frame.columns[-1]
        scores = (
            metric_frame.set_index("factor_name")[score_field].astype(float).abs().to_dict()
            if "factor_name" in metric_frame.columns and score_field in metric_frame.columns
            else {name: 1.0 for name in data_map}
        )

        method = str(self.spec.method).strip().lower()
        selected_names: list[str]
        if method in {"greedy", "default", "correlation", "non_redundant"}:
            selected_names = _select_from_correlation_matrix(correlation_matrix, scores, threshold=float(self.spec.threshold), comparator="max")
        elif method in {"average", "avg", "mean", "mean_corr", "average_corr"}:
            selected_names = _select_from_correlation_matrix(correlation_matrix, scores, threshold=float(self.spec.threshold), comparator="mean")
        elif method in {"cluster", "clustered", "cluster_representative"}:
            selected_names = select_cluster_representatives_from_correlation_matrix(correlation_matrix, scores, threshold=float(self.spec.threshold))
        elif method in {"graph", "independent_set", "max_independent_set", "graph_prune"}:
            selected_names = select_by_graph_independent_set_from_correlation_matrix(correlation_matrix, scores, threshold=float(self.spec.threshold))
        else:
            raise ValueError(f"unknown correlation screening method: {self.spec.method!r}")

        selection_summary = summary.copy()
        if not selection_summary.empty and "factor_name" in selection_summary.columns:
            selection_summary["selected"] = selection_summary["factor_name"].astype(str).isin(selected_names)
            if score_field in selection_summary.columns:
                selection_summary["selected_score"] = selection_summary[score_field]

        selected_return_series = {
            name: return_series_map[name]
            for name in selected_names
            if name in return_series_map
        }
        return_long_frames = [
            pd.DataFrame(
                {
                    "date_": pd.to_datetime(series.index, errors="coerce"),
                    "factor": name,
                    "return": pd.to_numeric(series, errors="coerce").to_numpy(),
                    "return_mode": source,
                }
            ).dropna(subset=["date_", "return"])
            for name, series in selected_return_series.items()
        ]
        return_long = (
            pd.concat(return_long_frames, ignore_index=True).sort_values(["date_", "factor"], kind="stable").reset_index(drop=True)
            if return_long_frames
            else pd.DataFrame(columns=["date_", "factor", "return", "return_mode"])
        )
        return_panel = (
            return_long.pivot_table(index="date_", columns="factor", values="return", aggfunc="last").sort_index()
            if not return_long.empty
            else pd.DataFrame()
        )
        if not return_panel.empty:
            return_panel.index = pd.to_datetime(return_panel.index, errors="coerce")
            return_panel = return_panel.loc[~return_panel.index.isna()].sort_index()

        if len(selected_return_series) > 1:
            correlation_matrix = pd.DataFrame(selected_return_series).dropna().corr()

        return CorrelationScreenerResult(
            spec=self.spec,
            factor_specs=self.factor_specs,
            screened_at=screened_at,
            summary=summary.reset_index(drop=True),
            selection_summary=selection_summary.reset_index(drop=True),
            return_series=selected_return_series,
            return_panel=return_panel,
            correlation_matrix=correlation_matrix,
        )


def run_correlation_screener(
    spec: CorrelationScreenerSpec,
    factor_specs: Iterable[FactorSpec],
    *,
    store: FactorStore | None = None,
) -> CorrelationScreenerResult:
    resolved_specs = tuple(factor_specs)
    return CorrelationScreener(
        spec,
        factor_specs=resolved_specs,
        store=store,
    ).run()


def run_ic_correlation_screener(
    spec: CorrelationScreenerSpec,
    factor_specs: Iterable[FactorSpec],
    *,
    store: FactorStore | None = None,
) -> CorrelationScreenerResult:
    ic_spec = CorrelationScreenerSpec(
        evaluation_source="ic",
        method=spec.method,
        threshold=spec.threshold,
        score_field=spec.score_field,
        ic_period=spec.ic_period,
    )
    return run_correlation_screener(ic_spec, factor_specs, store=store)


__all__ = [
    "CorrelationScreener",
    "CorrelationScreenerResult",
    "CorrelationScreenerSpec",
    "run_correlation_screener",
    "run_ic_correlation_screener",
]
