from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from tiger_factors.factor_store import TigerFactorLibrary
from tiger_factors.factor_evaluation.utils import period_to_label
from tiger_factors.factor_evaluation.performance import factor_returns
from tiger_factors.factor_evaluation.performance import mean_return_by_quantile
from tiger_factors.factor_evaluation.utils import get_clean_factor_and_forward_returns as tiger_clean_factor_and_forward_returns
from tiger_factors.factor_store import FactorStore
from tiger_factors.factor_store import FactorSpec
from tiger_factors.multifactor_evaluation.allocation import LongShortReturnConfig
from tiger_factors.multifactor_evaluation.allocation import compute_factor_long_short_returns
from tiger_factors.multifactor_evaluation._inputs import coerce_factor_series
from tiger_factors.multifactor_evaluation.screening import FactorMetricFilterConfig
from tiger_factors.multifactor_evaluation.screening import screen_factor_metrics
from tiger_factors.multifactor_evaluation.selection import select_non_redundant_factors


def _coerce_factor_names(factor_names: Iterable[str]) -> tuple[str, ...]:
    names = [str(name).strip() for name in factor_names if str(name).strip()]
    if not names:
        raise ValueError("factor_names must not be empty")
    return tuple(names)


def _normalize_return_series(series: pd.Series, *, factor_name: str) -> pd.Series:
    cleaned = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        return cleaned
    cleaned.index = pd.to_datetime(cleaned.index, errors="coerce")
    cleaned = cleaned[~cleaned.index.isna()].sort_index()
    cleaned.name = factor_name
    return cleaned


def _resolve_period(config: "TigerScreenerSpec") -> int | str | pd.Timedelta:
    if config.preferred_return_period is not None:
        return config.preferred_return_period
    return 1


def _pick_return_column(frame: pd.DataFrame, *, preferred_period: str | int | pd.Timedelta | None = None) -> str:
    if frame.empty:
        raise ValueError("return frame is empty")

    columns = [str(column) for column in frame.columns]
    lookup = {str(column): column for column in frame.columns}

    if preferred_period is not None:
        preferred_label = period_to_label(preferred_period)
        if preferred_label in lookup:
            return lookup[preferred_label]
        if str(preferred_period) in lookup:
            return lookup[str(preferred_period)]

    for candidate in ("long_short", "long_short_returns", "factor_portfolio_returns", "returns", "return", "1D"):
        if candidate in lookup:
            return lookup[candidate]

    if len(frame.columns) == 1:
        return frame.columns[0]

    numeric_columns = [
        column
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
    ]
    if numeric_columns:
        return numeric_columns[0]

    raise KeyError(
        "Could not infer a return column. "
        f"Available columns: {columns!r}"
    )


def _top_quantile_series_from_mean_returns(
    factor_data: pd.DataFrame,
    *,
    selected_period: str | int | pd.Timedelta,
) -> pd.Series | None:
    mean_ret, _ = mean_return_by_quantile(
        factor_data,
        by_date=True,
        by_group=False,
        demeaned=False,
    )
    period_label = period_to_label(selected_period)
    if period_label not in mean_ret.columns:
        return None
    frame = mean_ret[period_label].unstack("factor_quantile").sort_index()
    if frame.empty:
        return None
    quantile_cols = pd.Index(frame.columns)
    if quantile_cols.empty:
        return None
    top_quantile = pd.to_numeric(quantile_cols, errors="coerce").max()
    if pd.isna(top_quantile):
        top_quantile = quantile_cols[-1]
    series = pd.to_numeric(frame[top_quantile], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    series.index = pd.to_datetime(series.index, errors="coerce")
    series = series[~series.index.isna()].sort_index()
    series.name = "long_only"
    return series


def _build_factor_data(
    factor_panel: pd.DataFrame,
    price_panel: pd.DataFrame,
    *,
    selected_period: str | int | pd.Timedelta,
) -> Any:
    return tiger_clean_factor_and_forward_returns(
        factor=coerce_factor_series(factor_panel),
        prices=price_panel,
        quantiles=5,
        periods=(selected_period,),
        filter_zscore=20,
        max_loss=0.35,
        cumulative_returns=True,
    )


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


@dataclass(frozen=True)
class TigerScreenerSpec:
    factor_names: tuple[str, ...]
    provider: str = "tiger"
    region: str = "us"
    sec_type: str = "stock"
    freq: str = "1d"
    variant: str | None = None
    price_panel: pd.DataFrame | None = None
    summary_section: str = "summary"
    returns_section: str = "returns"
    summary_table_name: str | None = None
    return_table_name: str = "factor_portfolio_returns"
    preferred_return_period: str | int | pd.Timedelta | None = None
    return_modes: tuple[str, ...] = ("long_short", "long_only")
    selection_threshold: float | None = 0.75
    selection_score_field: str = "fitness"
    screening_config: FactorMetricFilterConfig = field(default_factory=FactorMetricFilterConfig)

    def normalized_factor_names(self) -> tuple[str, ...]:
        return _coerce_factor_names(self.factor_names)

    def factor_spec(self, factor_name: str) -> FactorSpec:
        return FactorSpec(
            provider=self.provider,
            region=self.region,
            sec_type=self.sec_type,
            freq=self.freq,
            table_name=str(factor_name).strip().lower(),
            variant=self.variant,
        )


@dataclass(frozen=True)
class TigerScreenerResult:
    spec: TigerScreenerSpec
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
    def screened_factor_names(self) -> list[str]:
        if self.summary.empty or "factor_name" not in self.summary.columns:
            return []
        return self.summary["factor_name"].astype(str).tolist()

    @property
    def rejected_factor_names(self) -> list[str]:
        if self.summary.empty or "usable" not in self.summary.columns:
            return []
        frame = self.summary.loc[~self.summary["usable"].fillna(False)]
        if "factor_name" not in frame.columns:
            return []
        return frame["factor_name"].astype(str).tolist()

    @property
    def time_range(self) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        if self.return_panel.empty:
            return None, None
        index = pd.DatetimeIndex(self.return_panel.index).dropna().sort_values()
        if index.empty:
            return None, None
        return index[0], index[-1]

    def to_summary(self) -> dict[str, Any]:
        start, end = self.time_range
        return {
            "screened_at": self.screened_at.isoformat(),
            "factor_count": int(len(self.spec.factor_names)),
            "screened_factor_names": self.screened_factor_names,
            "selected_factor_names": self.selected_factor_names,
            "selected_count": int(len(self.selected_factor_names)),
            "rejected_factor_names": self.rejected_factor_names,
            "missing_return_factors": list(self.missing_return_factors),
            "return_start": None if start is None else start.isoformat(),
            "return_end": None if end is None else end.isoformat(),
            "summary_rows": int(len(self.summary)),
            "return_long_rows": int(len(self.return_long)),
            "return_modes": list(self.spec.return_modes),
        }


class TigerScreener:
    def __init__(
        self,
        spec: TigerScreenerSpec,
        *,
        store: FactorStore | None = None,
    ) -> None:
        self.spec = spec
        self.store = store or FactorStore()
        self.library = TigerFactorLibrary(
            store=self.store,
            region=self.spec.region,
            sec_type=self.spec.sec_type,
            price_provider=self.spec.provider,
            verbose=False,
        )

    def _summary_frame(self, factor_name: str) -> pd.DataFrame:
        factor_spec = self.spec.factor_spec(factor_name)
        section = self.store.evaluation.section(factor_spec, self.spec.summary_section)
        if self.spec.summary_table_name is None:
            frame = section.get_table()
        else:
            frame = section.get_table(self.spec.summary_table_name)

        if frame.empty:
            raise ValueError(f"summary table is empty for factor {factor_name!r}")
        if len(frame) != 1:
            raise ValueError(
                "summary table must contain exactly one row per factor; "
                f"got {len(frame)} rows for {factor_name!r}"
            )

        normalized = frame.copy().reset_index(drop=True)
        if "factor_name" not in normalized.columns:
            normalized.insert(0, "factor_name", factor_name)
        else:
            normalized["factor_name"] = normalized["factor_name"].fillna(factor_name).astype(str)
        if "factor_name" not in normalized.columns:
            normalized.insert(0, "factor_name", factor_name)
        return normalized

    def _stored_long_short_series(self, factor_name: str) -> pd.Series | None:
        factor_spec = self.spec.factor_spec(factor_name)
        section = self.store.evaluation.section(factor_spec, self.spec.returns_section)
        try:
            return_frame = section.get_table(self.spec.return_table_name)
        except FileNotFoundError:
            return None
        if isinstance(return_frame, pd.Series):
            return _normalize_return_series(return_frame, factor_name=factor_name)
        if not isinstance(return_frame, pd.DataFrame) or return_frame.empty:
            return None
        column = _pick_return_column(return_frame, preferred_period=self.spec.preferred_return_period)
        series = return_frame[column]
        if isinstance(series, pd.DataFrame):
            series = series.squeeze(axis=1)
        return _normalize_return_series(series, factor_name=factor_name)

    def _stored_long_only_series(self, factor_name: str) -> pd.Series | None:
        factor_spec = self.spec.factor_spec(factor_name)
        section = self.store.evaluation.section(factor_spec, self.spec.returns_section)
        candidates = ("mean_return_by_quantile_by_date", "mean_return_by_quantile")
        period_label = period_to_label(_resolve_period(self.spec))
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
                return _normalize_return_series(quantile_frame[top_quantile], factor_name=factor_name)
            if isinstance(frame.columns, pd.MultiIndex) and "factor_quantile" in frame.columns.names:
                try:
                    level = frame.columns.names.index("factor_quantile")
                    columns = frame.columns.get_level_values(level)
                    if period_label in frame.columns:
                        quantile_frame = frame[period_label].copy()
                        if isinstance(quantile_frame, pd.Series):
                            return _normalize_return_series(quantile_frame, factor_name=factor_name)
                except Exception:
                    continue
        return None

    def _return_modes(self, factor_name: str) -> dict[str, pd.Series]:
        factor_panel = self.library.load_factor_panel(
            factor_name=factor_name,
            provider=self.spec.provider,
            freq=self.spec.freq,
            variant=self.spec.variant,
            engine="pandas",
        )
        return_modes: dict[str, pd.Series] = {}
        if isinstance(factor_panel, pd.DataFrame) and not factor_panel.empty and self.spec.price_panel is not None:
            selected_period = _resolve_period(self.spec)
            try:
                prepared = _build_factor_data(
                    factor_panel,
                    self.spec.price_panel,
                    selected_period=selected_period,
                )
                if "long_short" in self.spec.return_modes:
                    long_short = factor_returns(
                        prepared.factor_data,
                        demeaned=True,
                        group_adjust=False,
                        equal_weight=False,
                    )
                    period_label = period_to_label(selected_period)
                    if period_label in long_short.columns:
                        return_modes["long_short"] = _normalize_return_series(
                            long_short[period_label],
                            factor_name=factor_name,
                        )
                    else:
                        series = compute_factor_long_short_returns(
                            factor_panel,
                            self.spec.price_panel,
                            config=LongShortReturnConfig(
                                periods=(selected_period,),
                                selected_period=selected_period,
                            ),
                        )
                        if isinstance(series, pd.Series) and not series.empty:
                            return_modes["long_short"] = _normalize_return_series(series, factor_name=factor_name)
                if "long_only" in self.spec.return_modes:
                    long_only = _top_quantile_series_from_mean_returns(
                        prepared.factor_data,
                        selected_period=selected_period,
                    )
                    if long_only is not None and not long_only.empty:
                        return_modes["long_only"] = _normalize_return_series(long_only, factor_name=factor_name)
            except Exception:
                pass

        if "long_short" in self.spec.return_modes and "long_short" not in return_modes:
            stored = self._stored_long_short_series(factor_name)
            if stored is not None and not stored.empty:
                return_modes["long_short"] = stored
        if "long_only" in self.spec.return_modes and "long_only" not in return_modes:
            stored = self._stored_long_only_series(factor_name)
            if stored is not None and not stored.empty:
                return_modes["long_only"] = stored
        return return_modes

    def _build_return_long_table(self, selected_names: list[str]) -> tuple[dict[str, pd.Series], pd.DataFrame, list[str]]:
        return_series: dict[str, pd.Series] = {}
        missing_return_factors: list[str] = []
        return_long_frames: list[pd.DataFrame] = []

        for factor_name in selected_names:
            modes = self._return_modes(factor_name)
            if not modes:
                missing_return_factors.append(factor_name)
                continue
            for mode, series in modes.items():
                if series is None or series.empty:
                    continue
                key = f"{factor_name}:{mode}"
                return_series[key] = series
                return_long_frames.append(_series_to_long_frame(series, factor_name=factor_name, return_mode=mode))
            if factor_name not in {key.split(":")[0] for key in return_series}:
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

    def run(self) -> TigerScreenerResult:
        factor_names = self.spec.normalized_factor_names()
        summary_frames: list[pd.DataFrame] = []
        for factor_name in factor_names:
            summary_frames.append(self._summary_frame(factor_name))

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
            factor_panels = self.library.load_factor_panels(
                factor_names=screened_names,
                provider=self.spec.provider,
                freq=self.spec.freq,
                variant=self.spec.variant,
            )
            factor_panels = {name: panel for name, panel in factor_panels.items() if isinstance(panel, pd.DataFrame) and not panel.empty}

        if self.spec.selection_threshold is not None and len(factor_panels) > 1:
            score_field = self.spec.selection_score_field
            if score_field not in screened_summary.columns:
                score_field = "fitness" if "fitness" in screened_summary.columns else screened_summary.columns[-1]
            score_map = (
                screened_summary.set_index("factor_name")[score_field].astype(float).abs().to_dict()
                if "factor_name" in screened_summary.columns and score_field in screened_summary.columns
                else {name: 1.0 for name in factor_panels}
            )
            selected_names = select_non_redundant_factors(
                factor_panels,
                score_map,
                threshold=float(self.spec.selection_threshold),
            )
        else:
            selected_names = list(factor_panels.keys()) if factor_panels else screened_names

        if not selection_summary.empty and "factor_name" in selection_summary.columns:
            selection_summary = selection_summary.copy()
            selection_summary["selected"] = selection_summary["factor_name"].astype(str).isin(selected_names)
            if "selected_score" not in selection_summary.columns:
                score_field = self.spec.selection_score_field
                if score_field not in selection_summary.columns:
                    score_field = "fitness" if "fitness" in selection_summary.columns else None
                if score_field is not None:
                    selection_summary["selected_score"] = selection_summary[score_field]

        return_series, return_long, missing_return_factors = self._build_return_long_table(selected_names)

        primary_mode = "long_short" if "long_short" in self.spec.return_modes else (self.spec.return_modes[0] if self.spec.return_modes else "long_short")
        primary_series: dict[str, pd.Series] = {
            key.split(":", 1)[0]: series
            for key, series in return_series.items()
            if key.endswith(f":{primary_mode}")
        }
        return_panel = pd.DataFrame(primary_series).sort_index() if primary_series else pd.DataFrame()
        if not return_panel.empty:
            return_panel.index = pd.to_datetime(return_panel.index, errors="coerce")
            return_panel = return_panel.loc[~return_panel.index.isna()].sort_index()

        if not return_long.empty and "factor" in return_long.columns:
            range_table = return_long.groupby("factor")["date_"].agg(["min", "max"]).rename(columns={"min": "return_start", "max": "return_end"})
            if not screened_summary.empty and "factor_name" in screened_summary.columns:
                screened_summary = screened_summary.merge(range_table, how="left", left_on="factor_name", right_index=True)
            if not selection_summary.empty and "factor_name" in selection_summary.columns:
                selection_summary = selection_summary.merge(range_table, how="left", left_on="factor_name", right_index=True)
                selection_summary["screened_at"] = screened_at.isoformat()

        return TigerScreenerResult(
            spec=self.spec,
            screened_at=screened_at,
            summary=screened_summary.reset_index(drop=True) if not screened_summary.empty else screened_summary,
            selection_summary=selection_summary.reset_index(drop=True) if not selection_summary.empty else selection_summary,
            return_long=return_long,
            return_panel=return_panel,
            return_series=return_series,
            missing_return_factors=tuple(missing_return_factors),
        )


def run_tiger_screener(
    factor_names: Iterable[str],
    *,
    store: FactorStore | None = None,
    screening_config: FactorMetricFilterConfig | None = None,
    provider: str = "tiger",
    region: str = "us",
    sec_type: str = "stock",
    freq: str = "1d",
    variant: str | None = None,
    summary_section: str = "summary",
    returns_section: str = "returns",
    summary_table_name: str | None = None,
    return_table_name: str = "factor_portfolio_returns",
    preferred_return_period: str | int | pd.Timedelta | None = None,
    return_modes: tuple[str, ...] = ("long_short", "long_only"),
    selection_threshold: float | None = 0.75,
    selection_score_field: str = "fitness",
) -> TigerScreenerResult:
    spec = TigerScreenerSpec(
        factor_names=_coerce_factor_names(factor_names),
        provider=provider,
        region=region,
        sec_type=sec_type,
        freq=freq,
        variant=variant,
        summary_section=summary_section,
        returns_section=returns_section,
        summary_table_name=summary_table_name,
        return_table_name=return_table_name,
        preferred_return_period=preferred_return_period,
        return_modes=return_modes,
        selection_threshold=selection_threshold,
        selection_score_field=selection_score_field,
        screening_config=screening_config or FactorMetricFilterConfig(),
    )
    return TigerScreener(spec, store=store).run()


__all__ = [
    "TigerScreener",
    "TigerScreenerResult",
    "TigerScreenerSpec",
    "FactorScreener",
    "FactorScreenerResult",
    "FactorScreenerSpec",
    "run_tiger_screener",
    "run_factor_screener",
]

FactorScreener = TigerScreener
FactorScreenerResult = TigerScreenerResult
FactorScreenerSpec = TigerScreenerSpec
run_factor_screener = run_tiger_screener
