from __future__ import annotations

import json
import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from tiger_factors.factor_evaluation.core import FactorEvaluation
from tiger_factors.factor_evaluation.core import evaluate_factor_panel
from tiger_factors.factor_evaluation.input import build_tiger_evaluation_input
from tiger_factors.factor_evaluation.input import SeriesLike
from tiger_factors.factor_evaluation.input import load_group_labels
from tiger_factors.factor_evaluation.input import load_series
from tiger_factors.factor_evaluation.input import prewarm_group_labels_cache
from tiger_factors.factor_evaluation.input import TigerEvaluationInput
from tiger_factors.factor_evaluation.horizon import HoldingPeriodAnalyzer
from tiger_factors.factor_evaluation.horizon import summarize_best_horizon as summarize_best_horizon_result
from tiger_factors.factor_evaluation.native_tears import FactorTearSheetResult
from tiger_factors.factor_evaluation.native_tears import create_native_full_tear_sheet
from tiger_factors.factor_evaluation.tears import TigerEvaluationSheet
from tiger_factors.factor_evaluation.tears import prepare_shared_artifacts
from tiger_factors.factor_evaluation.tears import _summary_row_from_evaluation
from tiger_factors.factor_evaluation.tears import create_event_returns_tear_sheet
from tiger_factors.factor_evaluation.tears import create_event_study_tear_sheet
from tiger_factors.factor_evaluation.tears import create_full_tear_sheet
from tiger_factors.factor_evaluation.tears import create_information_tear_sheet
from tiger_factors.factor_evaluation.tears import create_returns_tear_sheet
from tiger_factors.factor_evaluation.tears import create_summary_tear_sheet
from tiger_factors.factor_evaluation.tears import create_turnover_tear_sheet
from tiger_factors.factor_evaluation.utils import TigerFactorData
from tiger_factors.factor_evaluation.utils import get_clean_factor_and_forward_returns
from tiger_factors.factor_evaluation.utils import PeriodLike
from tiger_factors.factor_store.store import FactorStore
from tiger_factors.factor_store.spec import FactorSpec
from tiger_factors.report_paths import report_output_root_for


@dataclass(frozen=True)
class ReportBundleSummary:
    factor_column: str
    report_dir: str
    full_manifest: str
    horizon_manifest: str | None
    figure_count: int
    table_count: int
    factor_rows: int | None
    price_rows: int | None
    evaluation: dict[str, Any] | None
    horizon_summary: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_column": self.factor_column,
            "report_dir": self.report_dir,
            "full_manifest": self.full_manifest,
            "horizon_manifest": self.horizon_manifest,
            "figure_count": self.figure_count,
            "table_count": self.table_count,
            "factor_rows": self.factor_rows,
            "price_rows": self.price_rows,
            "evaluation": self.evaluation,
            "horizon_summary": self.horizon_summary,
        }
@dataclass
class FactorEvaluationEngine:
    factor_frame: pd.DataFrame
    price_frame: pd.DataFrame
    factor_column: str
    spec: FactorSpec | None = None
    date_column: str = "date_"
    code_column: str = "code"
    price_column: str = "close"
    forward_days: int = 1
    benchmark_frame: SeriesLike | None = None
    benchmark_date_column: str = "date_"
    benchmark_value_column: str = "returns"
    group_labels: SeriesLike | None = None
    group_date_column: str = "date_"
    group_code_column: str = "code"
    group_value_column: str = "group"
    factor_store: FactorStore | None = None
    group_labels_cache: bool = True
    _factor_store: FactorStore | None = field(default=None, init=False, repr=False)
    _prepared_input: TigerEvaluationInput | None = field(default=None, init=False, repr=False)
    _factor_data_cache: dict[tuple[Any, ...], TigerFactorData] = field(default_factory=dict, init=False, repr=False)
    _benchmark_returns: pd.Series | None = field(default=None, init=False, repr=False)
    _group_frame: pd.DataFrame | pd.Series | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.group_labels_cache:
            return
        if isinstance(self.group_labels, (str, Path)):
            prewarm_group_labels_cache(
                [self.group_labels],
                date_column=self.group_date_column,
                code_column=self.group_code_column,
                value_column=self.group_value_column,
            )

    def _default_report_root(self) -> Path:
        if self.spec is not None:
            parts = [
                "factor_evaluation",
                self.spec.provider.lower(),
                self.spec.region.lower(),
                self.spec.sec_type.lower(),
                self.spec.freq.lower(),
            ]
            if getattr(self.spec, "group", None) is not None:
                parts.append(str(self.spec.group))
            parts.append(self.spec.data_stem())
            return report_output_root_for(*parts)
        return report_output_root_for("factor_evaluation", self.factor_column or "factor")

    def _resolve_report_section_dir(self, name: str, output_dir: str | Path | None) -> Path:
        if output_dir is not None:
            return Path(output_dir)
        return self._default_report_root() / name

    def _resolve_report_root(self, output_dir: str | Path | None) -> Path:
        if output_dir is not None:
            return Path(output_dir)
        return self._default_report_root()

    @staticmethod
    def _full_report_cache_key(
        *,
        periods: tuple[PeriodLike, ...],
        quantiles: int | list[float] | None,
        bins: int | list[float] | None,
        filter_zscore: float | None,
        max_loss: float | None,
        binning_by_group: bool,
        zero_aware: bool,
        cumulative_returns: bool,
        groupby_labels: dict[Any, str] | None,
        long_short: bool,
        group_neutral: bool,
        by_group: bool,
        turnover_periods: tuple[PeriodLike, ...] | None,
        avgretplot: tuple[int, int],
        include_horizon: bool,
        horizons: tuple[int, ...] | list[int],
        horizon_quantiles: int,
        horizon_periods_per_year: int,
        horizon_long_short_pct: float,
    ) -> tuple[Any, ...]:
        return (
            tuple(periods),
            tuple(quantiles) if isinstance(quantiles, list) else quantiles,
            tuple(bins) if isinstance(bins, list) else bins,
            filter_zscore,
            max_loss,
            binning_by_group,
            zero_aware,
            cumulative_returns,
            tuple(sorted(groupby_labels.items())) if groupby_labels else None,
            long_short,
            group_neutral,
            by_group,
            tuple(turnover_periods) if turnover_periods is not None else None,
            tuple(avgretplot),
            include_horizon,
            tuple(horizons),
            horizon_quantiles,
            horizon_periods_per_year,
            horizon_long_short_pct,
        )

    @staticmethod
    def _summary_cache_key(
        *,
        periods: tuple[PeriodLike, ...],
        quantiles: int | list[float] | None,
        bins: int | list[float] | None,
        filter_zscore: float | None,
        max_loss: float | None,
        binning_by_group: bool,
        zero_aware: bool,
        cumulative_returns: bool,
        groupby_labels: dict[Any, str] | None,
        long_short: bool,
        group_neutral: bool,
        by_group: bool,
    ) -> tuple[Any, ...]:
        return (
            tuple(periods),
            tuple(quantiles) if isinstance(quantiles, list) else quantiles,
            tuple(bins) if isinstance(bins, list) else bins,
            filter_zscore,
            max_loss,
            binning_by_group,
            zero_aware,
            cumulative_returns,
            tuple(sorted(groupby_labels.items())) if groupby_labels else None,
            long_short,
            group_neutral,
            by_group,
        )

    def _get_full_report_artifacts(
        self,
        *,
        periods: tuple[PeriodLike, ...],
        quantiles: int | list[float] | None,
        bins: int | list[float] | None,
        filter_zscore: float | None,
        max_loss: float | None,
        binning_by_group: bool,
        zero_aware: bool,
        cumulative_returns: bool,
        groupby_labels: dict[Any, str] | None,
        long_short: bool,
        group_neutral: bool,
        by_group: bool,
        turnover_periods: tuple[PeriodLike, ...] | None,
        avgretplot: tuple[int, int],
        include_horizon: bool,
        horizons: tuple[int, ...] | list[int],
        horizon_quantiles: int,
        horizon_periods_per_year: int,
        horizon_long_short_pct: float,
    ) -> dict[str, Any]:
        factor_data = self.get_clean_factor_and_forward_returns(
            periods=periods,
            quantiles=quantiles,
            bins=bins,
            filter_zscore=filter_zscore,
            max_loss=max_loss,
            binning_by_group=binning_by_group,
            zero_aware=zero_aware,
            cumulative_returns=cumulative_returns,
            groupby_labels=groupby_labels,
        )
        artifacts = prepare_shared_artifacts(
            factor_data,
            long_short=long_short,
            group_neutral=group_neutral,
            by_group=by_group,
            turnover_periods=turnover_periods,
            avgretplot=avgretplot,
        )
        artifacts["factor_data"] = factor_data
        artifacts["summary_cache_key"] = self._summary_cache_key(
            periods=periods,
            quantiles=quantiles,
            bins=bins,
            filter_zscore=filter_zscore,
            max_loss=max_loss,
            binning_by_group=binning_by_group,
            zero_aware=zero_aware,
            cumulative_returns=cumulative_returns,
            groupby_labels=groupby_labels,
            long_short=long_short,
            group_neutral=group_neutral,
            by_group=by_group,
        )
        artifacts["horizon_config"] = {
            "include_horizon": include_horizon,
            "horizons": list(horizons),
            "quantiles": horizon_quantiles,
            "periods_per_year": horizon_periods_per_year,
            "long_short_pct": horizon_long_short_pct,
        }
        if include_horizon:
            horizon_result = self.analyze_horizons(
                list(horizons),
                quantiles=horizon_quantiles,
                periods_per_year=horizon_periods_per_year,
                long_short_pct=horizon_long_short_pct,
            )
            artifacts["horizon_result"] = horizon_result
            artifacts["horizon_summary"] = summarize_best_horizon_result(horizon_result)
        else:
            artifacts["horizon_result"] = None
            artifacts["horizon_summary"] = None
        return artifacts

    def load_summary(self, *, output_dir: str | Path | None = None) -> pd.DataFrame:
        path = self._resolve_report_section_dir("summary", output_dir) / "summary.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        return pd.read_parquet(path)

    def load_section_table(
        self,
        section: str,
        table_name: str | None = None,
        *,
        output_dir: str | Path | None = None,
    ) -> pd.DataFrame:
        section_dir = self._resolve_report_section_dir(section, output_dir)
        if table_name is None:
            default_file = section_dir / f"{section}.parquet"
            if default_file.exists():
                return pd.read_parquet(default_file)
            parquet_files = sorted(section_dir.glob("*.parquet"))
            if len(parquet_files) == 1:
                return pd.read_parquet(parquet_files[0])
            raise FileNotFoundError(
                f"Unable to infer table for section '{section}'. "
                f"Available parquet files: {[path.name for path in parquet_files]!r}"
            )

        candidate = section_dir / f"{table_name}.parquet"
        if candidate.exists():
            return pd.read_parquet(candidate)
        if "/" not in table_name and "\\" not in table_name:
            matches = sorted(section_dir.rglob(f"{table_name}.parquet"))
            if len(matches) == 1:
                return pd.read_parquet(matches[0])
            if len(matches) > 1:
                raise FileNotFoundError(
                    f"Ambiguous table '{table_name}' under section '{section}': "
                    f"{[path.as_posix() for path in matches]!r}"
                )
        raise FileNotFoundError(candidate)

    @staticmethod
    def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return target

    @staticmethod
    def _created_at() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _read_json(path: str | Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def _input_data(self) -> TigerEvaluationInput:
        if self._prepared_input is None:
            self._prepared_input = build_tiger_evaluation_input(
                factor_frame=self.factor_frame,
                price_frame=self.price_frame,
                factor_column=self.factor_column,
                date_column=self.date_column,
                code_column=self.code_column,
                price_column=self.price_column,
                forward_days=self.forward_days,
            )
        return self._prepared_input

    def factor_panel(self) -> pd.DataFrame:
        return self._input_data().factor_panel

    def price_panel(self) -> pd.DataFrame:
        return self._input_data().price_panel

    def forward_returns(self) -> pd.DataFrame:
        return self._input_data().forward_returns

    def benchmark_returns(self) -> pd.Series | None:
        if self.benchmark_frame is None:
            return None
        if self._benchmark_returns is None:
            self._benchmark_returns = load_series(
                self.benchmark_frame,
                date_column=self.benchmark_date_column,
                value_column=self.benchmark_value_column,
            )
        return self._benchmark_returns

    def group_frame(self) -> pd.DataFrame | pd.Series | None:
        if self.group_labels is None:
            return None
        if self._group_frame is None:
            self._group_frame = load_group_labels(
                self.group_labels,
                date_column=self.group_date_column,
                code_column=self.group_code_column,
                value_column=self.group_value_column,
                use_cache=self.group_labels_cache,
            )
        return self._group_frame

    def _get_factor_store(self) -> FactorStore:
        if self._factor_store is None:
            self._factor_store = self.factor_store or FactorStore()
        return self._factor_store

    def get_clean_factor_and_forward_returns(
        self,
        *,
        periods: tuple[PeriodLike, ...] = (1, 5, 10),
        quantiles: int | list[float] | None = 5,
        bins: int | list[float] | None = None,
        filter_zscore: float | None = 20,
        max_loss: float | None = 0.35,
        binning_by_group: bool = False,
        zero_aware: bool = False,
        cumulative_returns: bool = True,
        groupby_labels: dict[Any, str] | None = None,
    ) -> TigerFactorData:
        cache_key = (
            tuple(periods),
            tuple(quantiles) if isinstance(quantiles, list) else quantiles,
            tuple(bins) if isinstance(bins, list) else bins,
            filter_zscore,
            max_loss,
            binning_by_group,
            zero_aware,
            cumulative_returns,
            tuple(sorted(groupby_labels.items())) if groupby_labels else None,
        )
        if cache_key not in self._factor_data_cache:
            factor_data = get_clean_factor_and_forward_returns(
                factor_frame=self.factor_frame,
                price_frame=self.price_frame,
                factor_column=self.factor_column,
                groupby=self.group_frame(),
                binning_by_group=binning_by_group,
                date_column=self.date_column,
                code_column=self.code_column,
                price_column=self.price_column,
                periods=periods,
                quantiles=quantiles,
                bins=bins,
                filter_zscore=filter_zscore,
                groupby_labels=groupby_labels,
                max_loss=max_loss,
                zero_aware=zero_aware,
                cumulative_returns=cumulative_returns,
            )
            self._factor_data_cache[cache_key] = factor_data
        return self._factor_data_cache[cache_key]

    def evaluate(
        self,
        *,
        save: bool = False,
        force_updated: bool = False,
        **kwargs: Any,
    ) -> FactorEvaluation:
        benchmark_returns = kwargs.pop("benchmark_returns", self.benchmark_returns())
        evaluation = evaluate_factor_panel(
            self.factor_panel(),
            self.forward_returns(),
            benchmark_returns=benchmark_returns,
            **kwargs,
        )
        if save:
            if self.spec is None:
                raise ValueError("evaluation spec is required when save=True")
            self._get_factor_store().evaluation_store.save_summary(
                evaluation,
                spec=self.spec,
                force_updated=force_updated,
            )
        return evaluation

    @staticmethod
    def _summary_frame_from_report(report: TigerEvaluationSheet) -> pd.DataFrame:
        summary = report.payload.get("summary") if report.payload is not None else None
        if isinstance(summary, pd.DataFrame):
            return summary.copy()
        if isinstance(summary, pd.Series):
            return summary.to_frame().T.reset_index(drop=True)
        if isinstance(summary, dict):
            return pd.DataFrame([summary])
        if report.evaluation is not None:
            return pd.DataFrame([report.evaluation.__dict__])
        return pd.DataFrame()

    def summary(
        self,
        *,
        output_dir: str | Path | None = None,
        periods: tuple[PeriodLike, ...] = (1, 5, 10),
        quantiles: int | list[float] | None = 5,
        bins: int | list[float] | None = None,
        filter_zscore: float | None = 20,
        max_loss: float | None = 0.35,
        binning_by_group: bool = False,
        zero_aware: bool = False,
        cumulative_returns: bool = True,
        groupby_labels: dict[Any, str] | None = None,
        long_short: bool = True,
        group_neutral: bool = False,
        highlights: dict[str, Any] | None = None,
        save: bool = True,
        figure: bool = False,
    ) -> pd.DataFrame:
        sheet = self._create_summary_tear_sheet(
            output_dir=output_dir,
            periods=periods,
            quantiles=quantiles,
            bins=bins,
            filter_zscore=filter_zscore,
            max_loss=max_loss,
            binning_by_group=binning_by_group,
            zero_aware=zero_aware,
            cumulative_returns=cumulative_returns,
            groupby_labels=groupby_labels,
            long_short=long_short,
            group_neutral=group_neutral,
            highlights=highlights,
            table=save,
            figure=figure,
        )
        return self._summary_frame_from_report(sheet)

    def _create_summary_tear_sheet(
        self,
        *,
        output_dir: str | Path | None = None,
        periods: tuple[PeriodLike, ...] = (1, 5, 10),
        quantiles: int | list[float] | None = 5,
        bins: int | list[float] | None = None,
        filter_zscore: float | None = 20,
        max_loss: float | None = 0.35,
        binning_by_group: bool = False,
        zero_aware: bool = False,
        cumulative_returns: bool = True,
        groupby_labels: dict[Any, str] | None = None,
        long_short: bool = True,
        group_neutral: bool = False,
        highlights: dict[str, Any] | None = None,
        table: bool = True,
        figure: bool = False,
    ) -> TigerEvaluationSheet:
        factor_data = self.get_clean_factor_and_forward_returns(
            periods=periods,
            quantiles=quantiles,
            bins=bins,
            filter_zscore=filter_zscore,
            max_loss=max_loss,
            binning_by_group=binning_by_group,
            zero_aware=zero_aware,
            cumulative_returns=cumulative_returns,
            groupby_labels=groupby_labels,
        )
        return create_summary_tear_sheet(
            factor_data,
            output_dir=self._resolve_report_section_dir("summary", output_dir),
            long_short=long_short,
            group_neutral=group_neutral,
            table=table,
            figure=figure,
            highlights=highlights,
        )

    def returns(
        self,
        *,
        output_dir: str | Path | None = None,
        periods: tuple[PeriodLike, ...] = (1, 5, 10),
        quantiles: int | list[float] | None = 5,
        bins: int | list[float] | None = None,
        filter_zscore: float | None = 20,
        max_loss: float | None = 0.35,
        binning_by_group: bool = False,
        zero_aware: bool = False,
        cumulative_returns: bool = True,
        groupby_labels: dict[Any, str] | None = None,
        rate_of_ret: bool = True,
        group_neutral: bool = False,
        by_group: bool = False,
        table: bool = True,
        figure: bool = False,
    ) -> TigerEvaluationSheet:
        return self._create_returns_tear_sheet(
            output_dir=output_dir,
            periods=periods,
            quantiles=quantiles,
            bins=bins,
            filter_zscore=filter_zscore,
            max_loss=max_loss,
            binning_by_group=binning_by_group,
            zero_aware=zero_aware,
            cumulative_returns=cumulative_returns,
            groupby_labels=groupby_labels,
            rate_of_ret=rate_of_ret,
            group_neutral=group_neutral,
            by_group=by_group,
            table=table,
            figure=figure,
        )

    def information(
        self,
        *,
        output_dir: str | Path | None = None,
        periods: tuple[PeriodLike, ...] = (1, 5, 10),
        quantiles: int | list[float] | None = 5,
        bins: int | list[float] | None = None,
        filter_zscore: float | None = 20,
        max_loss: float | None = 0.35,
        binning_by_group: bool = False,
        zero_aware: bool = False,
        cumulative_returns: bool = True,
        groupby_labels: dict[Any, str] | None = None,
        group_neutral: bool = False,
        by_group: bool = False,
        table: bool = True,
        figure: bool = False,
    ) -> TigerEvaluationSheet:
        return self._create_information_tear_sheet(
            output_dir=output_dir,
            periods=periods,
            quantiles=quantiles,
            bins=bins,
            filter_zscore=filter_zscore,
            max_loss=max_loss,
            binning_by_group=binning_by_group,
            zero_aware=zero_aware,
            cumulative_returns=cumulative_returns,
            groupby_labels=groupby_labels,
            group_neutral=group_neutral,
            by_group=by_group,
            table=table,
            figure=figure,
        )

    def turnover(
        self,
        *,
        output_dir: str | Path | None = None,
        periods: tuple[PeriodLike, ...] = (1, 5, 10),
        quantiles: int | list[float] | None = 5,
        bins: int | list[float] | None = None,
        filter_zscore: float | None = 20,
        max_loss: float | None = 0.35,
        binning_by_group: bool = False,
        zero_aware: bool = False,
        cumulative_returns: bool = True,
        groupby_labels: dict[Any, str] | None = None,
        turnover_periods: tuple[PeriodLike, ...] | None = None,
        table: bool = True,
        figure: bool = False,
    ) -> TigerEvaluationSheet:
        return self._create_turnover_tear_sheet(
            output_dir=output_dir,
            periods=periods,
            quantiles=quantiles,
            bins=bins,
            filter_zscore=filter_zscore,
            max_loss=max_loss,
            binning_by_group=binning_by_group,
            zero_aware=zero_aware,
            cumulative_returns=cumulative_returns,
            groupby_labels=groupby_labels,
            turnover_periods=turnover_periods,
            table=table,
            figure=figure,
        )

    def event_returns(
        self,
        *,
        output_dir: str | Path | None = None,
        periods: tuple[PeriodLike, ...] = (1, 5, 10),
        quantiles: int | list[float] | None = 5,
        bins: int | list[float] | None = None,
        filter_zscore: float | None = 20,
        max_loss: float | None = 0.35,
        binning_by_group: bool = False,
        zero_aware: bool = False,
        cumulative_returns: bool = True,
        groupby_labels: dict[Any, str] | None = None,
        returns: pd.DataFrame | None = None,
        avgretplot: tuple[int, int] = (5, 15),
        long_short: bool = True,
        group_neutral: bool = False,
        std_bar: bool = True,
        by_group: bool = False,
        table: bool = True,
        figure: bool = False,
    ) -> TigerEvaluationSheet:
        return self._create_event_returns_tear_sheet(
            output_dir=output_dir,
            periods=periods,
            quantiles=quantiles,
            bins=bins,
            filter_zscore=filter_zscore,
            max_loss=max_loss,
            binning_by_group=binning_by_group,
            zero_aware=zero_aware,
            cumulative_returns=cumulative_returns,
            groupby_labels=groupby_labels,
            returns=returns,
            avgretplot=avgretplot,
            long_short=long_short,
            group_neutral=group_neutral,
            std_bar=std_bar,
            by_group=by_group,
            table=table,
            figure=figure,
        )

    def event_study(
        self,
        *,
        output_dir: str | Path | None = None,
        periods: tuple[PeriodLike, ...] = (1, 5, 10),
        quantiles: int | list[float] | None = 5,
        bins: int | list[float] | None = None,
        filter_zscore: float | None = 20,
        max_loss: float | None = 0.35,
        binning_by_group: bool = False,
        zero_aware: bool = False,
        cumulative_returns: bool = True,
        groupby_labels: dict[Any, str] | None = None,
        returns: pd.DataFrame | None = None,
        avgretplot: tuple[int, int] | None = (5, 15),
        rate_of_ret: bool = True,
        n_bars: int = 50,
        long_short: bool = True,
        group_neutral: bool = False,
        by_group: bool = False,
        turnover_periods: tuple[PeriodLike, ...] | None = None,
        table: bool = True,
        figure: bool = False,
    ) -> TigerEvaluationSheet:
        return self._create_event_study_tear_sheet(
            output_dir=output_dir,
            periods=periods,
            quantiles=quantiles,
            bins=bins,
            filter_zscore=filter_zscore,
            max_loss=max_loss,
            binning_by_group=binning_by_group,
            zero_aware=zero_aware,
            cumulative_returns=cumulative_returns,
            groupby_labels=groupby_labels,
            returns=returns,
            avgretplot=avgretplot,
            rate_of_ret=rate_of_ret,
            n_bars=n_bars,
            long_short=long_short,
            group_neutral=group_neutral,
            by_group=by_group,
            turnover_periods=turnover_periods,
            table=table,
            figure=figure,
        )

    def full(
        self,
        *,
        output_dir: str | Path | None = None,
        periods: tuple[PeriodLike, ...] = (1, 5, 10),
        quantiles: int | list[float] | None = 5,
        bins: int | list[float] | None = None,
        filter_zscore: float | None = 20,
        max_loss: float | None = 0.35,
        binning_by_group: bool = False,
        zero_aware: bool = False,
        cumulative_returns: bool = True,
        groupby_labels: dict[Any, str] | None = None,
        long_short: bool = True,
        group_neutral: bool = False,
        by_group: bool = False,
        turnover_periods: tuple[int, ...] | None = None,
        avgretplot: tuple[int, int] = (5, 15),
        include_horizon: bool = True,
        horizons: tuple[int, ...] | list[int] = (1, 3, 5, 10, 20),
        horizon_quantiles: int = 5,
        horizon_periods_per_year: int = 252,
        horizon_long_short_pct: float = 0.2,
        figure: bool = True,
    ) -> TigerEvaluationSheet:
        return self._create_full_tear_sheet(
            output_dir=output_dir,
            periods=periods,
            quantiles=quantiles,
            bins=bins,
            filter_zscore=filter_zscore,
            max_loss=max_loss,
            binning_by_group=binning_by_group,
            zero_aware=zero_aware,
            cumulative_returns=cumulative_returns,
            groupby_labels=groupby_labels,
            long_short=long_short,
            group_neutral=group_neutral,
            by_group=by_group,
            turnover_periods=turnover_periods,
            avgretplot=avgretplot,
            include_horizon=include_horizon,
            horizons=horizons,
            horizon_quantiles=horizon_quantiles,
            horizon_periods_per_year=horizon_periods_per_year,
            horizon_long_short_pct=horizon_long_short_pct,
            figure=figure,
        )

    def _create_returns_tear_sheet(
        self,
        *,
        output_dir: str | Path | None = None,
        periods: tuple[PeriodLike, ...] = (1, 5, 10),
        quantiles: int | list[float] | None = 5,
        bins: int | list[float] | None = None,
        filter_zscore: float | None = 20,
        max_loss: float | None = 0.35,
        binning_by_group: bool = False,
        zero_aware: bool = False,
        cumulative_returns: bool = True,
        groupby_labels: dict[Any, str] | None = None,
        rate_of_ret: bool = True,
        group_neutral: bool = False,
        by_group: bool = False,
        table: bool = True,
        figure: bool = False,
    ) -> TigerEvaluationSheet:
        factor_data = self.get_clean_factor_and_forward_returns(
            periods=periods,
            quantiles=quantiles,
            bins=bins,
            filter_zscore=filter_zscore,
            max_loss=max_loss,
            binning_by_group=binning_by_group,
            zero_aware=zero_aware,
            cumulative_returns=cumulative_returns,
            groupby_labels=groupby_labels,
        )
        resolved_output_dir = self._resolve_report_section_dir("returns", output_dir)
        return create_returns_tear_sheet(
            factor_data,
            output_dir=resolved_output_dir,
            rate_of_ret=rate_of_ret,
            group_neutral=group_neutral,
            by_group=by_group,
            table=table,
            figure=figure,
        )

    def _create_information_tear_sheet(
        self,
        *,
        output_dir: str | Path | None = None,
        periods: tuple[PeriodLike, ...] = (1, 5, 10),
        quantiles: int | list[float] | None = 5,
        bins: int | list[float] | None = None,
        filter_zscore: float | None = 20,
        max_loss: float | None = 0.35,
        binning_by_group: bool = False,
        zero_aware: bool = False,
        cumulative_returns: bool = True,
        groupby_labels: dict[Any, str] | None = None,
        group_neutral: bool = False,
        by_group: bool = False,
        table: bool = True,
        figure: bool = False,
    ) -> TigerEvaluationSheet:
        factor_data = self.get_clean_factor_and_forward_returns(
            periods=periods,
            quantiles=quantiles,
            bins=bins,
            filter_zscore=filter_zscore,
            max_loss=max_loss,
            binning_by_group=binning_by_group,
            zero_aware=zero_aware,
            cumulative_returns=cumulative_returns,
            groupby_labels=groupby_labels,
        )
        return create_information_tear_sheet(
            factor_data,
            output_dir=self._resolve_report_section_dir("information", output_dir),
            group_neutral=group_neutral,
            by_group=by_group,
            table=table,
            figure=figure,
        )

    def _create_turnover_tear_sheet(
        self,
        *,
        output_dir: str | Path | None = None,
        periods: tuple[PeriodLike, ...] = (1, 5, 10),
        quantiles: int | list[float] | None = 5,
        bins: int | list[float] | None = None,
        filter_zscore: float | None = 20,
        max_loss: float | None = 0.35,
        binning_by_group: bool = False,
        zero_aware: bool = False,
        cumulative_returns: bool = True,
        groupby_labels: dict[Any, str] | None = None,
        turnover_periods: tuple[PeriodLike, ...] | None = None,
        table: bool = True,
        figure: bool = False,
    ) -> TigerEvaluationSheet:
        factor_data = self.get_clean_factor_and_forward_returns(
            periods=periods,
            quantiles=quantiles,
            bins=bins,
            filter_zscore=filter_zscore,
            max_loss=max_loss,
            binning_by_group=binning_by_group,
            zero_aware=zero_aware,
            cumulative_returns=cumulative_returns,
            groupby_labels=groupby_labels,
        )
        return create_turnover_tear_sheet(
            factor_data,
            output_dir=self._resolve_report_section_dir("turnover", output_dir),
            turnover_periods=turnover_periods,
            table=table,
            figure=figure,
        )

    def _create_event_returns_tear_sheet(
        self,
        *,
        output_dir: str | Path | None = None,
        periods: tuple[PeriodLike, ...] = (1, 5, 10),
        quantiles: int | list[float] | None = 5,
        bins: int | list[float] | None = None,
        filter_zscore: float | None = 20,
        max_loss: float | None = 0.35,
        binning_by_group: bool = False,
        zero_aware: bool = False,
        cumulative_returns: bool = True,
        groupby_labels: dict[Any, str] | None = None,
        returns: pd.DataFrame | None = None,
        avgretplot: tuple[int, int] = (5, 15),
        long_short: bool = True,
        group_neutral: bool = False,
        std_bar: bool = True,
        by_group: bool = False,
        table: bool = True,
        figure: bool = False,
    ) -> TigerEvaluationSheet:
        factor_data = self.get_clean_factor_and_forward_returns(
            periods=periods,
            quantiles=quantiles,
            bins=bins,
            filter_zscore=filter_zscore,
            max_loss=max_loss,
            binning_by_group=binning_by_group,
            zero_aware=zero_aware,
            cumulative_returns=cumulative_returns,
            groupby_labels=groupby_labels,
        )
        return create_event_returns_tear_sheet(
            factor_data,
            output_dir=self._resolve_report_section_dir("event_returns", output_dir),
            returns=returns,
            avgretplot=avgretplot,
            long_short=long_short,
            group_neutral=group_neutral,
            std_bar=std_bar,
            by_group=by_group,
            table=table,
            figure=figure,
        )

    def _create_event_study_tear_sheet(
        self,
        *,
        output_dir: str | Path | None = None,
        periods: tuple[PeriodLike, ...] = (1, 5, 10),
        quantiles: int | list[float] | None = 5,
        bins: int | list[float] | None = None,
        filter_zscore: float | None = 20,
        max_loss: float | None = 0.35,
        binning_by_group: bool = False,
        zero_aware: bool = False,
        cumulative_returns: bool = True,
        groupby_labels: dict[Any, str] | None = None,
        returns: pd.DataFrame | None = None,
        avgretplot: tuple[int, int] | None = (5, 15),
        rate_of_ret: bool = True,
        n_bars: int = 50,
        long_short: bool = True,
        group_neutral: bool = False,
        by_group: bool = False,
        turnover_periods: tuple[PeriodLike, ...] | None = None,
        table: bool = True,
        figure: bool = False,
    ) -> TigerEvaluationSheet:
        factor_data = self.get_clean_factor_and_forward_returns(
            periods=periods,
            quantiles=quantiles,
            bins=bins,
            filter_zscore=filter_zscore,
            max_loss=max_loss,
            binning_by_group=binning_by_group,
            zero_aware=zero_aware,
            cumulative_returns=cumulative_returns,
            groupby_labels=groupby_labels,
        )
        return create_event_study_tear_sheet(
            factor_data,
            output_dir=self._resolve_report_section_dir("event_study", output_dir),
            returns=returns,
            avgretplot=avgretplot,
            rate_of_ret=rate_of_ret,
            n_bars=n_bars,
            long_short=long_short,
            group_neutral=group_neutral,
            by_group=by_group,
            turnover_periods=turnover_periods,
            table=table,
            figure=figure,
        )

    def _create_full_tear_sheet(
        self,
        *,
        output_dir: str | Path | None = None,
        periods: tuple[PeriodLike, ...] = (1, 5, 10),
        quantiles: int | list[float] | None = 5,
        bins: int | list[float] | None = None,
        filter_zscore: float | None = 20,
        max_loss: float | None = 0.35,
        binning_by_group: bool = False,
        zero_aware: bool = False,
        cumulative_returns: bool = True,
        groupby_labels: dict[Any, str] | None = None,
        long_short: bool = True,
        group_neutral: bool = False,
        by_group: bool = False,
        turnover_periods: tuple[PeriodLike, ...] | None = None,
        avgretplot: tuple[int, int] = (5, 15),
        include_horizon: bool = True,
        horizons: tuple[int, ...] | list[int] = (1, 3, 5, 10, 20),
        horizon_quantiles: int = 5,
        horizon_periods_per_year: int = 252,
        horizon_long_short_pct: float = 0.2,
        figure: bool = True,
    ) -> TigerEvaluationSheet:
        artifacts = self._get_full_report_artifacts(
            periods=periods,
            quantiles=quantiles,
            bins=bins,
            filter_zscore=filter_zscore,
            max_loss=max_loss,
            binning_by_group=binning_by_group,
            zero_aware=zero_aware,
            cumulative_returns=cumulative_returns,
            groupby_labels=groupby_labels,
            long_short=long_short,
            group_neutral=group_neutral,
            by_group=by_group,
            turnover_periods=turnover_periods,
            avgretplot=avgretplot,
            include_horizon=include_horizon,
            horizons=horizons,
            horizon_quantiles=horizon_quantiles,
            horizon_periods_per_year=horizon_periods_per_year,
            horizon_long_short_pct=horizon_long_short_pct,
        )
        factor_data = artifacts["factor_data"]
        report_root = self._resolve_report_root(output_dir)
        report = create_full_tear_sheet(
            factor_data,
            output_dir=report_root,
            figure_output_dir=None,
            long_short=long_short,
            group_neutral=group_neutral,
            by_group=by_group,
            turnover_periods=turnover_periods,
            avgretplot=avgretplot,
            horizon_result=artifacts.get("horizon_result"),
            horizon_summary=artifacts.get("horizon_summary"),
            figure=figure,
        )
        self._write_json(
            report.output_dir / "manifest.json",
            {
                "created_at": self._created_at(),
                "factor_column": self.factor_column,
                "date_column": self.date_column,
                "code_column": self.code_column,
                "price_column": self.price_column,
                "forward_days": self.forward_days,
                "report_name": report.name,
                "output_dir": str(report.output_dir),
                "factor_rows": int(len(self.factor_frame)),
                "price_rows": int(len(self.price_frame)),
                "periods": list(periods),
                "quantiles": int(quantiles),
                "max_loss": max_loss,
                "long_short": long_short,
                "group_neutral": group_neutral,
                "by_group": by_group,
                "turnover_periods": list(turnover_periods) if turnover_periods is not None else None,
                "avgretplot": list(avgretplot),
                "horizon_config": artifacts.get("horizon_config"),
                "evaluation": report.evaluation.__dict__ if report.evaluation is not None else None,
                "figure_paths": {name: str(path) for name, path in report.figure_paths.items()},
                "table_paths": {name: str(path) for name, path in report.table_paths.items()},
                "includes_horizon": include_horizon,
            },
        )
        return report

    @staticmethod
    def _image_to_data_uri(path: Path) -> str:
        suffix = path.suffix.lower().lstrip(".") or "png"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/{suffix};base64,{data}"

    @staticmethod
    def _split_return_mode_image_name(image_name: str) -> tuple[str, str | None]:
        for mode in ("long_only", "long_short"):
            suffix = f"_{mode}"
            if image_name.endswith(suffix):
                return image_name[: -len(suffix)], mode
        return image_name, None

    def _render_paired_figure_html(self, figure_paths: dict[str, Path]) -> str:
        parts: list[str] = []
        used: set[str] = set()
        for name, path in sorted(figure_paths.items()):
            if name in used or not path.exists():
                continue
            base_name, mode = self._split_return_mode_image_name(name)
            if mode in {"long_only", "long_short"}:
                left_name = f"{base_name}_long_only"
                right_name = f"{base_name}_long_short"
                left_path = figure_paths.get(left_name)
                right_path = figure_paths.get(right_name)
                if left_path is not None and right_path is not None and left_path.exists() and right_path.exists():
                    parts.append(
                        "<section>"
                        f"<h2>{escape(base_name)}</h2>"
                        "<div class='comparison-pair'>"
                        f"<figure><figcaption>long_only</figcaption><img src='{self._image_to_data_uri(left_path)}' alt='{escape(left_name)}' /></figure>"
                        f"<figure><figcaption>long_short</figcaption><img src='{self._image_to_data_uri(right_path)}' alt='{escape(right_name)}' /></figure>"
                        "</div>"
                        "</section>"
                    )
                    used.update({left_name, right_name})
                    continue
            parts.append(
                "<section>"
                f"<h2>{escape(name)}</h2>"
                f"<img src='{self._image_to_data_uri(path)}' alt='{escape(name)}' />"
                "</section>"
            )
            used.add(name)
        return "".join(parts)

    def _render_full_report_html(self, report: TigerEvaluationSheet) -> str:
        parts = self._render_paired_figure_html(report.figure_paths)
        return (
            "<html><head><meta charset='utf-8'>"
            f"<title>{escape(report.name)}</title>"
            "<style>body{font-family:Arial,sans-serif;margin:24px;}"
            "img{max-width:100%;height:auto;display:block;margin:12px 0;}"
            "figure{margin:0;}"
            "figcaption{font-size:12px;color:#555;text-align:center;margin-top:4px;}"
            ".comparison-pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;align-items:start;margin:16px 0;}"
            "@media (max-width: 1100px){.comparison-pair{grid-template-columns:1fr;}}"
            "</style></head><body>"
            f"<h1>{escape(report.name)}</h1>"
            + parts
            + "</body></html>"
        )

    def _create_native_full_tear_sheet(
        self,
        *,
        output_dir: str | Path | None = None,
        quantiles: int = 5,
        portfolio_returns: pd.Series | None = None,
        benchmark_returns: pd.Series | None = None,
        group_labels: pd.DataFrame | pd.Series | None = None,
    ) -> FactorTearSheetResult:
        return create_native_full_tear_sheet(
            self.factor_column,
            self.factor_panel(),
            self.forward_returns(),
            output_dir=self._resolve_report_section_dir("native", output_dir),
            quantiles=quantiles,
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns if benchmark_returns is not None else self.benchmark_returns(),
            group_labels=group_labels if group_labels is not None else self.group_frame(),
        )

    def create_report_bundle_summary(
        self,
        report: TigerEvaluationSheet | None = None,
        *,
        output_dir: str | Path | None = None,
    ) -> ReportBundleSummary:
        if report is None:
            root = self._resolve_report_root(output_dir)
            full_manifest_path = root / "manifest.json"
            if not full_manifest_path.exists():
                raise FileNotFoundError(f"Full report manifest not found: {full_manifest_path}")
            full_manifest = self._read_json(full_manifest_path)
            horizon_manifest_path = root / "horizon" / "manifest.json"
            horizon_manifest = self._read_json(horizon_manifest_path) if horizon_manifest_path.exists() else None
            return ReportBundleSummary(
                factor_column=str(full_manifest.get("factor_column", self.factor_column)),
                report_dir=str(root),
                full_manifest=str(full_manifest_path),
                horizon_manifest=str(horizon_manifest_path) if horizon_manifest_path.exists() else None,
                figure_count=len(full_manifest.get("figure_paths", {})),
                table_count=len(full_manifest.get("table_paths", {})),
                factor_rows=full_manifest.get("factor_rows"),
                price_rows=full_manifest.get("price_rows"),
                evaluation=full_manifest.get("evaluation"),
                horizon_summary=None if horizon_manifest is None else horizon_manifest.get("summary"),
            )

        full_manifest_path = report.output_dir / "manifest.json"
        horizon_manifest_path = Path(report.table_paths.get("horizon_manifest", report.output_dir / "horizon" / "manifest.json"))
        full_manifest = self._read_json(full_manifest_path) if full_manifest_path.exists() else {}
        horizon_manifest = self._read_json(horizon_manifest_path) if horizon_manifest_path.exists() else None
        return ReportBundleSummary(
            factor_column=self.factor_column,
            report_dir=str(report.output_dir),
            full_manifest=str(full_manifest_path),
            horizon_manifest=str(horizon_manifest_path) if horizon_manifest_path.exists() else None,
            figure_count=len(report.figure_paths),
            table_count=len(report.table_paths),
            factor_rows=int(len(self.factor_frame)),
            price_rows=int(len(self.price_frame)),
            evaluation=full_manifest.get("evaluation") if full_manifest else (None if report.evaluation is None else report.evaluation.__dict__),
            horizon_summary=None if horizon_manifest is None else horizon_manifest.get("summary"),
        )

    def holding_period_analyzer(
        self,
        *,
        quantiles: int = 5,
        periods_per_year: int = 252,
        long_short_pct: float = 0.2,
    ) -> HoldingPeriodAnalyzer:
        return HoldingPeriodAnalyzer(
            self.factor_panel(),
            self.price_panel(),
            quantiles=quantiles,
            periods_per_year=periods_per_year,
            long_short_pct=long_short_pct,
        )

    def analyze_horizons(
        self,
        horizons: list[int] | tuple[int, ...],
        *,
        quantiles: int = 5,
        periods_per_year: int = 252,
        long_short_pct: float = 0.2,
    ) -> pd.DataFrame:
        analyzer = self.holding_period_analyzer(
            quantiles=quantiles,
            periods_per_year=periods_per_year,
            long_short_pct=long_short_pct,
        )
        return analyzer.run(horizons)

    def summarize_best_horizon(
        self,
        horizons: list[int] | tuple[int, ...],
        *,
        quantiles: int = 5,
        periods_per_year: int = 252,
        long_short_pct: float = 0.2,
    ) -> dict[str, int | str | float]:
        result = self.analyze_horizons(
            horizons,
            quantiles=quantiles,
            periods_per_year=periods_per_year,
            long_short_pct=long_short_pct,
        )
        return summarize_best_horizon_result(result)

    def plot_horizon_result(
        self,
        horizons: list[int] | tuple[int, ...],
        *,
        output_path: str | Path | None = None,
        quantiles: int = 5,
        periods_per_year: int = 252,
        long_short_pct: float = 0.2,
    ) -> Any:
        analyzer = self.holding_period_analyzer(
            quantiles=quantiles,
            periods_per_year=periods_per_year,
            long_short_pct=long_short_pct,
        )
        result = analyzer.run(horizons)
        if output_path is None:
            output_path = self._default_report_root() / "horizon" / "horizon_result.png"
        return analyzer.plot_advanced(result, output_path=output_path)


__all__ = [
    "FactorEvaluationEngine",
]
