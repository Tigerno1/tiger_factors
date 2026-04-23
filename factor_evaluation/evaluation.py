from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from tiger_factors.factor_evaluation.engine import ReportBundleSummary
from tiger_factors.factor_evaluation.engine import FactorEvaluationEngine
from tiger_factors.factor_evaluation.analysis import FactorEffectivenessConfig
from tiger_factors.factor_evaluation.analysis import analyze_factor_attribution
from tiger_factors.factor_evaluation.analysis import analyze_factor_exposure
from tiger_factors.factor_evaluation.analysis import monitor_factor_dynamics
from tiger_factors.factor_evaluation.analysis import test_factor_effectiveness
from tiger_factors.factor_evaluation.input import TableLike
from tiger_factors.factor_evaluation.input import build_tiger_evaluation_input
from tiger_factors.factor_evaluation.utils import TigerFactorData
from tiger_factors.factor_evaluation.utils import get_clean_factor_and_forward_returns
from tiger_factors.factor_evaluation.native_tears import FactorTearSheetResult
from tiger_factors.factor_evaluation.tears import TigerEvaluationSheet
from tiger_factors.factor_maker.pipeline import Pipeline
from tiger_factors.factor_maker.pipeline import PipelineEngine
from tiger_factors.factor_store.evaluation_store import EvaluationStore
from tiger_factors.factor_store.evaluation_store import EvaluationSectionAccessor
from tiger_factors.factor_store.spec import FactorSpec


@dataclass(frozen=True)
class SingleFactorEvaluationConfig:
    factor_column: str
    date_column: str = "date_"
    code_column: str = "code"
    price_column: str = "close"
    forward_days: int = 1
    periods: tuple[int, ...] = (1, 5, 10)
    quantiles: int | list[float] | None = 5
    bins: int | list[float] | None = None
    filter_zscore: float | None = 20
    max_loss: float | None = 0.35
    binning_by_group: bool = False
    zero_aware: bool = False
    cumulative_returns: bool = True
    groupby_labels: dict[Any, str] | None = None
    long_short: bool = True
    group_neutral: bool = False
    by_group: bool = False
    turnover_periods: tuple[int, ...] | None = None
    avgretplot: tuple[int, int] = (5, 15)
    include_horizon: bool = True
    horizons: tuple[int, ...] = (1, 3, 5, 10, 20)
    horizon_quantiles: int = 5
    horizon_periods_per_year: int = 252
    horizon_long_short_pct: float = 0.2


@dataclass(frozen=True)
class SingleFactorEvaluationResult:
    config: SingleFactorEvaluationConfig
    evaluation: Any
    report: TigerEvaluationSheet | None
    report_bundle: ReportBundleSummary | None
    native_report: FactorTearSheetResult | None
    horizon_result: pd.DataFrame | None
    horizon_summary: dict[str, Any] | None
    output_dir: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.__dict__,
            "evaluation": self.evaluation.__dict__ if hasattr(self.evaluation, "__dict__") else self.evaluation,
            "report": None if self.report is None else {
                "name": self.report.name,
                "output_dir": str(self.report.output_dir),
                "figure_paths": {k: str(v) for k, v in self.report.figure_paths.items()},
                "table_paths": {k: str(v) for k, v in self.report.table_paths.items()},
            },
            "report_bundle": None if self.report_bundle is None else self.report_bundle.to_dict(),
            "native_report": None if self.native_report is None else self.native_report.to_summary(),
            "horizon_result": None if self.horizon_result is None else self.horizon_result.to_dict(orient="records"),
            "horizon_summary": self.horizon_summary,
            "output_dir": self.output_dir,
        }


class EvaluationSectionCommand:
    """Callable section handle used by ``SingleFactorEvaluation``.

    The command exposes read helpers through the underlying evaluation store
    accessor, and the call operator persists the corresponding section in the
    new directory layout.
    """

    def __init__(self, owner: "SingleFactorEvaluation", section: str) -> None:
        self._owner = owner
        self._section = section

    def _accessor(self) -> EvaluationSectionAccessor:
        return self._owner._section_accessor(self._section)

    def tables(self) -> list[str]:
        return self._accessor().tables()

    def imgs(self) -> list[str]:
        return self._accessor().imgs()

    def report(self) -> str | None:
        return self._accessor().report()

    def get_table(self, table_name: str | None = None) -> pd.DataFrame:
        return self._accessor().get_table(table_name)

    def get_img(self, img_name: str):
        return self._accessor().get_img(img_name)

    def get_report(self, *, open_browser: bool = True):
        return self._accessor().get_report(open_browser=open_browser)

    def __call__(self, *, format: str | None = None, force_updated: bool = False, **kwargs: Any) -> Any:
        if (
            self._owner.factor_frame is None
            or self._owner.price_frame is None
            or not self._owner.factor_column
        ):
            return self
        self._owner._save_section(
            self._section,
            format=format,
            force_updated=force_updated,
            **kwargs,
        )
        return self


class SingleFactorEvaluation:
    """Evaluation façade for a single factor.

    This wraps the Tiger factor-evaluation engine with one opinionated entrypoint:

    - normalize factor and price inputs
    - evaluate IC / rank IC / turnover / Sharpe / fitness
    - optionally persist section tables through EvaluationStore
    - optionally run horizon analysis
    """

    def __init__(
        self,
        *,
        factor: pd.Series | None = None,
        prices: pd.DataFrame | None = None,
        factor_frame: TableLike | None = None,
        price_frame: TableLike | None = None,
        factor_column: str | None = None,
        spec: FactorSpec | None = None,
        date_column: str = "date_",
        code_column: str = "code",
        price_column: str = "close",
        forward_days: int = 1,
        benchmark_frame: TableLike | None = None,
        benchmark_date_column: str = "date_",
        benchmark_value_column: str = "returns",
        group_labels: TableLike | None = None,
        group_date_column: str = "date_",
        group_code_column: str = "code",
        group_value_column: str = "group",
        group_labels_cache: bool = True,
        periods: tuple[int, ...] = (1, 5, 10),
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
        horizons: tuple[int, ...] = (1, 3, 5, 10, 20),
        horizon_quantiles: int = 5,
        horizon_periods_per_year: int = 252,
        horizon_long_short_pct: float = 0.2,
    ) -> None:
        inferred_factor_column = factor_column or (spec.table_name if spec is not None else None)
        prepared: TigerFactorData | None = None
        if factor is not None or prices is not None:
            if factor is None or prices is None:
                raise ValueError("factor and prices must be provided together")
            prepared = get_clean_factor_and_forward_returns(
                factor=factor,
                prices=prices,
                factor_column=inferred_factor_column or getattr(factor, "name", None) or "factor",
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
            factor_frame = prepared.factor_frame
            price_frame = prepared.price_frame
            inferred_factor_column = prepared.factor_column
        self.factor_frame = factor_frame
        self.price_frame = price_frame
        self.factor_column = inferred_factor_column
        self.evaluation_store = EvaluationStore()
        self.date_column = date_column
        self.code_column = code_column
        self.price_column = price_column
        self.forward_days = forward_days
        self.benchmark_frame = benchmark_frame
        self.benchmark_date_column = benchmark_date_column
        self.benchmark_value_column = benchmark_value_column
        self.group_labels = group_labels
        self.group_date_column = group_date_column
        self.group_code_column = group_code_column
        self.group_value_column = group_value_column
        self.group_labels_cache = group_labels_cache
        self.config = SingleFactorEvaluationConfig(
            factor_column=inferred_factor_column or "",
            date_column=date_column,
            code_column=code_column,
            price_column=price_column,
            forward_days=forward_days,
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
        if prepared is None and self.factor_frame is not None and self.price_frame is not None and self.factor_column is not None:
            prepared = build_tiger_evaluation_input(
                factor_frame=self.factor_frame,
                price_frame=self.price_frame,
                factor_column=self.factor_column,
                date_column=date_column,
                code_column=code_column,
                price_column=price_column,
                forward_days=forward_days,
            )
            self.factor_frame = prepared.factor_frame
            self.price_frame = prepared.price_frame
        if spec is None and self.factor_column:
            spec = FactorSpec(
                provider="tiger",
                region="us",
                sec_type="stock",
                freq="1d",
                table_name=self.factor_column,
            )
        self.spec = spec
        self._engine = FactorEvaluationEngine(
            factor_frame=prepared.factor_frame if prepared is not None else pd.DataFrame(),
            price_frame=prepared.price_frame if prepared is not None else pd.DataFrame(),
            factor_column=self.factor_column or "",
            spec=spec,
            date_column=self.date_column,
            code_column=self.code_column,
            price_column=self.price_column,
            forward_days=self.forward_days,
            benchmark_frame=self.benchmark_frame,
            benchmark_date_column=self.benchmark_date_column,
            benchmark_value_column=self.benchmark_value_column,
            group_labels=self.group_labels,
            group_date_column=self.group_date_column,
            group_code_column=self.group_code_column,
            group_value_column=self.group_value_column,
            group_labels_cache=self.group_labels_cache,
        )
        self.summary = EvaluationSectionCommand(self, "summary")
        self.horizon = EvaluationSectionCommand(self, "horizon")
        self.full = EvaluationSectionCommand(self, "full")

    @property
    def engine(self) -> FactorEvaluationEngine:
        return self._engine

    def _evaluation_store(self) -> EvaluationStore:
        return self.evaluation_store

    def _section_accessor(self, section: str) -> EvaluationSectionAccessor:
        if self.spec is None:
            raise ValueError("spec is required to access saved evaluation artifacts")
        return self._evaluation_store().section(self.spec, section)

    @staticmethod
    def _html_page(title: str, body: str) -> str:
        return (
            "<html><head><meta charset='utf-8'>"
            f"<title>{title}</title>"
            "<style>body{font-family:Arial,sans-serif;margin:24px;}"
            "table{border-collapse:collapse;margin:16px 0;}td,th{border:1px solid #ccc;padding:6px 8px;}"
            "img{max-width:100%;height:auto;display:block;margin:12px 0;}"
            "figure{margin:0;}"
            "figcaption{font-size:12px;color:#555;text-align:center;margin-top:4px;}"
            ".comparison-pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;align-items:start;margin:16px 0;}"
            "@media (max-width: 1100px){.comparison-pair{grid-template-columns:1fr;}}"
            ".summary-block{margin:28px 0 40px;}"
            ".summary-grid{width:100%;border-collapse:separate;border-spacing:12px 12px;table-layout:fixed;}"
            ".summary-grid td{border:0;padding:0;vertical-align:top;}"
            ".summary-triplet{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px;align-items:start;}"
            "@media (max-width: 1100px){.summary-triplet{grid-template-columns:1fr;}}"
            "</style></head><body>"
            f"<h1>{title}</h1>"
            f"{body}"
            "</body></html>"
        )

    def _render_table_html(self, title: str, table: pd.DataFrame | pd.Series) -> str:
        frame = table.to_frame().T if isinstance(table, pd.Series) else table
        return f"<section><h2>{title}</h2>{frame.to_html(index=False, escape=False)}</section>"

    def _render_image_html(self, title: str, image_src: str) -> str:
        return (
            f"<section><h2>{title}</h2>"
            f"<img src='{image_src}' alt='{title}' />"
            "</section>"
        )

    def _render_image_block_html(self, title: str, image_src: str) -> str:
        return (
            f"<figure>"
            f"<img src='{image_src}' alt='{title}' />"
            f"<figcaption>{title}</figcaption>"
            "</figure>"
        )

    @staticmethod
    def _split_return_mode_image_name(image_name: str) -> tuple[str, str | None]:
        for mode in ("long_only", "long_short"):
            suffix = f"_{mode}"
            if image_name.endswith(suffix):
                return image_name[: -len(suffix)], mode
        return image_name, None

    def _render_image_collection_html(self, image_items: list[tuple[str, str]]) -> str:
        if not image_items:
            return ""
        image_map = {name: src for name, src in image_items}
        used: set[str] = set()
        body_parts: list[str] = []
        for image_name, image_src in image_items:
            if image_name in used:
                continue
            base_name, mode = self._split_return_mode_image_name(image_name)
            if mode in {"long_only", "long_short"}:
                left_name = f"{base_name}_long_only"
                right_name = f"{base_name}_long_short"
                left_src = image_map.get(left_name)
                right_src = image_map.get(right_name)
                if left_src is not None and right_src is not None and left_name not in used and right_name not in used:
                    body_parts.append(
                        (
                            "<section>"
                            f"<h2>{base_name}</h2>"
                            "<div class='comparison-pair'>"
                            f"<figure><figcaption>long_only</figcaption><img src='{left_src}' alt='{left_name}' /></figure>"
                            f"<figure><figcaption>long_short</figcaption><img src='{right_src}' alt='{right_name}' /></figure>"
                            "</div>"
                            "</section>"
                        )
                    )
                    used.update({left_name, right_name})
                    continue
            body_parts.append(self._render_image_html(image_name, image_src))
            used.add(image_name)
        return "\n".join(body_parts)

    def _write_section_report(
        self,
        *,
        section: str,
        title: str,
        tables: dict[str, pd.DataFrame | pd.Series] | None = None,
        imgs: list[str] | None = None,
        force_updated: bool = False,
    ) -> None:
        if self.spec is None:
            raise ValueError("spec is required to save evaluation reports")
        body_parts: list[str] = []
        for table_name, table in (tables or {}).items():
            body_parts.append(self._render_table_html(table_name, table))
        if imgs:
            rendered_images = self._render_image_collection_html([(img_name, f"{img_name}.png") for img_name in imgs])
            if rendered_images:
                body_parts.append(rendered_images)
        body = "\n".join(body_parts) or "<p>No artifacts.</p>"
        html = self._html_page(title, body)
        self._evaluation_store().save_section_report(
            html,
            spec=self.spec,
            section=section,
            report_name="report",
            force_updated=force_updated,
        )

    def _save_section_image_file(
        self,
        *,
        section: str,
        img_name: str,
        source_path: Path,
        force_updated: bool = False,
    ) -> Path:
        return self._evaluation_store().save_section_image(
            source_path,
            spec=self.spec,
            section=section,
            img_name=img_name,
            force_updated=force_updated,
        )

    def _full_report_body(self) -> str:
        if self.spec is None:
            raise ValueError("spec is required to save evaluation reports")
        section_titles = {
            "summary": "Summary",
            "returns": "Returns",
            "information": "Information",
            "turnover": "Turnover",
            "event_returns": "Event Returns",
            "horizon": "Horizon",
        }
        body_parts: list[str] = []
        for section, title in section_titles.items():
            if section == "summary":
                continue
            lookup_section = section
            section_parts = [f"<section><h2>{title}</h2>"]
            img_names = self._evaluation_store().list_imgs(self.spec, lookup_section)
            if img_names:
                rendered_images = self._render_image_collection_html([(img_name, f"./{section}/{img_name}.png") for img_name in img_names])
                if rendered_images:
                    section_parts.append(rendered_images)
            section_parts.append("</section>")
            if len(section_parts) > 2:
                body_parts.append("\n".join(section_parts))
        return "\n".join(body_parts) or "<p>No artifacts.</p>"

    def _load_report_table(self, path: Path) -> pd.DataFrame | pd.Series | None:
        if not path.exists():
            return None
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return None
            if isinstance(payload, dict):
                return pd.DataFrame([payload])
            if isinstance(payload, list):
                return pd.DataFrame(payload)
        return None

    def _persist_sheet_images(
        self,
        *,
        section: str,
        report: TigerEvaluationSheet,
        force_updated: bool = False,
    ) -> list[str]:
        saved_names: list[str] = []
        for img_name, path in report.figure_paths.items():
            source_path = Path(path)
            if not source_path.exists():
                continue
            self._save_section_image_file(
                section=section,
                img_name=img_name,
                source_path=source_path,
                force_updated=force_updated,
            )
            saved_names.append(img_name)
        return saved_names

    def _persist_sheet_tables(
        self,
        *,
        section: str,
        report: TigerEvaluationSheet,
        force_updated: bool = False,
    ) -> dict[str, pd.DataFrame | pd.Series]:
        saved_tables: dict[str, pd.DataFrame | pd.Series] = {}
        for table_name, path in sorted(report.table_paths.items()):
            table = self._load_report_table(Path(path))
            if table is None:
                continue
            self._persist_table(section, table_name, table, force_updated=force_updated)
            saved_tables[table_name] = table
        return saved_tables

    def _collect_sheet_tables_for_report(
        self,
        report: TigerEvaluationSheet,
    ) -> dict[str, pd.DataFrame | pd.Series]:
        collected: dict[str, pd.DataFrame | pd.Series] = {}
        for table_name, path in sorted(report.table_paths.items()):
            table = self._load_report_table(Path(path))
            if table is not None:
                collected[table_name] = table
        return collected

    def _persist_sheet_section(
        self,
        *,
        section: str,
        report: TigerEvaluationSheet,
        save_imgs: bool,
        save_tables: bool,
        save_report: bool,
        force_updated: bool = False,
        report_title: str | None = None,
    ) -> None:
        saved_img_names: list[str] = []
        if save_imgs or save_report:
            for img_name, path in report.figure_paths.items():
                source_path = Path(path)
                if not source_path.exists():
                    continue
                try:
                    self._save_section_image_file(
                        section=section,
                        img_name=img_name,
                        source_path=source_path,
                        force_updated=force_updated,
                    )
                except FileExistsError:
                    if force_updated:
                        raise
                saved_img_names.append(img_name)
        if save_tables:
            for table_name, path in sorted(report.table_paths.items()):
                table = self._load_report_table(Path(path))
                if table is None:
                    continue
                try:
                    self._persist_table(section, table_name, table, force_updated=force_updated)
                except FileExistsError:
                    if force_updated:
                        raise
        if save_report:
            try:
                self._write_section_report(
                    section=section,
                    title=report_title or section,
                    tables=None,
                    imgs=saved_img_names,
                    force_updated=force_updated,
                )
            except FileExistsError:
                if force_updated:
                    raise

    def _factor_data_for_analysis(
        self,
        *,
        periods: tuple[int, ...] | None = None,
        quantiles: int | list[float] | None = None,
        bins: int | list[float] | None = None,
        filter_zscore: float | None = None,
        max_loss: float | None = None,
        binning_by_group: bool | None = None,
        zero_aware: bool | None = None,
        cumulative_returns: bool | None = None,
        groupby_labels: dict[Any, str] | None = None,
    ) -> TigerFactorData:
        if self.factor_frame is None or self.price_frame is None or not self.factor_column:
            raise ValueError("factor_frame, price_frame, and factor_column are required for analysis")
        return self._engine.get_clean_factor_and_forward_returns(
            periods=periods if periods is not None else self.config.periods,
            quantiles=self.config.quantiles if quantiles is None else quantiles,
            bins=self.config.bins if bins is None else bins,
            filter_zscore=self.config.filter_zscore if filter_zscore is None else filter_zscore,
            max_loss=self.config.max_loss if max_loss is None else max_loss,
            binning_by_group=self.config.binning_by_group if binning_by_group is None else binning_by_group,
            zero_aware=self.config.zero_aware if zero_aware is None else zero_aware,
            cumulative_returns=self.config.cumulative_returns if cumulative_returns is None else cumulative_returns,
            groupby_labels=self.config.groupby_labels if groupby_labels is None else groupby_labels,
        )

    def _persist_summary(self, summary: pd.DataFrame, *, force_updated: bool = False) -> None:
        self._evaluation_store().save_summary(
            summary,
            spec=self.spec,
            force_updated=force_updated,
        )

    def _persist_table(self, section: str, table_name: str, table: pd.DataFrame | pd.Series, *, force_updated: bool = False) -> None:
        self._evaluation_store().save_section_table(
            table,
            spec=self.spec,
            section=section,
            table_name=table_name,
            force_updated=force_updated,
        )

    def _persist_report(self, report: TigerEvaluationSheet, *, force_updated: bool = False) -> None:
        payload = report.payload or {}
        if report.name == "summary":
            summary = payload.get("summary")
            if summary is None and report.evaluation is not None:
                summary = report.evaluation.__dict__
            if summary is not None:
                if isinstance(summary, pd.DataFrame):
                    self._persist_summary(summary, force_updated=force_updated)
                elif isinstance(summary, pd.Series):
                    self._persist_summary(summary.to_frame().T.reset_index(drop=True), force_updated=force_updated)
                elif isinstance(summary, dict):
                    self._persist_summary(pd.DataFrame([summary]), force_updated=force_updated)
                elif hasattr(summary, "__dict__"):
                    self._persist_summary(pd.DataFrame([summary.__dict__]), force_updated=force_updated)
            return

        if report.name == "returns":
            self._persist_sheet_tables(
                section="returns",
                report=report,
                force_updated=force_updated,
            )
            return

        if report.name == "information":
            section = "information"
            table_map = {
                "information_coefficient": payload.get("ic"),
                "mean_information_coefficient": payload.get("mean_ic"),
                "monthly_information_coefficient": payload.get("monthly_ic"),
                "information_coefficient_by_group": payload.get("ic_group"),
                "mean_information_coefficient_by_group": payload.get("mean_ic_by_group"),
            }
            for table_name, table in table_map.items():
                if isinstance(table, (pd.DataFrame, pd.Series)):
                    self._persist_table(section, table_name, table, force_updated=force_updated)
            return

        if report.name == "turnover":
            section = "turnover"
            table_map = {
                "turnover": payload.get("turnover"),
                "rank_autocorrelation": payload.get("rank_autocorrelation"),
                "turnover_summary": payload.get("turnover_summary"),
            }
            for table_name, table in table_map.items():
                if isinstance(table, (pd.DataFrame, pd.Series)):
                    self._persist_table(section, table_name, table, force_updated=force_updated)
            return

        if report.name == "event_returns":
            section = "event_returns"
            table_map = {
                "average_cumulative_return_by_quantile": payload.get("average_cumulative_return_by_quantile"),
                "average_cumulative_return_by_group": payload.get("average_cumulative_return_by_group"),
            }
            for table_name, table in table_map.items():
                if isinstance(table, (pd.DataFrame, pd.Series)):
                    self._persist_table(section, table_name, table, force_updated=force_updated)
            best_holding_period = payload.get("best_holding_period")
            if best_holding_period is not None:
                self._persist_table(
                    section,
                    "best_holding_period",
                    pd.DataFrame([best_holding_period]) if isinstance(best_holding_period, dict) else pd.DataFrame([best_holding_period]),
                    force_updated=force_updated,
                )
            return

        if report.name == "event_study":
            section = "event_study"
            table_map = {
                "quantile_statistics": payload.get("quantile_statistics"),
                "mean_return_by_quantile": payload.get("mean_return_by_quantile"),
                "mean_return_by_quantile_by_date": payload.get("mean_return_by_quantile_by_date"),
            }
            for table_name, table in table_map.items():
                if isinstance(table, (pd.DataFrame, pd.Series)):
                    self._persist_table(section, table_name, table, force_updated=force_updated)
            nested_event_returns = payload.get("event_returns")
            if isinstance(nested_event_returns, TigerEvaluationSheet):
                self._persist_report(nested_event_returns, force_updated=force_updated)
            return

        if report.name == "full":
            return

    def _save_section(
        self,
        section: str,
        *,
        format: str | None = None,
        force_updated: bool = False,
        **kwargs: Any,
    ) -> None:
        default_formats = {
            "summary": "table",
            "horizon": "img_table",
            "full": "all",
        }
        fmt = (format or default_formats.get(section, "table")).lower().strip()
        if fmt not in {"img", "table", "report", "all", "img_table"}:
            raise ValueError(f"unsupported format: {fmt!r}")
        save_imgs = fmt in {"img", "report", "all", "img_table"}
        save_tables = fmt in {"table", "all", "img_table"}
        save_report = fmt in {"report", "all"}
        if section == "summary":
            summary_sheet = self._engine._create_summary_tear_sheet(
                periods=self.config.periods,
                quantiles=self.config.quantiles,
                bins=self.config.bins,
                filter_zscore=self.config.filter_zscore,
                max_loss=self.config.max_loss,
                binning_by_group=self.config.binning_by_group,
                zero_aware=self.config.zero_aware,
                cumulative_returns=self.config.cumulative_returns,
                groupby_labels=self.config.groupby_labels,
                long_short=self.config.long_short,
                group_neutral=self.config.group_neutral,
                highlights=kwargs.pop("highlights", None),
                table=save_tables,
                figure=False,
            )
            summary = self._engine._summary_frame_from_report(summary_sheet)
            if save_tables or save_report:
                self._persist_summary(summary, force_updated=force_updated)
            return

        if section == "horizon":
            horizons = tuple(kwargs.pop("horizons", self.config.horizons))
            quantiles = int(kwargs.pop("quantiles", self.config.horizon_quantiles))
            periods_per_year = int(kwargs.pop("periods_per_year", self.config.horizon_periods_per_year))
            long_short_pct = float(kwargs.pop("long_short_pct", self.config.horizon_long_short_pct))
            result = self.analyze_horizons(
                horizons=horizons,
                quantiles=quantiles,
                periods_per_year=periods_per_year,
                long_short_pct=long_short_pct,
            )
            summary = self.summarize_best_horizon(
                horizons=horizons,
                quantiles=quantiles,
                periods_per_year=periods_per_year,
                long_short_pct=long_short_pct,
            )
            if save_imgs:
                report_path = self._evaluation_store()._section_dir(self.spec, "horizon") / "horizon_result.png"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                if not force_updated and report_path.exists():
                    raise FileExistsError(f"evaluation section image already exists: {report_path}")
                if force_updated:
                    report_path.unlink(missing_ok=True)
                self._engine.plot_horizon_result(
                    horizons,
                    output_path=report_path,
                    quantiles=quantiles,
                    periods_per_year=periods_per_year,
                    long_short_pct=long_short_pct,
                )
            if save_tables:
                self._evaluation_store().save_section_table(
                    result,
                    spec=self.spec,
                    section="horizon",
                    table_name="horizon_result",
                    force_updated=force_updated,
                )
                self._evaluation_store().save_section_table(
                    pd.DataFrame([summary]),
                    spec=self.spec,
                    section="horizon",
                    table_name="horizon_summary",
                    force_updated=force_updated,
                )
            if save_report:
                tables = {
                    "horizon_result": result,
                    "horizon_summary": pd.DataFrame([summary]),
                }
                self._write_section_report(
                    section="horizon",
                    title="horizon",
                    tables=tables,
                    imgs=["horizon_result"] if save_imgs else None,
                    force_updated=force_updated,
                )
            return

        if section == "full":
            summary_sheet = self._engine._create_summary_tear_sheet(
                periods=self.config.periods,
                quantiles=self.config.quantiles,
                bins=self.config.bins,
                filter_zscore=self.config.filter_zscore,
                max_loss=self.config.max_loss,
                binning_by_group=self.config.binning_by_group,
                zero_aware=self.config.zero_aware,
                cumulative_returns=self.config.cumulative_returns,
                groupby_labels=self.config.groupby_labels,
                long_short=self.config.long_short,
                group_neutral=self.config.group_neutral,
                highlights=kwargs.get("highlights"),
                table=save_tables,
                figure=False,
            )
            summary = self._engine._summary_frame_from_report(summary_sheet)
            if save_tables or save_report:
                self._persist_summary(summary, force_updated=force_updated)

            returns_report = self._engine.returns(
                periods=self.config.periods,
                quantiles=self.config.quantiles,
                bins=self.config.bins,
                filter_zscore=self.config.filter_zscore,
                max_loss=self.config.max_loss,
                binning_by_group=self.config.binning_by_group,
                zero_aware=self.config.zero_aware,
                cumulative_returns=self.config.cumulative_returns,
                groupby_labels=self.config.groupby_labels,
                rate_of_ret=True,
                group_neutral=self.config.group_neutral,
                by_group=self.config.by_group,
                table=save_tables,
                figure=save_imgs or save_report,
            )
            self._persist_sheet_section(
                section="returns",
                report=returns_report,
                save_imgs=save_imgs,
                save_tables=save_tables,
                save_report=save_report,
                force_updated=force_updated,
            )

            information_report = self._engine.information(
                periods=self.config.periods,
                quantiles=self.config.quantiles,
                bins=self.config.bins,
                filter_zscore=self.config.filter_zscore,
                max_loss=self.config.max_loss,
                binning_by_group=self.config.binning_by_group,
                zero_aware=self.config.zero_aware,
                cumulative_returns=self.config.cumulative_returns,
                groupby_labels=self.config.groupby_labels,
                group_neutral=self.config.group_neutral,
                by_group=self.config.by_group,
                table=save_tables,
                figure=save_imgs or save_report,
            )
            self._persist_sheet_section(
                section="information",
                report=information_report,
                save_imgs=save_imgs,
                save_tables=save_tables,
                save_report=save_report,
                force_updated=force_updated,
            )

            turnover_report = self._engine.turnover(
                periods=self.config.periods,
                quantiles=self.config.quantiles,
                bins=self.config.bins,
                filter_zscore=self.config.filter_zscore,
                max_loss=self.config.max_loss,
                binning_by_group=self.config.binning_by_group,
                zero_aware=self.config.zero_aware,
                cumulative_returns=self.config.cumulative_returns,
                groupby_labels=self.config.groupby_labels,
                turnover_periods=self.config.turnover_periods,
                table=save_tables,
                figure=save_imgs or save_report,
            )
            self._persist_sheet_section(
                section="turnover",
                report=turnover_report,
                save_imgs=save_imgs,
                save_tables=save_tables,
                save_report=save_report,
                force_updated=force_updated,
            )

            event_returns_report = self._engine.event_returns(
                periods=self.config.periods,
                quantiles=self.config.quantiles,
                bins=self.config.bins,
                filter_zscore=self.config.filter_zscore,
                max_loss=self.config.max_loss,
                binning_by_group=self.config.binning_by_group,
                zero_aware=self.config.zero_aware,
                cumulative_returns=self.config.cumulative_returns,
                groupby_labels=self.config.groupby_labels,
                avgretplot=self.config.avgretplot,
                long_short=self.config.long_short,
                group_neutral=self.config.group_neutral,
                by_group=self.config.by_group,
                table=save_tables,
                figure=save_imgs or save_report,
            )
            self._persist_sheet_section(
                section="event_returns",
                report=event_returns_report,
                save_imgs=save_imgs,
                save_tables=save_tables,
                save_report=save_report,
                force_updated=force_updated,
            )

            try:
                self._save_section(
                    "horizon",
                    format=fmt,
                    force_updated=force_updated,
                    **kwargs,
                )
            except FileExistsError:
                if force_updated:
                    raise

            if save_report:
                html = self._html_page("full", self._full_report_body())
                report_path = self._evaluation_store()._section_dir(self.spec, "full").parent / "report.html"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                if not force_updated and report_path.exists():
                    raise FileExistsError(f"evaluation section report already exists: {report_path}")
                if force_updated:
                    report_path.unlink(missing_ok=True)
                report_path.write_text(html, encoding="utf-8")
            return

        raise ValueError(f"unsupported evaluation section: {section!r}")

    @classmethod
    def open(
        cls,
        *,
        spec: FactorSpec,
    ) -> "SingleFactorEvaluation":
        return cls(
            factor_frame=None,
            price_frame=None,
            factor_column=spec.table_name,
            spec=spec,
        )

    def evaluate(self, **kwargs: Any):
        if self.factor_frame is None or self.price_frame is None or not self.factor_column:
            raise ValueError("factor_frame, price_frame, and factor_column are required to evaluate")
        return self._engine.evaluate(**kwargs)

    def summary(
        self,
        *,
        save: bool = False,
        force_updated: bool = False,
    ) -> pd.DataFrame:
        if self.factor_frame is None or self.price_frame is None or not self.factor_column:
            raise ValueError("factor_frame, price_frame, and factor_column are required to compute summary")
        summary = self._engine.summary(
            periods=self.config.periods,
            quantiles=self.config.quantiles,
            bins=self.config.bins,
            filter_zscore=self.config.filter_zscore,
            max_loss=self.config.max_loss,
            binning_by_group=self.config.binning_by_group,
            zero_aware=self.config.zero_aware,
            cumulative_returns=self.config.cumulative_returns,
            groupby_labels=self.config.groupby_labels,
            long_short=self.config.long_short,
            group_neutral=self.config.group_neutral,
            save=False,
        )
        if save:
            self._persist_summary(summary, force_updated=force_updated)
        return summary

    def returns(
        self,
        *,
        save: bool = False,
        force_updated: bool = False,
        periods: tuple[int, ...] = (1, 5, 10),
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
        table: bool = False,
        figure: bool = False,
    ) -> TigerEvaluationSheet:
        if self.factor_frame is None or self.price_frame is None or not self.factor_column:
            raise ValueError("factor_frame, price_frame, and factor_column are required to compute returns")
        report = self._engine.returns(
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
            table=table and not save,
            figure=figure,
        )
        if save:
            self._persist_report(report, force_updated=force_updated)
        return report

    def information(
        self,
        *,
        save: bool = False,
        force_updated: bool = False,
        periods: tuple[int, ...] = (1, 5, 10),
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
        table: bool = False,
        figure: bool = False,
    ) -> TigerEvaluationSheet:
        if self.factor_frame is None or self.price_frame is None or not self.factor_column:
            raise ValueError("factor_frame, price_frame, and factor_column are required to compute information")
        report = self._engine.information(
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
            table=table and not save,
            figure=figure,
        )
        if save:
            self._persist_report(report, force_updated=force_updated)
        return report

    def turnover(
        self,
        *,
        save: bool = False,
        force_updated: bool = False,
        periods: tuple[int, ...] = (1, 5, 10),
        quantiles: int | list[float] | None = 5,
        bins: int | list[float] | None = None,
        filter_zscore: float | None = 20,
        max_loss: float | None = 0.35,
        binning_by_group: bool = False,
        zero_aware: bool = False,
        cumulative_returns: bool = True,
        groupby_labels: dict[Any, str] | None = None,
        turnover_periods: tuple[int, ...] | None = None,
        table: bool = False,
        figure: bool = False,
    ) -> TigerEvaluationSheet:
        if self.factor_frame is None or self.price_frame is None or not self.factor_column:
            raise ValueError("factor_frame, price_frame, and factor_column are required to compute turnover")
        report = self._engine.turnover(
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
            table=table and not save,
            figure=figure,
        )
        if save:
            self._persist_report(report, force_updated=force_updated)
        return report

    def event_returns(
        self,
        *,
        save: bool = False,
        force_updated: bool = False,
        periods: tuple[int, ...] = (1, 5, 10),
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
        table: bool = False,
        figure: bool = False,
    ) -> TigerEvaluationSheet:
        if self.factor_frame is None or self.price_frame is None or not self.factor_column:
            raise ValueError("factor_frame, price_frame, and factor_column are required to compute event returns")
        report = self._engine.event_returns(
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
            table=table and not save,
            figure=figure,
        )
        if save:
            self._persist_report(report, force_updated=force_updated)
        return report

    def event_study(
        self,
        *,
        save: bool = False,
        force_updated: bool = False,
        periods: tuple[int, ...] = (1, 5, 10),
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
        turnover_periods: tuple[int, ...] | None = None,
        table: bool = False,
        figure: bool = False,
    ) -> TigerEvaluationSheet:
        if self.factor_frame is None or self.price_frame is None or not self.factor_column:
            raise ValueError("factor_frame, price_frame, and factor_column are required to compute event study")
        report = self._engine.event_study(
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
            table=table and not save,
            figure=figure,
        )
        if save:
            self._persist_report(report, force_updated=force_updated)
        return report

    def full(
        self,
        *,
        save: bool = False,
        force_updated: bool = False,
    ) -> TigerEvaluationSheet:
        if self.factor_frame is None or self.price_frame is None or not self.factor_column:
            raise ValueError("factor_frame, price_frame, and factor_column are required to compute full report")
        report = self._engine.full(
            periods=self.config.periods,
            quantiles=self.config.quantiles,
            bins=self.config.bins,
            filter_zscore=self.config.filter_zscore,
            max_loss=self.config.max_loss,
            binning_by_group=self.config.binning_by_group,
            zero_aware=self.config.zero_aware,
            cumulative_returns=self.config.cumulative_returns,
            groupby_labels=self.config.groupby_labels,
            long_short=self.config.long_short,
            group_neutral=self.config.group_neutral,
            by_group=self.config.by_group,
            turnover_periods=self.config.turnover_periods,
            avgretplot=self.config.avgretplot,
            include_horizon=self.config.include_horizon,
            horizons=self.config.horizons,
            horizon_quantiles=self.config.horizon_quantiles,
            horizon_periods_per_year=self.config.horizon_periods_per_year,
            horizon_long_short_pct=self.config.horizon_long_short_pct,
        )
        if save:
            self._persist_report(report, force_updated=force_updated)
        return report

    def exposure(
        self,
        *,
        periods: tuple[int, ...] | None = None,
        quantiles: int | list[float] | None = None,
        bins: int | list[float] | None = None,
        filter_zscore: float | None = None,
        max_loss: float | None = None,
        binning_by_group: bool | None = None,
        zero_aware: bool | None = None,
        cumulative_returns: bool | None = None,
        groupby_labels: dict[Any, str] | None = None,
        window: int = 20,
    ) -> dict[str, Any]:
        factor_data = self._factor_data_for_analysis(
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
        return analyze_factor_exposure(factor_data, window=window)

    def attribution(
        self,
        *,
        periods: tuple[int, ...] | None = None,
        quantiles: int | list[float] | None = None,
        bins: int | list[float] | None = None,
        filter_zscore: float | None = None,
        max_loss: float | None = None,
        binning_by_group: bool | None = None,
        zero_aware: bool | None = None,
        cumulative_returns: bool | None = None,
        groupby_labels: dict[Any, str] | None = None,
        benchmark_returns: pd.Series | None = None,
    ) -> dict[str, Any]:
        factor_data = self._factor_data_for_analysis(
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
        benchmark = benchmark_returns if benchmark_returns is not None else self._engine.benchmark_returns()
        return analyze_factor_attribution(factor_data, benchmark_returns=benchmark)

    def monitoring(
        self,
        *,
        periods: tuple[int, ...] | None = None,
        quantiles: int | list[float] | None = None,
        bins: int | list[float] | None = None,
        filter_zscore: float | None = None,
        max_loss: float | None = None,
        binning_by_group: bool | None = None,
        zero_aware: bool | None = None,
        cumulative_returns: bool | None = None,
        groupby_labels: dict[Any, str] | None = None,
        window: int = 60,
    ) -> pd.DataFrame:
        factor_data = self._factor_data_for_analysis(
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
        return monitor_factor_dynamics(factor_data, window=window)

    def effectiveness(
        self,
        *,
        periods: tuple[int, ...] | None = None,
        quantiles: int | list[float] | None = None,
        bins: int | list[float] | None = None,
        filter_zscore: float | None = None,
        max_loss: float | None = None,
        binning_by_group: bool | None = None,
        zero_aware: bool | None = None,
        cumulative_returns: bool | None = None,
        groupby_labels: dict[Any, str] | None = None,
        existing_factors: list[pd.DataFrame] | None = None,
        config: FactorEffectivenessConfig | None = None,
        benchmark_returns: pd.Series | None = None,
    ) -> dict[str, Any]:
        factor_data = self._factor_data_for_analysis(
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
        benchmark = benchmark_returns if benchmark_returns is not None else self._engine.benchmark_returns()
        return test_factor_effectiveness(
            factor_data,
            existing_factors=existing_factors,
            config=config,
            benchmark_returns=benchmark,
        )

    def native_full(
        self,
        *,
        quantiles: int = 5,
        portfolio_returns: pd.Series | None = None,
        benchmark_returns: pd.Series | None = None,
    ) -> FactorTearSheetResult:
        if self.factor_frame is None or self.price_frame is None or not self.factor_column:
            raise ValueError("factor_frame, price_frame, and factor_column are required to compute native full report")
        return self._engine._create_native_full_tear_sheet(
            quantiles=quantiles,
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
        )

    def analyze_horizons(
        self,
        *,
        horizons: tuple[int, ...] | list[int] | None = None,
        quantiles: int | None = None,
        periods_per_year: int | None = None,
        long_short_pct: float | None = None,
    ) -> pd.DataFrame:
        if self.factor_frame is None or self.price_frame is None or not self.factor_column:
            raise ValueError("factor_frame, price_frame, and factor_column are required to analyze horizons")
        return self._engine.analyze_horizons(
            tuple(horizons) if horizons is not None else self.config.horizons,
            quantiles=self.config.horizon_quantiles if quantiles is None else quantiles,
            periods_per_year=self.config.horizon_periods_per_year if periods_per_year is None else periods_per_year,
            long_short_pct=self.config.horizon_long_short_pct if long_short_pct is None else long_short_pct,
        )

    def summarize_best_horizon(
        self,
        *,
        horizons: tuple[int, ...] | list[int] | None = None,
        quantiles: int | None = None,
        periods_per_year: int | None = None,
        long_short_pct: float | None = None,
    ) -> dict[str, int | str | float]:
        if self.factor_frame is None or self.price_frame is None or not self.factor_column:
            raise ValueError("factor_frame, price_frame, and factor_column are required to summarize best horizon")
        return self._engine.summarize_best_horizon(
            tuple(horizons) if horizons is not None else self.config.horizons,
            quantiles=self.config.horizon_quantiles if quantiles is None else quantiles,
            periods_per_year=self.config.horizon_periods_per_year if periods_per_year is None else periods_per_year,
            long_short_pct=self.config.horizon_long_short_pct if long_short_pct is None else long_short_pct,
        )

    def run(
        self,
        *,
        include_native_report: bool = False,
    ) -> SingleFactorEvaluationResult:
        if self.factor_frame is None or self.price_frame is None or not self.factor_column:
            raise ValueError("factor_frame, price_frame, and factor_column are required to run evaluation")
        evaluation = self.evaluate()
        report = type(self).full(self, save=False)
        report_bundle = self._engine.create_report_bundle_summary(report)
        native_report = None
        if include_native_report:
            native_report = self.native_full()
        horizon_result = None
        horizon_summary = None
        if self.config.include_horizon:
            horizon_result = self.analyze_horizons()
            horizon_summary = self.summarize_best_horizon()
        return SingleFactorEvaluationResult(
            config=self.config,
            evaluation=evaluation,
            report=report,
            report_bundle=report_bundle,
            native_report=native_report,
            horizon_result=horizon_result,
            horizon_summary=horizon_summary,
            output_dir=str(report.output_dir),
        )

def evaluate_from_factor_frame(
    *,
    spec: FactorSpec,
    factor_frame: TableLike,
    price_frame: TableLike,
    factor_column: str | None = None,
    **kwargs: Any,
) -> SingleFactorEvaluationResult:
    include_native_report = bool(kwargs.pop("include_native_report", False))
    evaluation = SingleFactorEvaluation(
        factor_frame=factor_frame,
        price_frame=price_frame,
        factor_column=factor_column or spec.table_name,
        spec=spec,
        **kwargs,
    )
    return evaluation.run(include_native_report=include_native_report)


def evaluate_from_pipeline(
    *,
    spec: FactorSpec,
    pipeline: Pipeline,
    pipeline_engine: PipelineEngine,
    codes: list[str],
    start: str,
    end: str,
    factor_column: str | None = None,
    **kwargs: Any,
) -> SingleFactorEvaluationResult:
    factor_long = pipeline_engine.run_pipeline(
        pipeline,
        codes=codes,
        start=start,
        end=end,
    )
    if factor_long.empty:
        raise ValueError("Pipeline produced no factor rows.")
    resolved_factor_column = factor_column or spec.table_name
    if resolved_factor_column not in factor_long.columns:
        raise KeyError(f"Factor column {resolved_factor_column!r} not found in pipeline output.")

    factor_date_column = kwargs.pop("factor_date_column", "date_")
    factor_code_column = kwargs.pop("factor_code_column", "code")
    price_provider = kwargs.pop("price_provider", None)
    price_freq = kwargs.pop("price_freq", "1d")
    price_column = kwargs.pop("price_column", "close")
    forward_days = kwargs.pop("forward_days", 1)
    include_native_report = bool(kwargs.pop("include_native_report", False))

    factor_frame = factor_long[[factor_date_column, factor_code_column, resolved_factor_column]].copy()

    provider = price_provider or pipeline_engine.price_provider
    price_frame = pipeline_engine.library.fetch_price_data(
        codes=list(dict.fromkeys(map(str, codes))),
        start=start,
        end=end,
        provider=provider,
        freq=price_freq,
        as_ex=getattr(pipeline_engine, "as_ex", None),
    )
    if price_frame.empty:
        raise ValueError("Pipeline price fetch returned no rows.")

    evaluation = SingleFactorEvaluation(
        factor_frame=factor_frame,
        price_frame=price_frame,
        factor_column=resolved_factor_column,
        date_column=factor_date_column,
        code_column=factor_code_column,
        price_column=price_column,
        forward_days=forward_days,
        spec=spec,
        **kwargs,
    )
    return evaluation.run(include_native_report=include_native_report)


def evaluate_factor_frame(
    *,
    spec: FactorSpec,
    factor_frame: TableLike,
    price_frame: TableLike,
    factor_column: str | None = None,
    **kwargs: Any,
) -> None:
    evaluation = SingleFactorEvaluation(
        factor_frame=factor_frame,
        price_frame=price_frame,
        factor_column=factor_column or spec.table_name,
        spec=spec,
        **kwargs,
    )
    evaluation.summary()
    return None


__all__ = [
    "FactorEffectivenessConfig",
    "SingleFactorEvaluation",
    "SingleFactorEvaluationConfig",
    "SingleFactorEvaluationResult",
    "evaluate_from_factor_frame",
    "evaluate_from_pipeline",
    "evaluate_factor_frame",
]
