from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
from typing import Sequence

from dataclasses import dataclass
from dataclasses import field

import numpy as np
import pandas as pd

from tiger_factors.factor_screener._screener import FactorScreener
from tiger_factors.factor_screener._screener import FactorScreenerResult
from tiger_factors.factor_screener._screener import FactorScreenerSpec
from tiger_factors.factor_screener.selection import FactorMarginalSelectionConfig
from tiger_factors.factor_screener.selection import select_factors_by_marginal_gain
from tiger_factors.factor_screener.selection import select_non_redundant_factors
from tiger_factors.factor_store import FactorStore


class FactorSelectionMode:
    CORRELATION = "correlation"
    CONDITIONAL = "conditional"
    RETURN_GAIN = "return_gain"

    @classmethod
    def choices(cls) -> tuple[str, ...]:
        return (cls.CORRELATION, cls.CONDITIONAL, cls.RETURN_GAIN)

    @classmethod
    def normalize(cls, value: str | None) -> str:
        normalized = cls.CORRELATION if value is None else str(value).strip().lower()
        aliases = {
            "corr": cls.CORRELATION,
            "correlation": cls.CORRELATION,
            "conditional": cls.CONDITIONAL,
            "gain": cls.RETURN_GAIN,
            "return": cls.RETURN_GAIN,
            "return_gain": cls.RETURN_GAIN,
        }
        if normalized in aliases:
            return aliases[normalized]
        if normalized in cls.choices():
            return normalized
        raise ValueError(f"selection_mode must be one of: {', '.join(cls.choices())}")


def _resolve_selection_mode(selection_mode: str | None) -> str:
    return FactorSelectionMode.normalize(selection_mode)


@dataclass(frozen=True)
class FactorScreenerBatchItem:
    spec: FactorScreenerSpec
    label: str | None = None


@dataclass(frozen=True)
class FactorScreenerDetailManifest:
    root_dir: str
    artifacts_dir: str
    reports_dir: str
    pipeline_summary_parquet: str | None = None
    pipeline_summary_csv: str | None = None
    batch_summary_parquet: str | None = None
    batch_summary_csv: str | None = None
    summary_parquet: str | None = None
    summary_csv: str | None = None
    spec_summaries_parquet: str | None = None
    spec_summaries_csv: str | None = None
    selection_summary_parquet: str | None = None
    selection_summary_csv: str | None = None
    return_long_parquet: str | None = None
    return_long_csv: str | None = None
    return_panel_parquet: str | None = None
    return_panel_csv: str | None = None
    manifest_json: str | None = None
    report_json: str | None = None
    summary_json: str | None = None
    summary: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "root_dir": self.root_dir,
            "artifacts_dir": self.artifacts_dir,
            "reports_dir": self.reports_dir,
            "pipeline_summary_parquet": self.pipeline_summary_parquet,
            "pipeline_summary_csv": self.pipeline_summary_csv,
            "batch_summary_parquet": self.batch_summary_parquet,
            "batch_summary_csv": self.batch_summary_csv,
            "summary_parquet": self.summary_parquet,
            "summary_csv": self.summary_csv,
            "spec_summaries_parquet": self.spec_summaries_parquet,
            "spec_summaries_csv": self.spec_summaries_csv,
            "selection_summary_parquet": self.selection_summary_parquet,
            "selection_summary_csv": self.selection_summary_csv,
            "return_long_parquet": self.return_long_parquet,
            "return_long_csv": self.return_long_csv,
            "return_panel_parquet": self.return_panel_parquet,
            "return_panel_csv": self.return_panel_csv,
            "manifest_json": self.manifest_json,
            "report_json": self.report_json,
            "summary_json": self.summary_json,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class FactorReturnGainSelectionConfig:
    metric_fields: tuple[str, ...] = ("directional_fitness", "directional_ic_ir", "directional_sharpe")
    metric_weights: tuple[float, ...] = (0.50, 0.25, 0.25)
    metric_fallback_field: str = "selected_score"
    metric_penalty_fields: tuple[str, ...] = ("turnover", "max_drawdown")
    metric_penalty_weights: tuple[float, ...] = (0.20, 0.10)
    ann_return_weight: float = 1.0
    sharpe_weight: float = 1.0
    max_drawdown_weight: float = 0.5
    win_rate_weight: float = 0.0
    corr_weight: float = 0.50
    min_improvement: float = 0.0
    min_base_objective: float | None = None

    @classmethod
    def metric_focused(cls) -> FactorReturnGainSelectionConfig:
        return cls(
            metric_fields=("directional_fitness", "directional_ic_ir", "directional_sharpe"),
            metric_weights=(0.60, 0.25, 0.15),
            ann_return_weight=0.75,
            sharpe_weight=0.75,
            max_drawdown_weight=0.35,
            corr_weight=0.35,
        )

    @classmethod
    def return_focused(cls) -> FactorReturnGainSelectionConfig:
        return cls(
            metric_fields=("directional_fitness", "directional_ic_ir", "directional_sharpe"),
            metric_weights=(0.35, 0.25, 0.15),
            ann_return_weight=1.25,
            sharpe_weight=1.25,
            max_drawdown_weight=0.65,
            corr_weight=0.50,
        )

    @classmethod
    def robust(cls) -> FactorReturnGainSelectionConfig:
        return cls(
            metric_fields=("directional_fitness", "directional_ic_ir", "directional_sharpe"),
            metric_weights=(0.45, 0.30, 0.25),
            ann_return_weight=0.90,
            sharpe_weight=0.90,
            max_drawdown_weight=0.80,
            win_rate_weight=0.10,
            corr_weight=0.60,
        )

    @classmethod
    def balanced(cls) -> FactorReturnGainSelectionConfig:
        return cls()


def _resolve_return_gain_config(
    config: FactorReturnGainSelectionConfig | None,
    preset: str | None,
) -> FactorReturnGainSelectionConfig:
    if config is not None:
        return config
    if preset is None:
        return FactorReturnGainSelectionConfig()
    normalized = str(preset).strip().lower()
    if normalized in {"balanced", "default"}:
        return FactorReturnGainSelectionConfig.balanced()
    if normalized in {"metric", "metric_focused"}:
        return FactorReturnGainSelectionConfig.metric_focused()
    if normalized in {"return", "return_focused"}:
        return FactorReturnGainSelectionConfig.return_focused()
    if normalized in {"robust", "stable"}:
        return FactorReturnGainSelectionConfig.robust()
    raise ValueError(
        "preset must be one of: balanced, metric_focused, return_focused, robust"
    )


def _resolve_selection_mode(selection_mode: str | None) -> str:
    return FactorSelectionMode.normalize(selection_mode)


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

    def spec_summaries(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for index, (item, result) in enumerate(zip(self.spec.items, self.results)):
            label = _item_label(item, index)
            selected_names = result.selected_factor_names
            rows.append(
                {
                    "batch_index": index,
                    "batch_label": label,
                    "factor_count": int(len(result.spec.factor_names)),
                    "screened_factor_count": int(len(result.screened_factor_names)),
                    "selected_factor_count": int(len(selected_names)),
                    "missing_return_factor_count": int(len(result.missing_return_factors)),
                    "return_long_rows": int(len(result.return_long)),
                    "return_panel_columns": int(len(result.return_panel.columns)),
                    "return_modes": list(result.spec.return_modes),
                    "selected_factor_names": list(selected_names),
                    "screened_factor_names": list(result.screened_factor_names),
                    "rejected_factor_names": list(result.rejected_factor_names),
                }
            )
        return pd.DataFrame(rows)

    def to_summary(self) -> dict[str, object]:
        spec_summary = self.spec_summaries()
        selection_mode = _resolve_selection_mode(self.spec.selection_mode)
        return {
            "selection_mode": selection_mode,
            "return_gain_preset": self.spec.return_gain_preset,
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
                selection_mode=_resolve_selection_mode(self.spec.selection_mode),
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
            "return_gain_preset",
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
        target = Path(path)
        summary = self.to_summary()
        if target.suffix == "":
            target.mkdir(parents=True, exist_ok=True)
            reports_dir = target / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            summary_path = reports_dir / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
            (target / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
            return summary_path

        target.parent.mkdir(parents=True, exist_ok=True)
        suffix = target.suffix.lower()
        if suffix == ".json":
            target.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
            return target
        frame = self.to_summary_frame()
        if suffix == ".parquet":
            frame.to_parquet(target, index=False)
        elif suffix == ".csv":
            frame.to_csv(target, index=False)
        else:
            raise ValueError("summary path must end with .parquet, .csv, or .json")
        return target

    def save_detail(self, path: str | Path) -> FactorScreenerDetailManifest:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        artifacts_dir = target / "artifacts"
        reports_dir = target / "reports"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        summary_frame = self.to_summary_frame()
        if not summary_frame.empty:
            summary_frame.to_parquet(artifacts_dir / "pipeline_summary.parquet", index=False)
            summary_frame.to_csv(artifacts_dir / "pipeline_summary.csv", index=False)
            summary_frame.to_parquet(artifacts_dir / "batch_summary.parquet", index=False)
            summary_frame.to_csv(artifacts_dir / "batch_summary.csv", index=False)

        spec_summary = self.spec_summaries()
        if not spec_summary.empty:
            spec_summary.to_parquet(artifacts_dir / "spec_summaries.parquet", index=False)
            spec_summary.to_csv(artifacts_dir / "spec_summaries.csv", index=False)

        if not self.summary.empty:
            self.summary.to_parquet(artifacts_dir / "summary.parquet", index=False)
            self.summary.to_csv(artifacts_dir / "summary.csv", index=False)
        if not self.selection_summary.empty:
            self.selection_summary.to_parquet(artifacts_dir / "selection_summary.parquet", index=False)
            self.selection_summary.to_csv(artifacts_dir / "selection_summary.csv", index=False)
        if not self.return_long.empty:
            self.return_long.to_parquet(artifacts_dir / "return_long.parquet", index=False)
            self.return_long.to_csv(artifacts_dir / "return_long.csv", index=False)
        if not self.return_panel.empty:
            self.return_panel.to_parquet(artifacts_dir / "return_panel.parquet")
            self.return_panel.to_csv(artifacts_dir / "return_panel.csv")

        summary_payload = self.to_summary()
        manifest = FactorScreenerDetailManifest(
            root_dir=str(target),
            artifacts_dir=str(artifacts_dir),
            reports_dir=str(reports_dir),
            pipeline_summary_parquet=str(artifacts_dir / "pipeline_summary.parquet") if not summary_frame.empty else None,
            pipeline_summary_csv=str(artifacts_dir / "pipeline_summary.csv") if not summary_frame.empty else None,
            batch_summary_parquet=str(artifacts_dir / "batch_summary.parquet") if not summary_frame.empty else None,
            batch_summary_csv=str(artifacts_dir / "batch_summary.csv") if not summary_frame.empty else None,
            summary_parquet=str(artifacts_dir / "summary.parquet") if not self.summary.empty else None,
            summary_csv=str(artifacts_dir / "summary.csv") if not self.summary.empty else None,
            spec_summaries_parquet=str(artifacts_dir / "spec_summaries.parquet") if not spec_summary.empty else None,
            spec_summaries_csv=str(artifacts_dir / "spec_summaries.csv") if not spec_summary.empty else None,
            selection_summary_parquet=str(artifacts_dir / "selection_summary.parquet") if not self.selection_summary.empty else None,
            selection_summary_csv=str(artifacts_dir / "selection_summary.csv") if not self.selection_summary.empty else None,
            return_long_parquet=str(artifacts_dir / "return_long.parquet") if not self.return_long.empty else None,
            return_long_csv=str(artifacts_dir / "return_long.csv") if not self.return_long.empty else None,
            return_panel_parquet=str(artifacts_dir / "return_panel.parquet") if not self.return_panel.empty else None,
            return_panel_csv=str(artifacts_dir / "return_panel.csv") if not self.return_panel.empty else None,
            manifest_json=str(reports_dir / "manifest.json"),
            report_json=str(reports_dir / "report.json"),
            summary_json=str(reports_dir / "summary.json"),
            summary=summary_payload,
        )
        manifest_payload = manifest.to_dict()
        (reports_dir / "manifest.json").write_text(json.dumps(manifest_payload, indent=2, default=str), encoding="utf-8")
        (reports_dir / "report.json").write_text(json.dumps(summary_payload, indent=2, default=str), encoding="utf-8")
        (reports_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, default=str), encoding="utf-8")
        (target / "manifest.json").write_text(json.dumps(manifest_payload, indent=2, default=str), encoding="utf-8")
        (target / "report.json").write_text(json.dumps(summary_payload, indent=2, default=str), encoding="utf-8")
        (target / "summary.json").write_text(json.dumps(summary_payload, indent=2, default=str), encoding="utf-8")
        object.__setattr__(self, "detail_manifest", manifest)
        return manifest


def _annualized_return_stats(returns: pd.Series, annual_trading_days: int = 252) -> dict[str, float]:
    series = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return {
            "ann_return": 0.0,
            "ann_volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
        }
    cumulative = (1.0 + series).cumprod()
    total_return = float(cumulative.iloc[-1] - 1.0)
    ann_return = float((1.0 + total_return) ** (annual_trading_days / max(len(series), 1)) - 1.0)
    ann_vol = float(series.std(ddof=0) * np.sqrt(annual_trading_days))
    sharpe = float(ann_return / ann_vol) if ann_vol > 1e-12 else 0.0
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    win_rate = float((series > 0).mean())
    return {
        "ann_return": ann_return,
        "ann_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
    }


def _return_objective(returns: pd.Series, config: FactorReturnGainSelectionConfig) -> float:
    stats = _annualized_return_stats(returns)
    return (
        float(config.ann_return_weight) * float(stats["ann_return"])
        + float(config.sharpe_weight) * float(stats["sharpe"])
        - float(config.max_drawdown_weight) * abs(float(stats["max_drawdown"]))
        + float(config.win_rate_weight) * float(stats["win_rate"])
    )


def _metric_objective(frame: pd.DataFrame, config: FactorReturnGainSelectionConfig) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    if len(config.metric_fields) != len(config.metric_weights):
        raise ValueError("metric_fields and metric_weights must have the same length")
    if len(config.metric_penalty_fields) != len(config.metric_penalty_weights):
        raise ValueError("metric_penalty_fields and metric_penalty_weights must have the same length")

    metric_score = pd.Series(0.0, index=frame.index, dtype=float)
    metric_weight_total = 0.0
    for field, weight in zip(config.metric_fields, config.metric_weights):
        if field not in frame.columns:
            continue
        values = pd.to_numeric(frame[field], errors="coerce")
        metric_score = metric_score.add(values.fillna(0.0) * float(weight), fill_value=0.0)
        metric_weight_total += float(weight)
    if metric_weight_total > 0:
        metric_score = metric_score / metric_weight_total
    elif config.metric_fallback_field in frame.columns:
        metric_score = pd.to_numeric(frame[config.metric_fallback_field], errors="coerce")
    elif "fitness" in frame.columns:
        metric_score = pd.to_numeric(frame["fitness"], errors="coerce")

    metric_penalty = pd.Series(0.0, index=frame.index, dtype=float)
    for field, weight in zip(config.metric_penalty_fields, config.metric_penalty_weights):
        if field not in frame.columns:
            continue
        values = pd.to_numeric(frame[field], errors="coerce").abs()
        metric_penalty = metric_penalty.add(values.fillna(0.0) * float(weight), fill_value=0.0)

    return metric_score - metric_penalty


def _item_label(item: FactorScreenerBatchItem, index: int) -> str:
    label = str(item.label).strip() if item.label is not None else ""
    return label or f"spec_{index + 1}"


def _spec_signature(spec: FactorScreenerSpec) -> str:
    variant = spec.variant or "default"
    return ":".join(
        [
            str(spec.provider),
            str(spec.region),
            str(spec.sec_type),
            str(spec.freq),
            str(variant),
            str(spec.summary_section),
            str(spec.returns_section),
        ]
    )


def _tag_frame(
    frame: pd.DataFrame,
    *,
    batch_index: int,
    batch_label: str,
    batch_signature: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    tagged = frame.copy()
    tagged["batch_index"] = batch_index
    tagged["batch_label"] = batch_label
    tagged["batch_signature"] = batch_signature
    if "factor_name" in tagged.columns:
        tagged["batch_factor_key"] = tagged["factor_name"].astype(str).map(lambda name: f"{batch_label}::{name}")
    return tagged


def _score_series(
    frame: pd.DataFrame,
    *,
    score_field: str,
) -> pd.Series:
    if frame.empty or "factor_name" not in frame.columns:
        return pd.Series(dtype=float)

    if score_field in frame.columns:
        values = pd.to_numeric(frame[score_field], errors="coerce")
    elif score_field == "selected_score" and "selected_score" in frame.columns:
        values = pd.to_numeric(frame["selected_score"], errors="coerce")
    elif "selected_score" in frame.columns:
        values = pd.to_numeric(frame["selected_score"], errors="coerce")
    elif "fitness" in frame.columns:
        values = pd.to_numeric(frame["fitness"], errors="coerce").abs()
    else:
        values = pd.Series(1.0, index=frame.index, dtype=float)

    values.index = frame["factor_name"].astype(str)
    return values.replace([pd.NA, pd.NaT], pd.NA)


def _primary_return_mode(spec: FactorScreenerSpec) -> str:
    if "long_short" in spec.return_modes:
        return "long_short"
    if spec.return_modes:
        return spec.return_modes[0]
    return "long_short"


def _extract_primary_return_series(
    result: FactorScreenerResult,
    *,
    primary_mode: str,
) -> dict[str, pd.Series]:
    extracted: dict[str, pd.Series] = {}
    for key, series in result.return_series.items():
        if not key.endswith(f":{primary_mode}"):
            continue
        factor_name = key.split(":", 1)[0]
        if isinstance(series, pd.Series) and not series.empty:
            extracted[factor_name] = series.copy()
    return extracted


def _combined_return_panel(results: list[tuple[str, FactorScreenerResult]]) -> pd.DataFrame:
    frames: dict[str, pd.Series] = {}
    for batch_label, result in results:
        for key, series in result.return_series.items():
            if series.empty:
                continue
            composite_key = f"{batch_label}::{key}"
            frames[composite_key] = series
    if not frames:
        return pd.DataFrame()
    frame = pd.DataFrame(frames).sort_index()
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame.loc[~frame.index.isna()].sort_index()
    return frame


def _collect_candidate_panels(
    screeners: list[tuple[str, FactorScreener]],
    selection_summary: pd.DataFrame,
    *,
    score_field: str,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, float]]:
    candidate_panels: dict[str, pd.DataFrame] = {}
    candidate_rows: list[pd.DataFrame] = []
    candidate_scores: dict[str, float] = {}

    for label, screener in screeners:
        frame = selection_summary.loc[selection_summary["batch_label"] == label].copy()
        if frame.empty or "factor_name" not in frame.columns:
            continue
        if "selected" in frame.columns:
            frame = frame.loc[frame["selected"].fillna(False)]
        elif "usable" in frame.columns:
            frame = frame.loc[frame["usable"].fillna(False)]
        if frame.empty:
            continue

        factor_names = frame["factor_name"].astype(str).tolist()
        loaded_panels = screener.library.load_factor_panels(
            factor_names=factor_names,
            provider=screener.spec.provider,
            freq=screener.spec.freq,
            variant=screener.spec.variant,
        )
        scores = _score_series(frame, score_field=score_field)

        for factor_name, panel in loaded_panels.items():
            if not isinstance(panel, pd.DataFrame) or panel.empty:
                continue
            composite_key = f"{label}::{factor_name}"
            candidate_panels[composite_key] = panel
            score = scores.get(factor_name)
            candidate_scores[composite_key] = float(score) if pd.notna(score) else 0.0

        if "batch_factor_key" in frame.columns:
            candidate_rows.append(
                frame.assign(batch_factor_key=frame["batch_factor_key"].astype(str))
            )
        else:
            candidate_rows.append(
                frame.assign(batch_factor_key=frame["factor_name"].astype(str).map(lambda name: f"{label}::{name}"))
            )

    candidates = pd.concat(candidate_rows, ignore_index=True) if candidate_rows else pd.DataFrame()
    return candidate_panels, candidates, candidate_scores


def _collect_candidate_returns(
    screeners: list[tuple[str, FactorScreener]],
    results: list[tuple[str, FactorScreenerResult]],
    selection_summary: pd.DataFrame,
) -> tuple[dict[str, pd.Series], dict[str, str]]:
    candidate_returns: dict[str, pd.Series] = {}
    factor_to_label: dict[str, str] = {}

    result_lookup = {label: result for label, result in results}
    for label, screener in screeners:
        frame = selection_summary.loc[selection_summary["batch_label"] == label].copy()
        if frame.empty or "factor_name" not in frame.columns:
            continue
        if "selected" in frame.columns:
            frame = frame.loc[frame["selected"].fillna(False)]
        elif "usable" in frame.columns:
            frame = frame.loc[frame["usable"].fillna(False)]
        if frame.empty:
            continue

        result = result_lookup.get(label)
        if result is None:
            continue

        primary_mode = _primary_return_mode(screener.spec)
        extracted = _extract_primary_return_series(result, primary_mode=primary_mode)
        for factor_name, series in extracted.items():
            batch_key = f"{label}::{factor_name}"
            candidate_returns[batch_key] = series
            factor_to_label[batch_key] = label

    return candidate_returns, factor_to_label


def _greedy_select_by_return_gain(
    candidate_returns: dict[str, pd.Series],
    candidate_scores: dict[str, float],
    candidate_metrics: pd.DataFrame,
    *,
    config: FactorReturnGainSelectionConfig,
) -> list[str]:
    if not candidate_returns:
        return []

    ordered = sorted(
        candidate_returns.keys(),
        key=lambda key: (float(candidate_scores.get(key, 0.0)), key),
        reverse=True,
    )
    selected: list[str] = []
    current_objective = float("-inf")

    metric_frame = candidate_metrics.copy()
    if not metric_frame.empty and "batch_factor_key" in metric_frame.columns:
        metric_frame = metric_frame.set_index("batch_factor_key", drop=False)
    metric_frame = metric_frame.reindex(candidate_returns.keys())
    metric_scores = _metric_objective(metric_frame, config)
    metric_scores.index = metric_frame.index

    for key in ordered:
        trial = selected + [key]
        frame = pd.DataFrame({name: candidate_returns[name] for name in trial}).sort_index()
        combined = frame.mean(axis=1, skipna=True).dropna()
        if combined.empty:
            continue
        metric_component = float(pd.to_numeric(metric_scores.reindex(trial), errors="coerce").dropna().mean()) if trial else 0.0
        return_component = _return_objective(combined, config)
        corr_penalty = 0.0
        if len(trial) > 1:
            corr_frame = pd.DataFrame({name: candidate_returns[name] for name in trial}).corr().abs()
            pair_values: list[float] = []
            for idx, left in enumerate(trial):
                for right in trial[idx + 1 :]:
                    if left in corr_frame.index and right in corr_frame.columns:
                        value = corr_frame.loc[left, right]
                        if pd.notna(value):
                            pair_values.append(float(value))
            corr_penalty = float(np.mean(pair_values)) if pair_values else 0.0
        trial_objective = metric_component + return_component - float(config.corr_weight) * corr_penalty

        if not selected:
            if config.min_base_objective is not None and trial_objective < config.min_base_objective:
                continue
            if trial_objective >= config.min_improvement:
                selected.append(key)
                current_objective = trial_objective
            continue

        if trial_objective >= current_objective + config.min_improvement:
            selected.append(key)
            current_objective = trial_objective

    return selected


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
        label = _item_label(item, index)
        screener = FactorScreener(item.spec, store=self.store)
        result = screener.run()
        return label, screener, result

    def run(self) -> FactorScreenerBatchResult:
        items = self.spec.normalized_items()
        selection_mode = _resolve_selection_mode(self.spec.selection_mode)
        results: list[tuple[str, FactorScreenerResult]] = []
        screeners: list[tuple[str, FactorScreener]] = []

        summary_frames: list[pd.DataFrame] = []
        selection_frames: list[pd.DataFrame] = []
        return_long_frames: list[pd.DataFrame] = []

        for index, item in enumerate(items):
            label, screener, result = self._run_item(item, index=index)
            signature = _spec_signature(item.spec)
            results.append((label, result))
            screeners.append((label, screener))
            summary_frames.append(
                _tag_frame(
                    result.summary,
                    batch_index=index,
                    batch_label=label,
                    batch_signature=signature,
                )
            )
            selection_frames.append(
                _tag_frame(
                    result.selection_summary,
                    batch_index=index,
                    batch_label=label,
                    batch_signature=signature,
                )
            )
            return_long_frames.append(
                _tag_frame(
                    result.return_long,
                    batch_index=index,
                    batch_label=label,
                    batch_signature=signature,
                )
            )

        summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
        selection_summary = pd.concat(selection_frames, ignore_index=True) if selection_frames else pd.DataFrame()
        return_long = pd.concat(return_long_frames, ignore_index=True) if return_long_frames else pd.DataFrame()
        return_panel = _combined_return_panel(results)

        global_selected_keys: tuple[str, ...] = ()
        global_selected_names: tuple[str, ...] = ()
        if self.spec.cross_spec_selection_threshold is not None and not selection_summary.empty:
            candidate_panels, candidate_frame, candidate_scores = _collect_candidate_panels(
                screeners,
                selection_summary,
                score_field=self.spec.cross_spec_selection_score_field,
            )
            candidate_returns, _ = _collect_candidate_returns(screeners, results, selection_summary)

            if candidate_panels and not candidate_frame.empty:
                if selection_mode == FactorSelectionMode.CONDITIONAL:
                    selected_keys = select_factors_by_marginal_gain(
                        candidate_panels,
                        candidate_frame,
                        config=self.spec.marginal_selection_config,
                        key_column="batch_factor_key",
                    )
                elif selection_mode == FactorSelectionMode.RETURN_GAIN:
                    selected_keys = _greedy_select_by_return_gain(
                        candidate_returns,
                        candidate_scores,
                        candidate_frame,
                        config=self.spec.return_gain_config,
                    )
                else:
                    selected_keys = select_non_redundant_factors(
                        candidate_panels,
                        candidate_scores,
                        threshold=float(self.spec.cross_spec_selection_threshold),
                    )
                global_selected_keys = tuple(selected_keys)
                global_selected_names = tuple(key.split("::", 1)[-1] for key in selected_keys)

                if not selection_summary.empty and "batch_factor_key" in selection_summary.columns:
                    selection_summary = selection_summary.copy()
                    selection_summary["global_selected"] = selection_summary["batch_factor_key"].astype(str).isin(selected_keys)

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
        selection_mode=_resolve_selection_mode(selection_mode),
        marginal_selection_config=marginal_selection_config or FactorMarginalSelectionConfig(),
        return_gain_config=_resolve_return_gain_config(return_gain_config, return_gain_preset),
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
        selection_mode=_resolve_selection_mode(selection_mode),
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
