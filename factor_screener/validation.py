from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tiger_factors.factor_evaluation.validation import *  # noqa: F401,F403


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _parquet_safe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    safe = frame.copy()

    def _encode(value: Any) -> Any:
        if isinstance(value, (list, tuple, set, dict)):
            return json.dumps(value, default=str)
        return value

    for column in safe.columns:
        safe[column] = safe[column].map(_encode)
    return safe


@dataclass(frozen=True)
class ScreeningEffectivenessSpec:
    selected_flag_column: str = "selected"
    factor_name_column: str = "factor_name"
    metric_fields: tuple[str, ...] = (
        "directional_fitness",
        "directional_ic_ir",
        "directional_sharpe",
        "ic_mean_abs",
        "rank_ic_mean_abs",
        "turnover",
        "decay_score",
        "capacity_score",
        "correlation_penalty",
        "regime_robustness",
        "out_of_sample_stability",
    )
    higher_is_better_fields: tuple[str, ...] = (
        "directional_fitness",
        "directional_ic_ir",
        "directional_sharpe",
        "ic_mean_abs",
        "rank_ic_mean_abs",
        "decay_score",
        "capacity_score",
        "regime_robustness",
        "out_of_sample_stability",
    )
    lower_is_better_fields: tuple[str, ...] = (
        "turnover",
        "correlation_penalty",
    )
    min_selected_ratio: float = 0.0
    min_retained_ratio: float = 0.8
    min_selected_count: int = 1


@dataclass(frozen=True)
class ScreeningEffectivenessResult:
    spec: ScreeningEffectivenessSpec
    total_count: int
    selected_count: int
    selected_ratio: float
    metric_table: pd.DataFrame
    passed: bool
    failed_rules: tuple[str, ...]

    def to_summary(self) -> dict[str, Any]:
        return {
            "total_count": int(self.total_count),
            "selected_count": int(self.selected_count),
            "selected_ratio": float(self.selected_ratio),
            "passed": bool(self.passed),
            "failed_rules": list(self.failed_rules),
            "metric_table": self.metric_table.to_dict(orient="records"),
            "spec": asdict(self.spec),
        }

    def comparison_frame(self) -> pd.DataFrame:
        frame = self.metric_table.copy()
        if frame.empty:
            return frame
        frame = frame.assign(
            total_count=int(self.total_count),
            selected_count=int(self.selected_count),
            selected_ratio=float(self.selected_ratio),
            passed_overall=bool(self.passed),
            failed_rules=", ".join(self.failed_rules),
        )
        preferred_columns = [
            "metric_name",
            "direction",
            "before_mean",
            "after_mean",
            "delta",
            "retained_ratio",
            "passed",
            "total_count",
            "selected_count",
            "selected_ratio",
            "passed_overall",
            "failed_rules",
        ]
        existing = [column for column in preferred_columns if column in frame.columns]
        remaining = [column for column in frame.columns if column not in existing]
        return frame.loc[:, existing + remaining]

    def to_summary_frame(self) -> pd.DataFrame:
        overall = pd.DataFrame(
            [
                {
                    "record_type": "overall",
                    "total_count": int(self.total_count),
                    "selected_count": int(self.selected_count),
                    "selected_ratio": float(self.selected_ratio),
                    "passed": bool(self.passed),
                    "failed_rules": list(self.failed_rules),
                    "spec": asdict(self.spec),
                }
            ]
        )
        detail = self.comparison_frame()
        if not detail.empty:
            detail = detail.assign(record_type="metric", spec=asdict(self.spec))
        combined = pd.concat([overall, detail], ignore_index=True, sort=False) if not detail.empty else overall
        preferred_columns = [
            "record_type",
            "metric_name",
            "direction",
            "total_count",
            "selected_count",
            "selected_ratio",
            "before_mean",
            "after_mean",
            "delta",
            "retained_ratio",
            "passed",
            "passed_overall",
            "failed_rules",
            "spec",
        ]
        existing = [column for column in preferred_columns if column in combined.columns]
        remaining = [column for column in combined.columns if column not in existing]
        return combined.loc[:, existing + remaining]

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        if target.suffix == "":
            manifest = self.save_detail(target)
            summary_path = manifest.get("screening_effectiveness_summary_json")
            return Path(summary_path) if summary_path is not None else target / "reports" / "screening_effectiveness_summary.json"

        target.parent.mkdir(parents=True, exist_ok=True)
        suffix = target.suffix.lower()
        if suffix == ".json":
            target.write_text(json.dumps(self.to_summary(), indent=2, default=str), encoding="utf-8")
            return target
        if suffix == ".parquet":
            _parquet_safe_frame(self.comparison_frame()).to_parquet(target, index=False)
            return target
        if suffix == ".csv":
            self.comparison_frame().to_csv(target, index=False)
            return target
        raise ValueError("path must end with .json, .parquet, .csv, or be a directory")

    def save_detail(self, path: str | Path) -> dict[str, str | None]:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        artifacts_dir = target / "artifacts"
        reports_dir = target / "reports"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        summary_frame = self.to_summary_frame()
        comparison = self.comparison_frame()
        if not summary_frame.empty:
            _parquet_safe_frame(summary_frame).to_parquet(artifacts_dir / "screening_effectiveness_summary_frame.parquet", index=False)
            summary_frame.to_csv(artifacts_dir / "screening_effectiveness_summary_frame.csv", index=False)
        if not comparison.empty:
            _parquet_safe_frame(comparison).to_parquet(artifacts_dir / "screening_effectiveness_comparison.parquet", index=False)
            comparison.to_csv(artifacts_dir / "screening_effectiveness_comparison.csv", index=False)

        summary_path = reports_dir / "screening_effectiveness_summary.json"
        summary_payload = self.to_summary()
        summary_path.write_text(json.dumps(summary_payload, indent=2, default=str), encoding="utf-8")
        _write_json(reports_dir / "manifest.json", {
            "root_dir": str(target),
            "artifacts_dir": str(artifacts_dir),
            "reports_dir": str(reports_dir),
            "screening_effectiveness_summary_frame_parquet": str(artifacts_dir / "screening_effectiveness_summary_frame.parquet") if not summary_frame.empty else None,
            "screening_effectiveness_summary_frame_csv": str(artifacts_dir / "screening_effectiveness_summary_frame.csv") if not summary_frame.empty else None,
            "screening_effectiveness_comparison_parquet": str(artifacts_dir / "screening_effectiveness_comparison.parquet") if not comparison.empty else None,
            "screening_effectiveness_comparison_csv": str(artifacts_dir / "screening_effectiveness_comparison.csv") if not comparison.empty else None,
            "screening_effectiveness_summary_json": str(summary_path),
            "summary": summary_payload,
        })
        _write_json(reports_dir / "report.json", summary_payload)
        _write_json(target / "manifest.json", {
            "root_dir": str(target),
            "artifacts_dir": str(artifacts_dir),
            "reports_dir": str(reports_dir),
            "screening_effectiveness_summary_frame_parquet": str(artifacts_dir / "screening_effectiveness_summary_frame.parquet") if not summary_frame.empty else None,
            "screening_effectiveness_summary_frame_csv": str(artifacts_dir / "screening_effectiveness_summary_frame.csv") if not summary_frame.empty else None,
            "screening_effectiveness_comparison_parquet": str(artifacts_dir / "screening_effectiveness_comparison.parquet") if not comparison.empty else None,
            "screening_effectiveness_comparison_csv": str(artifacts_dir / "screening_effectiveness_comparison.csv") if not comparison.empty else None,
            "screening_effectiveness_summary_json": str(summary_path),
            "summary": summary_payload,
        })
        _write_json(target / "report.json", summary_payload)
        _write_json(target / "screening_effectiveness_summary.json", summary_payload)
        return {
            "root_dir": str(target),
            "artifacts_dir": str(artifacts_dir),
            "reports_dir": str(reports_dir),
            "screening_effectiveness_summary_frame_parquet": str(artifacts_dir / "screening_effectiveness_summary_frame.parquet") if not summary_frame.empty else None,
            "screening_effectiveness_summary_frame_csv": str(artifacts_dir / "screening_effectiveness_summary_frame.csv") if not summary_frame.empty else None,
            "screening_effectiveness_comparison_parquet": str(artifacts_dir / "screening_effectiveness_comparison.parquet") if not comparison.empty else None,
            "screening_effectiveness_comparison_csv": str(artifacts_dir / "screening_effectiveness_comparison.csv") if not comparison.empty else None,
            "screening_effectiveness_summary_json": str(summary_path),
        }

    def save_summary(self, path: str | Path) -> Path:
        target = Path(path)
        summary_frame = self.to_summary_frame()
        summary_payload = self.to_summary()
        if target.suffix == "":
            target.mkdir(parents=True, exist_ok=True)
            reports_dir = target / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            summary_path = reports_dir / "screening_effectiveness_summary.json"
            summary_path.write_text(json.dumps(summary_payload, indent=2, default=str), encoding="utf-8")
            _parquet_safe_frame(summary_frame).to_parquet(reports_dir / "screening_effectiveness_summary_frame.parquet", index=False)
            summary_frame.to_csv(reports_dir / "screening_effectiveness_summary_frame.csv", index=False)
            _write_json(reports_dir / "manifest.json", {
                "root_dir": str(target),
                "reports_dir": str(reports_dir),
                "screening_effectiveness_summary_json": str(summary_path),
                "screening_effectiveness_summary_frame_parquet": str(reports_dir / "screening_effectiveness_summary_frame.parquet"),
                "screening_effectiveness_summary_frame_csv": str(reports_dir / "screening_effectiveness_summary_frame.csv"),
                "summary": summary_payload,
            })
            _write_json(target / "manifest.json", {
                "root_dir": str(target),
                "reports_dir": str(reports_dir),
                "screening_effectiveness_summary_json": str(summary_path),
                "screening_effectiveness_summary_frame_parquet": str(reports_dir / "screening_effectiveness_summary_frame.parquet"),
                "screening_effectiveness_summary_frame_csv": str(reports_dir / "screening_effectiveness_summary_frame.csv"),
                "summary": summary_payload,
            })
            return summary_path

        target.parent.mkdir(parents=True, exist_ok=True)
        suffix = target.suffix.lower()
        if suffix == ".json":
            target.write_text(json.dumps(summary_payload, indent=2, default=str), encoding="utf-8")
            return target
        if suffix == ".parquet":
            _parquet_safe_frame(summary_frame).to_parquet(target, index=False)
            return target
        if suffix == ".csv":
            summary_frame.to_csv(target, index=False)
            return target
        raise ValueError("path must end with .json, .parquet, .csv, or be a directory")


def _coerce_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()
    if not isinstance(frame, pd.DataFrame):
        return pd.DataFrame(frame)
    return frame.copy()


def _mean_value(frame: pd.DataFrame, field: str) -> float:
    if frame.empty or field not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[field], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return float("nan")
    return float(values.mean())


def validate_screening_effectiveness(
    summary: pd.DataFrame,
    selection_summary: pd.DataFrame | None = None,
    *,
    spec: ScreeningEffectivenessSpec | None = None,
) -> ScreeningEffectivenessResult:
    cfg = spec if spec is not None else ScreeningEffectivenessSpec()
    summary_frame = _coerce_frame(summary)
    selection_frame = _coerce_frame(selection_summary)

    if summary_frame.empty:
        empty = pd.DataFrame(
            columns=[
                "metric_name",
                "direction",
                "before_mean",
                "after_mean",
                "delta",
                "retained_ratio",
                "passed",
            ]
        )
        return ScreeningEffectivenessResult(
            spec=cfg,
            total_count=0,
            selected_count=0,
            selected_ratio=0.0,
            metric_table=empty,
            passed=False,
            failed_rules=("empty_summary",),
        )

    if cfg.factor_name_column not in summary_frame.columns:
        raise KeyError(f"Missing required factor name column: {cfg.factor_name_column!r}")

    if selection_frame.empty:
        if cfg.selected_flag_column not in summary_frame.columns:
            raise KeyError(
                f"Selection summary is empty and summary frame does not contain {cfg.selected_flag_column!r}"
            )
        merged = summary_frame.copy()
    else:
        if cfg.factor_name_column not in selection_frame.columns:
            raise KeyError(f"Missing required factor name column: {cfg.factor_name_column!r}")
        if cfg.selected_flag_column not in selection_frame.columns:
            raise KeyError(f"Missing required selected flag column: {cfg.selected_flag_column!r}")
        merged = summary_frame.merge(
            selection_frame[[cfg.factor_name_column, cfg.selected_flag_column]],
            on=cfg.factor_name_column,
            how="left",
            suffixes=("", "_selection"),
        )

    if cfg.selected_flag_column not in merged.columns:
        raise KeyError(f"Missing required selected flag column: {cfg.selected_flag_column!r}")

    selected_mask = merged[cfg.selected_flag_column].fillna(False).astype(bool)
    total_count = int(len(merged))
    selected_count = int(selected_mask.sum())
    selected_ratio = float(selected_count / total_count) if total_count > 0 else 0.0

    failed_rules: list[str] = []
    if selected_count < int(cfg.min_selected_count):
        failed_rules.append(f"selected_count<{int(cfg.min_selected_count)}")
    if selected_ratio < float(cfg.min_selected_ratio):
        failed_rules.append(f"selected_ratio<{float(cfg.min_selected_ratio)}")

    selected_frame = merged.loc[selected_mask].copy()
    metric_rows: list[dict[str, object]] = []
    for field in cfg.metric_fields:
        if field not in merged.columns:
            continue
        before_mean = _mean_value(merged, field)
        after_mean = _mean_value(selected_frame, field)
        delta = after_mean - before_mean if pd.notna(before_mean) and pd.notna(after_mean) else float("nan")
        direction = "higher" if field in cfg.higher_is_better_fields else "lower" if field in cfg.lower_is_better_fields else "neutral"
        if direction == "lower":
            retained_ratio = (
                float(before_mean / after_mean)
                if pd.notna(before_mean) and pd.notna(after_mean) and abs(after_mean) > 1e-12
                else float("nan")
            )
        else:
            retained_ratio = (
                float(after_mean / before_mean)
                if pd.notna(before_mean) and pd.notna(after_mean) and abs(before_mean) > 1e-12
                else float("nan")
            )

        passed_metric = True
        if pd.notna(retained_ratio) and retained_ratio < float(cfg.min_retained_ratio):
            passed_metric = False
            failed_rules.append(f"{field}_retained_ratio<{float(cfg.min_retained_ratio)}")

        metric_rows.append(
            {
                "metric_name": field,
                "direction": direction,
                "before_mean": before_mean,
                "after_mean": after_mean,
                "delta": delta,
                "retained_ratio": retained_ratio,
                "passed": passed_metric,
            }
        )

    metric_table = pd.DataFrame(metric_rows)
    passed = len(failed_rules) == 0 and not metric_table.empty
    return ScreeningEffectivenessResult(
        spec=cfg,
        total_count=total_count,
        selected_count=selected_count,
        selected_ratio=selected_ratio,
        metric_table=metric_table,
        passed=passed,
        failed_rules=tuple(dict.fromkeys(failed_rules)),
    )


__all__ = [
    *[name for name in globals().keys() if not name.startswith("_") and name not in {"Any", "np", "pd", "asdict", "dataclass"}],
    "ScreeningEffectivenessSpec",
    "ScreeningEffectivenessResult",
    "validate_screening_effectiveness",
]
