from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def build_detail_manifest(
    *,
    target: Path,
    artifacts_dir: Path,
    reports_dir: Path,
    summary_frame: pd.DataFrame,
    summary: pd.DataFrame,
    selection_summary: pd.DataFrame,
    return_long: pd.DataFrame,
    return_panel: pd.DataFrame,
    spec_summaries: pd.DataFrame,
    summary_payload: dict[str, object],
) -> FactorScreenerDetailManifest:
    return FactorScreenerDetailManifest(
        root_dir=str(target),
        artifacts_dir=str(artifacts_dir),
        reports_dir=str(reports_dir),
        pipeline_summary_parquet=str(artifacts_dir / "pipeline_summary.parquet") if not summary_frame.empty else None,
        pipeline_summary_csv=str(artifacts_dir / "pipeline_summary.csv") if not summary_frame.empty else None,
        batch_summary_parquet=str(artifacts_dir / "batch_summary.parquet") if not summary_frame.empty else None,
        batch_summary_csv=str(artifacts_dir / "batch_summary.csv") if not summary_frame.empty else None,
        summary_parquet=str(artifacts_dir / "summary.parquet") if not summary.empty else None,
        summary_csv=str(artifacts_dir / "summary.csv") if not summary.empty else None,
        spec_summaries_parquet=str(artifacts_dir / "spec_summaries.parquet") if not spec_summaries.empty else None,
        spec_summaries_csv=str(artifacts_dir / "spec_summaries.csv") if not spec_summaries.empty else None,
        selection_summary_parquet=str(artifacts_dir / "selection_summary.parquet") if not selection_summary.empty else None,
        selection_summary_csv=str(artifacts_dir / "selection_summary.csv") if not selection_summary.empty else None,
        return_long_parquet=str(artifacts_dir / "return_long.parquet") if not return_long.empty else None,
        return_long_csv=str(artifacts_dir / "return_long.csv") if not return_long.empty else None,
        return_panel_parquet=str(artifacts_dir / "return_panel.parquet") if not return_panel.empty else None,
        return_panel_csv=str(artifacts_dir / "return_panel.csv") if not return_panel.empty else None,
        manifest_json=str(reports_dir / "manifest.json"),
        report_json=str(reports_dir / "report.json"),
        summary_json=str(reports_dir / "summary.json"),
        summary=summary_payload,
    )


def save_batch_summary(result: Any, path: str | Path) -> Path:
    target = Path(path)
    summary = result.to_summary()
    if target.suffix == "":
        target.mkdir(parents=True, exist_ok=True)
        reports_dir = target / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        summary_path = reports_dir / "summary.json"
        _write_json(summary_path, summary)
        _write_json(target / "summary.json", summary)
        return summary_path

    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix.lower()
    if suffix == ".json":
        _write_json(target, summary)
        return target
    frame = result.to_summary_frame()
    if suffix == ".parquet":
        frame.to_parquet(target, index=False)
    elif suffix == ".csv":
        frame.to_csv(target, index=False)
    else:
        raise ValueError("summary path must end with .parquet, .csv, or .json")
    return target


def save_batch_detail(result: Any, path: str | Path) -> FactorScreenerDetailManifest:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    artifacts_dir = target / "artifacts"
    reports_dir = target / "reports"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_frame = result.to_summary_frame()
    if not summary_frame.empty:
        summary_frame.to_parquet(artifacts_dir / "pipeline_summary.parquet", index=False)
        summary_frame.to_csv(artifacts_dir / "pipeline_summary.csv", index=False)
        summary_frame.to_parquet(artifacts_dir / "batch_summary.parquet", index=False)
        summary_frame.to_csv(artifacts_dir / "batch_summary.csv", index=False)

    spec_summary = result.spec_summaries()
    if not spec_summary.empty:
        spec_summary.to_parquet(artifacts_dir / "spec_summaries.parquet", index=False)
        spec_summary.to_csv(artifacts_dir / "spec_summaries.csv", index=False)

    if not result.summary.empty:
        result.summary.to_parquet(artifacts_dir / "summary.parquet", index=False)
        result.summary.to_csv(artifacts_dir / "summary.csv", index=False)
    if not result.selection_summary.empty:
        result.selection_summary.to_parquet(artifacts_dir / "selection_summary.parquet", index=False)
        result.selection_summary.to_csv(artifacts_dir / "selection_summary.csv", index=False)
    if not result.return_long.empty:
        result.return_long.to_parquet(artifacts_dir / "return_long.parquet", index=False)
        result.return_long.to_csv(artifacts_dir / "return_long.csv", index=False)
    if not result.return_panel.empty:
        result.return_panel.to_parquet(artifacts_dir / "return_panel.parquet")
        result.return_panel.to_csv(artifacts_dir / "return_panel.csv")

    summary_payload = result.to_summary()
    manifest = build_detail_manifest(
        target=target,
        artifacts_dir=artifacts_dir,
        reports_dir=reports_dir,
        summary_frame=summary_frame,
        summary=result.summary,
        selection_summary=result.selection_summary,
        return_long=result.return_long,
        return_panel=result.return_panel,
        spec_summaries=spec_summary,
        summary_payload=summary_payload,
    )
    manifest_payload = manifest.to_dict()
    _write_json(reports_dir / "manifest.json", manifest_payload)
    _write_json(reports_dir / "report.json", summary_payload)
    _write_json(reports_dir / "summary.json", summary_payload)
    _write_json(target / "manifest.json", manifest_payload)
    _write_json(target / "report.json", summary_payload)
    _write_json(target / "summary.json", summary_payload)
    object.__setattr__(result, "detail_manifest", manifest)
    return manifest


__all__ = [
    "FactorScreenerDetailManifest",
    "build_detail_manifest",
    "save_batch_detail",
    "save_batch_summary",
]
