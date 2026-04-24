from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import is_dataclass
import re
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Mapping
import webbrowser

import pandas as pd

from .conf import DEFAULT_EVALUATION_ROOT
from .spec import FactorSpec


_SUMMARY_IMAGE_KIND_ORDER = {
    "ic_ts": 0,
    "histogram": 1,
    "qq": 2,
    "rolling": 3,
    "monthly": 4,
}
_RETURNS_IMAGE_KIND_ORDER = {
    "quantile_returns_bar": 0,
    "quantile_returns_violin": 1,
    "mean_return_spread_1D": 2,
    "mean_return_spread_5D": 3,
    "mean_return_spread_10D": 4,
}
_SUMMARY_PERIOD_UNIT_ORDER = {
    "D": 0,
    "W": 1,
    "M": 2,
    "Q": 3,
    "Y": 4,
}


def _summary_period_sort_key(period: str) -> tuple[int, int, str]:
    match = re.fullmatch(r"(?P<num>\d+)(?P<unit>[A-Za-z]+)", period)
    if match is None:
        return (1, 0, period.lower())
    unit = match.group("unit").upper()
    return (0, _SUMMARY_PERIOD_UNIT_ORDER.get(unit, 99), int(match.group("num")))


def _summary_img_sort_key(stem: str) -> tuple[int, tuple[int, int, str], int, str]:
    prefix = "signal_overview_"
    if not stem.startswith(prefix):
        return (1, (1, 0, stem.lower()), 99, stem)
    suffix = stem[len(prefix) :]
    for kind, kind_rank in _SUMMARY_IMAGE_KIND_ORDER.items():
        suffix_token = f"_{kind}"
        if suffix.endswith(suffix_token):
            period = suffix[: -len(suffix_token)]
            return (0, _summary_period_sort_key(period), kind_rank, stem)
    return (0, (1, 0, suffix.lower()), 99, stem)


def _returns_img_sort_key(stem: str) -> tuple[int, int, str]:
    if stem in _RETURNS_IMAGE_KIND_ORDER:
        return (0, _RETURNS_IMAGE_KIND_ORDER[stem], stem)
    if stem.startswith("mean_return_spread_"):
        period = stem[len("mean_return_spread_") :]
        return (0, 10 + _summary_period_sort_key(period)[1], stem)
    return (1, 99, stem)


def _evaluation_dataset_dir(spec: FactorSpec, root: str | Path) -> Path:
    parts = [
        "evaluation",
        spec.provider,
        spec.region,
        spec.sec_type,
        spec.freq,
    ]
    if getattr(spec, "group", None) is not None:
        parts.append(str(spec.group))
    return Path(root).joinpath(*parts, spec.data_stem())


@dataclass(frozen=True, slots=True)
class EvaluationPathResult:
    root_dir: Path
    run_dir: Path


@dataclass(frozen=True, slots=True)
class EvaluationSaveResult:
    spec: FactorSpec
    data: Any
    saved: bool
    root_dir: Path | None = None
    run_dir: Path | None = None
    summary_path: Path | None = None
    manifest_path: Path | None = None


@dataclass(frozen=True, slots=True)
class EvaluationSectionAccessor:
    """Read and optionally save one evaluation section such as summary/full/horizon."""

    store: "EvaluationStore"
    spec: FactorSpec
    section: str
    save_callback: Callable[..., Any] | None = None

    def tables(self) -> list[str]:
        return self.store.list_tables(self.spec, self.section)

    def imgs(self) -> list[str]:
        return self.store.list_imgs(self.spec, self.section)

    def report(self) -> str | None:
        report_path = self.store.get_report_path(self.spec, self.section)
        return None if report_path is None else report_path.stem

    def get_table(self, table_name: str | None = None) -> pd.DataFrame:
        section_dir = self.store._section_dir(self.spec, self.section)
        if table_name is None:
            default_file = section_dir / f"{self.section}.parquet"
            if default_file.exists():
                return pd.read_parquet(default_file)
            parquet_files = sorted(section_dir.glob("*.parquet"))
            if len(parquet_files) == 1:
                return pd.read_parquet(parquet_files[0])
            raise FileNotFoundError(
                f"Unable to infer table for section '{self.section}'. "
                f"Available parquet files: {[path.name for path in section_dir.glob('*.parquet')]!r}"
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
                    f"Ambiguous table '{table_name}' under section '{self.section}': "
                    f"{[path.as_posix() for path in matches]!r}"
                )
        raise FileNotFoundError(section_dir / f"{table_name}.parquet")

    def get_img(self, img_name: str):
        return self.store.get_img(self.spec, self.section, img_name)

    def get_report(self, *, open_browser: bool = True) -> Path:
        report_path = self.store.get_report_path(self.spec, self.section)
        if report_path is None:
            raise FileNotFoundError(
                f"report not found for section '{self.section}' and spec {self.spec.to_dict()}"
            )
        if open_browser:
            webbrowser.open(report_path.as_uri())
        return report_path

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.save_callback is None:
            raise TypeError(f"section accessor '{self.section}' is read-only")
        self.save_callback(*args, **kwargs)
        return None


class EvaluationStore:
    def __init__(
        self,
        root_dir: str | Path | None = None,
    ):
        self.root_dir = Path(root_dir) if root_dir is not None else DEFAULT_EVALUATION_ROOT
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def ensure_run_dir(self, *parts: str) -> EvaluationPathResult:
        run_dir = self.root_dir.joinpath(*parts)
        run_dir.mkdir(parents=True, exist_ok=True)
        return EvaluationPathResult(root_dir=self.root_dir, run_dir=run_dir)

    def _section_dir(self, spec: FactorSpec, section: str) -> Path:
        return _evaluation_dataset_dir(spec, self.root_dir) / section

    def list_tables(self, spec: FactorSpec, section: str) -> list[str]:
        section_dir = self._section_dir(spec, section)
        if not section_dir.exists():
            return []
        return sorted(
            path.stem
            for path in section_dir.glob("*.parquet")
            if path.is_file()
        )

    def list_imgs(self, spec: FactorSpec, section: str) -> list[str]:
        section_dir = self._section_dir(spec, section)
        if not section_dir.exists():
            return []
        stems = [path.stem for path in section_dir.glob("*.png") if path.is_file()]
        if section == "summary":
            return sorted(stems, key=_summary_img_sort_key)
        if section == "returns":
            return sorted(stems, key=_returns_img_sort_key)
        return sorted(stems)

    def get_report_path(self, spec: FactorSpec, section: str) -> Path | None:
        section_dir = self._section_dir(spec, section)
        candidates = sorted(section_dir.glob("*.html"))
        if not candidates:
            return None
        if len(candidates) > 1:
            report_candidates = [path for path in candidates if path.stem == "report"]
            if len(report_candidates) == 1:
                return report_candidates[0]
            raise FileNotFoundError(
                f"ambiguous report files for section '{section}': {[path.name for path in candidates]!r}"
            )
        return candidates[0]

    def get_img(self, spec: FactorSpec, section: str, img_name: str):
        from PIL import Image

        img_path = self._section_dir(spec, section) / f"{img_name}.png"
        if not img_path.exists():
            raise FileNotFoundError(img_path)
        return Image.open(img_path)

    def save_section_image(
        self,
        img: Any,
        *,
        spec: FactorSpec,
        section: str,
        img_name: str,
        force_updated: bool = False,
    ) -> Path:
        section_dir = self._section_dir(spec, section)
        section_dir.mkdir(parents=True, exist_ok=True)
        img_path = section_dir / f"{img_name}.png"
        if not force_updated and img_path.exists():
            raise FileExistsError(f"evaluation section image already exists: {img_path}")
        if force_updated:
            img_path.unlink(missing_ok=True)
        if isinstance(img, (str, Path)):
            source = Path(img)
            if not source.exists():
                raise FileNotFoundError(source)
            import shutil

            shutil.copy2(source, img_path)
        elif hasattr(img, "save"):
            img.save(img_path)
        else:
            raise TypeError(f"evaluation image must support save(); got {type(img)!r}")
        return img_path

    def save_section_report(
        self,
        html: str,
        *,
        spec: FactorSpec,
        section: str,
        report_name: str = "report",
        force_updated: bool = False,
    ) -> Path:
        section_dir = self._section_dir(spec, section)
        section_dir.mkdir(parents=True, exist_ok=True)
        report_path = section_dir / f"{report_name}.html"
        if not force_updated and report_path.exists():
            raise FileExistsError(f"evaluation section report already exists: {report_path}")
        if force_updated:
            report_path.unlink(missing_ok=True)
        report_path.write_text(html, encoding="utf-8")
        return report_path

    @staticmethod
    def _to_summary_frame(summary: Any) -> pd.DataFrame:
        if isinstance(summary, pd.DataFrame):
            return summary.copy()
        if isinstance(summary, pd.Series):
            return summary.to_frame().T.reset_index(drop=True)
        raise TypeError(
            "evaluation summary must be a pandas DataFrame or Series; "
            f"got {type(summary)!r}"
        )

    @staticmethod
    def _encode_payload(value: Any) -> Any:
        if isinstance(value, pd.DataFrame):
            return {
                "__kind__": "dataframe",
                "value": value.to_dict(orient="split"),
            }
        if isinstance(value, pd.Series):
            return {
                "__kind__": "series",
                "value": value.to_dict(),
                "name": value.name,
            }
        if is_dataclass(value):
            return {
                "__kind__": "dataclass",
                "value": EvaluationStore._encode_payload(asdict(value)),
            }
        if isinstance(value, Mapping):
            return {str(key): EvaluationStore._encode_payload(item) for key, item in value.items()}
        if isinstance(value, list):
            return [EvaluationStore._encode_payload(item) for item in value]
        if isinstance(value, tuple):
            return {"__kind__": "tuple", "value": [EvaluationStore._encode_payload(item) for item in value]}
        return value

    @staticmethod
    def _decode_payload(value: Any) -> Any:
        if isinstance(value, list):
            return [EvaluationStore._decode_payload(item) for item in value]
        if isinstance(value, Mapping):
            kind = value.get("__kind__")
            if kind == "dataframe":
                table = value["value"]
                return pd.DataFrame(**table)
            if kind == "series":
                series = pd.Series(value["value"])
                series.name = value.get("name")
                return series
            if kind == "tuple":
                return tuple(EvaluationStore._decode_payload(item) for item in value["value"])
            if kind == "dataclass":
                return EvaluationStore._decode_payload(value["value"])
            return {str(key): EvaluationStore._decode_payload(item) for key, item in value.items()}
        return value

    def save_evaluation(
        self,
        summary: Any,
        *,
        save: bool = True,
        force_updated: bool = False,
        metadata: dict[str, Any] | None = None,
        spec: FactorSpec,
    ) -> EvaluationSaveResult:
        frame: pd.DataFrame | None = None
        is_table = isinstance(summary, (pd.DataFrame, pd.Series))
        if is_table:
            frame = self._to_summary_frame(summary)
        if not save:
            return EvaluationSaveResult(spec=spec, data=frame if is_table else summary, saved=False)

        run_dir = _evaluation_dataset_dir(spec, self.root_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_dir = run_dir / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / "summary.parquet"
        summary_json_path = summary_dir / "summary.json"
        manifest_path = run_dir / "manifest.json"
        if not force_updated and (summary_path.exists() or summary_json_path.exists() or manifest_path.exists()):
            raise FileExistsError(f"evaluation already exists: {run_dir}")
        if force_updated:
            summary_path.unlink(missing_ok=True)
            summary_json_path.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)
        if is_table and frame is not None:
            frame.to_parquet(summary_path, index=False)
            stored_summary_path = summary_path
            stored_format = "table"
            stored_columns = list(frame.columns)
            stored_rows = int(len(frame))
        else:
            encoded_summary = self._encode_payload(summary)
            summary_json_path.write_text(json.dumps(encoded_summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            stored_summary_path = summary_json_path
            stored_format = "summary"
            if isinstance(summary, Mapping):
                stored_columns = list(summary.keys())
            elif is_dataclass(summary):
                stored_columns = list(asdict(summary).keys())
            else:
                stored_columns = None
            stored_rows = 1
        manifest_payload: dict[str, Any] = {
            "spec": spec.to_dict(),
            "root_dir": str(self.root_dir),
            "run_dir": str(run_dir),
            "summary_path": str(stored_summary_path),
            "format": stored_format,
            "rows": stored_rows,
            "columns": stored_columns,
            "created_at": pd.Timestamp.utcnow().isoformat(),
        }
        if metadata:
            manifest_payload["metadata"] = metadata
        manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return EvaluationSaveResult(
            spec=spec,
            data=frame if is_table else summary,
            saved=True,
            root_dir=self.root_dir,
            run_dir=run_dir,
            summary_path=summary_path if is_table else summary_json_path,
            manifest_path=manifest_path,
        )

    def save_summary(
        self,
        summary: Any,
        *,
        spec: FactorSpec,
        save: bool = True,
        force_updated: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationSaveResult:
        return self.save_evaluation(
            summary,
            save=save,
            force_updated=force_updated,
            metadata=metadata,
            spec=spec,
        )

    def save_section_table(
        self,
        table: pd.DataFrame | pd.Series,
        *,
        spec: FactorSpec,
        section: str,
        table_name: str | None = None,
        force_updated: bool = False,
    ) -> Path:
        run_dir = _evaluation_dataset_dir(spec, self.root_dir)
        section_dir = run_dir / section
        section_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"{table_name or section}"
        table_path = section_dir / f"{file_name}.parquet"
        if not force_updated and table_path.exists():
            raise FileExistsError(f"evaluation section table already exists: {table_path}")
        if force_updated:
            table_path.unlink(missing_ok=True)
        if isinstance(table, pd.Series):
            table = table.to_frame().T.reset_index(drop=True)
        if not isinstance(table, pd.DataFrame):
            raise TypeError(f"evaluation section table must be a pandas DataFrame or Series; got {type(table)!r}")
        table.to_parquet(table_path, index=False)
        return table_path

    def save_returns(
        self,
        table: pd.DataFrame | pd.Series,
        *,
        spec: FactorSpec,
        table_name: str | None = None,
        force_updated: bool = False,
    ) -> Path:
        return self.save_section_table(
            table,
            spec=spec,
            section="returns",
            table_name=table_name,
            force_updated=force_updated,
        )

    def save_information(
        self,
        table: pd.DataFrame | pd.Series,
        *,
        spec: FactorSpec,
        table_name: str | None = None,
        force_updated: bool = False,
    ) -> Path:
        return self.save_section_table(
            table,
            spec=spec,
            section="information",
            table_name=table_name,
            force_updated=force_updated,
        )

    def save_turnover(
        self,
        table: pd.DataFrame | pd.Series,
        *,
        spec: FactorSpec,
        table_name: str | None = None,
        force_updated: bool = False,
    ) -> Path:
        return self.save_section_table(
            table,
            spec=spec,
            section="turnover",
            table_name=table_name,
            force_updated=force_updated,
        )

    def save_event_returns(
        self,
        table: pd.DataFrame | pd.Series,
        *,
        spec: FactorSpec,
        table_name: str | None = None,
        force_updated: bool = False,
    ) -> Path:
        return self.save_section_table(
            table,
            spec=spec,
            section="event_returns",
            table_name=table_name,
            force_updated=force_updated,
        )

    def save_event_study(
        self,
        table: pd.DataFrame | pd.Series,
        *,
        spec: FactorSpec,
        table_name: str | None = None,
        force_updated: bool = False,
    ) -> Path:
        return self.save_section_table(
            table,
            spec=spec,
            section="event_study",
            table_name=table_name,
            force_updated=force_updated,
        )

    def save_full(
        self,
        table: pd.DataFrame | pd.Series,
        *,
        spec: FactorSpec,
        table_name: str | None = None,
        force_updated: bool = False,
    ) -> Path:
        return self.save_section_table(
            table,
            spec=spec,
            section="full",
            table_name=table_name,
            force_updated=force_updated,
        )

    def section(self, spec: FactorSpec, section: str, *, save_callback: Callable[..., Any] | None = None) -> EvaluationSectionAccessor:
        return EvaluationSectionAccessor(self, spec, section, save_callback=save_callback)

__all__ = [
    "DEFAULT_EVALUATION_ROOT",
    "EvaluationPathResult",
    "EvaluationSaveResult",
    "EvaluationStore",
]
