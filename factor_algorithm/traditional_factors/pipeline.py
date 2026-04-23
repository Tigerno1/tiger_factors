from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Sequence

import pandas as pd

from tiger_factors.factor_store import FactorResult
from tiger_factors.factor_store import TigerFactorLibrary

from .factor_functions import available_factors
from .factor_functions import factor_metadata
from .factor_functions import run_original_factor


DEFAULT_OUTPUT_ROOT = Path("/Volumes/Quant_Disk/factors")
DEFAULT_REGION = "us"
DEFAULT_SEC_TYPE = "stock"
DEFAULT_FREQ = "1m"
DEFAULT_VARIANT = "raw"


@dataclass(frozen=True)
class TraditionalFactorPipelineResult:
    factor_frames: dict[str, pd.DataFrame]
    saved_factor_results: dict[str, FactorResult] | None = None
    factor_errors: dict[str, str] | None = None
    output_root: str = ""
    manifest_path: str | None = None


def _month_end_date(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError("date value is missing")
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.to_period("M").to_timestamp("M")


def _normalize_naive_dates(values: pd.Series | pd.Index | list[Any]) -> pd.Series:
    series = pd.to_datetime(values, errors="coerce")
    if hasattr(series, "dt") and getattr(series.dt, "tz", None) is not None:
        series = series.dt.tz_localize(None)
    return series.dt.normalize()


def _canonicalize_result_frame(
    factor_name: str,
    data: pd.DataFrame | None,
    result: pd.Series | pd.DataFrame,
) -> pd.DataFrame:
    if isinstance(result, pd.Series):
        frame = result.to_frame(name=factor_name)
    else:
        frame = result.copy()

    if data is not None:
        base = data.copy()
        if factor_name in frame.columns and len(frame.columns) == 1:
            values = frame[factor_name].to_numpy()
        else:
            candidate_columns = [column for column in frame.columns if column not in {"permno", "yyyymm", "time_avail_m"}]
            if len(candidate_columns) != 1:
                raise ValueError(
                    f"{factor_name} result must have exactly one value column. "
                    f"Found columns={list(frame.columns)!r}"
                )
            values = frame[candidate_columns[0]].to_numpy()

        if len(base) != len(values):
            raise ValueError(
                f"{factor_name} aligned result length {len(values)} does not match input length {len(base)}."
            )

        if "date_" in base.columns:
            date_values = _normalize_naive_dates(base["date_"])
        elif "time_avail_m" in base.columns:
            date_values = _normalize_naive_dates(base["time_avail_m"])
        elif "yyyymm" in base.columns:
            date_values = base["yyyymm"].apply(_month_end_date)
        elif "date" in base.columns:
            date_values = _normalize_naive_dates(base["date"])
        else:
            raise ValueError(
                f"{factor_name} needs one of date_/time_avail_m/yyyymm/date on the input frame to build canonical storage."
            )

        if "code" in base.columns:
            code_values = base["code"].astype(str)
        elif "permno" in base.columns:
            code_values = base["permno"].astype(str)
        else:
            raise ValueError(
                f"{factor_name} needs a code or permno column on the input frame to build canonical storage."
            )

        canonical = pd.DataFrame(
            {
                "date_": date_values,
                "code": code_values,
                "value": pd.to_numeric(values, errors="coerce"),
            }
        )
        return canonical.dropna(subset=["date_", "code", "value"]).reset_index(drop=True)

    if "date_" not in frame.columns:
        if "time_avail_m" in frame.columns:
            frame["date_"] = _normalize_naive_dates(frame["time_avail_m"])
        elif "yyyymm" in frame.columns:
            frame["date_"] = frame["yyyymm"].apply(_month_end_date)
        elif "date" in frame.columns:
            frame["date_"] = _normalize_naive_dates(frame["date"])
        else:
            raise ValueError(
                f"{factor_name} result must contain date_/time_avail_m/yyyymm/date when no input frame is provided."
            )

    if "code" not in frame.columns:
        if "permno" in frame.columns:
            frame["code"] = frame["permno"].astype(str)
        else:
            raise ValueError(
                f"{factor_name} result must contain code or permno when no input frame is provided."
            )

    value_columns = [column for column in frame.columns if column not in {"date_", "code", "permno", "yyyymm", "time_avail_m"}]
    if factor_name in frame.columns:
        value_column = factor_name
    elif len(value_columns) == 1:
        value_column = value_columns[0]
    else:
        raise ValueError(
            f"{factor_name} result must contain a single value column; found {value_columns!r}"
        )

    canonical = frame.loc[:, ["date_", "code", value_column]].copy()
    canonical = canonical.rename(columns={value_column: "value"})
    canonical["date_"] = pd.to_datetime(canonical["date_"], errors="coerce").dt.normalize()
    canonical["code"] = canonical["code"].astype(str)
    canonical["value"] = pd.to_numeric(canonical["value"], errors="coerce")
    return canonical.dropna(subset=["date_", "code", "value"]).reset_index(drop=True)


def _factor_runner_metadata(
    factor_name: str,
    *,
    source: str = "openassetpricing",
    input_name: str | None = None,
    datasets: Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source": source,
        "family": "traditional",
        "factor_name": factor_name,
        "script_relpath": factor_metadata(factor_name).script_relpath,
        "inputs": list(factor_metadata(factor_name).inputs),
    }
    if input_name is not None:
        payload["input_name"] = input_name
    if datasets is not None:
        payload["provided_datasets"] = sorted({str(key) for key in datasets})
    if extra_metadata:
        payload.update(dict(extra_metadata))
    return payload


class TraditionalFactorPipelineEngine:
    """Batch runner for the vendored OpenAssetPricing signal catalog.

    The engine executes the upstream signal scripts via ``run_original_factor``,
    normalizes their outputs to Tiger canonical factor storage, and writes them
    through ``TigerFactorLibrary.save_factor`` so the results land in
    ``tiger_factors.factor_store``.
    """

    def __init__(
        self,
        *,
        library: TigerFactorLibrary | None = None,
        output_root: str | Path = DEFAULT_OUTPUT_ROOT,
        region: str = DEFAULT_REGION,
        sec_type: str = DEFAULT_SEC_TYPE,
        freq: str = DEFAULT_FREQ,
        variant: str = DEFAULT_VARIANT,
        verbose: bool = False,
    ) -> None:
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.library = library or TigerFactorLibrary(
            output_dir=self.output_root,
            region=region,
            sec_type=sec_type,
            verbose=verbose,
        )
        self.region = region
        self.sec_type = sec_type
        self.freq = freq
        self.variant = variant
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    @staticmethod
    def available_factors() -> list[str]:
        return available_factors()

    def compute_factor(
        self,
        name: str,
        *,
        data: pd.DataFrame | None = None,
        datasets: Mapping[str, Any] | None = None,
        input_name: str | None = None,
    ) -> pd.DataFrame:
        result = run_original_factor(
            name,
            data,
            input_name=input_name,
            datasets=datasets,
            return_frame=False,
        )
        return _canonicalize_result_frame(name, data, result)

    def save_factor(
        self,
        name: str,
        factor_frame: pd.DataFrame,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> FactorResult:
        payload = _factor_runner_metadata(
            name,
            input_name=None,
            datasets=None,
            extra_metadata=metadata,
        )
        payload.setdefault("region", self.region)
        payload.setdefault("sec_type", self.sec_type)
        payload.setdefault("freq", self.freq)
        payload.setdefault("variant", self.variant)
        return self.library.save_factor(
            factor_name=name,
            factor_df=factor_frame,
            metadata=dict(payload),
        )

    def compute_and_save_factor(
        self,
        name: str,
        *,
        data: pd.DataFrame | None = None,
        datasets: Mapping[str, Any] | None = None,
        input_name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[pd.DataFrame, FactorResult]:
        factor_frame = self.compute_factor(
            name,
            data=data,
            datasets=datasets,
            input_name=input_name,
        )
        payload = _factor_runner_metadata(
            name,
            input_name=input_name,
            datasets=datasets,
            extra_metadata=metadata,
        )
        payload.setdefault("region", self.region)
        payload.setdefault("sec_type", self.sec_type)
        payload.setdefault("freq", self.freq)
        payload.setdefault("variant", self.variant)
        saved = self.library.save_factor(
            factor_name=name,
            factor_df=factor_frame,
            metadata=dict(payload),
        )
        return factor_frame, saved

    def compute_all(
        self,
        *,
        factor_names: Sequence[str] | None = None,
        data: pd.DataFrame | None = None,
        datasets: Mapping[str, Any] | None = None,
        input_name: str | None = None,
        save: bool = True,
        skip_failed: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> TraditionalFactorPipelineResult:
        names = list(factor_names or available_factors())
        factor_frames: dict[str, pd.DataFrame] = {}
        saved_factor_results: dict[str, FactorResult] = {}
        factor_errors: dict[str, str] = {}

        for name in names:
            try:
                factor_frame = self.compute_factor(
                    name,
                    data=data,
                    datasets=datasets,
                    input_name=input_name,
                )
                factor_frames[name] = factor_frame
                if save:
                    payload = _factor_runner_metadata(
                        name,
                        input_name=input_name,
                        datasets=datasets,
                        extra_metadata=metadata,
                    )
                    payload.setdefault("region", self.region)
                    payload.setdefault("sec_type", self.sec_type)
                    payload.setdefault("freq", self.freq)
                    payload.setdefault("variant", self.variant)
                    saved_factor_results[name] = self.library.save_factor(
                        factor_name=name,
                        factor_df=factor_frame,
                        metadata=dict(payload),
                    )
            except Exception as exc:
                factor_errors[name] = str(exc)
                if not skip_failed:
                    raise

        manifest_payload = {
            "engine": "TraditionalFactorPipelineEngine",
            "factor_count": int(len(factor_frames)),
            "saved_count": int(len(saved_factor_results)),
            "error_count": int(len(factor_errors)),
            "region": self.region,
            "sec_type": self.sec_type,
            "freq": self.freq,
            "variant": self.variant,
            "factors": list(factor_frames),
            "errors": factor_errors,
            "metadata": dict(metadata or {}),
        }
        manifest_path = self.output_root / "traditional_factor_pipeline_manifest.json"
        manifest_path.write_text(json.dumps(manifest_payload, indent=2, default=str), encoding="utf-8")

        return TraditionalFactorPipelineResult(
            factor_frames=factor_frames,
            saved_factor_results=saved_factor_results or None,
            factor_errors=factor_errors or None,
            output_root=str(self.output_root),
            manifest_path=str(manifest_path),
        )


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_FREQ",
    "DEFAULT_REGION",
    "DEFAULT_SEC_TYPE",
    "DEFAULT_VARIANT",
    "TraditionalFactorPipelineEngine",
    "TraditionalFactorPipelineResult",
]
