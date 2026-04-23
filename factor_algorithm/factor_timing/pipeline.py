from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Sequence

import pandas as pd

from tiger_factors.factor_store import FactorResult
from tiger_factors.factor_store import TigerFactorLibrary

from . import factor_timing_lib as timing_lib


DEFAULT_OUTPUT_ROOT = Path("/Volumes/Quant_Disk/factors")
DEFAULT_REGION = "us"
DEFAULT_SEC_TYPE = "stock"
DEFAULT_FREQ = "1m"
DEFAULT_VARIANT = "raw"


@dataclass(frozen=True)
class FactorTimingPipelineResult:
    factor_frames: dict[str, pd.DataFrame]
    saved_factor_results: dict[str, FactorResult] | None = None
    factor_errors: dict[str, str] | None = None
    output_root: str = ""
    manifest_path: str | None = None


def available_factors() -> list[str]:
    names: list[str] = []
    for name in dir(timing_lib):
        if not name.isupper():
            continue
        value = getattr(timing_lib, name)
        if callable(value):
            names.append(name)
    return sorted(names)


def _dataset_aliases(datasets: Mapping[str, Any] | None) -> dict[str, Any]:
    aliases: dict[str, Any] = {}
    for key, value in (datasets or {}).items():
        aliases[str(key)] = value
        aliases[Path(str(key)).name] = value
        aliases[Path(str(key)).stem] = value
    return aliases


def _coerce_wide_panel(price_df: pd.DataFrame) -> pd.DataFrame:
    frame = price_df.copy()
    if isinstance(frame.index, pd.DatetimeIndex):
        index = pd.to_datetime(frame.index)
        if getattr(index, "tz", None) is not None:
            index = index.tz_localize(None)
        frame.index = index.normalize()
        frame = frame.sort_index()
        frame.columns = frame.columns.astype(str)
        return frame
    if {"date_", "code"}.issubset(frame.columns):
        value_candidates = [column for column in frame.columns if column not in {"date_", "code"}]
        if len(value_candidates) != 1:
            raise ValueError(
                "Long-form factor timing input must contain exactly one value column besides date_/code."
            )
        value_column = value_candidates[0]
        pivot = (
            frame.loc[:, ["date_", "code", value_column]]
            .dropna(subset=["date_", "code"])
            .assign(date_=lambda df: pd.to_datetime(df["date_"], errors="coerce").dt.normalize())
            .pivot_table(index="date_", columns="code", values=value_column, aggfunc="last")
            .sort_index()
        )
        pivot.columns = pivot.columns.astype(str)
        return pivot
    raise ValueError(
        "factor timing expects either a wide DataFrame indexed by date or a long DataFrame with date_/code/value."
    )


def _resolve_extra_args(name: str, datasets: Mapping[str, Any] | None) -> list[Any]:
    fn = getattr(timing_lib, name)
    signature = inspect.signature(fn)
    params = list(signature.parameters.values())
    if len(params) <= 2:
        return []

    aliases = _dataset_aliases(datasets)
    args: list[Any] = []
    for param in params[2:]:
        value = aliases.get(param.name)
        if value is None:
            raise ValueError(f"{name} requires dataset {param.name!r}. Provide it via datasets={{...}}.")
        args.append(value)
    return args


def _normalize_result_frame(
    factor_name: str,
    price_df: pd.DataFrame,
    values: Sequence[pd.Series | pd.DataFrame | Any],
) -> pd.DataFrame:
    if not values:
        return pd.DataFrame(columns=["date_", "code", "value"])

    dates = pd.DatetimeIndex(pd.to_datetime(price_df.index, errors="coerce"))
    if getattr(dates, "tz", None) is not None:
        dates = dates.tz_localize(None)
    dates = dates.normalize()
    columns = pd.Index(price_df.columns.astype(str))
    frames: list[pd.DataFrame] = []
    for date_value, value in zip(dates, values, strict=False):
        if isinstance(value, pd.DataFrame):
            if value.shape[0] == 1:
                series = value.iloc[0]
            elif value.shape[1] == 1:
                series = value.iloc[:, 0]
            else:
                series = value.stack(dropna=False)
                if not isinstance(series, pd.Series):
                    series = pd.Series(series)
        elif isinstance(value, pd.Series):
            series = value
        else:
            series = pd.Series(value, index=columns)
        series = pd.Series(pd.to_numeric(series.reindex(columns), errors="coerce").to_numpy(), index=columns)
        frame = pd.DataFrame(
            {
                "date_": date_value,
                "code": columns,
                "value": series.to_numpy(),
            }
        )
        frame = frame.dropna(subset=["value"])
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["date_", "code", "value"])
    return pd.concat(frames, ignore_index=True).sort_values(["date_", "code"]).reset_index(drop=True)


class FactorTimingPipelineEngine:
    """Batch runner for the local timing-factor library.

    The engine consumes a wide price panel indexed by date and computes each
    timing factor into Tiger canonical factor storage using the shared
    ``TigerFactorLibrary.save_factor`` path.
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
        price_df: pd.DataFrame,
        *,
        datasets: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        wide = _coerce_wide_panel(price_df)
        fn = getattr(timing_lib, name)
        extra_args = _resolve_extra_args(name, datasets)
        values: list[pd.Series] = []
        for t in range(len(wide.index)):
            output = fn(wide, t, *extra_args)
            if isinstance(output, pd.DataFrame):
                if output.shape[0] == 1:
                    output = output.iloc[0]
                elif output.shape[1] == 1:
                    output = output.iloc[:, 0]
                else:
                    output = output.stack(dropna=False)
            if not isinstance(output, pd.Series):
                output = pd.Series(output, index=wide.columns)
            output = pd.Series(pd.to_numeric(output.reindex(wide.columns), errors="coerce").to_numpy(), index=wide.columns)
            values.append(output)
        return _normalize_result_frame(name, wide, values)

    def save_factor(
        self,
        name: str,
        factor_frame: pd.DataFrame,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> FactorResult:
        payload = {
            "source": "factor_timing",
            "family": "factor_timing",
            "factor_name": name,
            **dict(metadata or {}),
        }
        payload.setdefault("region", self.region)
        payload.setdefault("sec_type", self.sec_type)
        payload.setdefault("freq", self.freq)
        payload.setdefault("variant", self.variant)
        return self.library.save_factor(
            factor_name=name,
            factor_df=factor_frame,
            metadata=payload,
        )

    def compute_and_save_factor(
        self,
        name: str,
        price_df: pd.DataFrame,
        *,
        datasets: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[pd.DataFrame, FactorResult]:
        factor_frame = self.compute_factor(name, price_df, datasets=datasets)
        saved = self.save_factor(name, factor_frame, metadata=metadata)
        return factor_frame, saved

    def compute_all(
        self,
        price_df: pd.DataFrame,
        *,
        factor_names: Sequence[str] | None = None,
        datasets: Mapping[str, Any] | None = None,
        save: bool = True,
        skip_failed: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> FactorTimingPipelineResult:
        names = list(factor_names or available_factors())
        factor_frames: dict[str, pd.DataFrame] = {}
        saved_factor_results: dict[str, FactorResult] = {}
        factor_errors: dict[str, str] = {}

        for name in names:
            try:
                factor_frame = self.compute_factor(name, price_df, datasets=datasets)
                factor_frames[name] = factor_frame
                if save:
                    saved_factor_results[name] = self.save_factor(name, factor_frame, metadata=metadata)
            except Exception as exc:
                factor_errors[name] = str(exc)
                if not skip_failed:
                    raise

        manifest_payload = {
            "engine": "FactorTimingPipelineEngine",
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
        manifest_path = self.output_root / "factor_timing_pipeline_manifest.json"
        manifest_path.write_text(json.dumps(manifest_payload, indent=2, default=str), encoding="utf-8")

        return FactorTimingPipelineResult(
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
    "FactorTimingPipelineEngine",
    "FactorTimingPipelineResult",
    "available_factors",
]
