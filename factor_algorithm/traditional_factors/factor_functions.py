"""Wrappers around the vendored OpenAssetPricing factor scripts.

This module exposes the original Chen-Zimmermann signal builders as regular
Python callables inside ``tiger_factors.factor_algorithm.traditional_factors``.

The vendored source lives under ``traditional_factors/openassetpricing`` and is
kept in the same folder layout as the upstream project.  Each public function in
this module delegates to the corresponding upstream script, writes the required
intermediate datasets into a temporary workspace, executes the original script,
and then aligns the produced signal back to the caller's input frame.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import runpy
import shutil
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent
OPEN_ASSET_PRICING_ROOT = MODULE_DIR / "openassetpricing"
OPEN_ASSET_PRICING_PYCODE = OPEN_ASSET_PRICING_ROOT / "Signals" / "pyCode"
OPEN_ASSET_PRICING_METADATA = OPEN_ASSET_PRICING_ROOT / "metadata.json"

_SCRIPT_INPUT_PATTERN = re.compile(
    r"pyData/Intermediate/([A-Za-z0-9_.-]+\.(?:parquet|csv))"
)
_SCRIPT_INPUT_PATH_PATTERN = re.compile(
    r"(?:data_path|input_dir|path_data|pathDataIntermediate)\s*/\s*[\"']([A-Za-z0-9_.-]+\.(?:parquet|csv))[\"']"
)
_OPTIONAL_HELPERS = {
    "SignalMasterTable.parquet": {"permno", "time_avail_m"},
    "monthlyCRSP.parquet": {"permno", "time_avail_m", "ret"},
    "dailyCRSP.parquet": {"permno", "time_d", "ret"},
    "m_aCompustat.parquet": {"gvkey", "permno", "time_avail_m"},
    "a_aCompustat.parquet": {"gvkey", "permno", "time_avail_m", "datadate"},
    "monthlyFF.parquet": {"time_avail_m", "mktrf"},
}


@dataclass(frozen=True)
class SignalMetadata:
    name: str
    script_relpath: str
    output_dir: str
    inputs: tuple[str, ...]
    category: str
    authors: str | None
    year: str | None
    journal: str | None
    long_description: str | None
    detailed_definition: str | None

    @property
    def script_path(self) -> Path:
        return OPEN_ASSET_PRICING_ROOT / self.script_relpath

    @property
    def output_filename(self) -> str:
        return f"{self.name}.csv"


@lru_cache(maxsize=1)
def _load_signal_metadata() -> dict[str, SignalMetadata]:
    payload = json.loads(OPEN_ASSET_PRICING_METADATA.read_text())
    metadata: dict[str, SignalMetadata] = {}
    for name, raw in payload.items():
        metadata[name] = SignalMetadata(
            name=name,
            script_relpath=raw["script_relpath"],
            output_dir=raw["output_dir"],
            inputs=tuple(raw.get("inputs", [])),
            category=raw.get("category") or raw.get("Cat.Signal") or "Unknown",
            authors=raw.get("authors"),
            year=raw.get("year"),
            journal=raw.get("journal"),
            long_description=raw.get("long_description"),
            detailed_definition=raw.get("detailed_definition"),
        )
    return metadata


def available_factors() -> list[str]:
    """Return all vendored signal names."""

    return sorted(_load_signal_metadata())


def factor_metadata(name: str) -> SignalMetadata:
    """Return metadata for a vendored signal."""

    try:
        return _load_signal_metadata()[name]
    except KeyError as exc:
        raise KeyError(f"Unknown traditional factor: {name}") from exc


@lru_cache(maxsize=None)
def _required_inputs(meta: SignalMetadata) -> tuple[str, ...]:
    inferred = set(meta.inputs)
    try:
        script_text = meta.script_path.read_text()
    except FileNotFoundError:
        return meta.inputs
    inferred.update(_SCRIPT_INPUT_PATTERN.findall(script_text))
    inferred.update(_SCRIPT_INPUT_PATH_PATTERN.findall(script_text))
    return tuple(sorted(name for name in inferred if name))


def _build_docstring(meta: SignalMetadata) -> str:
    summary = meta.long_description or meta.detailed_definition or "OpenAssetPricing signal"
    parts = [summary.strip()]
    citation_bits = [bit for bit in (meta.authors, meta.year) if bit]
    if citation_bits:
        parts.append(f"Source: {', '.join(citation_bits)}")
    required_inputs = _required_inputs(meta)
    if required_inputs:
        parts.append(
            "Required upstream datasets: " + ", ".join(required_inputs)
        )
    parts.append(
        "Pass extra datasets via datasets={...}. If the factor uses multiple "
        "inputs, either pass input_name=... for the primary frame or supply all "
        "required upstream filenames explicitly."
    )
    return "\n\n".join(parts)


@contextlib.contextmanager
def _temporary_cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _normalize_dataset_aliases(
    datasets: Mapping[str, Any] | None,
) -> dict[str, Any]:
    aliases: dict[str, Any] = {}
    for key, value in (datasets or {}).items():
        path_key = Path(str(key))
        aliases[path_key.name] = value
        aliases[path_key.stem] = value
    return aliases


def _infer_input_name(meta: SignalMetadata, frame: pd.DataFrame | None) -> str | None:
    if frame is None:
        return None
    required_inputs = _required_inputs(meta)
    if len(required_inputs) == 1:
        return required_inputs[0]
    frame_cols = set(frame.columns)
    best_name: str | None = None
    best_score = 0
    for candidate in required_inputs:
        score = len(frame_cols & _OPTIONAL_HELPERS.get(candidate, set()))
        if score > best_score:
            best_name = candidate
            best_score = score
    if best_score > 0:
        return best_name
    if "SignalMasterTable.parquet" in required_inputs and {"permno", "time_avail_m"} <= frame_cols:
        return "SignalMasterTable.parquet"
    return None


def _resolve_required_datasets(
    meta: SignalMetadata,
    data: pd.DataFrame | None,
    *,
    input_name: str | None = None,
    datasets: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = _normalize_dataset_aliases(datasets)
    required_inputs = _required_inputs(meta)
    inferred_name = input_name or _infer_input_name(meta, data)
    if data is not None and inferred_name:
        resolved.setdefault(inferred_name, data)
        resolved.setdefault(Path(inferred_name).stem, data)

    missing: list[str] = []
    materialized: dict[str, Any] = {}
    for required_name in required_inputs:
        value = resolved.get(required_name)
        if value is None:
            value = resolved.get(Path(required_name).stem)
        if value is None:
            missing.append(required_name)
            continue
        materialized[required_name] = value

    if missing:
        raise ValueError(
            f"{meta.name} requires upstream datasets {missing}. "
            f"Provide them via datasets={{...}} using the original upstream "
            f"filenames. Upstream script: {meta.script_relpath}"
        )

    return materialized


def _ensure_datetime_columns(frame: pd.DataFrame) -> pd.DataFrame:
    converted = frame.copy()
    for col in converted.columns:
        if re.search(r"(date|time)", col, flags=re.IGNORECASE):
            try:
                converted[col] = pd.to_datetime(converted[col])
            except (TypeError, ValueError):
                continue
    return converted


def _write_frame_to_path(frame: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(frame, (str, Path)):
        shutil.copy2(Path(frame), destination)
        return

    if hasattr(frame, "to_pandas"):
        frame = frame.to_pandas()
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"Unsupported dataset type for {destination.name}: {type(frame)!r}. "
            "Pass a pandas DataFrame, a polars DataFrame, or a filesystem path."
        )

    frame = _ensure_datetime_columns(frame)
    if destination.suffix == ".parquet":
        frame.to_parquet(destination, index=False)
        return
    if destination.suffix == ".csv":
        frame.to_csv(destination, index=False)
        return
    raise ValueError(f"Unsupported upstream dataset format: {destination.suffix}")


def _prepare_workspace(meta: SignalMetadata, datasets: Mapping[str, Any]) -> Path:
    workspace_root = Path(tempfile.mkdtemp(prefix=f"tiger_traditional_{meta.name}_"))
    signals_root = workspace_root / "Signals"
    pycode_root = signals_root / "pyCode"
    pydata_root = signals_root / "pyData"
    pycode_root.mkdir(parents=True, exist_ok=True)
    (pydata_root / "Intermediate").mkdir(parents=True, exist_ok=True)
    (pydata_root / "Predictors").mkdir(parents=True, exist_ok=True)
    (pydata_root / "Placebos").mkdir(parents=True, exist_ok=True)

    # Keep the workspace pyCode directory real so relative ../pyData paths resolve
    # inside the temp workspace, while the script tree itself stays vendored only once.
    for source in OPEN_ASSET_PRICING_PYCODE.iterdir():
        os.symlink(source, pycode_root / source.name)

    for required_name, value in datasets.items():
        _write_frame_to_path(value, pydata_root / "Intermediate" / required_name)

    return workspace_root


def _execute_upstream_script(meta: SignalMetadata, workspace_root: Path) -> Path:
    pycode_dir = workspace_root / "Signals" / "pyCode"
    script_path = workspace_root / meta.script_relpath
    if not script_path.exists():
        raise FileNotFoundError(f"Vendored upstream script is missing: {script_path}")

    import sys

    sys_path_snapshot = list(sys.path)
    try:
        sys.path.insert(0, str(script_path.parent))
        sys.path.insert(0, str(pycode_dir))
        with _temporary_cwd(pycode_dir):
            runpy.run_path(str(script_path), run_name="__main__")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Running {meta.name} needs optional dependency {exc.name!r} "
            f"required by upstream script {meta.script_relpath}."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Vendored OpenAssetPricing script failed for {meta.name}: "
            f"{meta.script_relpath}"
        ) from exc
    finally:
        sys.path[:] = sys_path_snapshot

    output_path = workspace_root / "Signals" / "pyData" / meta.output_dir / meta.output_filename
    if not output_path.exists():
        raise FileNotFoundError(
            f"Upstream script completed but did not write {output_path.name} "
            f"for factor {meta.name}."
        )
    return output_path


def _yyyymm_from_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_integer_dtype(values):
        return values.astype("Int64")
    datetimes = pd.to_datetime(values)
    return (datetimes.dt.year * 100 + datetimes.dt.month).astype("Int64")


def _align_signal_to_input(
    signal_name: str,
    data: pd.DataFrame | None,
    result: pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    if data is None:
        return result

    base = data.reset_index(drop=False).copy()
    row_id = "__tiger_row_id__"
    if row_id in base.columns:
        raise ValueError(f"Input frame already contains reserved column {row_id!r}.")
    base[row_id] = range(len(base))

    if "permno" not in base.columns:
        raise ValueError(
            f"{signal_name} needs an input frame with a permno column so the "
            "upstream result can be aligned back to the caller."
        )
    if "yyyymm" in base.columns:
        base["yyyymm"] = _yyyymm_from_series(base["yyyymm"])
    elif "time_avail_m" in base.columns:
        base["yyyymm"] = _yyyymm_from_series(base["time_avail_m"])
    else:
        raise ValueError(
            f"{signal_name} needs either yyyymm or time_avail_m on the input "
            "frame so the upstream result can be aligned back to the caller."
        )

    merged = base.merge(
        result[["permno", "yyyymm", signal_name]],
        how="left",
        on=["permno", "yyyymm"],
        sort=False,
    ).sort_values(row_id)

    series = pd.Series(merged[signal_name].to_numpy(), index=data.index, name=signal_name)
    return series


def run_original_factor(
    signal_name: str,
    data: pd.DataFrame | None,
    *,
    input_name: str | None = None,
    datasets: Mapping[str, Any] | None = None,
    return_frame: bool = False,
) -> pd.Series | pd.DataFrame:
    """Run a vendored OpenAssetPricing signal and return aligned output."""

    meta = factor_metadata(signal_name)
    resolved_datasets = _resolve_required_datasets(
        meta,
        data,
        input_name=input_name,
        datasets=datasets,
    )
    workspace_root = _prepare_workspace(meta, resolved_datasets)
    try:
        output_path = _execute_upstream_script(meta, workspace_root)
        result = pd.read_csv(output_path)
    finally:
        shutil.rmtree(workspace_root, ignore_errors=True)

    if return_frame:
        return result
    return _align_signal_to_input(signal_name, data, result)


def _create_factor_function(signal_name: str):
    meta = factor_metadata(signal_name)

    def _factor(data: pd.DataFrame | None, **kwargs: Any) -> pd.Series | pd.DataFrame:
        return run_original_factor(signal_name, data, **kwargs)

    _factor.__name__ = signal_name
    _factor.__qualname__ = signal_name
    _factor.__doc__ = _build_docstring(meta)
    return _factor


for _signal_name in available_factors():
    globals()[_signal_name] = _create_factor_function(_signal_name)


__all__ = [
    *available_factors(),
    "SignalMetadata",
    "available_factors",
    "factor_metadata",
    "run_original_factor",
]
