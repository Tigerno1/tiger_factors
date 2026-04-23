from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Literal
from typing import Callable
from typing import Iterable 

import pandas as pd

try:
    import pyarrow as pa  # type: ignore[import-not-found]
    import pyarrow.dataset as ds  # type: ignore[import-not-found]
    import pyarrow.parquet as pq  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    pa = None
    ds = None
    pq = None



try:
    import polars as pl  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    pl = None

try:
    import duckdb  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    duckdb = None

try:
    import ibis
except Exception:  # pragma: no cover
    ibis = None

from .conf import DEFAULT_MACRO_DATA_SOURCE
from .conf import DEFAULT_FACTOR_STORE_ROOT
from .evaluation_store import EvaluationPathResult
from .evaluation_store import EvaluationStore
from .evaluation_store import EvaluationSectionAccessor
from .spec import AdjPriceData
from .spec import AdjPriceSpec
from .spec import DatasetSpec
from .spec import FactorData
from .spec import FactorSpec
from .spec import MacroData
from .spec import MacroSpec
from .spec import OthersSpec


IbisFilter = Callable[[Any], Any]

@dataclass(frozen=True, slots=True)
class DatasetSaveResult:
    dataset_dir: Path
    manifest_path: Path
    files: tuple[Path, ...]
    rows: int
    date_min: str | None
    date_max: str | None


@dataclass(frozen=True, slots=True)
class MacroBatchResult:
    manifest_path: Path
    results: dict[str, DatasetSaveResult]


LoadEngine = Literal["pandas", "polars", "duckdb", "ibis"]


class FactorStoreEvaluationReader:
    """Read-only evaluation access facade exposed as ``FactorStore.evaluation``."""

    def __init__(self, store: "FactorStore") -> None:
        self._store = store

    @property
    def _evaluation_store(self) -> EvaluationStore:
        return self._store.evaluation_store

    def section(self, spec: FactorSpec, section: str) -> EvaluationSectionAccessor:
        return self._evaluation_store.section(spec, section)

    def summary(self, spec: FactorSpec) -> EvaluationSectionAccessor:
        return self.section(spec, "summary")

    def returns(self, spec: FactorSpec) -> EvaluationSectionAccessor:
        return self.section(spec, "returns")

    def information(self, spec: FactorSpec) -> EvaluationSectionAccessor:
        return self.section(spec, "information")

    def turnover(self, spec: FactorSpec) -> EvaluationSectionAccessor:
        return self.section(spec, "turnover")

    def event_returns(self, spec: FactorSpec) -> EvaluationSectionAccessor:
        return self.section(spec, "event_returns")

    def event_study(self, spec: FactorSpec) -> EvaluationSectionAccessor:
        return self.section(spec, "event_study")

    def horizon(self, spec: FactorSpec) -> EvaluationSectionAccessor:
        return self.section(spec, "horizon")

    def full(self, spec: FactorSpec) -> EvaluationSectionAccessor:
        return self.section(spec, "full")


def _as_timestamp(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError("date_ contains missing values")
    return ts.normalize()


def _month_token(value: object) -> str:
    return _as_timestamp(value).strftime("%Y-%m")


def _month_range(start: object | None, end: object | None) -> list[str]:
    if start is None and end is None:
        return []
    if start is None:
        start = end
    if end is None:
        end = start
    start_ts = _as_timestamp(start).to_period("M")
    end_ts = _as_timestamp(end).to_period("M")
    months: list[str] = []
    current = start_ts
    while current <= end_ts:
        months.append(current.strftime("%Y-%m"))
        current += 1
    return months


def _to_pandas_frame(frame: Any) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame
    if pl is not None and isinstance(frame, pl.DataFrame):
        return frame.to_pandas()
    if pl is not None and isinstance(frame, pl.LazyFrame):
        return frame.collect().to_pandas()
    raise TypeError(f"Unsupported frame type: {type(frame)!r}")


def _to_polars_frame(frame: pd.DataFrame) -> Any:
    if pl is None:
        raise RuntimeError("polars is not installed")
    return pl.from_pandas(frame)


def _to_macro_frame(raw: Any, *, date_column: str = "date_", value_column: str = "value") -> pd.DataFrame:
    if isinstance(raw, pd.Series):
        frame = raw.to_frame(name=value_column)
    elif isinstance(raw, pd.DataFrame):
        frame = raw.copy()
    else:
        raise TypeError(f"Unsupported macro payload type: {type(raw)!r}")

    columns = list(frame.columns)
    rename_map = {}
    date_candidates = [
        column
        for column in columns
        if str(column).strip().lower() in {date_column.lower(), "date"}
    ]
    if len(date_candidates) == 1:
        rename_map[date_candidates[0]] = date_column
    if date_column not in frame.columns and rename_map:
        frame = frame.rename(columns=rename_map)

    if date_column not in frame.columns:
        frame = frame.reset_index()
        index_name = frame.columns[0]
        frame = frame.rename(columns={index_name: date_column})

    if value_column not in frame.columns:
        value_candidates = [column for column in frame.columns if str(column).strip().lower() != date_column.lower()]
        if len(value_candidates) == 1:
            frame = frame.rename(columns={value_candidates[0]: value_column})
        else:
            raise ValueError(
                "macro reader output must contain exactly one value column "
                f"besides '{date_column}', got {value_candidates!r}"
            )

    return frame.loc[:, [date_column, value_column]].copy()


def _quote_sql_literal(value: object) -> str:
    text = str(value).replace("'", "''")
    return f"'{text}'"


def _quote_sql_identifier(value: object) -> str:
    text = str(value).replace('"', '""')
    return f'"{text}"'


def _normalize_macro_freq(freq: str | None) -> str:
    if not freq:
        return "unknown"
    token = str(freq).strip().upper()
    if token.startswith(("B", "D")):
        return "1d"
    if token.startswith("W"):
        return "1w"
    if token.startswith("M"):
        return "1m"
    if token.startswith("Q"):
        return "1q"
    if token.startswith(("A", "Y")):
        return "1y"
    if token.startswith("H"):
        return "1h"
    if token.startswith(("T", "MIN")):
        return "1min"
    if token.startswith("S"):
        return "1s"
    return token.lower()


def _infer_macro_freq(frame: pd.DataFrame, *, date_column: str = "date_") -> str:
    if frame.empty or date_column not in frame.columns:
        return "unknown"

    dates = pd.DatetimeIndex(pd.to_datetime(frame[date_column], errors="coerce").dropna().sort_values().unique())
    if len(dates) < 2:
        return "unknown"

    if len(dates) >= 3:
        inferred = pd.infer_freq(dates)
        if inferred:
            return _normalize_macro_freq(inferred)

    deltas = pd.Series(dates).diff().dropna()
    if deltas.empty:
        return "unknown"

    median_days = deltas.median() / pd.Timedelta(days=1)
    if median_days <= 2.5:
        return "1d"
    if median_days <= 10:
        return "1w"
    if median_days <= 40:
        return "1m"
    if median_days <= 120:
        return "1q"
    if median_days <= 400:
        return "1y"
    return "unknown"


def _read_series_names_from_file(
    path: str | Path,
    *,
    column: str | int | None = None,
    has_header: bool = False,
) -> list[str]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    suffix = file_path.suffix.lower()
    if suffix in {".txt", ".list", ".names"}:
        names: list[str] = []
        for raw_line in file_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line and column is None:
                line = line.split(",", 1)[0].strip()
            names.append(line)
        return names

    frame = pd.read_csv(file_path, comment="#", sep=None, engine="python", header=0 if has_header else None)
    if column is None:
        series = frame.iloc[:, 0]
    else:
        series = frame[column]
    return [str(value).strip() for value in series.dropna().tolist() if str(value).strip()]


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    if pa is not None and pq is not None:
        try:
            table = pa.Table.from_pandas(frame, preserve_index=False)
            pq.write_table(table, path.as_posix())
            return
        except Exception:
            pass
    if duckdb is not None:
        try:
            con = duckdb.connect(database=":memory:")
            con.register("frame", frame)
            con.execute(f"COPY frame TO '{path.as_posix()}' (FORMAT PARQUET)")
            con.close()
            return
        except Exception:
            pass
    frame.to_parquet(path, index=False)


def _read_parquet(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    if len(paths) == 1:
        return pd.read_parquet(paths[0])
    if ds is not None:
        try:
            dataset = ds.dataset([path.as_posix() for path in paths], format="parquet")
            return dataset.to_table().to_pandas()
        except Exception:
            pass
    if duckdb is None:
        return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)

    def _quote_path(path: Path) -> str:
        return "'" + path.as_posix().replace("'", "''") + "'"

    sql_paths = "[" + ", ".join(_quote_path(path) for path in paths) + "]"
    query = f"SELECT * FROM read_parquet({sql_paths}, union_by_name=true)"
    con = duckdb.connect(database=":memory:")
    try:
        return con.execute(query).fetchdf()
    finally:
        con.close()


def _month_data_paths(spec: Any, root_dir: str | Path) -> list[Path]:
    dataset_dir = spec.dataset_dir(root_dir)
    return sorted(dataset_dir.glob(f"{spec.data_stem()}__????-??.parquet"))


def _existing_data_paths(spec: Any, root_dir: str | Path) -> list[Path]:
    dataset_dir = spec.dataset_dir(root_dir)
    if not dataset_dir.exists():
        return []

    exact_path = spec.data_path(root_dir)
    month_paths = _month_data_paths(spec, root_dir)

    paths: list[Path] = []
    if exact_path.exists():
        paths.append(exact_path)
    paths.extend(path for path in month_paths if path.exists())
    return paths


class FactorStore:
    def __init__(self, root_dir: str | Path | None = None):
        self.root_dir = Path(root_dir) if root_dir is not None else DEFAULT_FACTOR_STORE_ROOT
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._evaluation_store: EvaluationStore | None = None
        self._evaluation_reader: FactorStoreEvaluationReader | None = None

    @property
    def evaluation_store(self) -> EvaluationStore:
        if self._evaluation_store is None:
            self._evaluation_store = EvaluationStore(self.root_dir)
        return self._evaluation_store

    @property
    def evaluation(self) -> FactorStoreEvaluationReader:
        if self._evaluation_reader is None:
            self._evaluation_reader = FactorStoreEvaluationReader(self)
        return self._evaluation_reader

    def ensure_run_dir(self, *parts: str) -> EvaluationPathResult:
        run_dir = self.root_dir.joinpath(*parts)
        run_dir.mkdir(parents=True, exist_ok=True)
        return EvaluationPathResult(root_dir=self.root_dir, run_dir=run_dir)

    def load_manifest(self, spec: DatasetSpec) -> dict[str, Any]:
        path = spec.manifest_path(self.root_dir)
        if not path.exists():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def save_manifest(self, spec: DatasetSpec, payload: Mapping[str, Any]) -> Path:
        path = spec.manifest_path(self.root_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(dict(payload), indent=2, default=str),
            encoding="utf-8",
        )
        return path

    def update_manifest(
        self,
        spec: DatasetSpec,
        updater: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        current = self.load_manifest(spec) if spec.manifest_path(self.root_dir).exists() else {}
        updated = updater(current)
        self.save_manifest(spec, updated)
        return updated

    def load_meta(self, spec: DatasetSpec) -> dict[str, Any]:
        path = spec.meta_path(self.root_dir)
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def save_meta(self, spec: DatasetSpec, payload: Mapping[str, Any]) -> Path:
        path = spec.meta_path(self.root_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(dict(payload), indent=2, default=str),
            encoding="utf-8",
        )
        return path

    def update_meta(
        self,
        spec: DatasetSpec,
        updates: Mapping[str, Any] | None = None,
        *,
        updater: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        current = self.load_meta(spec)
        if updater is not None:
            new_meta = updater(current)
        else:
            new_meta = dict(current)
            if updates:
                new_meta.update(dict(updates))
        self.save_meta(spec, new_meta)
        return new_meta


    def save_factor(
        self,
        spec: FactorSpec,
        frame: Any,
        *,
        force_update: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetSaveResult:
        normalized = FactorData().normalize(_to_pandas_frame(frame), factor_name=spec.table_name)
        return self.save_dataset(
            spec,
            normalized,
            force_update=force_update,
            metadata=metadata,
        )

    def save_price(
        self,
        spec: AdjPriceSpec,
        frame: Any,
        *,
        force_update: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetSaveResult:
        return self.save_adj_price(
            spec,
            frame,
            force_update=force_update,
            metadata=metadata,
        )

    def save_adj_price(
        self,
        spec: AdjPriceSpec,
        frame: Any,
        *,
        force_update: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetSaveResult:
        frame = AdjPriceData().normalize(_to_pandas_frame(frame))
        return self.save_dataset(
            spec,
            frame,
            force_update=force_update,
            metadata=metadata,
        )

    def save_macro(
        self,
        spec: MacroSpec,
        frame: Any,
        *,
        force_update: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetSaveResult:
        frame = _to_macro_frame(
            _to_pandas_frame(frame),
            date_column=spec.date_column,
            value_column=getattr(spec, "value_column", "value"),
        )
        frame = MacroData().normalize(frame)

        return self.save_macro_dataset(
            spec, 
            frame, 
            force_update=force_update, 
            metadata=metadata)
    

    def save_others(
        self,
        spec: OthersSpec,
        frame: Any,
        *,
        force_update: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetSaveResult:
        normalized = self._to_summary_frame(frame)
        dataset_dir = spec.dataset_dir(self.root_dir)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        data_path = spec.data_path(self.root_dir)
        existing_parquet_files = [data_path] if data_path.exists() else []
        manifest_path = spec.manifest_path(self.root_dir)

        if not force_update and (existing_parquet_files or manifest_path.exists()):
            raise FileExistsError(f"others dataset already exists: {dataset_dir}")
        
        if force_update:
            for parquet_path in existing_parquet_files:
                parquet_path.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)

        _write_parquet(normalized, data_path)
        payload = self._build_others_manifest(spec, normalized, [data_path])
        manifest_path = self.save_manifest(spec, payload)

        if metadata:
            self.update_meta(spec, metadata)

        return DatasetSaveResult(
            dataset_dir=dataset_dir,
            manifest_path=manifest_path,
            files=(data_path,),
            rows=int(len(normalized)),
            date_min=payload["date_min"],
            date_max=payload["date_max"],
        )

    def save_fama3(
        self,
        spec: OthersSpec,
        frame: Any,
        *,
        force_update: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetSaveResult:
        return self.save_others(spec, frame, force_update=force_update, metadata=metadata)

    def download_macro(
        self,
        spec: MacroSpec,
        *,
        start: object | None = None,
        end: object | None = None,
        data_source: str | None = None,
        api_key: str | None = None,
        reader: Any | None = None,
    ) -> DatasetSaveResult:
        source = (data_source or spec.provider or DEFAULT_MACRO_DATA_SOURCE).lower()
        if reader is None:
            try:
                from pandas_datareader import data as pdr  # type: ignore[import-not-found]
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("pandas_datareader is required for macro downloads") from exc
            reader = pdr.DataReader

        start_ts = _as_timestamp(start) if start is not None else None
        end_ts = _as_timestamp(end) if end is not None else None
        raw = reader(spec.series_name, source, start=start_ts, end=end_ts, api_key=api_key)
        frame = _to_macro_frame(raw, date_column=spec.date_column, value_column=spec.value_column)
        return self.save_macro(spec, frame)

    def download_macro_batch(
        self,
        specs: Iterable[MacroSpec],
        *,
        start: object | None = None,
        end: object | None = None,
        data_source: str | None = None,
        api_key: str | None = None,
        reader: Any | None = None,
        skip_failed: bool = False,
    ) -> dict[str, DatasetSaveResult]:
        results: dict[str, DatasetSaveResult] = {}
        for spec in specs:
            try:
                result = self.download_macro(
                    spec,
                    start=start,
                    end=end,
                    data_source=data_source,
                    api_key=api_key,
                    reader=reader,
                )
                key = self._macro_result_key(result)
                results[key] = result
            except Exception:
                if not skip_failed:
                    raise
        return results

    def download_fred_macro_series(
        self,
        series_names: Iterable[str],
        *,
        region: str = "us",
        freq: str = "auto",
        variant: str | None = None,
        start: object | None = None,
        end: object | None = None,
        api_key: str | None = None,
        reader: Any | None = None,
        skip_failed: bool = False,
        write_manifest: bool = True,
    ) -> MacroBatchResult:
        specs = [
            MacroSpec(
                region=region,
                freq=freq,
                table_name=series_name,
                variant=variant,
                provider="fred",
                source_name=series_name,
            )
            for series_name in series_names
        ]
        results = self.download_macro_batch(
            specs,
            start=start,
            end=end,
            data_source="fred",
            api_key=api_key,
            reader=reader,
            skip_failed=skip_failed,
        )
        manifest_path = self.root_dir / "macro" / "fred_batch_manifest.json"
        if write_manifest:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "data_source": "fred",
                "region": region.lower(),
                "freq": freq.lower(),
                "variant": variant.lower() if variant is not None else None,
                "results": {
                    key: {
                        "dataset_dir": str(result.dataset_dir),
                        "manifest_path": str(result.manifest_path),
                        "files": [str(path) for path in result.files],
                        "rows": result.rows,
                        "date_min": result.date_min,
                        "date_max": result.date_max,
                    }
                    for key, result in results.items()
                },
            }
            manifest_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        else:
            manifest_path = Path()
        return MacroBatchResult(manifest_path=manifest_path, results=results)

    def download_fred_macro_series_from_file(
        self,
        series_list_path: str | Path,
        *,
        region: str = "us",
        freq: str = "auto",
        variant: str | None = None,
        start: object | None = None,
        end: object | None = None,
        api_key: str | None = None,
        reader: Any | None = None,
        skip_failed: bool = False,
        write_manifest: bool = True,
        column: str | int | None = None,
        has_header: bool = False,
    ) -> MacroBatchResult:
        series_names = _read_series_names_from_file(
            series_list_path,
            column=column,
            has_header=has_header,
        )
        return self.download_fred_macro_series(
            series_names,
            region=region,
            freq=freq,
            variant=variant,
            start=start,
            end=end,
            api_key=api_key,
            reader=reader,
            skip_failed=skip_failed,
            write_manifest=write_manifest,
        )

    def get_price(
        self,
        spec: AdjPriceSpec,
        *,
        start: object | None = None,
        end: object | None = None,
        engine: LoadEngine = "pandas",
    ) -> pd.DataFrame | Any:
        return self.load_dataset(spec, start=start, end=end, engine=engine)

    def get_adj_price(
        self,
        spec: AdjPriceSpec,
        *,
        start: object | None = None,
        end: object | None = None,
        engine: LoadEngine = "pandas",
    ) -> pd.DataFrame | Any:
        dataset_dir = spec.dataset_dir(self.root_dir)
        if not dataset_dir.exists():
            empty = pd.DataFrame(columns=spec.storage_columns())
            return _to_polars_frame(empty) if engine == "polars" else empty

        data_path = spec.data_path(self.root_dir)
        if data_path.exists():
            paths = [data_path]
        elif start is None and end is None:
            paths = _month_data_paths(spec, self.root_dir)
        else:
            months = _month_range(start, end)
            paths = [
                spec.data_path(self.root_dir, part=month)
                for month in months
                if spec.data_path(self.root_dir, part=month).exists()
            ]

        if not paths:
            empty = pd.DataFrame(columns=spec.storage_columns())
            return _to_polars_frame(empty) if engine == "polars" else empty

        frame = _read_parquet(paths)
        if frame.empty:
            return _to_polars_frame(frame) if engine == "polars" else frame

        frame = self._normalize_loaded_frame(spec, frame)
        if start is not None:
            frame = frame.loc[frame[spec.date_column] >= _as_timestamp(start)]
        if end is not None:
            frame = frame.loc[frame[spec.date_column] <= _as_timestamp(end)]
        frame = frame.sort_values([spec.code_column or "code", spec.date_column], kind="stable").reset_index(drop=True)
        return _to_polars_frame(frame) if engine == "polars" else frame

    def get_macro(
        self,
        spec: MacroSpec,
        *,
        start: object | None = None,
        end: object | None = None,
        engine: LoadEngine = "pandas",
    ) -> pd.DataFrame | Any:
        return self.load_dataset(spec, start=start, end=end, engine=engine)

    def get_others(
        self,
        spec: OthersSpec,
        *,
        engine: LoadEngine = "pandas",
    ) -> pd.DataFrame | Any:
        dataset_dir = spec.dataset_dir(self.root_dir)
        data_path = spec.data_path(self.root_dir)
        if not data_path.exists():
            empty = pd.DataFrame()
            return _to_polars_frame(empty) if engine == "polars" else empty
        frame = pd.read_parquet(data_path)
        return _to_polars_frame(frame) if engine == "polars" else frame

    def get_fama3(
        self,
        spec: OthersSpec,
        *,
        engine: LoadEngine = "pandas",
    ) -> pd.DataFrame | Any:
        return self.get_others(spec, engine=engine)
    
    def save_macro_dataset(
        self,
        spec: DatasetSpec,
        frame: Any,
        *,
        force_update: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetSaveResult:

        normalized = self._normalize_frame(spec, _to_pandas_frame(frame))

        inferred_freq = _infer_macro_freq(normalized, date_column=spec.date_column)
        if inferred_freq != "unknown":
            spec = replace(spec, freq=inferred_freq)

        dataset_dir = spec.dataset_dir(self.root_dir)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        data_path = spec.data_path(self.root_dir)
        existing_data_files = [data_path] if data_path.exists() else []
        manifest_path = spec.manifest_path(self.root_dir)

        if not force_update and (existing_data_files or manifest_path.exists()):
            raise FileExistsError(f"dataset already exists: {dataset_dir}")

        if force_update:
            for path in existing_data_files:
                path.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)

        _write_parquet(normalized, data_path)

        payload = self._build_manifest(spec, normalized, [data_path])
        manifest_path = self.save_manifest(spec, payload)

        if metadata:
            self.update_meta(spec, metadata)

        return DatasetSaveResult(
            dataset_dir=dataset_dir,
            manifest_path=manifest_path,
            files=(data_path,),
            rows=int(len(normalized)),
            date_min=payload["date_min"],
            date_max=payload["date_max"],
        )

    def save_dataset(
        self,
        spec: DatasetSpec,
        frame: Any,
        *,
        force_update: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetSaveResult:
        
        normalized = self._normalize_frame(spec, _to_pandas_frame(frame))

        dataset_dir = spec.dataset_dir(self.root_dir)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        existing_parquet_files = _existing_data_paths(spec, self.root_dir)
        manifest_path = spec.manifest_path(self.root_dir)

        if not force_update and (existing_parquet_files or manifest_path.exists()):
            raise FileExistsError(f"dataset already exists: {dataset_dir}")
        
        if force_update:
            for parquet_path in existing_parquet_files:
                parquet_path.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)

        month_files: list[Path] = []
        for month_token, part in normalized.groupby("__month__", sort=True):
            monthly_path = spec.data_path(self.root_dir, part=month_token)
            merged = self._merge_month_file(spec, monthly_path, part.drop(columns="__month__"))
            _write_parquet(merged, monthly_path)
            month_files.append(monthly_path)

        month_files = _month_data_paths(spec, self.root_dir)
        current_frame = _read_parquet(month_files)

        payload = self._build_manifest(spec, current_frame, month_files)
        self.save_manifest(spec, payload)

        if metadata:
            self.update_meta(spec, metadata)

        return DatasetSaveResult(
            dataset_dir=dataset_dir,
            manifest_path=manifest_path,
            files=tuple(month_files),
            rows=int(len(current_frame)),
            date_min=payload["date_min"],
            date_max=payload["date_max"],
        )

    def load_dataset(
        self,
        spec: DatasetSpec,
        *,
        start: object | None = None,
        end: object | None = None,
        engine: LoadEngine = "pandas",
    ) -> pd.DataFrame | Any:
        if spec.kind() == "factor" and ibis is not None and duckdb is not None:
            query = self.query_dataset(spec, start=start, end=end)
            if engine == "ibis":
                return query

            frame = query.to_pandas()
            if not frame.empty:
                frame = self._normalize_loaded_frame(spec, frame)
                if start is not None:
                    frame = frame.loc[frame[spec.date_column] >= _as_timestamp(start)]
                if end is not None:
                    frame = frame.loc[frame[spec.date_column] <= _as_timestamp(end)]
                key_columns = list(spec.key_columns())
                if key_columns and all(column in frame.columns for column in key_columns):
                    frame = frame.sort_values(key_columns, kind="mergesort")
                    frame = frame.drop_duplicates(key_columns, keep="last")
                frame = frame.reset_index(drop=True)
            return _to_polars_frame(frame) if engine == "polars" else frame

        dataset_dir = spec.dataset_dir(self.root_dir)
        if not dataset_dir.exists() and spec.kind() == "macro":
            resolved = self._find_macro_dataset_dir(spec)
            if resolved is not None:
                dataset_dir = resolved

        if not dataset_dir.exists():
            empty = pd.DataFrame(columns=spec.storage_columns())
            return _to_polars_frame(empty) if engine == "polars" else empty

        data_path = spec.data_path(self.root_dir)
        if data_path.exists():
            paths = [data_path]
        elif start is None and end is None:
            paths = _month_data_paths(spec, self.root_dir)
        else:
            months = _month_range(start, end)
            paths = [
                spec.data_path(self.root_dir, part=month)
                for month in months
                if spec.data_path(self.root_dir, part=month).exists()
            ]

        if not paths:
            empty = pd.DataFrame(columns=spec.storage_columns())
            return _to_polars_frame(empty) if engine == "polars" else empty

        frame = _read_parquet(paths)
        if frame.empty:
            return _to_polars_frame(frame) if engine == "polars" else frame

        frame = self._normalize_loaded_frame(spec, frame)
        if start is not None:
            frame = frame.loc[frame[spec.date_column] >= _as_timestamp(start)]
        if end is not None:
            frame = frame.loc[frame[spec.date_column] <= _as_timestamp(end)]

        key_columns = list(spec.key_columns())
        if key_columns and all(column in frame.columns for column in key_columns):
            frame = frame.sort_values(key_columns, kind="mergesort")
            frame = frame.drop_duplicates(key_columns, keep="last")
        frame = frame.reset_index(drop=True)
        return _to_polars_frame(frame) if engine == "polars" else frame

    def _normalize_frame(self, spec: DatasetSpec, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in spec.storage_columns() if column not in frame.columns]
        if missing:
            raise ValueError(f"frame is missing required columns: {missing}")

        columns = list(spec.storage_columns())
        normalized = frame.loc[:, columns].copy()

        date_series = pd.to_datetime(normalized[spec.date_column], errors="raise")
        if getattr(date_series.dt, "tz", None) is not None:
            date_series = date_series.dt.tz_localize(None)
        normalized[spec.date_column] = date_series.dt.normalize()

        key_columns = list(spec.key_columns())
        normalized = normalized.sort_values(key_columns, kind="mergesort")
        normalized = normalized.drop_duplicates(key_columns, keep="last")
        if spec.kind() != "macro":
            normalized["__month__"] = normalized[spec.date_column].dt.strftime("%Y-%m")
        return normalized.reset_index(drop=True)

    def _normalize_loaded_frame(self, spec: DatasetSpec, frame: pd.DataFrame) -> pd.DataFrame:
        if spec.date_column in frame.columns:
            frame = frame.copy()
            if spec.kind() == "macro":
                frame[spec.date_column] = pd.to_datetime(frame[spec.date_column], utc=True).dt.tz_localize(None)
                frame[spec.date_column] = pd.to_datetime(frame[spec.date_column]).dt.normalize()
            else:
                frame[spec.date_column] = pd.to_datetime(frame[spec.date_column]).dt.normalize()
        return frame

    @staticmethod
    def _to_summary_frame(summary: Any) -> pd.DataFrame:
        from dataclasses import asdict
        from dataclasses import is_dataclass
        from typing import Mapping

        if isinstance(summary, pd.DataFrame):
            return summary.copy()
        if isinstance(summary, pd.Series):
            return summary.to_frame().T.reset_index(drop=True)
        if is_dataclass(summary):
            return pd.DataFrame([asdict(summary)])
        if isinstance(summary, Mapping):
            return pd.DataFrame([dict(summary)])
        if hasattr(summary, "to_dict"):
            payload = summary.to_dict()
            if isinstance(payload, Mapping):
                return pd.DataFrame([dict(payload)])
        raise TypeError(
            "evaluation summary must be a pandas DataFrame/Series, dataclass, or mapping; "
            f"got {type(summary)!r}"
        )

    def _merge_month_file(self, spec: DatasetSpec, path: Path, new_part: pd.DataFrame) -> pd.DataFrame:
        if not path.exists():
            return new_part
        existing = pd.read_parquet(path)
        if existing.empty:
            return new_part
        existing = self._normalize_loaded_frame(spec, existing)
        key_columns = list(spec.key_columns())
        payload_columns = [column for column in new_part.columns if column not in key_columns]
        if payload_columns:
            new_part = new_part.loc[~new_part[payload_columns].isna().all(axis=1)].copy()
        payload_columns = [column for column in existing.columns if column not in key_columns]
        if payload_columns:
            existing = existing.loc[~existing[payload_columns].isna().all(axis=1)].copy()
        if existing.empty:
            return new_part
        if new_part.empty:
            return existing
        merged = pd.concat([existing, new_part], ignore_index=True, sort=False)
        merged = merged.groupby(list(spec.key_columns()), sort=False, as_index=False).last()
        return merged.reset_index(drop=True)

    def _build_manifest(
        self,
        spec: DatasetSpec,
        frame: pd.DataFrame,
        files: Iterable[Path],
    ) -> dict[str, object]:
        files = tuple(files)
        payload: dict[str, object] = {
            "spec": spec.to_dict(),
            "rows": int(len(frame)),
            "files": [path.name for path in files],
            "date_min": str(frame[spec.date_column].min()) if not frame.empty else None,
            "date_max": str(frame[spec.date_column].max()) if not frame.empty else None,
        }
        if spec.kind() == "factor":
            payload["factor_name"] = spec.data_stem()
        if spec.code_column and spec.code_column in frame.columns:
            payload["codes"] = int(frame[spec.code_column].nunique())
        else:
            payload["codes"] = None
        return payload

    def _build_others_manifest(
        self,
        spec: OthersSpec,
        frame: pd.DataFrame,
        files: Iterable[Path],
    ) -> dict[str, object]:
        files = tuple(files)
        payload: dict[str, object] = {
            "spec": spec.to_dict(),
            "rows": int(len(frame)),
            "files": [path.name for path in files],
            "columns": list(frame.columns),
            "date_min": str(frame["date_"].min()) if "date_" in frame.columns and not frame.empty else None,
            "date_max": str(frame["date_"].max()) if "date_" in frame.columns and not frame.empty else None,
        }
        if "code" in frame.columns:
            payload["codes"] = int(frame["code"].nunique())
        else:
            payload["codes"] = None
        return payload

    def _find_macro_dataset_dir(self, spec: MacroSpec) -> Path | None:
        base = self.root_dir / "macro"
        if not base.exists():
            return None

        provider = spec.provider or "*"
        region = spec.region
        if spec.freq == "auto":
            pattern = f"{provider}/{region}/*"
            matches = [path for path in base.glob(pattern) if path.is_dir()]
        else:
            matches = [base / provider / region / spec.freq]
        matches = [
            path
            for path in matches
            if path.is_dir()
            and (
                (path / f"{spec.data_stem()}.parquet").exists()
                or any(path.glob(f"{spec.data_stem()}__????-??.parquet"))
            )
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    def _macro_result_key(self, result: DatasetSaveResult) -> str:
        try:
            rel_path = result.dataset_dir.relative_to(self.root_dir)
        except Exception:
            return result.dataset_dir.as_posix()

        parts = rel_path.parts
        if len(parts) < 4 or parts[0] != "macro":
            return result.dataset_dir.as_posix()
        # expected: macro/provider/region/freq
        provider = parts[1]
        region = parts[2]
        freq = parts[3]
        stem = result.files[0].stem if result.files else "none"
        return f"{provider}/{region}/{freq}/{stem}"
    
    def _require_ibis(self):
        if ibis is None:
            raise RuntimeError(
                "ibis is not installed. Please install with: pip install ibis-framework[duckdb] duckdb"
            )

    def _dataset_paths_for_query(
        self,
        spec,
        *,
        start: object | None = None,
        end: object | None = None,
    ) -> list[str]:
        """
        复用你现有的按月切片逻辑，尽量只读取需要的 parquet 文件。
        """
        dataset_dir = spec.dataset_dir(self.root_dir)
        if not dataset_dir.exists():
            return []

        data_path = spec.data_path(self.root_dir)
        if data_path.exists():
            paths = [data_path]
        elif start is None and end is None:
            paths = _month_data_paths(spec, self.root_dir)
        else:
            months = _month_range(start, end)
            paths = [
                spec.data_path(self.root_dir, part=month)
                for month in months
                if spec.data_path(self.root_dir, part=month).exists()
            ]

        return [p.as_posix() for p in paths]

    def query_dataset(
        self,
        spec,
        *,
        start: object | None = None,
        end: object | None = None,
        codes: Iterable[str] | None = None,
        columns: Iterable[str] | None = None,
        value_min: float | None = None,
        value_max: float | None = None,
    ):
        """
        返回 Ibis expression，不立即 materialize。
        """
        self._require_ibis()

        paths = self._dataset_paths_for_query(spec, start=start, end=end)
        if not paths:
            # 返回一个空表表达式
            empty = pd.DataFrame(columns=spec.storage_columns())
            return ibis.memtable(empty)

        con = ibis.duckdb.connect()
        t = con.read_parquet(paths)

        # 时间过滤
        if start is not None and spec.date_column in t.columns:
            start_ts = _as_timestamp(start)
            t = t.filter(t[spec.date_column] >= start_ts)

        if end is not None and spec.date_column in t.columns:
            end_ts = _as_timestamp(end)
            t = t.filter(t[spec.date_column] <= end_ts)

        # code 过滤
        if codes is not None and getattr(spec, "code_column", None):
            code_column = spec.code_column or "code"
            normalized_codes = [str(c) for c in codes]
            t = t.filter(t[code_column].isin(normalized_codes))

        # value 过滤（只在有 value 列时启用）
        if value_min is not None and "value" in t.columns:
            t = t.filter(t["value"] >= value_min)

        if value_max is not None and "value" in t.columns:
            t = t.filter(t["value"] <= value_max)

        # 列裁剪
        if columns is not None:
            selected = [str(col) for col in columns]
            existing = [col for col in selected if col in t.columns]
            if existing:
                t = t.select(*existing)

        return t

    def get_factor_query(
        self,
        spec,
        *,
        start: object | None = None,
        end: object | None = None,
        codes: Iterable[str] | None = None,
        columns: Iterable[str] | None = None,
        value_min: float | None = None,
        value_max: float | None = None,
    ):
        """
        因子专用查询入口：返回 Ibis expression。
        """
        return self.query_dataset(
            spec,
            start=start,
            end=end,
            codes=codes,
            columns=columns,
            value_min=value_min,
            value_max=value_max,
        )

    @staticmethod
    def _factor_join_output_name(spec: FactorSpec, position: int) -> str:
        name = str(spec.data_stem()).strip()
        return name or f"factor_{position}"

    def _build_joined_factor_sql(
        self,
        specs: list[FactorSpec],
        *,
        start: object | None = None,
        end: object | None = None,
        codes: Iterable[str] | None = None,
        join_keys: Iterable[str] | None = None,
        how: str = "inner",
    ) -> str:
        if not specs:
            raise ValueError("specs must not be empty")

        normalized_how = str(how).strip().lower()
        if normalized_how not in {"inner", "left", "outer", "full", "full_outer"}:
            raise ValueError(f"unsupported join type: {how!r}")
        sql_join_type = {
            "inner": "INNER JOIN",
            "left": "LEFT JOIN",
            "outer": "FULL OUTER JOIN",
            "full": "FULL OUTER JOIN",
            "full_outer": "FULL OUTER JOIN",
        }[normalized_how]

        join_key_list = [str(key) for key in join_keys] if join_keys is not None else [specs[0].code_column or "code", specs[0].date_column]
        if not join_key_list:
            raise ValueError("join_keys must not be empty")

        normalized_codes = None if codes is None else list(dict.fromkeys(str(code) for code in codes))
        ctes: list[str] = []
        included_specs: list[tuple[int, FactorSpec, str]] = []
        seen_output_names: set[str] = set()

        for idx, spec in enumerate(specs):
            output_name = self._factor_join_output_name(spec, idx)
            if output_name in seen_output_names:
                raise ValueError(f"duplicate output column name for joined factors: {output_name!r}")
            paths = self._dataset_paths_for_query(spec, start=start, end=end)
            if not paths:
                continue
            seen_output_names.add(output_name)
            included_specs.append((idx, spec, output_name))
            path_list = ", ".join(_quote_sql_literal(path) for path in paths)

            conditions: list[str] = []
            if start is not None:
                conditions.append(f"{spec.date_column} >= TIMESTAMP {_quote_sql_literal(_as_timestamp(start))}")
            if end is not None:
                conditions.append(f"{spec.date_column} <= TIMESTAMP {_quote_sql_literal(_as_timestamp(end))}")
            if normalized_codes is not None:
                code_list = ", ".join(_quote_sql_literal(code) for code in normalized_codes)
                conditions.append(f"{spec.code_column or 'code'} IN ({code_list})")
            where_clause = ""
            if conditions:
                where_clause = "\n    WHERE " + " AND ".join(conditions)

            select_lines: list[str] = []
            for key in join_key_list:
                if key == spec.date_column:
                    select_lines.append(f"        CAST({key} AS TIMESTAMP) AS {_quote_sql_identifier(key)}")
                elif key == (spec.code_column or "code"):
                    select_lines.append(f"        CAST({key} AS VARCHAR) AS {_quote_sql_identifier(key)}")
                else:
                    select_lines.append(f"        {key} AS {_quote_sql_identifier(key)}")
            select_lines.append(
                f"        CAST({spec.value_column} AS DOUBLE) AS {_quote_sql_identifier(output_name)}"
            )

            ctes.append(
                "\n".join(
                    [
                        f"factor_{idx} AS (",
                        "    SELECT",
                        ",\n".join(select_lines),
                        f"    FROM read_parquet([{path_list}], union_by_name=true){where_clause}",
                        ")",
                    ]
                )
            )

        if not included_specs:
            raise FileNotFoundError("none of the requested factor specs have readable parquet files")

        base_keys = [f"base.{_quote_sql_identifier(key)}" for key in join_key_list]
        projected_columns: list[str] = []
        for position, (original_idx, _, output_name) in enumerate(included_specs):
            table_alias = "base" if position == 0 else f"factor_{original_idx}"
            projected_columns.append(f"{table_alias}.{_quote_sql_identifier(output_name)}")

        select_lines = ["SELECT"]
        if len(base_keys) > 1:
            select_lines.extend(f"    {column}," for column in base_keys[:-1])
        select_lines.append(f"    {base_keys[-1]}{',' if projected_columns else ''}")
        select_lines.extend(
            f"    {column}{',' if idx < len(projected_columns) - 1 else ''}"
            for idx, column in enumerate(projected_columns)
        )

        using_clause = ", ".join(_quote_sql_identifier(key) for key in join_key_list)
        join_lines = [f"FROM factor_{included_specs[0][0]} AS base"]
        for original_idx, _, _ in included_specs[1:]:
            join_lines.append(f"{sql_join_type} factor_{original_idx} USING ({using_clause})")

        order_clause = ", ".join(base_keys)
        return "\n".join(
            [
                "WITH",
                ",\n".join(ctes),
                *select_lines,
                *join_lines,
                f"ORDER BY {order_clause}",
            ]
        )
    

    def _get_ibis_con(self):
        self._require_ibis()
        if duckdb is None:
            raise RuntimeError("duckdb is not installed")
        if getattr(self, "_ibis_con", None) is None:
            self._ibis_con = ibis.duckdb.connect()
        return self._ibis_con


    def get_factors_query(
        self,
        specs: Iterable[FactorSpec],
        *,
        start: object | None = None,
        end: object | None = None,
        codes: Iterable[str] | None = None,
        join_keys: Iterable[str] | None = None,
        how: str = "inner",
        filters: Iterable[IbisFilter] | None = None,
    ):
        """
        多因子查询入口：传入多个 FactorSpec，用 Ibis + DuckDB 在 parquet 上直接 join。
        默认按 code + date_ 连接，返回 Ibis relation。
        """
        con = self._get_ibis_con()

        spec_list = list(specs)
        if not spec_list:
            raise ValueError("specs must not be empty")

        sql = self._build_joined_factor_sql(
            spec_list,
            start=start,
            end=end,
            codes=codes,
            join_keys=join_keys,
            how=how,
        )
        query = con.sql(sql, dialect="duckdb")
        if filters:
            for i, func in enumerate(filters):
                if not callable(func):
                    raise TypeError(f"filters[{i}] must be callable")
                expr = func(query)
                query = query.filter(expr)
        return query

    def get_factors(
        self,
        specs: Iterable[FactorSpec],
        *,
        start: object | None = None,
        end: object | None = None,
        codes: Iterable[str] | None = None,
        join_keys: Iterable[str] | None = None,
        how: str = "inner",
        filters: Iterable[IbisFilter] | None = None,
        engine: LoadEngine = "pandas",
        as_query: bool = False,
    ) -> pd.DataFrame | Any:
        """
        多因子读取入口：
        - specs: 多个 FactorSpec 组成的列表
        - 默认按 code + date_ join
        - engine=\"ibis\" 或 as_query=True 返回 Ibis relation
        - 其他 engine 物化为 pandas / polars
        """
        query = self.get_factors_query(
            specs,
            start=start,
            end=end,
            codes=codes,
            join_keys=join_keys,
            how=how,
            filters=filters
        )
        if as_query or engine == "ibis":
            return query

        frame = query.to_pandas()
        if not frame.empty:
            if "date_" in frame.columns:
                frame["date_"] = pd.to_datetime(frame["date_"], errors="coerce").dt.tz_localize(None)
            if "code" in frame.columns:
                frame["code"] = frame["code"].astype(str)
            sort_keys = [column for column in ("code", "date_") if column in frame.columns]
            if sort_keys:
                frame = frame.sort_values(sort_keys, kind="stable").reset_index(drop=True)
        return _to_polars_frame(frame) if engine == "polars" else frame

    def get_factor(
        self,
        spec,
        *,
        start: object | None = None,
        end: object | None = None,
        codes: Iterable[str] | None = None,
        columns: Iterable[str] | None = None,
        value_min: float | None = None,
        value_max: float | None = None,
        engine: LoadEngine = "pandas",
        as_query: bool = False,
    ) -> pd.DataFrame | Any:
        """
        Factor 专用读取入口：
        - 默认返回 pandas
        - engine="polars" 返回 polars
        - engine="ibis" 或 as_query=True 返回 query relation
        - engine="duckdb" 通过 DuckDB + Ibis 物化后返回 pandas / polars
        - factor 数据优先走 Ibis + DuckDB 查询路径
        """
        if as_query or engine == "ibis":
            return self.get_factor_query(
                spec,
                start=start,
                end=end,
                codes=codes,
                columns=columns,
                value_min=value_min,
                value_max=value_max,
            )

        query_backend_available = ibis is not None and duckdb is not None
        if spec.kind() == "factor" and query_backend_available:
            query = self.get_factor_query(
                spec,
                start=start,
                end=end,
                codes=codes,
                columns=columns,
                value_min=value_min,
                value_max=value_max,
            )
            frame = query.to_pandas()
            if not frame.empty:
                frame = self._normalize_loaded_frame(spec, frame)
                sort_columns: list[str] = []
                if getattr(spec, "code_column", None):
                    sort_columns.append(spec.code_column or "code")
                if getattr(spec, "date_column", None):
                    sort_columns.append(spec.date_column)
                if sort_columns:
                    frame = frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)
            return _to_polars_frame(frame) if engine == "polars" else frame

        frame = self.load_dataset(spec, start=start, end=end, engine="pandas")
        if not isinstance(frame, pd.DataFrame):
            frame = _to_pandas_frame(frame)

        if spec.kind() == "factor":
            if codes is not None and getattr(spec, "code_column", None) and spec.code_column in frame.columns:
                normalized_codes = {str(code) for code in codes}
                frame = frame.loc[frame[spec.code_column].astype(str).isin(normalized_codes)]
            if columns is not None:
                selected = [str(column) for column in columns if str(column) in frame.columns]
                if selected:
                    frame = frame.loc[:, selected]
            factor_value_column = getattr(spec, "value_column", "value")
            if value_min is not None and factor_value_column in frame.columns:
                frame = frame.loc[pd.to_numeric(frame[factor_value_column], errors="coerce") >= value_min]
            if value_max is not None and factor_value_column in frame.columns:
                frame = frame.loc[pd.to_numeric(frame[factor_value_column], errors="coerce") <= value_max]
            frame = self._normalize_loaded_frame(spec, frame)
            sort_columns: list[str] = []
            if getattr(spec, "code_column", None) and spec.code_column in frame.columns:
                sort_columns.append(spec.code_column)
            if getattr(spec, "date_column", None) and spec.date_column in frame.columns:
                sort_columns.append(spec.date_column)
            if sort_columns:
                frame = frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)

        if engine == "polars":
            return _to_polars_frame(frame)
        return frame
__all__ = ["DatasetSaveResult", "FactorStore", "MacroBatchResult"]
