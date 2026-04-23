from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal, Sequence

import pandas as pd

from tiger_factors.utils.calculation.time_engine import FactorTimeEngine
from tiger_factors.utils.calculation.types import Interval
from tiger_factors.utils.merge import merge_code_frames as util_merge_code_frames
from tiger_factors.utils.merge import merge_other_frames as util_merge_other_frames
from tiger_reference.adjustments import adj_df

try:  # pragma: no cover - optional dependency
    import polars as pl
except Exception:  # pragma: no cover - optional dependency
    pl = None

FrameBackend = Literal["auto", "pandas", "polars"]


@dataclass(frozen=True)
class VectorDatasetSpec:
    name: str
    frame: pd.DataFrame | Any
    time_column: str = "date_"
    code_column: str | None = None
    join_keys: Sequence[str] | None = None
    prefix: str | None = None
    forward_fill: bool = False
    value_columns: Sequence[str] | None = None


@dataclass(frozen=True)
class VectorMergeResult:
    frame: pd.DataFrame | Any
    calendar: pd.DataFrame | Any
    key_column: str
    eff_column: str
    time_kind: str
    join_keys: tuple[str, ...] = ()
    output_path: str | None = None

    @property
    def available_column(self) -> str:
        """Backward-compatible alias for the effective-time column name."""
        return self.eff_column


class FactorVectorizationTransformer:
    """
    Vectorized factor engine with a trading-calendar master table.

    The engine builds a single trading-calendar table with ``exchange_date_``
    (the exchange/session date), a backward-compatible ``date_`` alias, and
    ``eff_at`` (the shifted effective time) first, then left-joins every
    dataset onto that table. Missing rows stay null by default. Optional
    forward fill can be enabled per dataset.

    Merge entry points are intentionally split by join shape:

    - ``merge_code_list``: merge code-only lookup tables.
    - ``merge_date_list``: merge date-only series.
    - ``merge_code_date_list``: merge panel data and attach it to the trading
      calendar.
    - ``merge_other_list``: merge custom-key tables and return the intermediate
      result without attaching to the calendar.

    The calendar shift is controlled by ``lag``:

    - ``lag=1`` means factor rows become available one trading step later.
    - ``lag=3`` means factor rows become available three trading steps later.
    """

    def __init__(
        self,
        *,
        calendar: str = "XNYS",
        start: date | str | pd.Timestamp,
        end: date | str | pd.Timestamp | None = None,
        interval: Interval | None = None,
        lag: int = 1,
    ) -> None:
        self.time_engine = FactorTimeEngine(
            calendar=calendar,
            start=start,
            end=end,
            interval=interval,
        )
        self.set_lag(lag)

    @property
    def interval(self) -> Interval:
        return self.time_engine.interval

    @property
    def key_column(self) -> str:
        return "exchange_date_"

    @property
    def eff_column(self) -> str:
        return "eff_at"

    @property
    def available_column(self) -> str:
        """Backward-compatible alias for the effective-time column name."""
        return self.eff_column

    @property
    def time_kind(self) -> str:
        return "daily" if self.interval.is_daily else "intraday"

    @property
    def lag(self) -> int:
        return self._lag

    def set_interval(self, interval: Interval | None = None, **components: int) -> Interval:
        return self.time_engine.set_interval(interval, **components)

    def set_lag(self, lag: int) -> int:
        lag_value = int(lag)
        if lag_value < 0:
            raise ValueError("lag must be non-negative.")
        self._lag = lag_value
        return self._lag

    def build_calendar_frame(
        self,
        *,
        start: date | str | pd.Timestamp | None = None,
        end: date | str | pd.Timestamp | None = None,
        trading_days: Sequence[date | str | pd.Timestamp] | None = None,
        backend: FrameBackend = "pandas",
        set_freq: bool = False,   # 👈 新增
    ) -> pd.DataFrame | Any:
        steps = self.time_engine.resolve_schedule_points(start=start, end=end, trading_days=trading_days)
        dates = [step.at for step in steps]
        available_times = self._shift_forward_times(dates)
        rows: list[dict[str, Any]] = []
        for step, eff_at in zip(steps, available_times):
            row: dict[str, Any] = {
                self.key_column: step.at,
                "date_": step.at,
                self.eff_column: eff_at,
                "trading_day": step.trading_day,
                "step_index": step.step_index,
                "step_kind": step.step_kind,
            }
            if self.time_kind == "intraday":
                row.update(
                    {
                        "session_open": step.session_open,
                        "session_close": step.session_close,
                        "is_session_open": step.is_session_open,
                        "is_session_close": step.is_session_close,
                    }
                )
            rows.append(row)

        base_columns = [self.key_column, "date_", self.eff_column, "trading_day", "step_index", "step_kind"]
        if self.time_kind == "intraday":
            base_columns.extend(["session_open", "session_close", "is_session_open", "is_session_close"])
        frame = pd.DataFrame(rows, columns=base_columns)
        if set_freq:
            pandas_freq = self.interval.get_pandas_freq()
            frame = frame.set_index(self.key_column).asfreq(pandas_freq).reset_index()
        if backend == "pandas" or frame.empty:
            return frame
        if backend == "polars":
            return self._to_polars(frame)
        if backend == "auto":
            return frame
        raise ValueError("backend must be one of: auto, pandas, polars")

    def merge(
        self,
        datasets: Sequence[pd.DataFrame | Any | VectorDatasetSpec],
        *,
        time_column: str | None = None,
        code_column: str | None = None,
        names: Sequence[str] | None = None,
        forward_fill: bool | Sequence[bool | None] | None = None,
        start: date | str | pd.Timestamp | None = None,
        end: date | str | pd.Timestamp | None = None,
        trading_days: Sequence[date | str | pd.Timestamp] | None = None,
        codes: Sequence[str] | None = None,
        backend: FrameBackend = "auto",
        output_path: str | Path | None = None,
    ) -> VectorMergeResult:
        if not datasets:
            raise ValueError("datasets must not be empty.")

        specs = self._coerce_specs(
            datasets,
            time_column=time_column,
            code_column=code_column,
            names=names,
            forward_fill=forward_fill,
        )

        calendar = self.build_calendar_frame(start=start, end=end, trading_days=trading_days, backend="pandas")
        merged = calendar.copy()
        key_column = self.key_column
        eff_column = self.eff_column
        code_column = self._resolve_code_column(specs)

        if code_column is not None:
            code_values = self._resolve_codes(specs, code_column, codes)
            merged = self._expand_calendar_for_codes(merged, code_column, code_values)

        for spec in specs:
            source = self._to_pandas(spec.frame)
            source = self._normalize_source_frame(
                source,
                key_column=key_column,
                spec=spec,
                code_column=code_column,
            )
            source_columns = self._value_columns_for_spec(source, spec, code_column)
            if not source_columns:
                continue

            source = source[[key_column] + ([code_column] if code_column is not None else []) + source_columns].copy()
            source = source.dropna(subset=[key_column])
            join_keys = [key_column] + ([code_column] if code_column is not None else [])
            source = source.sort_values(join_keys, kind="stable")
            source = source.drop_duplicates(subset=join_keys, keep="last")

            prefix = spec.prefix if spec.prefix is not None else spec.name
            rename_map = {column: f"{prefix}__{column}" for column in source_columns if prefix != ""}
            source = source.rename(columns=rename_map)
            joined_columns = [rename_map.get(column, column) for column in source_columns]

            merged = merged.merge(source, how="left", on=join_keys, sort=False)
            if spec.forward_fill:
                merged = self._forward_fill(merged, columns=joined_columns, key_column=key_column, code_column=code_column)

        merged = self._sort_merged(merged, key_column=key_column, code_column=code_column)
        merged = self._reorder_columns(merged, code_column=code_column)
        output_backend = self._resolve_backend(backend, specs)
        output_frame = self._convert_backend(merged, output_backend)
        calendar_frame = self._convert_backend(calendar, output_backend)

        saved_path = self._save_frame(output_frame, output_path)
        return VectorMergeResult(
            frame=output_frame,
            calendar=calendar_frame,
            key_column=key_column,
            eff_column=eff_column,
            time_kind=self.time_kind,
            join_keys=tuple((key_column,) + ((code_column,) if code_column is not None else ())),
            output_path=saved_path,
        )

    def merge_code_date_frames(
        self,
        datasets: Sequence[pd.DataFrame | Any | VectorDatasetSpec],
        *,
        time_column: str | None = None,
        code_column: str | None = None,
        names: Sequence[str] | None = None,
        forward_fill: bool | Sequence[bool | None] | None = None,
        start: date | str | pd.Timestamp | None = None,
        end: date | str | pd.Timestamp | None = None,
        trading_days: Sequence[date | str | pd.Timestamp] | None = None,
        codes: Sequence[str] | None = None,
        backend: FrameBackend = "auto",
        output_path: str | Path | None = None,
    ) -> VectorMergeResult:
        """Merge panel-style datasets and attach them to the trading calendar."""
        return self.merge(
            datasets,
            time_column=time_column,
            code_column=code_column,
            names=names,
            forward_fill=forward_fill,
            start=start,
            end=end,
            trading_days=trading_days,
            codes=codes,
            backend=backend,
            output_path=output_path,
        )

    def merge_code_date_list(
        self,
        datasets: Sequence[pd.DataFrame | Any | VectorDatasetSpec],
        **kwargs: Any,
    ) -> VectorMergeResult:
        return self.merge_code_date_frames(datasets, **kwargs)

    def merge_date_frames(
        self,
        datasets: Sequence[pd.DataFrame | Any | VectorDatasetSpec],
        *,
        time_column: str = "date_",
        names: Sequence[str] | None = None,
        forward_fill: bool | Sequence[bool | None] | None = None,
        start: date | str | pd.Timestamp | None = None,
        end: date | str | pd.Timestamp | None = None,
        trading_days: Sequence[date | str | pd.Timestamp] | None = None,
        backend: FrameBackend = "auto",
        output_path: str | Path | None = None,
    ) -> VectorMergeResult:
        """Merge date-only datasets onto the trading-calendar axis."""
        return self.merge(
            datasets,
            time_column=time_column,
            code_column=None,
            names=names,
            forward_fill=forward_fill,
            start=start,
            end=end,
            trading_days=trading_days,
            backend=backend,
            output_path=output_path,
        )

    def merge_date_list(
        self,
        datasets: Sequence[pd.DataFrame | Any | VectorDatasetSpec],
        **kwargs: Any,
    ) -> VectorMergeResult:
        return self.merge_date_frames(datasets, **kwargs)

    def merge_code_frames(
        self,
        datasets: Sequence[pd.DataFrame | Any | VectorDatasetSpec],
        *,
        code_column: str = "code",
        names: Sequence[str] | None = None,
        backend: FrameBackend = "auto",
        output_path: str | Path | None = None,
    ) -> VectorMergeResult:
        """Merge code-only lookup tables into a code-domain intermediate frame."""
        return self._merge_keyed_frames(
            datasets,
            join_keys=(code_column,),
            names=names,
            backend=backend,
            output_path=output_path,
            key_aliases={code_column: (code_column, "symbol", "ticker", "code_")},
        )

    def merge_code_list(
        self,
        datasets: Sequence[pd.DataFrame | Any | VectorDatasetSpec],
        **kwargs: Any,
    ) -> VectorMergeResult:
        return self.merge_code_frames(datasets, **kwargs)

    def merge_other_frames(
        self,
        datasets: Sequence[pd.DataFrame | Any | VectorDatasetSpec],
        *,
        join_keys: Sequence[str],
        names: Sequence[str] | None = None,
        backend: FrameBackend = "auto",
        output_path: str | Path | None = None,
        key_aliases: dict[str, Sequence[str]] | None = None,
    ) -> VectorMergeResult:
        """Merge custom-key tables and return an intermediate keyed result."""
        normalized_keys = tuple(str(key) for key in join_keys)
        if not normalized_keys:
            raise ValueError("join_keys must not be empty.")
        return self._merge_keyed_frames(
            datasets,
            join_keys=normalized_keys,
            names=names,
            backend=backend,
            output_path=output_path,
            key_aliases=key_aliases,
        )

    def merge_other_list(
        self,
        datasets: Sequence[pd.DataFrame | Any | VectorDatasetSpec],
        **kwargs: Any,
    ) -> VectorMergeResult:
        return self.merge_other_frames(datasets, **kwargs)

    def adjust(
        self,
        frame: pd.DataFrame | Any,
        *,
        time_column: str | None = None,
        code_column: str = "code",
        drop_adj_close: bool = True,
        dividends: bool = False,
        history: bool = False,
        backend: FrameBackend = "auto",
    ) -> pd.DataFrame | Any:
        """
        Adjust one long OHLCV table by delegating to ``tiger_reference.adj_df``.

        The method only normalizes the table to the reference ``date_`` / ``code``
        shape, runs the price adjustment on the whole table, and then restores the
        original column names if needed.
        """
        source = self._to_pandas(frame)
        resolved_time_column = time_column or self.key_column
        rename_map: dict[str, str] = {}
        if resolved_time_column != "date_" and resolved_time_column in source.columns:
            rename_map[resolved_time_column] = "date_"
        if code_column != "code" and code_column in source.columns:
            rename_map[code_column] = "code"
        if rename_map:
            source = source.rename(columns=rename_map)

        adjusted = adj_df(source, drop_adj_close=drop_adj_close, dividends=dividends, history=history)

        if resolved_time_column != "date_" and "date_" in adjusted.columns:
            adjusted = adjusted.rename(columns={"date_": resolved_time_column})
        if code_column != "code" and "code" in adjusted.columns:
            adjusted = adjusted.rename(columns={"code": code_column})
        return self._convert_backend(adjusted, self._resolve_wide_backend(backend, frame))

    def adjust_raw_ohlcv(self, *args: object, **kwargs: object) -> pd.DataFrame | Any:
        return self.adjust(*args, **kwargs)

    def long_to_wide(
        self,
        frame: pd.DataFrame | Any,
        *,
        time_column: str | None = None,
        code_column: str = "code",
        value_columns: Sequence[str] | None = None,
        aggfunc: str = "last",
        fill_value: Any | None = None,
        separator: str = "__",
        flatten: bool = True,
        backend: FrameBackend = "auto",
    ) -> pd.DataFrame | Any:
        """
        Convert a long panel frame into a wide matrix.

        - Single value column -> one column per code.
        - Multiple value columns -> ``{code}{separator}{value}`` columns when
          ``flatten`` is True, or a MultiIndex when ``flatten`` is False.
        """
        source = self._to_pandas(frame)
        if source.empty:
            empty = pd.DataFrame()
            return self._convert_backend(empty, self._resolve_wide_backend(backend, frame))

        resolved_time_column = time_column or self.key_column
        if resolved_time_column not in source.columns:
            raise KeyError(f"frame is missing time column {resolved_time_column!r}.")
        if code_column not in source.columns:
            raise KeyError(f"frame is missing code column {code_column!r}.")

        source = source.copy()
        source[code_column] = source[code_column].astype(str)

        if value_columns is None:
            excluded = {
                resolved_time_column,
                code_column,
                self.key_column,
                self.eff_column,
                "trading_day",
                "step_index",
                "step_kind",
                "session_open",
                "session_close",
                "is_session_open",
                "is_session_close",
            }
            value_columns = [column for column in source.columns if column not in excluded]
        else:
            value_columns = [column for column in value_columns if column in source.columns and column not in {resolved_time_column, code_column}]

        if not value_columns:
            raise ValueError("value_columns must contain at least one usable column.")

        code_order = list(dict.fromkeys(source[code_column].dropna().astype(str).tolist()))
        if not code_order:
            raise ValueError("frame does not contain any usable codes.")

        source[resolved_time_column] = pd.to_datetime(source[resolved_time_column], errors="coerce")
        source = source.dropna(subset=[resolved_time_column, code_column])
        source = source.sort_values([resolved_time_column, code_column], kind="stable")

        if len(value_columns) == 1:
            value_column = value_columns[0]
            wide = source.pivot_table(
                index=resolved_time_column,
                columns=code_column,
                values=value_column,
                aggfunc=aggfunc,
                fill_value=fill_value,
                sort=False,
            )
            wide = wide.reindex(columns=code_order)
            wide.index.name = resolved_time_column
            if flatten:
                wide.columns = [str(column) for column in wide.columns]
            output = wide.sort_index()
            return self._convert_backend(output, self._resolve_wide_backend(backend, frame))

        wide_pieces: dict[str, pd.DataFrame] = {}
        for value_column in value_columns:
            piece = source.pivot_table(
                index=resolved_time_column,
                columns=code_column,
                values=value_column,
                aggfunc=aggfunc,
                fill_value=fill_value,
                sort=False,
            )
            piece = piece.reindex(columns=code_order)
            wide_pieces[value_column] = piece

        combined = pd.concat([wide_pieces[value] for value in value_columns], axis=1, keys=list(value_columns))
        combined.index.name = resolved_time_column
        combined = combined.sort_index()

        if flatten:
            flat_columns: list[str] = []
            for code in code_order:
                for value_column in value_columns:
                    flat_columns.append(f"{code}{separator}{value_column}")
            combined = combined.reindex(columns=pd.MultiIndex.from_tuples([(value, code) for code in code_order for value in value_columns]))
            combined.columns = flat_columns

        return self._convert_backend(combined, self._resolve_wide_backend(backend, frame))

    def _coerce_specs(
        self,
        datasets: Sequence[pd.DataFrame | Any | VectorDatasetSpec],
        *,
        time_column: str | None,
        code_column: str | None,
        names: Sequence[str] | None,
        forward_fill: bool | Sequence[bool | None] | None,
    ) -> list[VectorDatasetSpec]:
        forward_fill_values = self._normalize_forward_fill(datasets, forward_fill)
        specs: list[VectorDatasetSpec] = []
        for index, dataset in enumerate(datasets):
            name = names[index] if names is not None and index < len(names) else f"dataset_{index}"
            if isinstance(dataset, VectorDatasetSpec):
                dataset_forward_fill = dataset.forward_fill
                if forward_fill_values[index] is not None:
                    dataset_forward_fill = bool(forward_fill_values[index])
                specs.append(
                    VectorDatasetSpec(
                        name=dataset.name,
                        frame=dataset.frame,
                        time_column=dataset.time_column,
                        code_column=dataset.code_column,
                        join_keys=dataset.join_keys,
                        prefix=dataset.prefix,
                        forward_fill=dataset_forward_fill,
                        value_columns=dataset.value_columns,
                    )
                )
                continue
            dataset_forward_fill = bool(forward_fill_values[index]) if forward_fill_values[index] is not None else False
            specs.append(
                VectorDatasetSpec(
                    name=name,
                    frame=dataset,
                    time_column=time_column or self.key_column,
                    code_column=code_column,
                    join_keys=None,
                    prefix=name,
                    forward_fill=dataset_forward_fill,
                )
            )
        return specs

    def _merge_keyed_frames(
        self,
        datasets: Sequence[pd.DataFrame | Any | VectorDatasetSpec],
        *,
        join_keys: Sequence[str],
        names: Sequence[str] | None = None,
        backend: FrameBackend = "auto",
        output_path: str | Path | None = None,
        key_aliases: dict[str, Sequence[str]] | None = None,
    ) -> VectorMergeResult:
        if not datasets:
            raise ValueError("datasets must not be empty.")

        normalized_keys = tuple(str(key) for key in join_keys)
        specs = self._coerce_key_specs(
            datasets,
            join_keys=normalized_keys,
            names=names,
        )
        frames = [self._to_pandas(spec.frame) for spec in specs]
        merge_names = [spec.prefix if spec.prefix is not None else spec.name for spec in specs]
        if len(normalized_keys) == 1 and normalized_keys[0] == "code":
            merged = util_merge_code_frames(frames, code_column=normalized_keys[0], names=merge_names)
        else:
            merged = util_merge_other_frames(frames, join_keys=normalized_keys, names=merge_names)

        merged = merged.sort_values(list(normalized_keys), kind="stable").reset_index(drop=True)
        output_backend = self._resolve_backend(backend, specs)
        output_frame = self._convert_backend(merged, output_backend)
        saved_path = self._save_frame(output_frame, output_path)
        return VectorMergeResult(
            frame=output_frame,
            calendar=merged,
            key_column=normalized_keys[0],
            eff_column="",
            time_kind="static",
            join_keys=normalized_keys,
            output_path=saved_path,
        )

    def _coerce_key_specs(
        self,
        datasets: Sequence[pd.DataFrame | Any | VectorDatasetSpec],
        *,
        join_keys: Sequence[str],
        names: Sequence[str] | None,
    ) -> list[VectorDatasetSpec]:
        specs: list[VectorDatasetSpec] = []
        for index, dataset in enumerate(datasets):
            name = names[index] if names is not None and index < len(names) else f"dataset_{index}"
            if isinstance(dataset, VectorDatasetSpec):
                specs.append(
                    VectorDatasetSpec(
                        name=dataset.name,
                        frame=dataset.frame,
                        time_column=dataset.time_column,
                        code_column=dataset.code_column,
                        join_keys=tuple(dataset.join_keys) if dataset.join_keys is not None else tuple(join_keys),
                        prefix=dataset.prefix,
                        forward_fill=dataset.forward_fill,
                        value_columns=dataset.value_columns,
                    )
                )
            else:
                specs.append(
                    VectorDatasetSpec(
                        name=name,
                        frame=dataset,
                        join_keys=tuple(join_keys),
                        prefix=name,
                    )
                )
        return specs

    def _normalize_keyed_source(
        self,
        source: pd.DataFrame,
        *,
        join_keys: Sequence[str],
        key_aliases: dict[str, Sequence[str]] | None = None,
    ) -> pd.DataFrame:
        if source.empty:
            return source

        result = source.copy()
        rename_map: dict[str, str] = {}
        aliases = key_aliases or {}
        for key in join_keys:
            if key in result.columns:
                continue
            for alias in aliases.get(key, ()):  # explicit first, then direct matches
                if alias in result.columns:
                    rename_map[alias] = key
                    break
        if rename_map:
            result = result.rename(columns=rename_map)

        missing = [key for key in join_keys if key not in result.columns]
        if missing:
            raise KeyError(f"frame is missing join columns: {missing}")

        for key in join_keys:
            if key in {"code", "symbol", "ticker", "code_"}:
                result[key] = result[key].astype(str)
            elif key.lower() in {"date", "date_", "datetime", "available_time", "eff_at"}:
                result[key] = pd.to_datetime(result[key], errors="coerce").dt.tz_localize(None)

        return result

    def _value_columns_for_keyed_spec(
        self,
        source: pd.DataFrame,
        *,
        join_keys: Sequence[str],
    ) -> list[str]:
        excluded = set(join_keys)
        return [column for column in source.columns if column not in excluded]

    def _normalize_forward_fill(
        self,
        datasets: Sequence[pd.DataFrame | Any | VectorDatasetSpec],
        forward_fill: bool | Sequence[bool | None] | None,
    ) -> list[bool | None]:
        if forward_fill is None:
            return [None] * len(datasets)
        if isinstance(forward_fill, Sequence) and not isinstance(forward_fill, (str, bytes)):
            if len(forward_fill) != len(datasets):
                raise ValueError("forward_fill sequence must have the same length as datasets.")
            values: list[bool | None] = []
            for item in forward_fill:
                if item is None:
                    values.append(None)
                else:
                    values.append(bool(item))
            return values
        return [bool(forward_fill) for _ in datasets]

    def _resolve_wide_backend(self, backend: FrameBackend, frame: pd.DataFrame | Any) -> str:
        if backend in {"pandas", "polars"}:
            return backend
        if backend != "auto":
            raise ValueError("backend must be one of: auto, pandas, polars")
        if self._is_polars_frame(frame):
            return "polars"
        return "pandas"

    def _resolve_backend(self, backend: FrameBackend, datasets: Sequence[VectorDatasetSpec]) -> str:
        if backend in {"pandas", "polars"}:
            return backend
        if backend != "auto":
            raise ValueError("backend must be one of: auto, pandas, polars")
        for spec in datasets:
            if self._is_polars_frame(spec.frame):
                return "polars"
        return "pandas"

    def _resolve_code_column(self, datasets: Sequence[VectorDatasetSpec]) -> str | None:
        code_columns = {spec.code_column for spec in datasets if spec.code_column}
        if not code_columns:
            return None
        if len(code_columns) != 1:
            raise ValueError("all datasets must use the same code_column when panel data is merged.")
        return next(iter(code_columns))

    def _resolve_codes(
        self,
        datasets: Sequence[VectorDatasetSpec],
        code_column: str,
        codes: Sequence[str] | None,
    ) -> tuple[str, ...]:
        if codes is not None:
            return tuple(dict.fromkeys(str(code) for code in codes))

        gathered: list[str] = []
        for spec in datasets:
            if spec.code_column != code_column:
                continue
            source = self._to_pandas(spec.frame)
            if code_column not in source.columns:
                continue
            values = source[code_column].dropna().astype(str).unique().tolist()
            for value in values:
                if value not in gathered:
                    gathered.append(value)
        return tuple(gathered)

    def _normalize_source_frame(
        self,
        source: pd.DataFrame,
        *,
        key_column: str,
        spec: VectorDatasetSpec,
        code_column: str | None,
    ) -> pd.DataFrame:
        if source.empty:
            return source

        source = source.copy()
        if spec.time_column not in source.columns:
            raise KeyError(f"{spec.name!r} is missing time column {spec.time_column!r}.")

        if code_column is not None:
            if code_column not in source.columns:
                raise KeyError(f"{spec.name!r} is missing code column {code_column!r}.")
            source[code_column] = source[code_column].astype(str)

        if self.time_kind == "daily":
            source[key_column] = pd.to_datetime(source[spec.time_column], errors="coerce").dt.date
        else:
            source[key_column] = self._to_utc_timestamp_series(source[spec.time_column])

        return source

    def _value_columns_for_spec(
        self,
        source: pd.DataFrame,
        spec: VectorDatasetSpec,
        code_column: str | None,
    ) -> list[str]:
        excluded = {spec.time_column, self.key_column, self.eff_column}
        if code_column is not None:
            excluded.add(code_column)
        if spec.value_columns is not None:
            return [column for column in spec.value_columns if column in source.columns and column not in excluded]
        return [column for column in source.columns if column not in excluded]

    def _expand_calendar_for_codes(
        self,
        calendar: pd.DataFrame,
        code_column: str,
        codes: Sequence[str],
    ) -> pd.DataFrame:
        if not codes:
            raise ValueError("codes must not be empty when panel data is merged.")
        left = calendar.copy()
        left["__vector_join_key__"] = 1
        right = pd.DataFrame({code_column: list(codes)})
        right["__vector_join_key__"] = 1
        merged = left.merge(right, on="__vector_join_key__", how="left").drop(columns=["__vector_join_key__"])
        return merged

    def _forward_fill(
        self,
        frame: pd.DataFrame,
        *,
        columns: Sequence[str],
        key_column: str,
        code_column: str | None,
    ) -> pd.DataFrame:
        if not columns:
            return frame
        sorted_frame = frame.sort_values(([code_column, key_column] if code_column else [key_column]), kind="stable")
        if code_column is not None:
            sorted_frame.loc[:, list(columns)] = sorted_frame.groupby(code_column, sort=False)[list(columns)].ffill()
        else:
            sorted_frame.loc[:, list(columns)] = sorted_frame.loc[:, list(columns)].ffill()
        return sorted_frame.sort_values(([key_column, code_column] if code_column else [key_column]), kind="stable")

    def _sort_merged(self, frame: pd.DataFrame, *, key_column: str, code_column: str | None) -> pd.DataFrame:
        sort_keys = [key_column]
        if code_column is not None:
            sort_keys.append(code_column)
        return frame.sort_values(sort_keys, kind="stable").reset_index(drop=True)

    def _reorder_columns(self, frame: pd.DataFrame, *, code_column: str | None) -> pd.DataFrame:
        calendar_columns = [
            self.key_column,
            "date_",
            self.eff_column,
            "trading_day",
            "step_index",
            "step_kind",
        ]
        if self.time_kind == "intraday":
            calendar_columns.extend(["session_open", "session_close", "is_session_open", "is_session_close"])
        ordered = [column for column in calendar_columns if column in frame.columns]
        if code_column is not None and code_column in frame.columns:
            ordered.append(code_column)
        ordered.extend([column for column in frame.columns if column not in ordered])
        return frame.loc[:, ordered]

    @staticmethod
    def _is_polars_frame(frame: Any) -> bool:
        return pl is not None and isinstance(frame, pl.DataFrame)

    @staticmethod
    def _to_pandas(frame: pd.DataFrame | Any) -> pd.DataFrame:
        if isinstance(frame, pd.DataFrame):
            return frame.copy()
        if pl is not None and isinstance(frame, pl.DataFrame):  # pragma: no cover - optional dependency
            return frame.to_pandas()
        raise TypeError("frame must be a pandas.DataFrame or polars.DataFrame.")

    @staticmethod
    def _to_polars(frame: pd.DataFrame) -> Any:
        if pl is None:  # pragma: no cover - optional dependency
            raise ImportError("polars is not installed.")
        return pl.from_pandas(frame)

    def _convert_backend(self, frame: pd.DataFrame, backend: str) -> pd.DataFrame | Any:
        if backend == "pandas":
            return frame
        if backend == "polars":
            return self._to_polars(frame)
        raise ValueError("backend must be one of: pandas, polars")

    def _save_frame(self, frame: pd.DataFrame | Any, output_path: str | Path | None) -> str | None:
        if output_path is None:
            return None

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()

        if isinstance(frame, pd.DataFrame):
            if suffix in {".parquet", ".pq"}:
                frame.to_parquet(path, index=False)
            elif suffix == ".csv":
                frame.to_csv(path, index=False)
            elif suffix == ".feather":
                frame.to_feather(path)
            else:
                raise ValueError("output_path must end with .parquet, .pq, .csv, or .feather.")
            return str(path)

        if pl is not None and isinstance(frame, pl.DataFrame):  # pragma: no cover - optional dependency
            if suffix in {".parquet", ".pq"}:
                frame.write_parquet(path)
            elif suffix == ".csv":
                frame.write_csv(path)
            elif suffix == ".feather":
                frame.write_ipc(path)
            else:
                raise ValueError("output_path must end with .parquet, .pq, .csv, or .feather.")
            return str(path)

        raise TypeError("frame must be a pandas or polars DataFrame.")

    @staticmethod
    def _to_utc_timestamp_series(values: pd.Series) -> pd.Series:
        timestamps = pd.to_datetime(values, errors="coerce")
        if getattr(timestamps.dt, "tz", None) is None:
            return timestamps.dt.tz_localize("UTC")
        return timestamps.dt.tz_convert("UTC")

    def _shift_forward_times(self, values: Sequence[Any]) -> list[Any]:
        time_values = list(values)
        if not time_values:
            return []
        if self._lag == 0:
            return list(time_values)

        missing_value = pd.NaT if self.time_kind == "intraday" else None
        if self._lag >= len(time_values):
            return [missing_value for _ in time_values]
        return list(time_values[self._lag :]) + [missing_value for _ in range(self._lag)]


__all__ = ["FactorVectorizationTransformer", "VectorDatasetSpec", "VectorMergeResult"]
