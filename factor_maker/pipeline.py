from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from tiger_factors.utils.calculation.time_engine import EXCHANGE_CALENDAR_ALIASES
from tiger_reference.calendar import build_trading_sessions
from tiger_factors.factor_store import TigerFactorLibrary


def _as_frame(value: pd.DataFrame | pd.Series | float | int, index: pd.Index, columns: pd.Index) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.reindex(index=index, columns=columns)
    if isinstance(value, pd.Series):
        if value.index.equals(index):
            return pd.DataFrame({column: value for column in columns}, index=index)
        return pd.DataFrame([value.reindex(columns)] * len(index), index=index, columns=columns)
    return pd.DataFrame(value, index=index, columns=columns, dtype=float)


def _resolve_trading_calendar(
    *,
    trading_calendar: str | None,
) -> str | None:
    if trading_calendar is None:
        return None
    normalized = str(trading_calendar).strip().upper()
    if not normalized:
        return None
    return EXCHANGE_CALENDAR_ALIASES.get(normalized, normalized)


@dataclass(frozen=True)
class ColumnSpec:
    provider: str
    dataset: str
    field: str
    freq: str
    source_type: str

    def bind(self) -> "BoundColumn":
        return BoundColumn(self)


class DataSet:
    provider: str
    dataset: str
    freq: str
    source_type: str
    fields: tuple[str, ...]

    @classmethod
    def column(cls, field: str) -> "BoundColumn":
        if field not in cls.fields:
            raise AttributeError(f"{field!r} is not declared on dataset {cls.__name__}.")
        return ColumnSpec(
            provider=cls.provider,
            dataset=cls.dataset,
            field=field,
            freq=cls.freq,
            source_type=cls.source_type,
        ).bind()


class USEquityPricing(DataSet):
    provider = "yahoo"
    dataset = "price"
    freq = "1d"
    source_type = "price"
    fields = ("open", "high", "low", "close", "adj_close", "volume")


class SimFinBalanceSheet(DataSet):
    provider = "simfin"
    dataset = "balance_sheet"
    freq = "1q"
    source_type = "fundamental"
    fields = (
        "total_assets",
        "total_equity",
        "total_liabilities",
        "total_current_assets",
        "total_current_liabilities",
        "shares_basic",
    )


class SimFinIncomeStatement(DataSet):
    provider = "simfin"
    dataset = "income_statement"
    freq = "1q"
    source_type = "fundamental"
    fields = ("net_income", "revenue", "operating_income")


class SimFinCashflowStatement(DataSet):
    provider = "simfin"
    dataset = "cashflow_statement"
    freq = "1q"
    source_type = "fundamental"
    fields = ("net_cfo",)


class PipelineContext:
    def __init__(
        self,
        *,
        library: TigerFactorLibrary,
        codes: list[str],
        start: str,
        end: str,
        price_provider: str | None = None,
        extra_lookback_days: int = 400,
        trading_calendar: str | None = None,
        calendar_source: str = "auto",
        provider_overrides: dict[str, str] | None = None,
        fundamental_use_point_in_time: bool = True,
        fundamental_availability_column: str | None = None,
        fundamental_lag_sessions: int = 1,
        as_ex: bool | None = None,
    ) -> None:
        self.library = library
        self.codes = list(dict.fromkeys(map(str, codes)))
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.extended_start = self.start - pd.Timedelta(days=extra_lookback_days)
        self.price_provider = price_provider or library.price_provider
        self.trading_calendar = _resolve_trading_calendar(trading_calendar=trading_calendar)
        self.calendar_source = calendar_source
        self.provider_overrides = provider_overrides or {}
        self.fundamental_use_point_in_time = fundamental_use_point_in_time
        self.fundamental_availability_column = fundamental_availability_column
        self.fundamental_lag_sessions = int(fundamental_lag_sessions)
        self.as_ex = as_ex
        self._column_cache: dict[tuple[str, str, str, str, str], pd.DataFrame] = {}
        self._trading_dates: pd.DatetimeIndex | None = None

    def _resolve_provider(self, column: ColumnSpec) -> str:
        # Priority: source_type override -> dataset override -> global default -> column default.
        source_key = f"source_type:{column.source_type}"
        dataset_key = f"dataset:{column.dataset}"
        return (
            self.provider_overrides.get(source_key)
            or self.provider_overrides.get(dataset_key)
            or self.provider_overrides.get("default")
            or column.provider
        )

    @property
    def trading_dates(self) -> pd.DatetimeIndex:
        if self._trading_dates is None:
            close = self.library.price_panel(
                codes=self.codes,
                start=str(self.extended_start.date()),
                end=str(self.end.date()),
                provider=self.price_provider,
                field="close",
                as_ex=self.as_ex,
            )
            fallback_index = pd.DatetimeIndex(close.index)
            if self.trading_calendar:
                sessions = build_trading_sessions(
                    start=self.extended_start,
                    end=self.end,
                    calendar_name=self.trading_calendar,
                    fallback_dates=fallback_index,
                )
                self._trading_dates = sessions if len(sessions) > 0 else fallback_index
            else:
                self._trading_dates = fallback_index
        return self._trading_dates

    @property
    def output_dates(self) -> pd.DatetimeIndex:
        return self.trading_dates[self.trading_dates >= self.start]

    def load_column(self, column: ColumnSpec) -> pd.DataFrame:
        provider = self._resolve_provider(column)
        key = (
            provider,
            column.dataset,
            column.field,
            column.freq,
            column.source_type,
            str(self.fundamental_use_point_in_time),
            str(self.fundamental_availability_column),
            str(self.fundamental_lag_sessions),
        )
        cached = self._column_cache.get(key)
        if cached is not None:
            return cached

        if column.source_type == "price":
            frame = self.library.price_panel(
                codes=self.codes,
                start=str(self.extended_start.date()),
                end=str(self.end.date()),
                provider=provider,
                field=column.field,
                as_ex=self.as_ex,
            )
        else:
            fundamentals = self.library.fetch_fundamental_data(
                provider=provider,
                name=column.dataset,
                freq=column.freq,
                codes=self.codes,
                start=str(self.extended_start.date()),
                end=str(self.end.date()),
                as_ex=self.as_ex,
            )
            aligned = self.library.align_fundamental_to_trading_dates(
                fundamentals,
                self.trading_dates,
                value_columns=[column.field],
                use_point_in_time=self.fundamental_use_point_in_time,
                availability_column=self.fundamental_availability_column,
                lag_sessions=self.fundamental_lag_sessions,
            )
            frame = aligned[column.field]

        frame = frame.reindex(index=self.trading_dates, columns=self.codes)
        self._column_cache[key] = frame
        return frame


class Term:
    def evaluate(self, context: PipelineContext) -> pd.DataFrame:
        raise NotImplementedError

    def _binary_op(self, other: Any, op) -> "ComputedTerm":
        return ComputedTerm(lambda ctx: op(self.evaluate(ctx), _coerce_term(other, ctx)))

    def __add__(self, other: Any) -> "ComputedTerm":
        return self._binary_op(other, lambda a, b: a + b)

    def __sub__(self, other: Any) -> "ComputedTerm":
        return self._binary_op(other, lambda a, b: a - b)

    def __mul__(self, other: Any) -> "ComputedTerm":
        return self._binary_op(other, lambda a, b: a * b)

    def __truediv__(self, other: Any) -> "ComputedTerm":
        return self._binary_op(other, lambda a, b: a / b.replace(0, np.nan) if isinstance(b, pd.DataFrame) else a / b)

    def __gt__(self, other: Any) -> "Filter":
        return Filter(lambda ctx: self.evaluate(ctx) > _coerce_term(other, ctx))

    def __lt__(self, other: Any) -> "Filter":
        return Filter(lambda ctx: self.evaluate(ctx) < _coerce_term(other, ctx))

    def __ge__(self, other: Any) -> "Filter":
        return Filter(lambda ctx: self.evaluate(ctx) >= _coerce_term(other, ctx))

    def __le__(self, other: Any) -> "Filter":
        return Filter(lambda ctx: self.evaluate(ctx) <= _coerce_term(other, ctx))

    def rank(self, ascending: bool = True) -> "Factor":
        return Factor(lambda ctx: self.evaluate(ctx).rank(axis=1, ascending=ascending, method="average"))

    def zscore(self) -> "Factor":
        return Factor(
            lambda ctx: self.evaluate(ctx)
            .sub(self.evaluate(ctx).mean(axis=1), axis=0)
            .div(self.evaluate(ctx).std(axis=1, ddof=0).replace(0, np.nan), axis=0)
        )

    def notnull(self) -> "Filter":
        return Filter(lambda ctx: self.evaluate(ctx).notna())

    def top(self, n: int) -> "Filter":
        return Filter(lambda ctx: self.evaluate(ctx).rank(axis=1, ascending=False, method="first") <= n)

    def bottom(self, n: int) -> "Filter":
        return Filter(lambda ctx: self.evaluate(ctx).rank(axis=1, ascending=True, method="first") <= n)


def _coerce_term(other: Any, context: PipelineContext) -> pd.DataFrame:
    if isinstance(other, Term):
        return other.evaluate(context)
    return _as_frame(other, context.trading_dates, pd.Index(context.codes))


class BoundColumn(Term):
    def __init__(self, spec: ColumnSpec) -> None:
        self.spec = spec

    def evaluate(self, context: PipelineContext) -> pd.DataFrame:
        return context.load_column(self.spec)


class ComputedTerm(Term):
    def __init__(self, fn) -> None:
        self.fn = fn

    def evaluate(self, context: PipelineContext) -> pd.DataFrame:
        return self.fn(context)


class Factor(ComputedTerm):
    pass


class Filter(ComputedTerm):
    def __and__(self, other: "Filter") -> "Filter":
        return Filter(lambda ctx: self.evaluate(ctx) & other.evaluate(ctx))

    def __or__(self, other: "Filter") -> "Filter":
        return Filter(lambda ctx: self.evaluate(ctx) | other.evaluate(ctx))

    def __invert__(self) -> "Filter":
        return Filter(lambda ctx: ~self.evaluate(ctx))


class CustomFactor(Term):
    inputs: tuple[BoundColumn, ...] = ()
    window_length: int = 1

    def compute(self, today, assets, out, *inputs) -> None:
        raise NotImplementedError

    def evaluate(self, context: PipelineContext) -> pd.DataFrame:
        input_frames = [column.evaluate(context).reindex(index=context.trading_dates, columns=context.codes) for column in self.inputs]
        output = pd.DataFrame(np.nan, index=context.trading_dates, columns=context.codes, dtype=float)
        assets = np.array(context.codes, dtype=object)

        for row_idx in range(self.window_length - 1, len(context.trading_dates)):
            today = context.trading_dates[row_idx]
            windows = [frame.iloc[row_idx - self.window_length + 1 : row_idx + 1].to_numpy(dtype=float) for frame in input_frames]
            out = np.full(len(context.codes), np.nan, dtype=float)
            self.compute(today, assets, out, *windows)
            output.iloc[row_idx] = out
        return output


class Returns(CustomFactor):
    inputs = (USEquityPricing.column("close"),)

    def __init__(self, window_length: int) -> None:
        self.window_length = window_length

    def compute(self, today, assets, out, close) -> None:
        start = close[0]
        end = close[-1]
        out[:] = end / start - 1.0


class SimpleMovingAverage(CustomFactor):
    def __init__(self, inputs: Iterable[BoundColumn], window_length: int) -> None:
        self.inputs = tuple(inputs)
        self.window_length = window_length

    def compute(self, today, assets, out, values) -> None:
        out[:] = np.nanmean(values, axis=0)


class RollingStdDev(CustomFactor):
    def __init__(self, inputs: Iterable[BoundColumn], window_length: int) -> None:
        self.inputs = tuple(inputs)
        self.window_length = window_length

    def compute(self, today, assets, out, values) -> None:
        finite_counts = np.isfinite(values).sum(axis=0)
        out[:] = np.nan
        valid = finite_counts >= 2
        if np.any(valid):
            out[valid] = np.nanstd(values[:, valid], axis=0)


@dataclass
class Pipeline:
    columns: dict[str, Term]
    screen: Filter | None = None


class PipelineEngine:
    def __init__(
        self,
        *,
        library: TigerFactorLibrary | None = None,
        region: str = "us",
        sec_type: str = "stock",
        price_provider: str = "yahoo",
        extra_lookback_days: int = 400,
        trading_calendar: str | None = None,
        calendar_source: str = "auto",
        provider_overrides: dict[str, str] | None = None,
        fundamental_use_point_in_time: bool = True,
        fundamental_availability_column: str | None = None,
        fundamental_lag_sessions: int = 1,
        as_ex: bool | None = None,
    ) -> None:
        self.library = library or TigerFactorLibrary(region=region, sec_type=sec_type, price_provider=price_provider)
        self.region = region
        self.sec_type = sec_type
        self.price_provider = price_provider
        self.extra_lookback_days = extra_lookback_days
        self.trading_calendar = _resolve_trading_calendar(trading_calendar=trading_calendar)
        self.calendar_source = calendar_source
        self.provider_overrides = provider_overrides or {}
        self.fundamental_use_point_in_time = fundamental_use_point_in_time
        self.fundamental_availability_column = fundamental_availability_column
        self.fundamental_lag_sessions = int(fundamental_lag_sessions)
        self.as_ex = as_ex

    def run_pipeline(
        self,
        pipeline: Pipeline,
        *,
        codes: list[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        context = PipelineContext(
            library=self.library,
            codes=codes,
            start=start,
            end=end,
            price_provider=self.price_provider,
            extra_lookback_days=self.extra_lookback_days,
            trading_calendar=self.trading_calendar,
            provider_overrides=self.provider_overrides,
            fundamental_use_point_in_time=self.fundamental_use_point_in_time,
            fundamental_availability_column=self.fundamental_availability_column,
            fundamental_lag_sessions=self.fundamental_lag_sessions,
            as_ex=self.as_ex,
        )
        index = context.output_dates
        columns = pd.Index(context.codes)

        screen = (
            pipeline.screen.evaluate(context).reindex(index=index, columns=columns).fillna(False)
            if pipeline.screen is not None
            else pd.DataFrame(True, index=index, columns=columns)
        )

        outputs: list[pd.DataFrame] = []
        for name, term in pipeline.columns.items():
            frame = term.evaluate(context).reindex(index=index, columns=columns)
            frame = frame.where(screen)
            long_df = (
                frame.rename_axis(index="date_")
                .reset_index()
                .melt(id_vars="date_", var_name="code", value_name=name)
            )
            outputs.append(long_df)

        if not outputs:
            return pd.DataFrame(columns=["date_", "code"])

        result = outputs[0]
        for frame in outputs[1:]:
            result = result.merge(frame, on=["date_", "code"], how="outer")
        result["date_"] = pd.to_datetime(result["date_"])
        result["code"] = result["code"].astype(str)
        return result.sort_values(["date_", "code"]).reset_index(drop=True)


__all__ = [
    "BoundColumn",
    "ColumnSpec",
    "ComputedTerm",
    "DataSet",
    "Filter",
    "Pipeline",
    "PipelineContext",
    "PipelineEngine",
    "Returns",
    "RollingStdDev",
    "SimpleMovingAverage",
    "SimFinBalanceSheet",
    "SimFinCashflowStatement",
    "SimFinIncomeStatement",
    "USEquityPricing",
]
