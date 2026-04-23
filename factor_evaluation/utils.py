from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.tseries.offsets import BDay, CustomBusinessDay, DateOffset

from tiger_factors.factor_evaluation.input import TableLike
from tiger_factors.factor_evaluation.input import load_factor_frame
from tiger_factors.factor_evaluation.input import load_group_labels
from tiger_factors.factor_evaluation.input import load_price_frame
from tiger_factors.factor_evaluation.input import to_panel

PeriodLike = int | str | pd.Timedelta


@dataclass(frozen=True)
class TigerFactorData:
    factor_data: pd.DataFrame
    factor_frame: pd.DataFrame
    price_frame: pd.DataFrame
    factor_series: pd.Series
    prices: pd.DataFrame
    factor_panel: pd.DataFrame
    forward_returns: pd.DataFrame
    factor_column: str
    date_column: str
    code_column: str
    price_column: str
    periods: tuple[PeriodLike, ...]
    quantiles: int | None
    group_column: str | None = None


class MaxLossExceededError(ValueError):
    pass


class NonMatchingTimezoneError(ValueError):
    pass


def rethrow(exception: Exception, additional_message: str):
    raise type(exception)(f"{exception}\n{additional_message}") from exception


def non_unique_bin_edges_error(func):
    def _wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as exc:
            message = str(exc).lower()
            if "bin edges must be unique" not in message and "bin labels must be one fewer" not in message:
                raise
            raise ValueError(
                "Unable to assign quantiles/bins because the factor has too many repeated values. "
                "Try fewer quantiles, explicit bins, or zero_aware/binning_by_group settings."
            ) from exc

    return _wrapped


def _date_level_name(index: pd.Index) -> str:
    if isinstance(index, pd.MultiIndex) and index.names and index.names[0]:
        return str(index.names[0])
    return "date_"


def _code_level_name(index: pd.Index) -> str:
    if isinstance(index, pd.MultiIndex) and len(index.names) > 1 and index.names[1]:
        return str(index.names[1])
    return "code"


def _factor_name(series: pd.Series, fallback: str = "factor") -> str:
    return str(series.name) if getattr(series, "name", None) else fallback


def _coerce_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if all(is_numeric_dtype(dtype) for dtype in frame.dtypes):
        return frame.astype(float, copy=False)
    return frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return series.copy()
    if is_numeric_dtype(series.dtype):
        return series.astype(float, copy=False)
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _normalize_datetime_like_index(values: pd.Index | pd.Series) -> pd.DatetimeIndex:
    if isinstance(values, pd.DatetimeIndex):
        return values
    if isinstance(values, pd.Series) and isinstance(values.dtype, pd.DatetimeTZDtype):
        return pd.DatetimeIndex(values)
    if isinstance(values, pd.Index) and values.dtype.kind == "M":
        return pd.DatetimeIndex(values)
    return pd.DatetimeIndex(pd.to_datetime(values, errors="coerce"))


def _require_matching_timezones(
    factor_idx: pd.Index | pd.DatetimeIndex,
    prices_idx: pd.Index | pd.DatetimeIndex,
) -> None:
    factor_dt = pd.DatetimeIndex(factor_idx)
    prices_dt = pd.DatetimeIndex(prices_idx)
    factor_tz = factor_dt.tz
    prices_tz = prices_dt.tz
    if factor_tz is None and prices_tz is None:
        return
    if factor_tz is None or prices_tz is None or str(factor_tz) != str(prices_tz):
        raise NonMatchingTimezoneError(
            f"factor timezone ({factor_tz}) and prices timezone ({prices_tz}) must match"
        )


def _rowwise_cross_sectional_corr(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    rank: bool = False,
) -> pd.Series:
    if left.empty or right.empty:
        return pd.Series(dtype=float)

    common_index = left.index[left.index.isin(right.index)]
    common_columns = left.columns[left.columns.isin(right.columns)]
    if len(common_index) == 0 or len(common_columns) == 0:
        return pd.Series(dtype=float, index=common_index)

    left = left.loc[common_index, common_columns]
    right = right.loc[common_index, common_columns]
    if rank:
        left = left.rank(axis=1, method="average")
        right = right.rank(axis=1, method="average")

    left_values = left.to_numpy(dtype=np.float64, copy=False)
    right_values = right.to_numpy(dtype=np.float64, copy=False)
    mask = np.isfinite(left_values) & np.isfinite(right_values)
    counts = mask.sum(axis=1).astype(float)
    result = np.full(len(common_index), np.nan, dtype=float)
    valid = counts >= 3
    if not valid.any():
        return pd.Series(result, index=common_index)

    left_abs = np.where(mask, np.abs(left_values), -np.inf)
    right_abs = np.where(mask, np.abs(right_values), -np.inf)
    left_scale = np.max(left_abs, axis=1)
    right_scale = np.max(right_abs, axis=1)
    scale = np.maximum(np.maximum(left_scale, right_scale), 1.0)
    scale[~np.isfinite(scale)] = 1.0

    scaled_left = np.zeros_like(left_values, dtype=np.float64)
    scaled_right = np.zeros_like(right_values, dtype=np.float64)
    np.divide(left_values, scale[:, None], out=scaled_left, where=mask)
    np.divide(right_values, scale[:, None], out=scaled_right, where=mask)
    with np.errstate(over="ignore", invalid="ignore"):
        sum_left = scaled_left.sum(axis=1)
        sum_right = scaled_right.sum(axis=1)
        sum_left_sq = np.square(scaled_left).sum(axis=1)
        sum_right_sq = np.square(scaled_right).sum(axis=1)
        sum_cross = (scaled_left * scaled_right).sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        numerator = sum_cross - (sum_left * sum_right) / counts
        denom_left = sum_left_sq - (sum_left * sum_left) / counts
        denom_right = sum_right_sq - (sum_right * sum_right) / counts
        denom = np.sqrt(np.maximum(denom_left, 0.0) * np.maximum(denom_right, 0.0))
        corr = numerator / denom

    corr[~valid | ~np.isfinite(corr) | (denom <= 1e-12)] = np.nan
    return pd.Series(corr, index=common_index)


def factor_frame_to_series(
    factor_frame: pd.DataFrame,
    *,
    factor_column: str,
    date_column: str = "date_",
    code_column: str = "code",
) -> pd.Series:
    series = factor_frame.set_index([date_column, code_column])[factor_column].sort_index()
    series.name = factor_column
    return series


def price_frame_to_wide(
    price_frame: pd.DataFrame,
    *,
    date_column: str = "date_",
    code_column: str = "code",
    price_column: str = "close",
) -> pd.DataFrame:
    return to_panel(
        price_frame,
        value_column=price_column,
        date_column=date_column,
        code_column=code_column,
    )


def _normalize_factor_series(
    factor: pd.Series,
    *,
    date_column: str = "date_",
    code_column: str = "code",
) -> pd.Series:
    if not isinstance(factor.index, pd.MultiIndex) or factor.index.nlevels < 2:
        raise ValueError("factor must be a MultiIndex Series indexed by date and asset")
    series = _coerce_numeric_series(factor)
    if isinstance(series.index.levels[0], pd.DatetimeIndex):
        dates = pd.DatetimeIndex(series.index.levels[0].take(series.index.codes[0]))
    else:
        dates = _normalize_datetime_like_index(series.index.get_level_values(0))
    codes = series.index.get_level_values(1).astype(str)
    valid = series.notna() & dates.notna()
    series = series.loc[valid].copy()
    series.index = pd.MultiIndex.from_arrays(
        [dates[valid], pd.Index(codes[valid], dtype=object)],
        names=[date_column, code_column],
    )
    series.name = _factor_name(series)
    return series.sort_index()


def _normalize_prices_wide(
    prices: pd.DataFrame,
    *,
    date_column: str = "date_",
    code_column: str = "code",
    price_column: str = "close",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if {date_column, code_column, price_column}.issubset(prices.columns):
        price_frame = load_price_frame(
            prices,
            date_column=date_column,
            code_column=code_column,
            price_column=price_column,
        )
        wide = price_frame_to_wide(
            price_frame,
            date_column=date_column,
            code_column=code_column,
            price_column=price_column,
        )
        return wide, price_frame

    wide = prices.copy()
    wide.index = pd.to_datetime(wide.index, errors="coerce")
    wide = wide[~wide.index.isna()].sort_index()
    wide.columns = wide.columns.astype(str)
    wide = _coerce_numeric_frame(wide)
    if wide.empty:
        return wide, pd.DataFrame(columns=[date_column, code_column, price_column])

    row_count, col_count = wide.shape
    stacked = pd.DataFrame(
        {
            date_column: np.repeat(wide.index.to_numpy(), col_count),
            code_column: np.tile(wide.columns.to_numpy(dtype=object), row_count),
            price_column: wide.to_numpy(dtype=float, copy=False).reshape(-1),
        }
    )
    stacked = stacked.dropna(subset=[price_column]).reset_index(drop=True)
    stacked[code_column] = stacked[code_column].astype(str)
    return wide, stacked[[date_column, code_column, price_column]]


def _normalize_groupby_data(
    groupby: Any,
    factor_index: pd.MultiIndex,
    *,
    date_column: str = "date_",
    code_column: str = "code",
    groupby_labels: dict[Any, str] | None = None,
) -> pd.Series | None:
    if groupby is None:
        return None

    if isinstance(groupby, dict):
        group_series = pd.Series(groupby)
    elif isinstance(groupby, pd.DataFrame) and {date_column, code_column, "group"}.issubset(groupby.columns):
        frame = groupby[[date_column, code_column, "group"]]
        valid = frame[date_column].notna() & frame[code_column].notna() & frame["group"].notna()
        frame = frame.loc[valid]
        if frame.empty:
            empty_index = pd.MultiIndex.from_arrays(
                [pd.DatetimeIndex([], name=date_column), pd.Index([], dtype=object, name=code_column)],
                names=[date_column, code_column],
            )
            group_series = pd.Series(dtype=object, index=empty_index, name="group")
        else:
            if frame.duplicated(subset=[date_column, code_column]).any():
                raise ValueError("group labels must not contain duplicate date/code pairs")
            dates = _normalize_datetime_like_index(frame[date_column])
            codes = frame[code_column].astype(str).to_numpy(dtype=object, copy=False)
            factor_dates = factor_index.get_level_values(0)
            factor_codes = factor_index.get_level_values(1).astype(str).to_numpy(dtype=object, copy=False)
            values = frame["group"].astype(str).to_numpy(dtype=object, copy=False)
            if len(frame) == len(factor_index) and np.array_equal(dates.to_numpy(), factor_dates.to_numpy()) and np.array_equal(codes, factor_codes):
                group_series = pd.Series(values, index=factor_index, name="group")
                if groupby_labels:
                    group_series = group_series.map(lambda value: groupby_labels.get(value, value) if pd.notna(value) else value)
                return group_series.rename("group")
            group_series = pd.Series(
                values,
                index=pd.MultiIndex.from_arrays(
                    [dates, pd.Index(codes, dtype=object)],
                    names=[date_column, code_column],
                ),
                name="group",
            )
    elif isinstance(groupby, (str, Path, pd.DataFrame, pd.Series)):
        loaded = load_group_labels(
            groupby,
            date_column=date_column,
            code_column=code_column,
            value_column="group",
        )
        if isinstance(loaded, pd.DataFrame):
            row_indexer = loaded.index.get_indexer(factor_index.get_level_values(0))
            col_indexer = loaded.columns.get_indexer(factor_index.get_level_values(1).astype(str))
            aligned_values = np.empty(len(factor_index), dtype=object)
            aligned_values[:] = np.nan
            valid = (row_indexer >= 0) & (col_indexer >= 0)
            if valid.any():
                loaded_values = loaded.to_numpy(dtype=object, copy=False)
                aligned_values[valid] = loaded_values[row_indexer[valid], col_indexer[valid]]
            group_series = pd.Series(aligned_values, index=factor_index, name="group")
        else:
            group_series = loaded
    else:
        raise ValueError("Unsupported groupby type")

    if isinstance(group_series.index, pd.MultiIndex):
        if group_series.index.equals(factor_index):
            aligned = group_series.copy()
            aligned.index = factor_index
            aligned = aligned.astype("object")
            if groupby_labels:
                aligned = aligned.map(lambda value: groupby_labels.get(value, value) if pd.notna(value) else value)
            return aligned.rename("group")
        normalized = group_series.copy()
        if isinstance(normalized.index.levels[0], pd.DatetimeIndex):
            dates = pd.DatetimeIndex(normalized.index.levels[0].take(normalized.index.codes[0]))
        else:
            dates = _normalize_datetime_like_index(normalized.index.get_level_values(0))
        codes = normalized.index.get_level_values(1).astype(str)
        valid = dates.notna()
        normalized = normalized.loc[valid].copy()
        normalized.index = pd.MultiIndex.from_arrays(
            [dates[valid], pd.Index(codes[valid], dtype=object)],
            names=[date_column, code_column],
        )
        aligned = normalized.reindex(factor_index)
    else:
        code_map = group_series.astype(str)
        aligned = pd.Series(
            factor_index.get_level_values(1).map(code_map),
            index=factor_index,
            name="group",
        )

    aligned = aligned.astype("object")
    if groupby_labels:
        aligned = aligned.map(lambda value: groupby_labels.get(value, value) if pd.notna(value) else value)
    return aligned.rename("group")


@non_unique_bin_edges_error
def _cut_values(values: pd.Series, bins: int | list[float]) -> pd.Series:
    result = pd.cut(values, bins=bins, labels=False, duplicates="drop")
    return result.astype(float) + 1.0


@non_unique_bin_edges_error
def _qcut_values(values: pd.Series, quantiles: int | list[float]) -> pd.Series:
    result = pd.qcut(values, q=quantiles, labels=False, duplicates="drop")
    return result.astype(float) + 1.0


def _zero_aware_quantize(
    values: pd.Series,
    *,
    quantiles: int | list[float] | None,
    bins: int | list[float] | None,
) -> pd.Series:
    output = pd.Series(index=values.index, dtype=float)
    negatives = values[values < 0]
    positives = values[values > 0]
    zeros = values[values == 0]

    if bins is not None:
        if isinstance(bins, int):
            neg_bins = max(int(bins) // 2, 1)
            pos_bins = max(int(bins) - neg_bins, 1)
            center = neg_bins + 1
        else:
            neg_bins = bins
            pos_bins = bins
            center = 1
        if not negatives.empty:
            neg = _cut_values(negatives, neg_bins)
            output.loc[negatives.index] = neg
            center = int(np.nanmax(neg.to_numpy())) + 1
        if not zeros.empty:
            output.loc[zeros.index] = center
        if not positives.empty:
            pos = _cut_values(positives, pos_bins) + center
            output.loc[positives.index] = pos
        return output

    if quantiles is None:
        raise ValueError("quantiles or bins must be provided")

    if isinstance(quantiles, int):
        negative_buckets = max(int(quantiles) // 2, 1)
        positive_buckets = max(int(quantiles) // 2, 1)
        center_bucket = negative_buckets + 1 if int(quantiles) % 2 else None
        positive_offset = negative_buckets if center_bucket is None else center_bucket
    else:
        negative_buckets = quantiles
        positive_buckets = quantiles
        center_bucket = None
        positive_offset = 0

    if not negatives.empty:
        output.loc[negatives.index] = _qcut_values(negatives, negative_buckets)
    if center_bucket is not None and not zeros.empty:
        output.loc[zeros.index] = center_bucket
    if not positives.empty:
        output.loc[positives.index] = _qcut_values(positives, positive_buckets) + positive_offset
    return output


def quantize_factor(
    factor_data: pd.Series | pd.DataFrame,
    *,
    quantiles: int | list[float] | None = 5,
    bins: int | list[float] | None = None,
    by_group: bool = False,
    no_raise: bool = False,
    zero_aware: bool = False,
    groupby: pd.Series | None = None,
) -> pd.Series:
    if quantiles is not None and bins is not None:
        raise ValueError("quantiles and bins are mutually exclusive")
    if quantiles is None and bins is None:
        raise ValueError("either quantiles or bins must be provided")

    if isinstance(factor_data, pd.DataFrame):
        if "factor" not in factor_data.columns:
            raise ValueError("factor_data DataFrame must contain a 'factor' column")
        factor_series = pd.to_numeric(factor_data["factor"], errors="coerce")
        if groupby is None and "group" in factor_data.columns:
            groupby = factor_data["group"]
    else:
        factor_series = pd.to_numeric(factor_data, errors="coerce")

    frame = pd.DataFrame({"factor": factor_series})
    if groupby is not None:
        frame["group"] = groupby.reindex(frame.index)
        if by_group:
            frame["group"] = frame["group"].astype("category")

    grouper: list[Any] = [frame.index.get_level_values(0)]
    if by_group and "group" in frame.columns:
        grouper.append(frame["group"])

    if isinstance(quantiles, int) and bins is None and not zero_aware:
        grouped = frame.groupby(grouper, observed=True, sort=False)["factor"]
        ranks = grouped.rank(method="first", pct=True)
        quantized = np.ceil(ranks * int(quantiles)).clip(1, int(quantiles))
        return quantized.rename("factor_quantile").sort_index()

    def _quantize(group: pd.DataFrame) -> pd.Series:
        values = pd.to_numeric(group["factor"], errors="coerce").dropna()
        if values.empty:
            return pd.Series(index=group.index, dtype=float)
        try:
            if zero_aware:
                result = _zero_aware_quantize(values, quantiles=quantiles, bins=bins)
            elif bins is not None:
                result = _cut_values(values, bins)
            else:
                result = _qcut_values(values, quantiles if quantiles is not None else 5)
        except ValueError:
            if not no_raise:
                raise
            return pd.Series(index=group.index, dtype=float)
        output = pd.Series(index=group.index, dtype=float)
        output.loc[result.index] = result.astype(float)
        return output

    grouped = frame.groupby(grouper, group_keys=False, observed=True, sort=False)
    quantized = grouped.apply(_quantize, include_groups=False)
    return quantized.rename("factor_quantile").sort_index()


def get_forward_returns_columns(
    columns: pd.Index | list[str],
    *,
    require_exact_day_multiple: bool = False,
) -> list[str]:
    results: list[str] = []
    for column in list(columns):
        text = str(column)
        if require_exact_day_multiple:
            if text.endswith("D") and text[:-1].isdigit():
                results.append(text)
            continue
        try:
            pd.Timedelta(text)
        except (TypeError, ValueError):
            continue
        results.append(text)
    return results


def _coerce_period_timedelta(period: PeriodLike) -> pd.Timedelta:
    if isinstance(period, int):
        return pd.Timedelta(days=max(int(period), 1))
    if isinstance(period, pd.Timedelta):
        return period
    text = str(period).strip()
    if text.endswith("D") and text[:-1].isdigit():
        return pd.Timedelta(days=max(int(text[:-1]), 1))
    if text.isdigit():
        return pd.Timedelta(days=max(int(text), 1))
    return pd.Timedelta(text)


def period_to_label(period: PeriodLike) -> str:
    if isinstance(period, int):
        return f"{max(int(period), 1)}D"
    if isinstance(period, str):
        text = period.strip()
        if text.endswith("D") and text[:-1].isdigit():
            return f"{max(int(text[:-1]), 1)}D"
    return timedelta_to_string(_coerce_period_timedelta(period))


def infer_trading_calendar(
    factor_idx: pd.Index | pd.DatetimeIndex,
    prices_idx: pd.Index | pd.DatetimeIndex,
):
    factor_dt = pd.DatetimeIndex(pd.to_datetime(factor_idx, errors="coerce")).dropna().unique().sort_values()
    prices_dt = pd.DatetimeIndex(pd.to_datetime(prices_idx, errors="coerce")).dropna().unique().sort_values()
    if factor_dt.empty and prices_dt.empty:
        return BDay()
    _require_matching_timezones(factor_dt, prices_dt)
    full_idx = factor_dt.union(prices_dt).sort_values()
    if full_idx.empty:
        return BDay()
    traded_weekdays: list[str] = []
    holidays: list[object] = []
    days_of_week = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for weekday, weekday_name in enumerate(days_of_week):
        weekday_mask = full_idx.dayofweek == weekday
        if not weekday_mask.any():
            continue
        traded_weekdays.append(weekday_name)
        used_weekdays = full_idx[weekday_mask].normalize()
        all_weekdays = pd.date_range(
            full_idx.min(),
            full_idx.max(),
            freq=CustomBusinessDay(weekmask=weekday_name),
        ).normalize()
        weekday_holidays = all_weekdays.difference(used_weekdays)
        holidays.extend([timestamp.date() for timestamp in weekday_holidays])
    weekmask = " ".join(traded_weekdays) or "Mon Tue Wed Thu Fri"
    try:
        return CustomBusinessDay(weekmask=weekmask, holidays=holidays)
    except ValueError:
        return BDay()


def _period_days(period: str | pd.Timedelta | int) -> int:
    try:
        delta = _coerce_period_timedelta(period)
    except ValueError:
        return 1
    return max(int(np.ceil(delta / pd.Timedelta(days=1))), 1)


def rate_of_return(period_ret: pd.Series, *, base_period: str) -> pd.Series:
    period_days = _period_days(getattr(period_ret, "name", None) or base_period)
    base_days = _period_days(base_period)
    scale = base_days / max(period_days, 1)
    return (1.0 + period_ret).pow(scale) - 1.0


def std_conversion(period_std: pd.Series, *, base_period: str) -> pd.Series:
    period_days = _period_days(getattr(period_std, "name", None) or base_period)
    base_days = _period_days(base_period)
    scale = np.sqrt(base_days / max(period_days, 1))
    return period_std * scale


def timedelta_to_string(value: pd.Timedelta) -> str:
    delta = pd.Timedelta(value)
    components: list[str] = []
    days = int(delta / pd.Timedelta(days=1))
    if days:
        components.append(f"{days}D")
        delta -= pd.Timedelta(days=days)
    hours = int(delta / pd.Timedelta(hours=1))
    if hours:
        components.append(f"{hours}h")
        delta -= pd.Timedelta(hours=hours)
    minutes = int(delta / pd.Timedelta(minutes=1))
    if minutes:
        components.append(f"{minutes}m")
    return "".join(components) or "0D"


def timedelta_strings_to_integers(sequence: list[str] | tuple[str, ...]) -> list[int]:
    values: list[int] = []
    for item in sequence:
        try:
            values.append(_period_days(item))
        except ValueError:
            continue
    return values


def make_naive_ts(value: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        return ts.tz_convert(None)
    return ts


def add_custom_calendar_timedelta(
    value: pd.Timestamp,
    timedelta: pd.Timedelta,
    freq=None,
) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    delta = pd.Timedelta(timedelta)
    if freq is None or delta == pd.Timedelta(0):
        return ts + delta
    days = int(delta / pd.Timedelta(days=1))
    remainder = delta - pd.Timedelta(days=days)
    step = freq if isinstance(freq, DateOffset) else BDay()
    return ts + (step * days) + remainder


def diff_custom_calendar_timedeltas(
    start: pd.Timestamp,
    end: pd.Timestamp,
    freq=None,
) -> pd.Timedelta:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if freq is None:
        return end_ts - start_ts
    if start_ts == end_ts:
        return pd.Timedelta(0)
    step = freq if isinstance(freq, DateOffset) else BDay()
    if end_ts < start_ts:
        return -diff_custom_calendar_timedeltas(end_ts, start_ts, step)
    start_day = start_ts.normalize()
    end_day = end_ts.normalize()
    if start_day == end_day:
        return end_ts - start_ts
    business_days = len(pd.date_range(start_day, end_day, freq=step)) - 1
    intraday = (end_ts - end_day) - (start_ts - start_day)
    return pd.Timedelta(days=business_days) + intraday


def _rolling_forward_non_cumulative_returns(prices: pd.DataFrame, period: int) -> pd.DataFrame:
    daily = prices.pct_change(fill_method=None)
    horizon_returns = []
    for offset in range(period):
        shifted = daily.shift(-offset)
        horizon_returns.append(shifted)
    if not horizon_returns:
        return daily * np.nan
    combined = sum(frame for frame in horizon_returns) / float(len(horizon_returns))
    return combined


def _forward_returns_for_timedelta(
    prices: pd.DataFrame,
    *,
    period: PeriodLike,
    freq: DateOffset | None,
    cumulative_returns: bool,
) -> pd.DataFrame:
    delta = _coerce_period_timedelta(period)
    if delta <= pd.Timedelta(0):
        raise ValueError("period must be positive")
    index = pd.DatetimeIndex(prices.index)
    desired_end = pd.DatetimeIndex([add_custom_calendar_timedelta(ts, delta, freq) for ts in index])
    end_positions = index.searchsorted(desired_end, side="left")
    start_prices = prices.to_numpy(dtype=float)
    forward = np.full(start_prices.shape, np.nan, dtype=float)

    if cumulative_returns:
        for row_idx, end_idx in enumerate(end_positions):
            if end_idx >= len(index):
                continue
            base = start_prices[row_idx]
            future = start_prices[end_idx]
            with np.errstate(divide="ignore", invalid="ignore"):
                forward[row_idx] = future / base - 1.0
        return pd.DataFrame(forward, index=index, columns=prices.columns)

    daily_returns = prices.pct_change(fill_method=None).to_numpy(dtype=float)
    for row_idx, end_idx in enumerate(end_positions):
        if end_idx >= len(index) or end_idx <= row_idx:
            continue
        window = daily_returns[row_idx + 1 : end_idx + 1]
        if len(window) == 0:
            continue
        forward[row_idx] = np.nanmean(window, axis=0)
    return pd.DataFrame(forward, index=index, columns=prices.columns)


def compute_forward_returns(
    factor: pd.Series | pd.DataFrame,
    prices: pd.DataFrame | None = None,
    *,
    periods: tuple[PeriodLike, ...] = (1, 5, 10),
    filter_zscore: float | None = None,
    cumulative_returns: bool = True,
    date_column: str = "date_",
    code_column: str = "code",
) -> pd.DataFrame:
    if prices is None:
        if not isinstance(factor, pd.DataFrame):
            raise ValueError("prices must be provided when factor is not a prices DataFrame")
        factor_idx = factor.index
        wide_prices = factor.copy()
    else:
        if not isinstance(factor, pd.Series):
            raise ValueError("factor must be a MultiIndex Series when prices are provided")
        factor_idx = factor.index.get_level_values(0)
        wide_prices = prices.copy()

    wide_prices.index = pd.to_datetime(wide_prices.index, errors="coerce")
    wide_prices = wide_prices[~wide_prices.index.isna()].sort_index()
    wide_prices.columns = wide_prices.columns.astype(str)
    wide_prices = _coerce_numeric_frame(wide_prices)
    if isinstance(factor_idx, pd.DatetimeIndex):
        factor_dt = factor_idx
    else:
        factor_dt = pd.DatetimeIndex(pd.to_datetime(factor_idx, errors="coerce"))
    _require_matching_timezones(factor_dt, wide_prices.index)
    freq = wide_prices.index.freq
    if freq is None:
        inferred_freq = pd.infer_freq(wide_prices.index)
        if inferred_freq is not None:
            freq = pd.tseries.frequencies.to_offset(inferred_freq)
    if freq is None:
        freq = infer_trading_calendar(factor_dt, wide_prices.index)

    product_index = pd.MultiIndex.from_product(
        [wide_prices.index, wide_prices.columns],
        names=[date_column, code_column],
    )
    frames: list[pd.Series] = []
    for period in periods:
        label = period_to_label(period)
        if isinstance(period, int):
            current_period = max(int(period), 1)
            if cumulative_returns:
                prices_values = wide_prices.to_numpy(dtype=float, copy=False)
                forward_values = np.full(prices_values.shape, np.nan, dtype=float)
                if current_period < len(wide_prices.index):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        forward_values[:-current_period] = (
                            prices_values[current_period:] / prices_values[:-current_period] - 1.0
                        )
                forward = pd.DataFrame(forward_values, index=wide_prices.index, columns=wide_prices.columns)
            else:
                forward = _rolling_forward_non_cumulative_returns(wide_prices, current_period)
        else:
            forward = _forward_returns_for_timedelta(
                wide_prices,
                period=period,
                freq=freq,
                cumulative_returns=cumulative_returns,
            )
        if filter_zscore is not None:
            mean = forward.mean(axis=1)
            std = forward.std(axis=1, ddof=0).replace(0.0, np.nan)
            zscore = forward.sub(mean, axis=0).div(std, axis=0)
            forward = forward.mask(zscore.abs() > float(filter_zscore))
        stacked = pd.Series(
            forward.to_numpy(dtype=float, copy=False).reshape(-1),
            index=product_index,
            name=label,
        )
        frames.append(stacked)

    result = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    if prices is not None and isinstance(factor, pd.Series):
        aligned_index = factor.index.intersection(result.index)
        result = result.loc[aligned_index]
    return result.sort_index()


def demean_forward_returns(
    factor_data: pd.DataFrame,
    grouper: list[Any] | None = None,
) -> pd.DataFrame:
    results = factor_data.copy()
    return_cols = get_forward_returns_columns(results.columns)
    if not return_cols:
        return results
    groupers = list(grouper or [results.index.get_level_values(0)])
    demeaned = results.groupby(groupers, observed=True)[return_cols].transform(lambda x: x - x.mean())
    results[return_cols] = demeaned
    return results


def backshift_returns_series(series: pd.Series, N: int) -> pd.Series:
    if isinstance(series.index, pd.MultiIndex) and series.index.nlevels >= 2:
        unstacked = series.unstack()
        shifted = unstacked.shift(-int(N))
        return shifted.stack(future_stack=True).rename(series.name)
    return series.shift(-int(N))


def print_table(table: pd.DataFrame | pd.Series, name: str | None = None, fmt: str | None = None) -> None:
    if name:
        print(name)
    if fmt is not None:
        with pd.option_context("display.float_format", lambda x: fmt.format(x)):
            print(table)
        return
    print(table)


def get_clean_factor(
    factor_series: pd.Series,
    forward_returns: pd.DataFrame,
    *,
    groupby: Any = None,
    binning_by_group: bool = False,
    quantiles: int | list[float] | None = 5,
    bins: int | list[float] | None = None,
    max_loss: float | None = 0.35,
    groupby_labels: dict[Any, str] | None = None,
    zero_aware: bool = False,
    date_column: str = "date_",
    code_column: str = "code",
) -> pd.DataFrame:
    factor = _normalize_factor_series(
        factor_series,
        date_column=date_column,
        code_column=code_column,
    ).rename("factor")
    original_count = len(factor)
    if factor.index.equals(forward_returns.index):
        clean = forward_returns.copy()
        clean["factor"] = factor.to_numpy(dtype=float, copy=False)
    else:
        common_index = factor.index.intersection(forward_returns.index)
        factor = factor.loc[common_index]
        clean = forward_returns.loc[common_index].copy()
        clean["factor"] = factor.to_numpy(dtype=float, copy=False)
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna(subset=["factor"])

    group_series = _normalize_groupby_data(
        groupby,
        clean.index,
        date_column=date_column,
        code_column=code_column,
        groupby_labels=groupby_labels,
    )
    if group_series is not None:
        clean["group"] = group_series

    effective_quantiles = None if bins is not None else quantiles
    quantized = quantize_factor(
        clean,
        quantiles=effective_quantiles,
        bins=bins,
        by_group=binning_by_group,
        zero_aware=zero_aware,
        no_raise=False,
    )
    clean["factor_quantile"] = quantized

    before = original_count
    after = int(clean["factor_quantile"].notna().sum())
    if max_loss is not None and before > 0:
        loss = 1.0 - (after / before)
        if loss > max_loss:
            raise MaxLossExceededError(f"max_loss ({max_loss:.1%}) exceeded {loss:.1%}")

    clean = clean.dropna(subset=["factor_quantile"]).copy()
    clean["factor_quantile"] = clean["factor_quantile"].astype(int)
    return clean.sort_index()


def _build_tiger_inputs_from_long_frames(
    *,
    factor_frame: TableLike,
    price_frame: TableLike,
    factor_column: str,
    date_column: str,
    code_column: str,
    price_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    factor_frame_df = load_factor_frame(
        factor_frame,
        factor_column=factor_column,
        date_column=date_column,
        code_column=code_column,
    )
    price_frame_df = load_price_frame(
        price_frame,
        date_column=date_column,
        code_column=code_column,
        price_column=price_column,
    )
    factor_series = factor_frame_to_series(
        factor_frame_df,
        factor_column=factor_column,
        date_column=date_column,
        code_column=code_column,
    )
    prices = price_frame_to_wide(
        price_frame_df,
        date_column=date_column,
        code_column=code_column,
        price_column=price_column,
    )
    return factor_frame_df, price_frame_df, factor_series, prices


def get_clean_factor_and_forward_returns(
    *,
    factor_frame: TableLike | None = None,
    price_frame: TableLike | None = None,
    factor_column: str | None = None,
    factor: pd.Series | None = None,
    prices: pd.DataFrame | None = None,
    groupby: Any = None,
    binning_by_group: bool = False,
    quantiles: int | list[float] | None = 5,
    bins: int | list[float] | None = None,
    periods: tuple[PeriodLike, ...] = (1, 5, 10),
    filter_zscore: float | None = 20,
    groupby_labels: dict[Any, str] | None = None,
    max_loss: float | None = 0.35,
    zero_aware: bool = False,
    cumulative_returns: bool = True,
    date_column: str = "date_",
    code_column: str = "code",
    price_column: str = "close",
) -> TigerFactorData:
    if factor is not None or prices is not None:
        if factor is None or prices is None:
            raise ValueError("factor and prices must be provided together")
        factor_series = _normalize_factor_series(factor, date_column=date_column, code_column=code_column)
        prices_wide, price_frame_df = _normalize_prices_wide(
            prices,
            date_column=date_column,
            code_column=code_column,
            price_column=price_column,
        )
        factor_name = _factor_name(factor_series)
        factor_frame_df = (
            factor_series.rename(factor_name)
            .reset_index()
            .rename(columns={0: factor_name})
        )
    else:
        if factor_frame is None or price_frame is None or factor_column is None:
            raise ValueError("factor_frame, price_frame, and factor_column are required")
        factor_frame_df, price_frame_df, factor_series, prices_wide = _build_tiger_inputs_from_long_frames(
            factor_frame=factor_frame,
            price_frame=price_frame,
            factor_column=factor_column,
            date_column=date_column,
            code_column=code_column,
            price_column=price_column,
        )
        factor_name = factor_column

    forward_returns = compute_forward_returns(
        factor_series,
        prices_wide,
        periods=periods,
        filter_zscore=filter_zscore,
        cumulative_returns=cumulative_returns,
        date_column=date_column,
        code_column=code_column,
    )
    factor_data = get_clean_factor(
        factor_series,
        forward_returns,
        groupby=groupby,
        binning_by_group=binning_by_group,
        quantiles=None if bins is not None else quantiles,
        bins=bins,
        max_loss=max_loss,
        groupby_labels=groupby_labels,
        zero_aware=zero_aware,
        date_column=date_column,
        code_column=code_column,
    )
    primary_period = period_to_label(periods[0])
    primary_forward_returns = (
        factor_data[primary_period]
        .unstack(code_column)
        .sort_index()
        .reindex(index=prices_wide.index, columns=prices_wide.columns)
    )
    return TigerFactorData(
        factor_data=factor_data,
        factor_frame=factor_frame_df,
        price_frame=price_frame_df,
        factor_series=factor_series,
        prices=prices_wide,
        factor_panel=factor_series.unstack(code_column).sort_index().reindex(index=prices_wide.index, columns=prices_wide.columns),
        forward_returns=primary_forward_returns,
        factor_column=factor_name,
        date_column=date_column,
        code_column=code_column,
        price_column=price_column,
        periods=periods,
        quantiles=int(quantiles) if isinstance(quantiles, int) else None,
        group_column="group" if "group" in factor_data.columns else None,
    )


__all__ = [
    "TigerFactorData",
    "MaxLossExceededError",
    "NonMatchingTimezoneError",
    "non_unique_bin_edges_error",
    "rethrow",
    "compute_forward_returns",
    "demean_forward_returns",
    "factor_frame_to_series",
    "get_clean_factor",
    "get_clean_factor_and_forward_returns",
    "get_forward_returns_columns",
    "infer_trading_calendar",
    "price_frame_to_wide",
    "period_to_label",
    "quantize_factor",
    "add_custom_calendar_timedelta",
    "backshift_returns_series",
    "diff_custom_calendar_timedeltas",
    "make_naive_ts",
    "print_table",
    "rate_of_return",
    "std_conversion",
    "timedelta_strings_to_integers",
    "timedelta_to_string",
]
