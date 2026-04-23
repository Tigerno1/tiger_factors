from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from tiger_factors.utils.cross_sectional import demean as _cross_sectional_demean
from tiger_factors.utils.cross_sectional import l1_normalize as _cross_sectional_l1_normalize
from tiger_factors.utils.cross_sectional import l2_normalize as _cross_sectional_l2_normalize
from tiger_factors.utils.cross_sectional import minmax_scale as _cross_sectional_minmax_scale
from tiger_factors.utils.cross_sectional import neutralize_by_group
from tiger_factors.utils.cross_sectional import winsorize_quantile as _winsorize_quantile
from tiger_factors.utils.cross_sectional import zscore as _cross_sectional_zscore
from tiger_factors.utils.group_operators import group_rank as _group_rank_by_group
from tiger_factors.utils.group_operators import group_scale as _group_scale_by_group
from tiger_factors.utils.group_operators import group_zscore as _group_zscore_by_group
from tiger_factors.utils.time_series import lag as _time_series_lag
from tiger_factors.utils.time_series import rolling_beta as _time_series_rolling_beta
from tiger_factors.utils.time_series import rolling_corr as _time_series_rolling_corr
from tiger_factors.utils.time_series import rolling_max as _time_series_rolling_max
from tiger_factors.utils.time_series import rolling_min as _time_series_rolling_min
from tiger_factors.utils.time_series import rolling_median as _time_series_rolling_median
from tiger_factors.utils.time_series import rolling_rank_pct as _time_series_rolling_rank_pct
from tiger_factors.utils.time_series import rolling_kurt as _time_series_rolling_kurt
from tiger_factors.utils.time_series import rolling_skew as _time_series_rolling_skew
from tiger_factors.utils.time_series import rolling_zscore as _time_series_rolling_zscore
from tiger_factors.utils.time_series import ewm_mean as _time_series_ewm_mean
from tiger_factors.utils.time_series import ewm_std as _time_series_ewm_std
from tiger_factors.utils.time_series import rolling_sharpe as _time_series_rolling_sharpe
from tiger_factors.utils.time_series import rolling_information_ratio as _time_series_rolling_ir
from tiger_factors.utils.time_series import ts_ema as _time_series_ts_ema
from tiger_factors.utils.time_series import ts_wma as _time_series_ts_wma
from tiger_factors.utils.time_series import ts_product as _time_series_ts_product


def _resolve_operand(value: Any, ctx: Any) -> Any:
    if isinstance(value, FactorFrameExpr):
        return value.evaluate(ctx)
    if isinstance(value, FactorFrameFactor):
        return value(ctx)
    return value


def _to_numeric_like(values: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return values.apply(pd.to_numeric, errors="coerce")


def _apply_math(values: pd.Series | pd.DataFrame, func: Callable[[Any], Any]) -> pd.Series | pd.DataFrame:
    numeric = _to_numeric_like(values)
    if isinstance(numeric, pd.Series):
        return pd.Series(func(numeric), index=numeric.index, name=numeric.name)
    return numeric.apply(func)


def _cumulative_values(values: pd.Series | pd.DataFrame, *, kind: str) -> pd.Series | pd.DataFrame:
    numeric = _to_numeric_like(values)
    if kind == "cumsum":
        return numeric.cumsum()
    if kind == "cumprod":
        return numeric.cumprod()
    raise ValueError(f"Unsupported cumulative kind={kind!r}")


def _apply_rolling(
    values: pd.Series | pd.DataFrame,
    *,
    kind: str,
    window: int,
    min_periods: int | None = None,
    ddof: int = 0,
    quantile: float | None = None,
) -> pd.Series | pd.DataFrame:
    rolling = _to_numeric_like(values).rolling(window=window, min_periods=min_periods)
    if kind == "mean":
        return rolling.mean()
    if kind == "sum":
        return rolling.sum()
    if kind == "std":
        return rolling.std(ddof=ddof)
    if kind == "var":
        return rolling.var(ddof=ddof)
    if kind == "min":
        return rolling.min()
    if kind == "max":
        return rolling.max()
    if kind == "quantile":
        if quantile is None:
            raise ValueError("quantile is required for rolling quantile.")
        return rolling.quantile(quantile)
    raise ValueError(f"Unsupported rolling kind={kind!r}")


def _rolling_rank_values(
    values: pd.Series | pd.DataFrame,
    *,
    window: int,
    min_periods: int | None = None,
) -> pd.Series | pd.DataFrame:
    numeric = _to_numeric_like(values)
    if isinstance(numeric, pd.Series):
        return _time_series_rolling_rank_pct(numeric, window=window, min_periods=min_periods)
    return numeric.apply(lambda column: _time_series_rolling_rank_pct(column, window=window, min_periods=min_periods))


def _rolling_cov_values(
    x: pd.Series | pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    *,
    window: int,
    min_periods: int | None = None,
) -> pd.Series | pd.DataFrame:
    if isinstance(x, pd.Series) and isinstance(y, pd.Series):
        return _to_numeric_like(x).rolling(window, min_periods=min_periods).cov(_to_numeric_like(y))

    if isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame):
        left, right = x.align(y, join="inner", axis=0)
        left, right = left.align(right, join="inner", axis=1)
        result = {
            column: _to_numeric_like(left[column]).rolling(window, min_periods=min_periods).cov(_to_numeric_like(right[column]))
            for column in left.columns
        }
        return pd.DataFrame(result, index=left.index)

    if isinstance(x, pd.DataFrame) and isinstance(y, pd.Series):
        y_numeric = _to_numeric_like(y).reindex(x.index)
        return pd.DataFrame(
            {
                column: _to_numeric_like(x[column]).rolling(window, min_periods=min_periods).cov(y_numeric)
                for column in x.columns
            },
            index=x.index,
        )

    if isinstance(x, pd.Series) and isinstance(y, pd.DataFrame):
        x_numeric = _to_numeric_like(x).reindex(y.index)
        return pd.DataFrame(
            {
                column: x_numeric.rolling(window, min_periods=min_periods).cov(_to_numeric_like(y[column]))
                for column in y.columns
            },
            index=y.index,
        )

    raise TypeError("rolling_cov expects Series/Series, DataFrame/DataFrame, or Series/DataFrame inputs.")


def _rolling_beta_values(
    y: pd.Series | pd.DataFrame,
    x: pd.Series | pd.DataFrame,
    *,
    window: int,
    min_periods: int | None = None,
) -> pd.Series | pd.DataFrame:
    if isinstance(y, pd.Series) and isinstance(x, pd.Series):
        return _time_series_rolling_beta(y, x, window=window, min_periods=min_periods)

    if isinstance(y, pd.DataFrame) and isinstance(x, pd.DataFrame):
        left, right = y.align(x, join="inner", axis=0)
        left, right = left.align(right, join="inner", axis=1)
        result = {
            column: _time_series_rolling_beta(left[column], right[column], window=window, min_periods=min_periods)
            for column in left.columns
        }
        return pd.DataFrame(result, index=left.index)

    if isinstance(y, pd.DataFrame) and isinstance(x, pd.Series):
        x_numeric = pd.to_numeric(x, errors="coerce").reindex(y.index)
        return pd.DataFrame(
            {
                column: _time_series_rolling_beta(y[column], x_numeric, window=window, min_periods=min_periods)
                for column in y.columns
            },
            index=y.index,
        )

    if isinstance(y, pd.Series) and isinstance(x, pd.DataFrame):
        y_numeric = pd.to_numeric(y, errors="coerce").reindex(x.index)
        return pd.DataFrame(
            {
                column: _time_series_rolling_beta(y_numeric, x[column], window=window, min_periods=min_periods)
                for column in x.columns
            },
            index=x.index,
        )

    raise TypeError("ts_beta expects Series/Series, DataFrame/DataFrame, or Series/DataFrame inputs.")


def _ts_corr_values(
    x: pd.Series | pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    *,
    window: int,
    min_periods: int | None = None,
) -> pd.Series | pd.DataFrame:
    return _rolling_corr_values(x, y, window=window, min_periods=min_periods)


def _ts_var_values(
    values: pd.Series | pd.DataFrame,
    *,
    window: int,
    min_periods: int | None = None,
    ddof: int = 0,
) -> pd.Series | pd.DataFrame:
    return _apply_rolling(values, kind="var", window=window, min_periods=min_periods, ddof=ddof)


def _rolling_skew_values(
    values: pd.Series | pd.DataFrame,
    *,
    window: int,
    min_periods: int | None = None,
) -> pd.Series | pd.DataFrame:
    numeric = _to_numeric_like(values)
    if isinstance(numeric, pd.Series):
        return _time_series_rolling_skew(numeric, window=window, min_periods=min_periods)
    return numeric.apply(lambda column: _time_series_rolling_skew(column, window=window, min_periods=min_periods))


def _rolling_kurt_values(
    values: pd.Series | pd.DataFrame,
    *,
    window: int,
    min_periods: int | None = None,
) -> pd.Series | pd.DataFrame:
    numeric = _to_numeric_like(values)
    if isinstance(numeric, pd.Series):
        return _time_series_rolling_kurt(numeric, window=window, min_periods=min_periods)
    return numeric.apply(lambda column: _time_series_rolling_kurt(column, window=window, min_periods=min_periods))


def _rolling_median_values(
    values: pd.Series | pd.DataFrame,
    *,
    window: int,
    min_periods: int | None = None,
) -> pd.Series | pd.DataFrame:
    numeric = _to_numeric_like(values)
    if isinstance(numeric, pd.Series):
        return _time_series_rolling_median(numeric, window=window, min_periods=min_periods)
    return numeric.apply(lambda column: _time_series_rolling_median(column, window=window, min_periods=min_periods))


def _rolling_sharpe_values(
    values: pd.Series | pd.DataFrame,
    *,
    window: int,
    risk_free_rate: float = 0.0,
    annualization: int = 252,
    min_periods: int | None = None,
) -> pd.Series | pd.DataFrame:
    if isinstance(values, pd.Series):
        return _time_series_rolling_sharpe(
            values,
            window=window,
            risk_free_rate=risk_free_rate,
            annualization=annualization,
            min_periods=min_periods,
        )
    return values.apply(
        lambda column: _time_series_rolling_sharpe(
            column,
            window=window,
            risk_free_rate=risk_free_rate,
            annualization=annualization,
            min_periods=min_periods,
        )
    )


def _rolling_ir_values(
    values: pd.Series | pd.DataFrame,
    *,
    window: int,
    annualization: int = 252,
    min_periods: int | None = None,
) -> pd.Series | pd.DataFrame:
    if isinstance(values, pd.Series):
        return _time_series_rolling_ir(values, window=window, annualization=annualization, min_periods=min_periods)
    return values.apply(
        lambda column: _time_series_rolling_ir(
            column,
            window=window,
            annualization=annualization,
            min_periods=min_periods,
        )
    )


def _rolling_median_series_values(
    values: pd.Series | pd.DataFrame,
    *,
    window: int,
    min_periods: int | None = None,
) -> pd.Series | pd.DataFrame:
    numeric = _to_numeric_like(values)
    if isinstance(numeric, pd.Series):
        return _time_series_rolling_median(numeric, window=window, min_periods=min_periods)
    return numeric.apply(lambda column: _time_series_rolling_median(column, window=window, min_periods=min_periods))


def _ts_wma_values(values: pd.Series | pd.DataFrame, window: int) -> pd.Series | pd.DataFrame:
    numeric = _to_numeric_like(values)
    if isinstance(numeric, pd.Series):
        return _time_series_ts_wma(numeric, d=window)
    return numeric.apply(lambda column: _time_series_ts_wma(column, d=window))


def _ts_ema_values(values: pd.Series | pd.DataFrame, window: int) -> pd.Series | pd.DataFrame:
    numeric = _to_numeric_like(values)
    if isinstance(numeric, pd.Series):
        return _time_series_ts_ema(numeric, d=window)
    return numeric.apply(lambda column: _time_series_ts_ema(column, d=window))


def _ts_prod_values(values: pd.Series | pd.DataFrame, window: int) -> pd.Series | pd.DataFrame:
    numeric = _to_numeric_like(values)
    if isinstance(numeric, pd.Series):
        return _time_series_ts_product(numeric, d=window)
    return numeric.apply(lambda column: _time_series_ts_product(column, d=window))


def _ewm_mean_values(
    values: pd.Series | pd.DataFrame,
    *,
    span: int | None = None,
    halflife: float | None = None,
    alpha: float | None = None,
) -> pd.Series | pd.DataFrame:
    return _time_series_ewm_mean(values, span=span, halflife=halflife, alpha=alpha)


def _ewm_std_values(
    values: pd.Series | pd.DataFrame,
    *,
    span: int | None = None,
    halflife: float | None = None,
    alpha: float | None = None,
) -> pd.Series | pd.DataFrame:
    return _time_series_ewm_std(values, span=span, halflife=halflife, alpha=alpha)


def _ts_zscore_values(
    values: pd.Series | pd.DataFrame,
    *,
    window: int,
    min_periods: int | None = None,
    ddof: int = 1,
) -> pd.Series | pd.DataFrame:
    numeric = _to_numeric_like(values)
    if isinstance(numeric, pd.Series):
        return _time_series_rolling_zscore(numeric, window=window, min_periods=min_periods, ddof=ddof)
    return numeric.apply(lambda column: _time_series_rolling_zscore(column, window=window, min_periods=min_periods, ddof=ddof))


def _top_bottom_n(
    values: pd.Series | pd.DataFrame,
    n: int,
    *,
    axis: int,
    largest: bool,
) -> pd.Series | pd.DataFrame:
    if n <= 0:
        raise ValueError("n must be positive.")
    numeric = _to_numeric_like(values)

    if isinstance(numeric, pd.Series):
        ranks = numeric.rank(ascending=not largest, method="first")
        return numeric.where(ranks <= n)

    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1.")
    ranks = numeric.rank(axis=axis, ascending=not largest, method="first")
    return numeric.where(ranks <= n)


def _where_values(
    values: pd.Series | pd.DataFrame,
    condition: Any,
    *,
    other: Any = None,
    ctx: Any = None,
) -> pd.Series | pd.DataFrame:
    cond = _resolve_operand(condition, ctx) if ctx is not None else condition
    replacement = _resolve_operand(other, ctx) if ctx is not None else other
    if replacement is None:
        return values.where(cond)
    return values.where(cond, replacement)


def _ifelse_values(
    condition: Any,
    *,
    true_value: Any = None,
    false_value: Any = None,
    ctx: Any = None,
) -> pd.Series | pd.DataFrame:
    cond = _resolve_operand(condition, ctx) if ctx is not None else condition
    true_resolved = _resolve_operand(true_value, ctx) if ctx is not None else true_value
    false_resolved = _resolve_operand(false_value, ctx) if ctx is not None else false_value
    result = np.where(cond, true_resolved, false_resolved)
    if isinstance(cond, pd.Series):
        return pd.Series(result, index=cond.index, name=getattr(true_resolved, "name", None))
    if isinstance(cond, pd.DataFrame):
        return pd.DataFrame(result, index=cond.index, columns=cond.columns)
    return result


def _fillna_values(
    values: pd.Series | pd.DataFrame,
    value: Any = None,
    *,
    method: str | None = None,
    limit: int | None = None,
    ctx: Any = None,
) -> pd.Series | pd.DataFrame:
    replacement = _resolve_operand(value, ctx) if ctx is not None else value
    if method in {"ffill", "pad"}:
        return values.ffill(limit=limit)
    if method in {"bfill", "backfill"}:
        return values.bfill(limit=limit)
    return values.fillna(value=replacement, method=method, limit=limit)


def _mask_values(
    values: pd.Series | pd.DataFrame,
    condition: Any,
    *,
    other: Any = None,
    ctx: Any = None,
) -> pd.Series | pd.DataFrame:
    cond = _resolve_operand(condition, ctx) if ctx is not None else condition
    replacement = _resolve_operand(other, ctx) if ctx is not None else other
    if replacement is None:
        return values.mask(cond)
    return values.mask(cond, replacement)


def _replace_values(
    values: pd.Series | pd.DataFrame,
    to_replace: Any,
    *,
    value: Any = None,
    ctx: Any = None,
    method: str | None = None,
    limit: int | None = None,
    regex: bool = False,
) -> pd.Series | pd.DataFrame:
    resolved_to_replace = _resolve_operand(to_replace, ctx) if ctx is not None else to_replace
    resolved_value = _resolve_operand(value, ctx) if ctx is not None else value
    replace_kwargs: dict[str, Any] = {
        "to_replace": resolved_to_replace,
        "value": resolved_value,
        "regex": regex,
    }
    if method is not None:
        replace_kwargs["method"] = method
    if limit is not None:
        replace_kwargs["limit"] = limit
    return values.replace(**replace_kwargs)


def _neutralize_values(
    values: pd.Series | pd.DataFrame,
    groups: pd.Series | pd.DataFrame,
    *,
    method: str = "demean",
) -> pd.Series | pd.DataFrame:
    if isinstance(values, pd.Series):
        if not isinstance(groups, pd.Series):
            raise TypeError("Series neutralize expects a Series of groups.")
        return neutralize_by_group(values, groups, method=method)

    if isinstance(groups, pd.Series):
        if groups.index.equals(values.columns):
            return values.apply(lambda row: neutralize_by_group(row, groups, method=method), axis=1)
        raise ValueError(
            "DataFrame neutralize with a Series of groups expects groups indexed by the frame columns."
        )

    if isinstance(groups, pd.DataFrame):
        if not values.index.equals(groups.index) or not values.columns.equals(groups.columns):
            raise ValueError("DataFrame neutralize expects groups to have the same index and columns.")
        rows = []
        for idx in values.index:
            rows.append(neutralize_by_group(values.loc[idx], groups.loc[idx], method=method))
        return pd.DataFrame(rows, index=values.index, columns=values.columns)

    raise TypeError("Neutralize groups must be a Series or DataFrame.")


def _groupwise_apply(
    values: pd.Series | pd.DataFrame,
    groups: pd.Series | pd.DataFrame,
    *,
    transform: Callable[[pd.Series, pd.Series], pd.Series],
    series_error: str,
    dataframe_series_error: str,
    dataframe_df_error: str,
) -> pd.Series | pd.DataFrame:
    if isinstance(values, pd.Series):
        if not isinstance(groups, pd.Series):
            raise TypeError(series_error)
        return transform(values, groups)

    if isinstance(groups, pd.Series):
        if groups.index.equals(values.columns):
            return values.apply(lambda row: transform(row, groups), axis=1)
        raise ValueError(dataframe_series_error)

    if isinstance(groups, pd.DataFrame):
        if not values.index.equals(groups.index) or not values.columns.equals(groups.columns):
            raise ValueError(dataframe_df_error)
        rows = [transform(values.loc[idx], groups.loc[idx]) for idx in values.index]
        return pd.DataFrame(rows, index=values.index, columns=values.columns)

    raise TypeError(series_error)


@dataclass(frozen=True)
class FactorFrameExpr:
    fn: Callable[[Any], pd.DataFrame | pd.Series]
    name: str | None = None

    def evaluate(self, ctx: Any) -> pd.DataFrame | pd.Series:
        return self.fn(ctx)

    def __call__(self, ctx: Any) -> pd.DataFrame | pd.Series:
        return self.evaluate(ctx)

    def _wrap(self, fn: Callable[[pd.DataFrame | pd.Series], pd.DataFrame | pd.Series], *, name: str | None = None) -> "FactorFrameExpr":
        return FactorFrameExpr(lambda ctx: fn(self.evaluate(ctx)), name=name or self.name)

    def _binary_op(self, other: Any, op: Callable[[Any, Any], Any]) -> "FactorFrameExpr":
        if isinstance(other, FactorFrameExpr):
            return FactorFrameExpr(lambda ctx: op(self.evaluate(ctx), other.evaluate(ctx)), name=self.name)
        return FactorFrameExpr(lambda ctx: op(self.evaluate(ctx), other), name=self.name)

    def __add__(self, other: Any) -> "FactorFrameExpr":
        return self._binary_op(other, lambda left, right: left + right)

    def __sub__(self, other: Any) -> "FactorFrameExpr":
        return self._binary_op(other, lambda left, right: left - right)

    def __mul__(self, other: Any) -> "FactorFrameExpr":
        return self._binary_op(other, lambda left, right: left * right)

    def __truediv__(self, other: Any) -> "FactorFrameExpr":
        return self._binary_op(other, lambda left, right: left / right)

    def __pow__(self, other: Any) -> "FactorFrameExpr":
        return self._binary_op(other, lambda left, right: left**right)

    def __neg__(self) -> "FactorFrameExpr":
        return self._wrap(lambda frame: -frame)

    def abs(self) -> "FactorFrameExpr":
        return self._wrap(lambda frame: _apply_math(frame, np.abs))

    def rolling_abs(self) -> "FactorFrameExpr":
        return self.abs()

    def ts_abs(self) -> "FactorFrameExpr":
        return self.abs()

    def log(self) -> "FactorFrameExpr":
        return self._wrap(lambda frame: _apply_math(frame, np.log))

    def exp(self) -> "FactorFrameExpr":
        return self._wrap(lambda frame: _apply_math(frame, np.exp))

    def sqrt(self) -> "FactorFrameExpr":
        return self._wrap(lambda frame: _apply_math(frame, np.sqrt))

    def sign(self) -> "FactorFrameExpr":
        return self._wrap(lambda frame: _apply_math(frame, np.sign))

    def rolling_sign(self) -> "FactorFrameExpr":
        return self.sign()

    def ts_sign(self) -> "FactorFrameExpr":
        return self.sign()

    def pow(self, exponent: Any) -> "FactorFrameExpr":
        return self._binary_op(exponent, lambda left, right: left**right)

    def __lt__(self, other: Any) -> "FactorFrameExpr":
        return self._binary_op(other, lambda left, right: left < right)

    def __le__(self, other: Any) -> "FactorFrameExpr":
        return self._binary_op(other, lambda left, right: left <= right)

    def __gt__(self, other: Any) -> "FactorFrameExpr":
        return self._binary_op(other, lambda left, right: left > right)

    def __ge__(self, other: Any) -> "FactorFrameExpr":
        return self._binary_op(other, lambda left, right: left >= right)

    def __eq__(self, other: Any) -> "FactorFrameExpr":  # type: ignore[override]
        return self._binary_op(other, lambda left, right: left == right)

    def __ne__(self, other: Any) -> "FactorFrameExpr":  # type: ignore[override]
        return self._binary_op(other, lambda left, right: left != right)

    def __and__(self, other: Any) -> "FactorFrameExpr":
        return self._binary_op(other, lambda left, right: left & right)

    def __or__(self, other: Any) -> "FactorFrameExpr":
        return self._binary_op(other, lambda left, right: left | right)

    def __invert__(self) -> "FactorFrameExpr":
        return self._wrap(lambda frame: ~frame)

    def pct_change(self, periods: int = 1, *, fill_method: str | None = None) -> "FactorFrameExpr":
        return self._wrap(lambda frame: frame.pct_change(periods=periods, fill_method=fill_method))

    def rolling_pct_change(self, periods: int = 1, *, fill_method: str | None = None) -> "FactorFrameExpr":
        return self.pct_change(periods=periods, fill_method=fill_method)

    def ts_pct_change(self, periods: int = 1, *, fill_method: str | None = None) -> "FactorFrameExpr":
        return self.pct_change(periods=periods, fill_method=fill_method)

    def rank(
        self,
        *,
        axis: int = 0,
        ascending: bool = True,
        pct: bool = True,
        method: str = "average",
    ) -> "FactorFrameExpr":
        return self._wrap(lambda frame: frame.rank(axis=axis, ascending=ascending, pct=pct, method=method))

    def rank_desc(
        self,
        *,
        axis: int = 0,
        pct: bool = True,
        method: str = "average",
    ) -> "FactorFrameExpr":
        return self.rank(axis=axis, ascending=False, pct=pct, method=method)

    def rolling_mean(self, window: int, *, min_periods: int | None = None) -> "FactorFrameExpr":
        return self._wrap(lambda frame: _apply_rolling(frame, kind="mean", window=window, min_periods=min_periods))

    def mean(self, window: int, *, min_periods: int | None = None) -> "FactorFrameExpr":
        return self.rolling_mean(window, min_periods=min_periods)

    def rolling_std(
        self,
        window: int,
        *,
        min_periods: int | None = None,
        ddof: int = 0,
    ) -> "FactorFrameExpr":
        return self._wrap(lambda frame: _apply_rolling(frame, kind="std", window=window, min_periods=min_periods, ddof=ddof))

    def std(self, window: int, *, min_periods: int | None = None, ddof: int = 0) -> "FactorFrameExpr":
        return self.rolling_std(window, min_periods=min_periods, ddof=ddof)

    def rolling_var(
        self,
        window: int,
        *,
        min_periods: int | None = None,
        ddof: int = 0,
    ) -> "FactorFrameExpr":
        return self._wrap(lambda frame: _apply_rolling(frame, kind="var", window=window, min_periods=min_periods, ddof=ddof))

    def var(self, window: int, *, min_periods: int | None = None, ddof: int = 0) -> "FactorFrameExpr":
        return self.rolling_var(window, min_periods=min_periods, ddof=ddof)

    def rolling_sum(self, window: int, *, min_periods: int | None = None) -> "FactorFrameExpr":
        return self._wrap(lambda frame: _apply_rolling(frame, kind="sum", window=window, min_periods=min_periods))

    def sum(self, window: int, *, min_periods: int | None = None) -> "FactorFrameExpr":
        return self.rolling_sum(window, min_periods=min_periods)

    def rolling_quantile(self, window: int, quantile: float) -> "FactorFrameExpr":
        return self._wrap(lambda frame: _apply_rolling(frame, kind="quantile", window=window, quantile=quantile))

    def rolling_min(self, window: int, *, min_periods: int | None = None) -> "FactorFrameExpr":
        return self._wrap(lambda frame: _time_series_rolling_min(frame, window=window, min_periods=min_periods))

    def min(self, window: int, *, min_periods: int | None = None) -> "FactorFrameExpr":
        return self.rolling_min(window, min_periods=min_periods)

    def rolling_max(self, window: int, *, min_periods: int | None = None) -> "FactorFrameExpr":
        return self._wrap(lambda frame: _time_series_rolling_max(frame, window=window, min_periods=min_periods))

    def max(self, window: int, *, min_periods: int | None = None) -> "FactorFrameExpr":
        return self.rolling_max(window, min_periods=min_periods)

    def rolling_corr(
        self,
        other: Any,
        window: int,
        *,
        min_periods: int | None = None,
    ) -> "FactorFrameExpr":
        return FactorFrameExpr(
            lambda ctx: _rolling_corr_values(
                self.evaluate(ctx),
                _resolve_operand(other, ctx),
                window=window,
                min_periods=min_periods,
            ),
            name=self.name,
        )

    def corr(self, other: Any, window: int, *, min_periods: int | None = None) -> "FactorFrameExpr":
        return self.rolling_corr(other, window, min_periods=min_periods)

    def rolling_cov(
        self,
        other: Any,
        window: int,
        *,
        min_periods: int | None = None,
    ) -> "FactorFrameExpr":
        return FactorFrameExpr(
            lambda ctx: _rolling_cov_values(
                self.evaluate(ctx),
                _resolve_operand(other, ctx),
                window=window,
                min_periods=min_periods,
            ),
            name=self.name,
        )

    def cov(self, other: Any, window: int, *, min_periods: int | None = None) -> "FactorFrameExpr":
        return self.rolling_cov(other, window, min_periods=min_periods)

    def rolling_rank(self, window: int, *, min_periods: int | None = None) -> "FactorFrameExpr":
        return self._wrap(lambda frame: _rolling_rank_values(frame, window=window, min_periods=min_periods))

    def ts_rank(self, window: int, *, min_periods: int | None = None) -> "FactorFrameExpr":
        return self.rolling_rank(window, min_periods=min_periods)

    def rolling_skew(self, window: int, *, min_periods: int | None = None) -> "FactorFrameExpr":
        return self._wrap(lambda frame: _rolling_skew_values(frame, window=window, min_periods=min_periods))

    def skew(self, window: int, *, min_periods: int | None = None) -> "FactorFrameExpr":
        return self.rolling_skew(window, min_periods=min_periods)

    def rolling_kurt(self, window: int, *, min_periods: int | None = None) -> "FactorFrameExpr":
        return self._wrap(lambda frame: _rolling_kurt_values(frame, window=window, min_periods=min_periods))

    def kurt(self, window: int, *, min_periods: int | None = None) -> "FactorFrameExpr":
        return self.rolling_kurt(window, min_periods=min_periods)

    def rolling_median(self, window: int, *, min_periods: int | None = None) -> "FactorFrameExpr":
        return self._wrap(lambda frame: _rolling_median_values(frame, window=window, min_periods=min_periods))

    def median(self, window: int, *, min_periods: int | None = None) -> "FactorFrameExpr":
        return self.rolling_median(window, min_periods=min_periods)

    def ts_median(self, window: int, *, min_periods: int | None = None) -> "FactorFrameExpr":
        return self.rolling_median(window, min_periods=min_periods)

    def rolling_wma(self, window: int) -> "FactorFrameExpr":
        return self._wrap(lambda frame: _ts_wma_values(frame, window))

    def ts_wma(self, window: int) -> "FactorFrameExpr":
        return self.rolling_wma(window)

    def rolling_ema(self, window: int) -> "FactorFrameExpr":
        return self._wrap(lambda frame: _ts_ema_values(frame, window))

    def ts_ema(self, window: int) -> "FactorFrameExpr":
        return self.rolling_ema(window)

    def ts_corr(
        self,
        other: Any,
        window: int,
        *,
        min_periods: int | None = None,
    ) -> "FactorFrameExpr":
        return self.rolling_corr(other, window, min_periods=min_periods)

    def ts_var(self, window: int, *, min_periods: int | None = None, ddof: int = 0) -> "FactorFrameExpr":
        return self.rolling_var(window, min_periods=min_periods, ddof=ddof)

    def ts_beta(
        self,
        other: Any,
        window: int,
        *,
        min_periods: int | None = None,
    ) -> "FactorFrameExpr":
        return FactorFrameExpr(
            lambda ctx: _rolling_beta_values(
                self.evaluate(ctx),
                _resolve_operand(other, ctx),
                window=window,
                min_periods=min_periods,
            ),
            name=self.name,
        )

    def ts_skew(self, window: int, *, min_periods: int | None = None) -> "FactorFrameExpr":
        return self.rolling_skew(window, min_periods=min_periods)

    def ts_kurt(self, window: int, *, min_periods: int | None = None) -> "FactorFrameExpr":
        return self.rolling_kurt(window, min_periods=min_periods)

    def ewm_mean(
        self,
        *,
        span: int | None = None,
        halflife: float | None = None,
        alpha: float | None = None,
    ) -> "FactorFrameExpr":
        return self._wrap(lambda frame: _ewm_mean_values(frame, span=span, halflife=halflife, alpha=alpha))

    def ewm_std(
        self,
        *,
        span: int | None = None,
        halflife: float | None = None,
        alpha: float | None = None,
    ) -> "FactorFrameExpr":
        return self._wrap(lambda frame: _ewm_std_values(frame, span=span, halflife=halflife, alpha=alpha))

    def rolling_sharpe(
        self,
        window: int,
        *,
        risk_free_rate: float = 0.0,
        annualization: int = 252,
        min_periods: int | None = None,
    ) -> "FactorFrameExpr":
        return self._wrap(
            lambda frame: _rolling_sharpe_values(
                frame,
                window=window,
                risk_free_rate=risk_free_rate,
                annualization=annualization,
                min_periods=min_periods,
            )
        )

    def rolling_ir(
        self,
        window: int,
        *,
        annualization: int = 252,
        min_periods: int | None = None,
    ) -> "FactorFrameExpr":
        return self._wrap(
            lambda frame: _rolling_ir_values(
                frame,
                window=window,
                annualization=annualization,
                min_periods=min_periods,
            )
        )

    def shift(self, periods: int = 1) -> "FactorFrameExpr":
        return self._wrap(lambda frame: frame.shift(periods))

    def diff(self, periods: int = 1) -> "FactorFrameExpr":
        return self._wrap(lambda frame: frame.diff(periods))

    def rolling_delta(self, periods: int = 1) -> "FactorFrameExpr":
        return self.diff(periods=periods)

    def ts_delta(self, periods: int = 1) -> "FactorFrameExpr":
        return self.diff(periods=periods)

    def cumsum(self) -> "FactorFrameExpr":
        return self._wrap(lambda frame: _cumulative_values(frame, kind="cumsum"))

    def cumprod(self) -> "FactorFrameExpr":
        return self._wrap(lambda frame: _cumulative_values(frame, kind="cumprod"))

    def rolling_prod(self, window: int) -> "FactorFrameExpr":
        return self._wrap(lambda frame: _ts_prod_values(frame, window))

    def ts_prod(self, window: int) -> "FactorFrameExpr":
        return self.rolling_prod(window)

    def lag(self, periods: int = 1) -> "FactorFrameExpr":
        return self._wrap(lambda frame: _time_series_lag(frame, periods=periods))

    def rolling_delay(self, periods: int = 1) -> "FactorFrameExpr":
        return self.lag(periods=periods)

    def ts_delay(self, periods: int = 1) -> "FactorFrameExpr":
        return self.lag(periods=periods)

    def clip(self, *, lower: float | None = None, upper: float | None = None) -> "FactorFrameExpr":
        return self._wrap(lambda frame: frame.clip(lower=lower, upper=upper))

    def clip_lower(self, lower: float) -> "FactorFrameExpr":
        return self._wrap(lambda frame: frame.clip(lower=lower))

    def clip_upper(self, upper: float) -> "FactorFrameExpr":
        return self._wrap(lambda frame: frame.clip(upper=upper))

    def fillna(
        self,
        value: Any = None,
        *,
        method: str | None = None,
        limit: int | None = None,
    ) -> "FactorFrameExpr":
        return FactorFrameExpr(
            lambda ctx: _fillna_values(self.evaluate(ctx), value, method=method, limit=limit, ctx=ctx),
            name=self.name,
        )

    def where(self, condition: Any, *, other: Any = None) -> "FactorFrameExpr":
        return FactorFrameExpr(
            lambda ctx: _where_values(
                self.evaluate(ctx),
                _resolve_operand(condition, ctx),
                other=_resolve_operand(other, ctx) if other is not None else None,
                ctx=None,
            ),
            name=self.name,
        )

    def mask(self, condition: Any, *, other: Any = None) -> "FactorFrameExpr":
        return FactorFrameExpr(
            lambda ctx: _mask_values(
                self.evaluate(ctx),
                _resolve_operand(condition, ctx),
                other=_resolve_operand(other, ctx) if other is not None else None,
                ctx=None,
            ),
            name=self.name,
        )

    def ifelse(self, *, true_value: Any = None, false_value: Any = None) -> "FactorFrameExpr":
        return FactorFrameExpr(
            lambda ctx: _ifelse_values(
                self.evaluate(ctx),
                true_value=true_value,
                false_value=false_value,
                ctx=ctx,
            ),
            name=self.name,
        )

    def replace(
        self,
        to_replace: Any,
        value: Any = None,
        *,
        method: str | None = None,
        limit: int | None = None,
        regex: bool = False,
    ) -> "FactorFrameExpr":
        return FactorFrameExpr(
            lambda ctx: _replace_values(
                self.evaluate(ctx),
                to_replace,
                value=value,
                ctx=ctx,
                method=method,
                limit=limit,
                regex=regex,
            ),
            name=self.name,
        )

    def isna(self) -> "FactorFrameExpr":
        return self._wrap(lambda frame: frame.isna())

    def notna(self) -> "FactorFrameExpr":
        return self._wrap(lambda frame: frame.notna())

    def top_n(self, n: int, *, axis: int = 1) -> "FactorFrameExpr":
        return self._wrap(lambda frame: _top_bottom_n(frame, n, axis=axis, largest=True))

    def bottom_n(self, n: int, *, axis: int = 1) -> "FactorFrameExpr":
        return self._wrap(lambda frame: _top_bottom_n(frame, n, axis=axis, largest=False))

    def winsorize(
        self,
        *,
        lower: float = 0.01,
        upper: float = 0.99,
        axis: int = 1,
    ) -> "FactorFrameExpr":
        return self._wrap(lambda frame: _winsorize_quantile(frame, lower=lower, upper=upper, axis=axis))

    def zscore(
        self,
        *,
        axis: int = 1,
        ddof: int = 0,
        clip: tuple[float, float] | None = None,
    ) -> "FactorFrameExpr":
        return self._wrap(lambda frame: _cross_sectional_zscore(frame, axis=axis, ddof=ddof, clip=clip))

    def demean(self, *, axis: int = 1) -> "FactorFrameExpr":
        return self._wrap(lambda frame: _cross_sectional_demean(frame, axis=axis))

    def minmax_scale(
        self,
        *,
        axis: int = 1,
        feature_range: tuple[float, float] = (0.0, 1.0),
    ) -> "FactorFrameExpr":
        return self._wrap(
            lambda frame: _cross_sectional_minmax_scale(frame, axis=axis, feature_range=feature_range)
        )

    def cs_scale(
        self,
        *,
        axis: int = 1,
        feature_range: tuple[float, float] = (0.0, 1.0),
    ) -> "FactorFrameExpr":
        return self.minmax_scale(axis=axis, feature_range=feature_range)

    def l1_normalize(self, *, axis: int = 1) -> "FactorFrameExpr":
        return self._wrap(lambda frame: _cross_sectional_l1_normalize(frame, axis=axis))

    def l2_normalize(self, *, axis: int = 1) -> "FactorFrameExpr":
        return self._wrap(lambda frame: _cross_sectional_l2_normalize(frame, axis=axis))

    def ffill(self, *, limit: int | None = None) -> "FactorFrameExpr":
        return self._wrap(lambda frame: frame.ffill(limit=limit))

    def bfill(self, *, limit: int | None = None) -> "FactorFrameExpr":
        return self._wrap(lambda frame: frame.bfill(limit=limit))

    def cs_rank(self, *, ascending: bool = True, pct: bool = True) -> "FactorFrameExpr":
        return self.rank(axis=1, ascending=ascending, pct=pct)

    def cs_zscore(self) -> "FactorFrameExpr":
        def _apply(frame: pd.DataFrame | pd.Series):
            if isinstance(frame, pd.Series):
                return (frame - frame.mean()) / frame.std(ddof=0)
            mean = frame.mean(axis=1)
            std = frame.std(axis=1, ddof=0).replace(0, pd.NA)
            return frame.sub(mean, axis=0).div(std, axis=0)

        return self._wrap(_apply)

    def cs_demean(self) -> "FactorFrameExpr":
        return self.demean(axis=1)

    def neutralize(
        self,
        groups: Any,
        *,
        method: str = "demean",
    ) -> "FactorFrameExpr":
        return FactorFrameExpr(
            lambda ctx: _neutralize_values(
                self.evaluate(ctx),
                (
                    groups.evaluate(ctx)
                    if isinstance(groups, FactorFrameExpr)
                    else groups(ctx)
                    if isinstance(groups, FactorFrameFactor)
                    else groups
                ),
                method=method,
            ),
            name=self.name,
        )

    def group_neutralize(
        self,
        groups: Any,
        *,
        method: str = "demean",
    ) -> "FactorFrameExpr":
        return self.neutralize(groups, method=method)

    def group_demean(self, groups: Any) -> "FactorFrameExpr":
        return self.neutralize(groups, method="demean")

    def group_rank(self, groups: Any) -> "FactorFrameExpr":
        return FactorFrameExpr(
            lambda ctx: _groupwise_apply(
                self.evaluate(ctx),
                (
                    groups.evaluate(ctx)
                    if isinstance(groups, FactorFrameExpr)
                    else groups(ctx)
                    if isinstance(groups, FactorFrameFactor)
                    else groups
                ),
                transform=_group_rank_by_group,
                series_error="Series group_rank expects a Series of groups.",
                dataframe_series_error="DataFrame group_rank with a Series of groups expects groups indexed by the frame columns.",
                dataframe_df_error="DataFrame group_rank expects groups to have the same index and columns.",
            ),
            name=self.name,
        )

    def group_zscore(self, groups: Any) -> "FactorFrameExpr":
        return FactorFrameExpr(
            lambda ctx: _groupwise_apply(
                self.evaluate(ctx),
                (
                    groups.evaluate(ctx)
                    if isinstance(groups, FactorFrameExpr)
                    else groups(ctx)
                    if isinstance(groups, FactorFrameFactor)
                    else groups
                ),
                transform=_group_zscore_by_group,
                series_error="Series group_zscore expects a Series of groups.",
                dataframe_series_error="DataFrame group_zscore with a Series of groups expects groups indexed by the frame columns.",
                dataframe_df_error="DataFrame group_zscore expects groups to have the same index and columns.",
            ),
            name=self.name,
        )

    def group_scale(self, groups: Any) -> "FactorFrameExpr":
        return FactorFrameExpr(
            lambda ctx: _groupwise_apply(
                self.evaluate(ctx),
                (
                    groups.evaluate(ctx)
                    if isinstance(groups, FactorFrameExpr)
                    else groups(ctx)
                    if isinstance(groups, FactorFrameFactor)
                    else groups
                ),
                transform=_group_scale_by_group,
                series_error="Series group_scale expects a Series of groups.",
                dataframe_series_error="DataFrame group_scale with a Series of groups expects groups indexed by the frame columns.",
                dataframe_df_error="DataFrame group_scale expects groups to have the same index and columns.",
            ),
            name=self.name,
        )

    def ts_zscore(self, window: int, *, min_periods: int | None = None, ddof: int = 1) -> "FactorFrameExpr":
        return self._wrap(
            lambda frame: _ts_zscore_values(frame, window=window, min_periods=min_periods, ddof=ddof)
        )


@dataclass(frozen=True)
class FactorFrameFactor:
    name: str
    fn: Callable[[Any], pd.DataFrame | pd.Series]
    save: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __call__(self, ctx: Any) -> pd.DataFrame | pd.Series:
        return self.fn(ctx)


@dataclass(frozen=True)
class FactorFrameTemplate:
    """Reusable factor constructor that can be instantiated with parameters."""

    name: str
    builder: Callable[..., FactorFrameFactor | FactorFrameExpr | Callable[[Any], pd.DataFrame | pd.Series]]
    defaults: dict[str, Any] = field(default_factory=dict)
    save: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def build(self, *, factor_name: str | None = None, **kwargs: Any) -> FactorFrameFactor:
        params = {**self.defaults, **kwargs}
        built = self.builder(**params)
        output_name = factor_name or self.name
        if isinstance(built, FactorFrameFactor):
            return FactorFrameFactor(
                name=built.name or output_name,
                fn=built.fn,
                save=self.save or built.save,
                metadata={**self.metadata, **built.metadata},
            )
        if isinstance(built, FactorFrameExpr):
            return factor(output_name, built, save=self.save, metadata=self.metadata)
        if callable(built):
            return FactorFrameFactor(
                name=output_name,
                fn=built,
                save=self.save,
                metadata=dict(self.metadata),
            )
        raise TypeError(
            f"Factor template {self.name!r} must build a FactorFrameFactor, "
            "FactorFrameExpr, or callable."
        )

    def __call__(self, **kwargs: Any) -> FactorFrameFactor:
        return self.build(**kwargs)


def factor_template(
    name: str,
    builder: Callable[..., FactorFrameFactor | FactorFrameExpr | Callable[[Any], pd.DataFrame | pd.Series]],
    *,
    defaults: dict[str, Any] | None = None,
    save: bool = False,
    metadata: dict[str, Any] | None = None,
) -> FactorFrameTemplate:
    return FactorFrameTemplate(
        name=str(name),
        builder=builder,
        defaults=dict(defaults or {}),
        save=save,
        metadata=dict(metadata or {}),
    )


def factor(
    name: str,
    fn: FactorFrameExpr | Callable[[Any], pd.DataFrame | pd.Series],
    *,
    save: bool = False,
    metadata: dict[str, Any] | None = None,
) -> FactorFrameFactor:
    if isinstance(fn, FactorFrameExpr):
        callable_fn = fn.evaluate
    else:
        callable_fn = fn
    return FactorFrameFactor(name=str(name), fn=callable_fn, save=save, metadata=dict(metadata or {}))


def feed_wide(ctx: Any, feed_name: str, value_column: str | None = None) -> pd.DataFrame:
    return ctx.feed_wide(feed_name, value_column)


def feed_series(ctx: Any, feed_name: str, value_column: str | None = None) -> pd.Series:
    return ctx.feed_series(feed_name, value_column)


def _feed_expr(feed_name: str, value_column: str | None, *, is_series: bool = False) -> FactorFrameExpr:
    if is_series:
        return FactorFrameExpr(lambda ctx: ctx.feed_series(feed_name, value_column), name=feed_name)
    return FactorFrameExpr(lambda ctx: ctx.feed_wide(feed_name, value_column), name=feed_name)


def price(
    ctx: Any | None = None,
    *,
    value_column: str = "close",
    feed_name: str = "price",
) -> pd.DataFrame | FactorFrameExpr:
    if ctx is not None and hasattr(ctx, "feed_wide"):
        return ctx.feed_wide(feed_name, value_column)
    return _feed_expr(feed_name, value_column)


def financial(
    ctx: Any | None = None,
    *,
    value_column: str,
    feed_name: str = "financial",
) -> pd.DataFrame | FactorFrameExpr:
    if ctx is not None and hasattr(ctx, "feed_wide"):
        return ctx.feed_wide(feed_name, value_column)
    return _feed_expr(feed_name, value_column)


def valuation(
    ctx: Any | None = None,
    *,
    value_column: str,
    feed_name: str = "valuation",
) -> pd.DataFrame | FactorFrameExpr:
    if ctx is not None and hasattr(ctx, "feed_wide"):
        return ctx.feed_wide(feed_name, value_column)
    return _feed_expr(feed_name, value_column)


def macro(
    ctx: Any | None = None,
    *,
    value_column: str,
    feed_name: str = "macro",
) -> pd.Series | FactorFrameExpr:
    if ctx is not None and hasattr(ctx, "feed_series"):
        return ctx.feed_series(feed_name, value_column)
    return _feed_expr(feed_name, value_column, is_series=True)


def news(
    ctx: Any | None = None,
    *,
    value_column: str,
    feed_name: str = "news",
) -> pd.DataFrame | FactorFrameExpr:
    if ctx is not None and hasattr(ctx, "feed_wide"):
        return ctx.feed_wide(feed_name, value_column)
    return _feed_expr(feed_name, value_column)


def _rolling_corr_values(
    x: pd.Series | pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    *,
    window: int,
    min_periods: int | None = None,
) -> pd.Series | pd.DataFrame:
    if isinstance(x, pd.Series) and isinstance(y, pd.Series):
        return _time_series_rolling_corr(x, y, window=window, min_periods=min_periods)

    if isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame):
        left, right = x.align(y, join="inner", axis=0)
        left, right = left.align(right, join="inner", axis=1)
        result = {
            column: _time_series_rolling_corr(left[column], right[column], window=window, min_periods=min_periods)
            for column in left.columns
        }
        return pd.DataFrame(result, index=left.index)

    if isinstance(x, pd.DataFrame) and isinstance(y, pd.Series):
        return pd.DataFrame(
            {
                column: _time_series_rolling_corr(x[column], y.reindex(x.index), window=window, min_periods=min_periods)
                for column in x.columns
            },
            index=x.index,
        )

    if isinstance(x, pd.Series) and isinstance(y, pd.DataFrame):
        return pd.DataFrame(
            {
                column: _time_series_rolling_corr(x.reindex(y.index), y[column], window=window, min_periods=min_periods)
                for column in y.columns
            },
            index=y.index,
        )

    raise TypeError("rolling_corr expects Series/Series, DataFrame/DataFrame, or Series/DataFrame inputs.")


def rolling_min(frame: pd.DataFrame | pd.Series, window: int, *, min_periods: int | None = None) -> pd.DataFrame | pd.Series:
    return _time_series_rolling_min(frame, window=window, min_periods=min_periods)


def rolling_max(frame: pd.DataFrame | pd.Series, window: int, *, min_periods: int | None = None) -> pd.DataFrame | pd.Series:
    return _time_series_rolling_max(frame, window=window, min_periods=min_periods)


def rolling_corr(
    x: pd.Series | pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    window: int,
    *,
    min_periods: int | None = None,
) -> pd.Series | pd.DataFrame:
    return _rolling_corr_values(x, y, window=window, min_periods=min_periods)


def rolling_cov(
    x: pd.Series | pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    window: int,
    *,
    min_periods: int | None = None,
) -> pd.Series | pd.DataFrame:
    return _rolling_cov_values(x, y, window=window, min_periods=min_periods)


def rolling_rank(
    frame: pd.Series | pd.DataFrame,
    window: int,
    *,
    min_periods: int | None = None,
) -> pd.Series | pd.DataFrame:
    return _rolling_rank_values(frame, window=window, min_periods=min_periods)


def rolling_skew(
    frame: pd.Series | pd.DataFrame,
    window: int,
    *,
    min_periods: int | None = None,
) -> pd.Series | pd.DataFrame:
    return _rolling_skew_values(frame, window=window, min_periods=min_periods)


def rolling_kurt(
    frame: pd.Series | pd.DataFrame,
    window: int,
    *,
    min_periods: int | None = None,
) -> pd.Series | pd.DataFrame:
    return _rolling_kurt_values(frame, window=window, min_periods=min_periods)


def rolling_median(
    frame: pd.Series | pd.DataFrame,
    window: int,
    *,
    min_periods: int | None = None,
) -> pd.Series | pd.DataFrame:
    return _rolling_median_values(frame, window=window, min_periods=min_periods)


def rolling_mean(
    frame: pd.DataFrame | pd.Series,
    window: int,
    *,
    min_periods: int | None = None,
) -> pd.DataFrame | pd.Series:
    return _apply_rolling(frame, kind="mean", window=window, min_periods=min_periods)


def rolling_std(
    frame: pd.DataFrame | pd.Series,
    window: int,
    *,
    min_periods: int | None = None,
    ddof: int = 0,
) -> pd.DataFrame | pd.Series:
    return _apply_rolling(frame, kind="std", window=window, min_periods=min_periods, ddof=ddof)


def rolling_sum(
    frame: pd.DataFrame | pd.Series,
    window: int,
    *,
    min_periods: int | None = None,
) -> pd.DataFrame | pd.Series:
    return _apply_rolling(frame, kind="sum", window=window, min_periods=min_periods)


def rolling_var(
    frame: pd.DataFrame | pd.Series,
    window: int,
    *,
    min_periods: int | None = None,
    ddof: int = 0,
) -> pd.DataFrame | pd.Series:
    return _apply_rolling(frame, kind="var", window=window, min_periods=min_periods, ddof=ddof)


def ts_zscore(
    frame: pd.Series | pd.DataFrame,
    window: int,
    *,
    min_periods: int | None = None,
    ddof: int = 1,
) -> pd.Series | pd.DataFrame:
    return _ts_zscore_values(frame, window=window, min_periods=min_periods, ddof=ddof)


def rolling_pct_change(
    frame: pd.Series | pd.DataFrame,
    periods: int = 1,
    *,
    fill_method: str | None = None,
) -> pd.Series | pd.DataFrame:
    return _to_numeric_like(frame).pct_change(periods=periods, fill_method=fill_method)


def ts_pct_change(
    frame: pd.Series | pd.DataFrame,
    periods: int = 1,
    *,
    fill_method: str | None = None,
) -> pd.Series | pd.DataFrame:
    return rolling_pct_change(frame, periods=periods, fill_method=fill_method)


def rolling_delta(frame: pd.Series | pd.DataFrame, periods: int = 1) -> pd.Series | pd.DataFrame:
    return _to_numeric_like(frame).diff(periods)


def ts_delta(frame: pd.Series | pd.DataFrame, periods: int = 1) -> pd.Series | pd.DataFrame:
    return rolling_delta(frame, periods=periods)


def rolling_delay(frame: pd.Series | pd.DataFrame, periods: int = 1) -> pd.Series | pd.DataFrame:
    return _time_series_lag(frame, periods=periods)


def ts_delay(frame: pd.Series | pd.DataFrame, periods: int = 1) -> pd.Series | pd.DataFrame:
    return rolling_delay(frame, periods=periods)


def ewm_mean(
    frame: pd.Series | pd.DataFrame,
    *,
    span: int | None = None,
    halflife: float | None = None,
    alpha: float | None = None,
) -> pd.Series | pd.DataFrame:
    return _ewm_mean_values(frame, span=span, halflife=halflife, alpha=alpha)


def ewm_std(
    frame: pd.Series | pd.DataFrame,
    *,
    span: int | None = None,
    halflife: float | None = None,
    alpha: float | None = None,
) -> pd.Series | pd.DataFrame:
    return _ewm_std_values(frame, span=span, halflife=halflife, alpha=alpha)


def rolling_sharpe(
    frame: pd.Series | pd.DataFrame,
    window: int,
    *,
    risk_free_rate: float = 0.0,
    annualization: int = 252,
    min_periods: int | None = None,
) -> pd.Series | pd.DataFrame:
    return _rolling_sharpe_values(
        frame,
        window=window,
        risk_free_rate=risk_free_rate,
        annualization=annualization,
        min_periods=min_periods,
    )


def rolling_information_ratio(
    frame: pd.Series | pd.DataFrame,
    window: int,
    *,
    annualization: int = 252,
    min_periods: int | None = None,
) -> pd.Series | pd.DataFrame:
    return _rolling_ir_values(frame, window=window, annualization=annualization, min_periods=min_periods)


def ts_median(
    frame: pd.Series | pd.DataFrame,
    window: int,
    *,
    min_periods: int | None = None,
) -> pd.Series | pd.DataFrame:
    return _rolling_median_values(frame, window=window, min_periods=min_periods)


def rolling_prod(frame: pd.Series | pd.DataFrame, window: int) -> pd.Series | pd.DataFrame:
    return _ts_prod_values(frame, window)


def ts_prod(frame: pd.Series | pd.DataFrame, window: int) -> pd.Series | pd.DataFrame:
    return rolling_prod(frame, window)


def rolling_abs(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return abs(frame)


def ts_abs(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return abs(frame)


def rolling_sign(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return sign(frame)


def ts_sign(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return sign(frame)


def rolling_wma(frame: pd.Series | pd.DataFrame, window: int) -> pd.Series | pd.DataFrame:
    return _ts_wma_values(frame, window)


def ts_wma(frame: pd.Series | pd.DataFrame, window: int) -> pd.Series | pd.DataFrame:
    return rolling_wma(frame, window)


def rolling_ema(frame: pd.Series | pd.DataFrame, window: int) -> pd.Series | pd.DataFrame:
    return _ts_ema_values(frame, window)


def ts_ema(frame: pd.Series | pd.DataFrame, window: int) -> pd.Series | pd.DataFrame:
    return rolling_ema(frame, window)


def mean(frame: pd.DataFrame | pd.Series, window: int, *, min_periods: int | None = None) -> pd.DataFrame | pd.Series:
    return _apply_rolling(frame, kind="mean", window=window, min_periods=min_periods)


def sum(frame: pd.DataFrame | pd.Series, window: int, *, min_periods: int | None = None) -> pd.DataFrame | pd.Series:
    return _apply_rolling(frame, kind="sum", window=window, min_periods=min_periods)


def std(
    frame: pd.DataFrame | pd.Series,
    window: int,
    *,
    min_periods: int | None = None,
    ddof: int = 0,
) -> pd.DataFrame | pd.Series:
    return _apply_rolling(frame, kind="std", window=window, min_periods=min_periods, ddof=ddof)


def var(
    frame: pd.DataFrame | pd.Series,
    window: int,
    *,
    min_periods: int | None = None,
    ddof: int = 0,
) -> pd.DataFrame | pd.Series:
    return _apply_rolling(frame, kind="var", window=window, min_periods=min_periods, ddof=ddof)


def min(frame: pd.DataFrame | pd.Series, window: int, *, min_periods: int | None = None) -> pd.DataFrame | pd.Series:
    return _time_series_rolling_min(frame, window=window, min_periods=min_periods)


def max(frame: pd.DataFrame | pd.Series, window: int, *, min_periods: int | None = None) -> pd.DataFrame | pd.Series:
    return _time_series_rolling_max(frame, window=window, min_periods=min_periods)


def corr(
    x: pd.Series | pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    window: int,
    *,
    min_periods: int | None = None,
) -> pd.Series | pd.DataFrame:
    return _rolling_corr_values(x, y, window=window, min_periods=min_periods)


def cov(
    x: pd.Series | pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    window: int,
    *,
    min_periods: int | None = None,
) -> pd.Series | pd.DataFrame:
    return _rolling_cov_values(x, y, window=window, min_periods=min_periods)


def ts_corr(
    x: pd.Series | pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    window: int,
    *,
    min_periods: int | None = None,
) -> pd.Series | pd.DataFrame:
    return _ts_corr_values(x, y, window=window, min_periods=min_periods)


def ts_beta(
    y: pd.Series | pd.DataFrame,
    x: pd.Series | pd.DataFrame,
    window: int,
    *,
    min_periods: int | None = None,
) -> pd.Series | pd.DataFrame:
    return _rolling_beta_values(y, x, window=window, min_periods=min_periods)


def ts_var(
    frame: pd.Series | pd.DataFrame,
    window: int,
    *,
    min_periods: int | None = None,
    ddof: int = 0,
) -> pd.Series | pd.DataFrame:
    return _ts_var_values(frame, window=window, min_periods=min_periods, ddof=ddof)


def ts_skew(
    frame: pd.Series | pd.DataFrame,
    window: int,
    *,
    min_periods: int | None = None,
) -> pd.Series | pd.DataFrame:
    return _rolling_skew_values(frame, window=window, min_periods=min_periods)


def ts_kurt(
    frame: pd.Series | pd.DataFrame,
    window: int,
    *,
    min_periods: int | None = None,
) -> pd.Series | pd.DataFrame:
    return _rolling_kurt_values(frame, window=window, min_periods=min_periods)


def ifelse(
    condition: Any,
    *,
    true_value: Any = None,
    false_value: Any = None,
) -> pd.Series | pd.DataFrame | Any:
    return _ifelse_values(condition, true_value=true_value, false_value=false_value)


def cs_scale(
    frame: pd.DataFrame | pd.Series,
    *,
    axis: int = 1,
    feature_range: tuple[float, float] = (0.0, 1.0),
) -> pd.DataFrame | pd.Series:
    return _cross_sectional_minmax_scale(frame, axis=axis, feature_range=feature_range)


def minmax_scale(
    frame: pd.DataFrame | pd.Series,
    *,
    axis: int = 1,
    feature_range: tuple[float, float] = (0.0, 1.0),
) -> pd.DataFrame | pd.Series:
    return _cross_sectional_minmax_scale(frame, axis=axis, feature_range=feature_range)


def l1_normalize(frame: pd.DataFrame | pd.Series, *, axis: int = 1) -> pd.DataFrame | pd.Series:
    return _cross_sectional_l1_normalize(frame, axis=axis)


def l2_normalize(frame: pd.DataFrame | pd.Series, *, axis: int = 1) -> pd.DataFrame | pd.Series:
    return _cross_sectional_l2_normalize(frame, axis=axis)


def lag(frame: pd.DataFrame | pd.Series, periods: int = 1) -> pd.DataFrame | pd.Series:
    return _time_series_lag(frame, periods=periods)


def diff(frame: pd.DataFrame | pd.Series, periods: int = 1) -> pd.DataFrame | pd.Series:
    return frame.diff(periods)


def cumsum(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return _cumulative_values(frame, kind="cumsum")


def cumprod(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return _cumulative_values(frame, kind="cumprod")


def winsorize(
    frame: pd.DataFrame | pd.Series,
    *,
    lower: float = 0.01,
    upper: float = 0.99,
    axis: int = 1,
) -> pd.DataFrame | pd.Series:
    return _winsorize_quantile(frame, lower=lower, upper=upper, axis=axis)


def zscore(
    frame: pd.DataFrame | pd.Series,
    *,
    axis: int = 1,
    ddof: int = 0,
    clip: tuple[float, float] | None = None,
) -> pd.DataFrame | pd.Series:
    return _cross_sectional_zscore(frame, axis=axis, ddof=ddof, clip=clip)


def demean(frame: pd.DataFrame | pd.Series, *, axis: int = 1) -> pd.DataFrame | pd.Series:
    return _cross_sectional_demean(frame, axis=axis)


def neutralize(
    values: pd.DataFrame | pd.Series,
    groups: pd.DataFrame | pd.Series,
    *,
    method: str = "demean",
) -> pd.DataFrame | pd.Series:
    return _neutralize_values(values, groups, method=method)


def group_neutralize(
    values: pd.DataFrame | pd.Series,
    groups: pd.DataFrame | pd.Series,
    *,
    method: str = "demean",
) -> pd.DataFrame | pd.Series:
    return neutralize(values, groups, method=method)


def group_demean(values: pd.DataFrame | pd.Series, groups: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return neutralize(values, groups, method="demean")


def group_rank(values: pd.DataFrame | pd.Series, groups: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return _groupwise_apply(
        values,
        groups,
        transform=_group_rank_by_group,
        series_error="Series group_rank expects a Series of groups.",
        dataframe_series_error="DataFrame group_rank with a Series of groups expects groups indexed by the frame columns.",
        dataframe_df_error="DataFrame group_rank expects groups to have the same index and columns.",
    )


def group_zscore(values: pd.DataFrame | pd.Series, groups: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return _groupwise_apply(
        values,
        groups,
        transform=_group_zscore_by_group,
        series_error="Series group_zscore expects a Series of groups.",
        dataframe_series_error="DataFrame group_zscore with a Series of groups expects groups indexed by the frame columns.",
        dataframe_df_error="DataFrame group_zscore expects groups to have the same index and columns.",
    )


def group_scale(values: pd.DataFrame | pd.Series, groups: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return _groupwise_apply(
        values,
        groups,
        transform=_group_scale_by_group,
        series_error="Series group_scale expects a Series of groups.",
        dataframe_series_error="DataFrame group_scale with a Series of groups expects groups indexed by the frame columns.",
        dataframe_df_error="DataFrame group_scale expects groups to have the same index and columns.",
    )


def ts_change(frame: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    return frame.pct_change(periods)


def ts_return(frame: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    return frame.pct_change(periods)


def ts_mean(frame: pd.DataFrame, window: int, *, min_periods: int | None = None) -> pd.DataFrame:
    return frame.rolling(window=window, min_periods=min_periods).mean()


def ts_std(frame: pd.DataFrame, window: int, *, min_periods: int | None = None, ddof: int = 0) -> pd.DataFrame:
    return frame.rolling(window=window, min_periods=min_periods).std(ddof=ddof)


def ts_rank(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window=window).rank(pct=True)


def ts_momentum(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.pct_change(window)


def cs_rank(frame: pd.DataFrame, *, ascending: bool = True, pct: bool = True) -> pd.DataFrame:
    ranked = frame.rank(axis=1, ascending=ascending, pct=pct, method="average")
    return ranked


def rank_desc(frame: pd.DataFrame | pd.Series, *, axis: int = 0, pct: bool = True, method: str = "average") -> pd.DataFrame | pd.Series:
    return frame.rank(axis=axis, ascending=False, pct=pct, method=method)


def cs_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1, ddof=0).replace(0, pd.NA)
    return frame.sub(mean, axis=0).div(std, axis=0)


def cs_demean(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sub(frame.mean(axis=1), axis=0)


def clip(frame: pd.DataFrame, *, lower: float | None = None, upper: float | None = None) -> pd.DataFrame:
    return frame.clip(lower=lower, upper=upper)


def clip_lower(frame: pd.DataFrame | pd.Series, lower: float) -> pd.DataFrame | pd.Series:
    return frame.clip(lower=lower)


def clip_upper(frame: pd.DataFrame | pd.Series, upper: float) -> pd.DataFrame | pd.Series:
    return frame.clip(upper=upper)


def abs(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return _apply_math(frame, np.abs)


def log(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return _apply_math(frame, np.log)


def exp(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return _apply_math(frame, np.exp)


def sqrt(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return _apply_math(frame, np.sqrt)


def sign(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return _apply_math(frame, np.sign)


def pow(frame: pd.DataFrame | pd.Series, exponent: Any) -> pd.DataFrame | pd.Series:
    numeric = _to_numeric_like(frame)
    return numeric**exponent


def fillna(
    frame: pd.DataFrame | pd.Series,
    value: Any = None,
    *,
    method: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame | pd.Series:
    return _fillna_values(frame, value, method=method, limit=limit)


def where(
    frame: pd.DataFrame | pd.Series,
    condition: Any,
    *,
    other: Any = None,
) -> pd.DataFrame | pd.Series:
    return _where_values(frame, condition, other=other)


def mask(
    frame: pd.DataFrame | pd.Series,
    condition: Any,
    *,
    other: Any = None,
) -> pd.DataFrame | pd.Series:
    return _mask_values(frame, condition, other=other)


def replace(
    frame: pd.DataFrame | pd.Series,
    to_replace: Any,
    value: Any = None,
    *,
    method: str | None = None,
    limit: int | None = None,
    regex: bool = False,
) -> pd.DataFrame | pd.Series:
    return _replace_values(
        frame,
        to_replace,
        value=value,
        method=method,
        limit=limit,
        regex=regex,
    )


def isna(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return frame.isna()


def notna(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return frame.notna()


def top_n(frame: pd.DataFrame | pd.Series, n: int, *, axis: int = 1) -> pd.DataFrame | pd.Series:
    return _top_bottom_n(frame, n, axis=axis, largest=True)


def bottom_n(frame: pd.DataFrame | pd.Series, n: int, *, axis: int = 1) -> pd.DataFrame | pd.Series:
    return _top_bottom_n(frame, n, axis=axis, largest=False)


def rolling_quantile(frame: pd.DataFrame, window: int, quantile: float) -> pd.DataFrame:
    return frame.rolling(window=window).quantile(quantile)


def rolling_momentum(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.pct_change(window)


def rolling_volatility(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.pct_change().rolling(window=window).std(ddof=0)


__all__ = [
    "FactorFrameExpr",
    "FactorFrameFactor",
    "factor",
    "feed_wide",
    "feed_series",
    "price",
    "financial",
    "valuation",
    "macro",
    "news",
    "lag",
    "diff",
    "cumsum",
    "cumprod",
    "winsorize",
    "zscore",
    "demean",
    "neutralize",
    "group_neutralize",
    "group_demean",
    "group_rank",
    "group_zscore",
    "group_scale",
    "abs",
    "log",
    "exp",
    "sqrt",
    "sign",
    "pow",
    "fillna",
    "where",
    "mask",
    "replace",
    "isna",
    "notna",
    "clip_lower",
    "clip_upper",
    "top_n",
    "bottom_n",
    "rolling_min",
    "rolling_max",
    "rolling_corr",
    "rolling_cov",
    "rolling_rank",
    "rolling_skew",
    "rolling_kurt",
    "rolling_median",
    "rolling_abs",
    "ts_abs",
    "rolling_sign",
    "ts_sign",
    "rolling_wma",
    "ts_wma",
    "rolling_ema",
    "ts_ema",
    "rolling_delay",
    "ts_delay",
    "rolling_delta",
    "ts_delta",
    "rolling_pct_change",
    "ts_pct_change",
    "rolling_prod",
    "ts_prod",
    "rolling_mean",
    "rolling_std",
    "rolling_var",
    "rolling_sum",
    "mean",
    "sum",
    "std",
    "var",
    "min",
    "max",
    "corr",
    "cov",
    "ifelse",
    "ts_zscore",
    "ewm_mean",
    "ewm_std",
    "rolling_sharpe",
    "rolling_information_ratio",
    "ts_median",
    "ts_corr",
    "ts_beta",
    "ts_var",
    "ts_skew",
    "ts_kurt",
    "minmax_scale",
    "cs_scale",
    "l1_normalize",
    "l2_normalize",
    "ts_change",
    "ts_return",
    "ts_mean",
    "ts_std",
    "ts_rank",
    "ts_momentum",
    "cs_rank",
    "rank_desc",
    "cs_zscore",
    "cs_demean",
    "clip",
    "rolling_quantile",
    "rolling_momentum",
    "rolling_volatility",
]
