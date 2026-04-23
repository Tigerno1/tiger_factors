from __future__ import annotations

import numpy as np
import pandas as pd
import warnings
from pandas.tseries.offsets import BDay

from tiger_factors.factor_evaluation.utils import add_custom_calendar_timedelta
from tiger_factors.factor_evaluation.utils import demean_forward_returns
from tiger_factors.factor_evaluation.utils import get_forward_returns_columns
from tiger_factors.factor_evaluation.utils import period_to_label
from tiger_factors.factor_evaluation.utils import _rowwise_cross_sectional_corr


def _safe_series_corr(left: pd.Series, right: pd.Series, *, method: str = "pearson") -> float:
    joined = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    if len(joined) < 3:
        return float("nan")
    x = pd.to_numeric(joined["left"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(joined["right"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if method == "spearman":
        x = pd.Series(x).rank(method="average").to_numpy(dtype=float)
        y = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    x_std = float(np.std(x, ddof=0))
    y_std = float(np.std(y, ddof=0))
    if x_std <= 1e-12 or y_std <= 1e-12:
        return float("nan")
    x_centered = x - float(np.mean(x))
    y_centered = y - float(np.mean(y))
    denom = float(np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(x_centered * y_centered) / denom)


def _factor_dates(factor_data: pd.DataFrame) -> pd.Index:
    return pd.Index(factor_data.index.get_level_values(0).unique(), name=factor_data.index.names[0])


def _period_days(period: str | pd.Timedelta | int) -> int:
    if isinstance(period, int):
        return max(period, 1)
    if isinstance(period, pd.Timedelta):
        return max(int(period / pd.Timedelta(days=1)), 1)
    try:
        return max(int(pd.Timedelta(str(period)) / pd.Timedelta(days=1)), 1)
    except ValueError:
        text = str(period)
        if text.endswith("D") and text[:-1].isdigit():
            return max(int(text[:-1]), 1)
        return 1


def _filter_factor_data(
    factor_data: pd.DataFrame,
    *,
    quantiles: list[int] | tuple[int, ...] | set[int] | None = None,
    groups: list[str] | tuple[str, ...] | set[str] | None = None,
) -> pd.DataFrame:
    filtered = factor_data.copy()
    if quantiles is not None:
        quantile_values = {int(value) for value in quantiles}
        filtered = filtered[filtered["factor_quantile"].isin(quantile_values)]
    if groups is not None:
        if "group" not in filtered.columns:
            return filtered.iloc[0:0].copy()
        group_values = {str(value) for value in groups}
        filtered = filtered[filtered["group"].astype(str).isin(group_values)]
    return filtered


def _regression_alpha_beta(
    y: pd.Series,
    x: pd.Series,
) -> tuple[float, float, float]:
    joined = pd.concat([pd.to_numeric(y, errors="coerce"), pd.to_numeric(x, errors="coerce")], axis=1).dropna()
    if len(joined) < 3:
        return float("nan"), float("nan"), float("nan")
    y_values = joined.iloc[:, 0].to_numpy(dtype=float)
    x_values = joined.iloc[:, 1].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x_values)), x_values])
    coef, *_ = np.linalg.lstsq(design, y_values, rcond=None)
    residual = y_values - design @ coef
    dof = max(len(y_values) - design.shape[1], 1)
    sigma2 = float(np.sum(residual**2) / dof)
    xtx_inv = np.linalg.pinv(design.T @ design)
    alpha_se = float(np.sqrt(max(xtx_inv[0, 0] * sigma2, 0.0)))
    alpha_tstat = float(coef[0] / alpha_se) if alpha_se > 1e-12 else float("nan")
    return float(coef[0]), float(coef[1]), alpha_tstat


def factor_information_coefficient(
    factor_data: pd.DataFrame,
    group_adjust: bool = False,
    by_group: bool = False,
) -> pd.DataFrame:
    periods = get_forward_returns_columns(factor_data.columns)
    working = factor_data.copy()
    grouper: list[object] = [working.index.get_level_values(0)]
    if group_adjust and "group" in working.columns:
        working = demean_forward_returns(working, grouper + ["group"])
    if by_group and "group" in working.columns:
        grouper.append("group")
    if not by_group:
        ordered = working.sort_index()
        factor_wide = ordered["factor"].unstack()
        ic_columns: dict[str, pd.Series] = {}
        for period in periods:
            period_wide = ordered[period].unstack()
            ic_columns[period] = _rowwise_cross_sectional_corr(factor_wide, period_wide, rank=True)
        result = pd.DataFrame(ic_columns)
        result.index.name = ordered.index.names[0]
        return result

    def _src_ic(group: pd.DataFrame) -> pd.Series:
        factor = group["factor"]
        return group[periods].apply(lambda values: _safe_series_corr(values, factor, method="spearman"))

    selected_columns = ["factor", *periods]
    if "group" in working.columns:
        selected_columns.append("group")
    selected = working.loc[:, selected_columns]
    grouped = selected.groupby(grouper, observed=True)
    return grouped.apply(_src_ic, include_groups=False)


def mean_information_coefficient(
    factor_data: pd.DataFrame,
    group_adjust: bool = False,
    by_group: bool = False,
    by_time: str | None = None,
) -> pd.DataFrame | pd.Series:
    ic = factor_information_coefficient(factor_data, group_adjust=group_adjust, by_group=by_group)
    grouper: list[object] = []
    if by_time is not None:
        grouper.append(pd.Grouper(freq=by_time))
    if by_group and "group" in factor_data.columns:
        grouper.append("group")
    if not grouper:
        return ic.mean()
    return ic.reset_index().set_index("date_").groupby(grouper).mean()


def factor_weights(
    factor_data: pd.DataFrame,
    demeaned: bool = True,
    group_adjust: bool = False,
    equal_weight: bool = False,
) -> pd.Series:
    if not group_adjust and not equal_weight:
        values = pd.to_numeric(factor_data["factor"], errors="coerce").astype(float)
        date_level = factor_data.index.get_level_values(0)
        if demeaned:
            values = values - values.groupby(date_level).transform("mean")
        abs_sum = values.abs().groupby(date_level).transform("sum")
        weights = pd.Series(np.nan, index=values.index, dtype=float)
        valid = abs_sum > 0
        weights.loc[valid] = values.loc[valid] / abs_sum.loc[valid]
        zero_denom = ~valid & values.notna()
        weights.loc[zero_denom] = 0.0
        weights.name = "factor_weight"
        return weights

    def _to_weights(group: pd.Series, _demeaned: bool, _equal_weight: bool) -> pd.Series:
        values = group.astype(float).copy()
        if _equal_weight:
            if _demeaned:
                values = values - values.median()
            negative = values < 0
            positive = values > 0
            values[negative] = -1.0
            values[positive] = 1.0
            if _demeaned:
                if negative.any():
                    values[negative] /= negative.sum()
                if positive.any():
                    values[positive] /= positive.sum()
                return values
        elif _demeaned:
            values = values - values.mean()
        denom = values.abs().sum()
        if denom <= 0 or not np.isfinite(denom):
            return values * 0.0
        return values / denom

    grouper: list[object] = [factor_data.index.get_level_values(0)]
    if group_adjust and "group" in factor_data.columns:
        grouper.append("group")

    weights = factor_data.groupby(grouper, group_keys=False, observed=True)["factor"].apply(
        _to_weights,
        demeaned,
        equal_weight,
    )
    if group_adjust and "group" in factor_data.columns:
        weights = weights.groupby(level=0, group_keys=False).apply(_to_weights, False, False)
    weights.name = "factor_weight"
    return weights


def factor_returns(
    factor_data: pd.DataFrame,
    demeaned: bool = True,
    group_adjust: bool = False,
    equal_weight: bool = False,
    by_asset: bool = False,
) -> pd.DataFrame:
    periods = get_forward_returns_columns(factor_data.columns)
    weights = factor_weights(
        factor_data,
        demeaned=demeaned,
        group_adjust=group_adjust,
        equal_weight=equal_weight,
    )
    weighted = factor_data[periods].mul(weights, axis=0)
    if by_asset:
        return weighted
    return weighted.groupby(level=0).sum()


def factor_alpha_beta(
    factor_data: pd.DataFrame | pd.Series,
    returns: pd.DataFrame | None = None,
    demeaned: bool = True,
    group_adjust: bool = False,
    equal_weight: bool = False,
) -> pd.DataFrame | dict[str, float | int]:
    if isinstance(factor_data, pd.Series):
        from tiger_factors.factor_evaluation.core import alpha_beta_regression as _alpha_beta_regression

        if returns is None or not isinstance(returns, pd.Series):
            raise ValueError("Series factor_alpha_beta expects benchmark returns as a Series.")
        return _alpha_beta_regression(factor_data, returns)

    periods = get_forward_returns_columns(factor_data.columns)
    portfolio_returns = returns
    if portfolio_returns is None:
        portfolio_returns = factor_returns(
            factor_data,
            demeaned=demeaned,
            group_adjust=group_adjust,
            equal_weight=equal_weight,
        )
    universe = factor_data.groupby(level=0)[periods].mean().reindex(portfolio_returns.index)
    result = pd.DataFrame(index=["Ann. alpha", "beta", "alpha_tstat"], columns=periods, dtype=float)
    for period in periods:
        joined = pd.concat([portfolio_returns[period], universe[period]], axis=1).dropna()
        if len(joined) < 3:
            continue
        alpha, beta, alpha_tstat = _regression_alpha_beta(joined.iloc[:, 0], joined.iloc[:, 1])
        period_days = _period_days(period)
        annualization = max(252 / period_days, 1.0)
        result.loc["Ann. alpha", period] = (1.0 + alpha) ** annualization - 1.0
        result.loc["beta", period] = beta
        result.loc["alpha_tstat", period] = alpha_tstat
    return result


def cumulative_returns(
    returns: pd.Series,
    period: str | pd.Timedelta | int | None = None,
    freq: str | pd.offsets.BaseOffset | None = None,
) -> pd.Series:
    series = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if period is not None and _period_days(period) > 1:
        steps = _period_days(period)
        staggered: list[pd.Series] = []
        for offset in range(steps):
            base = series.shift(offset).fillna(0.0)
            staggered.append(base)
        series = pd.concat(staggered, axis=1).mean(axis=1)
    cumulative = (1.0 + series).cumprod() - 1.0
    cumulative.index = pd.DatetimeIndex(cumulative.index)
    return cumulative


def positions(
    weights: pd.DataFrame | pd.Series,
    period: int | str | pd.Timedelta,
    freq: str | pd.offsets.BaseOffset | None = None,
) -> pd.DataFrame:
    if isinstance(weights, pd.Series):
        frame = weights.unstack().sort_index()
    else:
        frame = weights.copy().sort_index()
    frame = frame.fillna(0.0)
    if frame.empty:
        return frame.copy()
    if not isinstance(period, pd.Timedelta):
        period = pd.Timedelta(period)
    if freq is None:
        freq = frame.index.freq
    if freq is None:
        inferred = pd.infer_freq(frame.index)
        if inferred is not None:
            freq = pd.tseries.frequencies.to_offset(inferred)
    if freq is None:
        freq = BDay()
        warnings.warn("'freq' not set, using business day calendar", UserWarning)

    trades_idx = frame.index.copy()
    returns_idx = pd.DatetimeIndex(
        [add_custom_calendar_timedelta(timestamp, period, freq) for timestamp in trades_idx]
    )
    weights_idx = trades_idx.union(returns_idx)

    portfolio_weights = pd.DataFrame(index=weights_idx, columns=frame.columns, dtype=float)
    active_weights: list[tuple[pd.Timestamp, pd.Series]] = []

    for current_time in weights_idx:
        if current_time in frame.index:
            assets_weights = frame.loc[current_time]
            expire_ts = add_custom_calendar_timedelta(current_time, period, freq)
            active_weights.append((pd.Timestamp(expire_ts), assets_weights))

        if active_weights:
            expire_ts, _ = active_weights[0]
            if expire_ts <= current_time:
                active_weights.pop(0)

        if not active_weights:
            continue

        total_weights = [weights_row for _, weights_row in active_weights]
        total_weights = pd.concat(total_weights, axis=1).sum(axis=1)
        denom = total_weights.abs().sum()
        if denom > 0 and np.isfinite(denom):
            total_weights = total_weights / denom
        portfolio_weights.loc[current_time] = total_weights

    return portfolio_weights.fillna(0.0)


def mean_return_by_quantile(
    factor_data: pd.DataFrame,
    by_date: bool = False,
    by_group: bool = False,
    demeaned: bool = True,
    group_adjust: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    periods = get_forward_returns_columns(factor_data.columns)
    working = factor_data.copy()
    grouper: list[object] = ["factor_quantile"]
    if group_adjust and "group" in working.columns:
        working = demean_forward_returns(working, [working.index.get_level_values(0), "group"])
    elif demeaned:
        working = demean_forward_returns(working, [working.index.get_level_values(0)])
    if by_date:
        grouper.insert(0, working.index.get_level_values(0))
    if by_group and "group" in working.columns:
        grouper.append("group")
    grouped = working.groupby(grouper, observed=True)[periods]
    mean_ret = grouped.mean()
    std_err = grouped.std(ddof=0) / np.sqrt(grouped.count().replace(0.0, np.nan))
    return mean_ret, std_err


def mean_return_by_quantile_by_date(
    factor_data: pd.DataFrame,
    demeaned: bool = True,
    group_adjust: bool = False,
) -> dict[str, pd.DataFrame]:
    mean_ret, _ = mean_return_by_quantile(
        factor_data,
        by_date=True,
        by_group=False,
        demeaned=demeaned,
        group_adjust=group_adjust,
    )
    results: dict[str, pd.DataFrame] = {}
    for period in get_forward_returns_columns(factor_data.columns):
        results[period] = mean_ret[period].unstack("factor_quantile").sort_index()
    return results


def compute_mean_returns_spread(
    mean_returns: pd.DataFrame,
    upper_quant: int,
    lower_quant: int,
    std_err: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.Series | None]:
    if isinstance(mean_returns.index, pd.MultiIndex) and "factor_quantile" in mean_returns.index.names:
        spread = mean_returns.xs(upper_quant, level="factor_quantile") - mean_returns.xs(
            lower_quant,
            level="factor_quantile",
        )
    else:
        spread = mean_returns[upper_quant] - mean_returns[lower_quant]
    if isinstance(spread, pd.Series):
        spread.name = "mean_return_spread"
    if std_err is None:
        return spread, None
    if isinstance(std_err.index, pd.MultiIndex) and "factor_quantile" in std_err.index.names:
        std1 = std_err.xs(upper_quant, level="factor_quantile")
        std2 = std_err.xs(lower_quant, level="factor_quantile")
        spread_err = np.sqrt(std1.pow(2) + std2.pow(2))
    else:
        spread_err = np.sqrt(std_err[upper_quant].pow(2) + std_err[lower_quant].pow(2))
    if isinstance(spread_err, pd.Series):
        spread_err.name = "mean_return_spread_std"
    return spread, spread_err


def quantile_turnover(
    quantile_factor: pd.Series | pd.DataFrame,
    quantile: int,
    period: int = 1,
) -> pd.Series:
    if isinstance(quantile_factor, pd.DataFrame):
        if "factor_quantile" not in quantile_factor.columns:
            raise ValueError("quantile_factor DataFrame must contain factor_quantile column")
        series = quantile_factor["factor_quantile"]
    else:
        series = quantile_factor
    members = series[series == quantile].reset_index()
    if members.empty:
        return pd.Series(dtype=float, name=f"quantile_{quantile}_turnover")
    date_col, code_col = members.columns[:2]
    grouped = members.groupby(date_col, observed=True)[code_col].apply(lambda values: set(values.astype(str)))
    dates = list(grouped.index)
    values: list[float] = []
    output_index: list[pd.Timestamp] = []
    for idx in range(period, len(dates)):
        current = grouped.iloc[idx]
        previous = grouped.iloc[idx - period]
        if not current:
            values.append(np.nan)
        else:
            overlap = len(current.intersection(previous))
            values.append(1.0 - overlap / max(len(current), 1))
        output_index.append(pd.Timestamp(dates[idx]))
    return pd.Series(values, index=pd.DatetimeIndex(output_index), name=f"quantile_{quantile}_turnover")


def factor_rank_autocorrelation(factor_data: pd.DataFrame, period: int = 1) -> pd.Series:
    if "factor" in factor_data.columns:
        ranked = factor_data["factor"].unstack().sort_index().rank(axis=1, method="average", pct=True)
    else:
        ranked = factor_data.sort_index().rank(axis=1, method="average", pct=True)
    shifted = ranked.shift(period)
    result = _rowwise_cross_sectional_corr(ranked.iloc[period:], shifted.iloc[period:])
    result.index = pd.DatetimeIndex(result.index)
    result.name = "factor_rank_autocorrelation"
    return result


def _coerce_returns_frame(returns: pd.DataFrame) -> pd.DataFrame:
    frame = returns.copy()
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame[~frame.index.isna()].sort_index()
    frame.columns = frame.columns.astype(str)
    frame = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if not frame.empty:
        positive_ratio = float((frame > 0).sum().sum()) / float(frame.notna().sum().sum() or 1)
        median_abs = float(np.nanmedian(np.abs(frame.to_numpy(dtype=float))))
        if positive_ratio > 0.95 and median_abs > 1.0:
            frame = frame.pct_change(fill_method=None)
    return frame


def common_start_returns(
    factor_data: pd.DataFrame,
    returns: pd.DataFrame,
    before: int,
    after: int,
    cumulative: bool = False,
    mean_by_date: bool = False,
    demean_by: pd.Series | pd.DataFrame | None = None,
) -> pd.DataFrame:
    wide_returns = _coerce_returns_frame(returns)
    if not cumulative:
        wide_returns = wide_returns.apply(cumulative_returns, axis=0)
    code_to_pos = {str(code): pos for pos, code in enumerate(wide_returns.columns.astype(str))}

    all_returns: list[pd.DataFrame | pd.Series] = []
    code_name = factor_data.index.names[1]

    for timestamp, frame in factor_data.groupby(level=0, observed=True):
        event_date = pd.Timestamp(timestamp)
        try:
            day_zero_index = wide_returns.index.get_loc(event_date)
        except KeyError:
            continue

        if isinstance(day_zero_index, slice):
            day_zero_index = day_zero_index.start
        elif isinstance(day_zero_index, np.ndarray):
            matches = np.flatnonzero(day_zero_index)
            if len(matches) == 0:
                continue
            day_zero_index = int(matches[0])
        if not isinstance(day_zero_index, (int, np.integer)):
            continue

        starting_index = max(int(day_zero_index) - int(before), 0)
        ending_index = min(int(day_zero_index) + int(after) + 1, len(wide_returns.index))

        event_codes = pd.Index(frame.index.get_level_values(code_name).astype(str)).unique()
        selected_codes = set(event_codes.tolist())
        demean_codes = pd.Index([], dtype=object)
        if demean_by is not None:
            try:
                demean_slice = demean_by.loc[event_date]
            except KeyError:
                demean_slice = None
            if demean_slice is not None:
                if isinstance(demean_slice.index, pd.MultiIndex):
                    demean_codes = pd.Index(demean_slice.index.get_level_values(-1).astype(str)).unique()
                else:
                    demean_codes = pd.Index(demean_slice.index.astype(str)).unique()
                selected_codes |= set(demean_codes.tolist())

        available_codes = [code for code in selected_codes if code in code_to_pos]
        if not available_codes:
            continue

        row_slice = slice(starting_index, ending_index)
        column_positions = [code_to_pos[code] for code in available_codes]
        values = wide_returns.iloc[row_slice, column_positions].to_numpy(copy=True)
        series = pd.DataFrame(
            values,
            index=range(starting_index - int(day_zero_index), ending_index - int(day_zero_index)),
            columns=available_codes,
        )

        event_columns = [code for code in event_codes if code in series.columns]
        if demean_by is not None and len(demean_codes) > 0:
            demean_columns = [code for code in demean_codes if code in series.columns]
            if demean_columns:
                mean = series.loc[:, demean_columns].mean(axis=1)
                series = series.loc[:, event_columns].sub(mean, axis=0)
            else:
                series = series.loc[:, event_columns]
        else:
            series = series.loc[:, event_columns]

        if mean_by_date:
            series = series.mean(axis=1)

        all_returns.append(series)

    if not all_returns:
        return pd.DataFrame()

    return pd.concat(all_returns, axis=1).sort_index()


def average_cumulative_return_by_quantile(
    factor_data: pd.DataFrame,
    returns: pd.DataFrame,
    periods_before: int = 10,
    periods_after: int = 15,
    demeaned: bool = True,
    group_adjust: bool = False,
    by_group: bool = False,
) -> pd.DataFrame:
    def cumulative_return_around_event(q_fact: pd.DataFrame | pd.Series, demean_by: pd.DataFrame | pd.Series | None):
        return common_start_returns(
            q_fact,
            returns,
            periods_before,
            periods_after,
            cumulative=True,
            mean_by_date=True,
            demean_by=demean_by,
        )

    def average_cumulative_return(q_fact: pd.DataFrame | pd.Series, demean_by: pd.DataFrame | pd.Series | None):
        q_returns = cumulative_return_around_event(q_fact, demean_by)
        q_returns = q_returns.replace([np.inf, -np.inf], np.nan)
        return pd.DataFrame(
            {
                "mean": q_returns.mean(skipna=True, axis=1),
                "std": q_returns.std(skipna=True, axis=1),
            }
        ).T

    if by_group and "group" in factor_data.columns:
        returns_by_group: list[pd.DataFrame] = []
        for group, group_data in factor_data.groupby("group", observed=True):
            group_quantiles = group_data["factor_quantile"]
            if group_adjust:
                demean_by = group_quantiles
            elif demeaned:
                demean_by = factor_data["factor_quantile"]
            else:
                demean_by = None
            avg_cumret = group_quantiles.groupby(group_quantiles).apply(average_cumulative_return, demean_by)
            if len(avg_cumret) == 0:
                continue
            avg_cumret["group"] = group
            avg_cumret.set_index("group", append=True, inplace=True)
            returns_by_group.append(avg_cumret)
        return pd.concat(returns_by_group, axis=0) if returns_by_group else pd.DataFrame()

    quantiles = factor_data["factor_quantile"]
    if group_adjust and "group" in factor_data.columns:
        all_returns: list[pd.DataFrame] = []
        for _, group_data in factor_data.groupby("group", observed=True):
            group_quantiles = group_data["factor_quantile"]
            avg_cumret = group_quantiles.groupby(group_quantiles).apply(
                cumulative_return_around_event,
                group_quantiles,
            )
            all_returns.append(avg_cumret)
        if not all_returns:
            return pd.DataFrame()
        q_returns = pd.concat(all_returns, axis=1)
        q_returns = pd.DataFrame({"mean": q_returns.mean(axis=1), "std": q_returns.std(axis=1)})
        return q_returns.unstack(level=1).stack(level=0)
    if demeaned:
        return quantiles.groupby(quantiles).apply(average_cumulative_return, quantiles)
    return quantiles.groupby(quantiles).apply(average_cumulative_return, None)


def factor_cumulative_returns(
    factor_data: pd.DataFrame,
    period: int | str | pd.Timedelta = "1D",
    long_short: bool = True,
    group_neutral: bool = False,
    equal_weight: bool = False,
    quantiles: list[int] | tuple[int, ...] | set[int] | None = None,
    groups: list[str] | tuple[str, ...] | set[str] | None = None,
) -> pd.Series:
    filtered = _filter_factor_data(factor_data, quantiles=quantiles, groups=groups)
    returns = factor_returns(
        filtered,
        demeaned=long_short,
        group_adjust=group_neutral,
        equal_weight=equal_weight,
    )
    period_label = period_to_label(period)
    if period_label not in returns.columns:
        raise ValueError(f"Unknown forward return period: {period_label}")
    return cumulative_returns(returns[period_label], period=period)


def factor_positions(
    factor_data: pd.DataFrame,
    period: int | str | pd.Timedelta = "1D",
    long_short: bool = True,
    group_neutral: bool = False,
    equal_weight: bool = False,
    quantiles: list[int] | tuple[int, ...] | set[int] | None = None,
    groups: list[str] | tuple[str, ...] | set[str] | None = None,
) -> pd.DataFrame:
    filtered = _filter_factor_data(factor_data, quantiles=quantiles, groups=groups)
    weights = factor_weights(
        filtered,
        demeaned=long_short,
        group_adjust=group_neutral,
        equal_weight=equal_weight,
    )
    return positions(weights, period)


def create_portfolio_input(
    factor_data: pd.DataFrame,
    period: int | str | pd.Timedelta = "1D",
    capital: float | None = None,
    long_short: bool = True,
    group_neutral: bool = False,
    equal_weight: bool = False,
    quantiles: list[int] | tuple[int, ...] | set[int] | None = None,
    groups: list[str] | tuple[str, ...] | set[str] | None = None,
    benchmark_period: int | str | pd.Timedelta = "1D",
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    filtered = _filter_factor_data(factor_data, quantiles=quantiles, groups=groups)
    cumrets = factor_cumulative_returns(
        filtered,
        period=period,
        long_short=long_short,
        group_neutral=group_neutral,
        equal_weight=equal_weight,
        quantiles=quantiles,
        groups=groups,
    )
    returns = cumrets.resample("1D").last().ffill().pct_change().fillna(0.0)
    benchmark_label = period_to_label(benchmark_period)
    positions_frame = factor_positions(
        filtered,
        period=period,
        long_short=long_short,
        group_neutral=group_neutral,
        equal_weight=equal_weight,
        quantiles=quantiles,
        groups=groups,
    )
    positions_frame = positions_frame.resample("1D").sum().ffill()
    positions_frame = positions_frame.div(positions_frame.abs().sum(axis=1), axis=0).fillna(0.0)
    positions_frame["cash"] = 1.0 - positions_frame.sum(axis=1)
    if capital is not None:
        positions_frame = positions_frame.mul(cumrets.reindex(positions_frame.index).ffill() * float(capital), axis=0)
    if benchmark_label not in filtered.columns:
        raise ValueError(f"Unknown benchmark period: {benchmark_label}")
    benchmark_data = filtered.copy()
    benchmark_data["factor"] = benchmark_data["factor"].abs()
    benchmark = factor_cumulative_returns(
        benchmark_data,
        period=benchmark_period,
        long_short=False,
        group_neutral=False,
        equal_weight=True,
    )
    benchmark = benchmark.resample("1D").last().ffill().pct_change().fillna(0.0)
    benchmark.name = "benchmark"
    return returns, positions_frame, benchmark


__all__ = [
    "average_cumulative_return_by_quantile",
    "common_start_returns",
    "compute_mean_returns_spread",
    "create_portfolio_input",
    "cumulative_returns",
    "factor_alpha_beta",
    "factor_cumulative_returns",
    "factor_information_coefficient",
    "factor_positions",
    "factor_rank_autocorrelation",
    "factor_returns",
    "factor_weights",
    "mean_information_coefficient",
    "mean_return_by_quantile",
    "mean_return_by_quantile_by_date",
    "positions",
    "quantile_turnover",
]
