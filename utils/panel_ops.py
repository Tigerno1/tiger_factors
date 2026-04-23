from __future__ import annotations

import numpy as np
import pandas as pd


def _group_transform(frame: pd.DataFrame, values: pd.Series, *, group_col: str, func) -> pd.Series:
    result = values.groupby(frame[group_col], sort=False).transform(func)
    result.index = values.index
    return result


def delay(frame: pd.DataFrame, values: pd.Series, periods: int, *, group_col: str = "symbol") -> pd.Series:
    return _group_transform(frame, values, group_col=group_col, func=lambda s: s.shift(int(periods)))


def ts_sum(frame: pd.DataFrame, values: pd.Series, window: int, *, group_col: str = "symbol") -> pd.Series:
    return _group_transform(frame, values, group_col=group_col, func=lambda s: s.rolling(int(window)).sum())


def ts_min(frame: pd.DataFrame, values: pd.Series, window: int, *, group_col: str = "symbol") -> pd.Series:
    return _group_transform(frame, values, group_col=group_col, func=lambda s: s.rolling(int(window)).min())


def ts_max(frame: pd.DataFrame, values: pd.Series, window: int, *, group_col: str = "symbol") -> pd.Series:
    return _group_transform(frame, values, group_col=group_col, func=lambda s: s.rolling(int(window)).max())


def ts_var(frame: pd.DataFrame, values: pd.Series, window: int, *, group_col: str = "symbol") -> pd.Series:
    return _group_transform(frame, values, group_col=group_col, func=lambda s: s.rolling(int(window)).var(ddof=0))


def ts_mean(frame: pd.DataFrame, values: pd.Series, window: int, *, group_col: str = "symbol") -> pd.Series:
    return _group_transform(frame, values, group_col=group_col, func=lambda s: s.rolling(int(window)).mean())


def ts_median(frame: pd.DataFrame, values: pd.Series, window: int, *, group_col: str = "symbol") -> pd.Series:
    return _group_transform(frame, values, group_col=group_col, func=lambda s: s.rolling(int(window)).median())


def ts_std(frame: pd.DataFrame, values: pd.Series, window: int, *, group_col: str = "symbol") -> pd.Series:
    return _group_transform(frame, values, group_col=group_col, func=lambda s: s.rolling(int(window)).std(ddof=0))


def ts_ema(frame: pd.DataFrame, values: pd.Series, span: int, *, group_col: str = "symbol") -> pd.Series:
    return _group_transform(frame, values, group_col=group_col, func=lambda s: s.ewm(span=int(span), adjust=False).mean())


def ts_wma(frame: pd.DataFrame, values: pd.Series, window: int, *, group_col: str = "symbol") -> pd.Series:
    weights = np.arange(1, int(window) + 1, dtype=float)

    def _weighted(series: pd.Series) -> pd.Series:
        def _apply(arr: np.ndarray) -> float:
            local_weights = weights[-len(arr) :]
            return float(np.dot(arr, local_weights) / local_weights.sum())

        return series.rolling(int(window)).apply(_apply, raw=True)

    return _group_transform(frame, values, group_col=group_col, func=_weighted)


def ts_pctchange(frame: pd.DataFrame, values: pd.Series, periods: int, *, group_col: str = "symbol") -> pd.Series:
    return _group_transform(frame, values, group_col=group_col, func=lambda s: s.ffill().pct_change(int(periods), fill_method=None))


def ts_delta(frame: pd.DataFrame, values: pd.Series, periods: int, *, group_col: str = "symbol") -> pd.Series:
    return _group_transform(frame, values, group_col=group_col, func=lambda s: s - s.shift(int(periods)))


def ts_quantile(
    frame: pd.DataFrame,
    values: pd.Series,
    window: int,
    percentile: float,
    *,
    group_col: str = "symbol",
) -> pd.Series:
    q = float(percentile) / 100.0
    return _group_transform(frame, values, group_col=group_col, func=lambda s: s.rolling(int(window)).quantile(q))


def ts_rank_pct(frame: pd.DataFrame, values: pd.Series, window: int, *, group_col: str = "symbol") -> pd.Series:
    def _rank_last(arr: np.ndarray) -> float:
        ranked = pd.Series(arr).rank(pct=True)
        return float(ranked.iloc[-1])

    return _group_transform(frame, values, group_col=group_col, func=lambda s: s.rolling(int(window)).apply(_rank_last, raw=True))


def ts_av_diff(frame: pd.DataFrame, values: pd.Series, window: int, *, group_col: str = "symbol") -> pd.Series:
    return _group_transform(frame, values, group_col=group_col, func=lambda s: s - s.rolling(int(window)).mean())


def ts_returns(frame: pd.DataFrame, values: pd.Series, periods: int, *, mode: int = 1, group_col: str = "symbol") -> pd.Series:
    previous = delay(frame, values, periods, group_col=group_col).replace(0, np.nan)
    numeric = pd.to_numeric(values, errors="coerce")
    if mode == 1:
        return numeric / previous - 1.0
    if mode == 0:
        return np.log(numeric / previous)
    raise ValueError("Unsupported return mode; expected 0 or 1.")


def ts_decay_linear(frame: pd.DataFrame, values: pd.Series, window: int, *, group_col: str = "symbol") -> pd.Series:
    weights = np.arange(1, int(window) + 1, dtype=float)

    def _weighted(series: pd.Series) -> pd.Series:
        def _apply(arr: np.ndarray) -> float:
            local_weights = weights[-len(arr) :]
            return float(np.dot(arr, local_weights) / local_weights.sum())

        return series.rolling(int(window)).apply(_apply, raw=True)

    return _group_transform(frame, values, group_col=group_col, func=_weighted)


def ts_corr(frame: pd.DataFrame, left: pd.Series, right: pd.Series, window: int, *, group_col: str = "symbol") -> pd.Series:
    result = pd.Series(index=frame.index, dtype=float)
    for _, idx in frame.groupby(group_col, sort=False).groups.items():
        x = pd.to_numeric(left.loc[idx], errors="coerce")
        y = pd.to_numeric(right.loc[idx], errors="coerce")
        result.loc[idx] = x.rolling(int(window)).corr(y)
    result.index = left.index
    return result


def ts_cov(frame: pd.DataFrame, left: pd.Series, right: pd.Series, window: int, *, group_col: str = "symbol") -> pd.Series:
    result = pd.Series(index=frame.index, dtype=float)
    for _, idx in frame.groupby(group_col, sort=False).groups.items():
        x = pd.to_numeric(left.loc[idx], errors="coerce")
        y = pd.to_numeric(right.loc[idx], errors="coerce")
        result.loc[idx] = x.rolling(int(window)).cov(y)
    result.index = left.index
    return result


def cs_rank(frame: pd.DataFrame, values: pd.Series, *, date_col: str = "date", pct: bool = True) -> pd.Series:
    return values.groupby(frame[date_col], sort=False).rank(pct=pct)


def cs_skew(frame: pd.DataFrame, values: pd.Series, *, date_col: str = "date") -> pd.Series:
    temp = pd.DataFrame({"date": frame[date_col], "value": pd.to_numeric(values, errors="coerce")})
    skew = temp.groupby("date", sort=False)["value"].transform(
        lambda s: float(s.skew()) if s.dropna().size >= 3 else np.nan
    )
    skew.index = values.index
    return skew


def cs_standardize(frame: pd.DataFrame, values: pd.Series, *, date_col: str = "date", eps: float = 1e-8) -> pd.Series:
    temp = pd.DataFrame({"date": frame[date_col], "value": pd.to_numeric(values, errors="coerce")})

    def _standardize(group: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(group, errors="coerce")
        std = float(numeric.std(ddof=0))
        if not np.isfinite(std) or std <= eps:
            return pd.Series(0.0, index=group.index)
        centered = numeric - float(numeric.mean())
        return centered / std

    standardized = temp.groupby("date", sort=False)["value"].transform(_standardize)
    standardized.index = values.index
    return standardized


def rolling_volatility(frame: pd.DataFrame, returns: pd.Series, window: int, *, group_col: str = "symbol") -> pd.Series:
    return ts_std(frame, returns, window, group_col=group_col)


def rolling_sharpe(
    frame: pd.DataFrame,
    returns: pd.Series,
    window: int,
    *,
    group_col: str = "symbol",
    annualization: int = 252,
) -> pd.Series:
    mean = ts_mean(frame, returns, window, group_col=group_col)
    std = ts_std(frame, returns, window, group_col=group_col).replace(0, np.nan)
    sharpe = mean / std * np.sqrt(annualization)
    return sharpe.replace([np.inf, -np.inf], np.nan)


def ts_ir(frame: pd.DataFrame, values: pd.Series, window: int, *, group_col: str = "symbol") -> pd.Series:
    mean = ts_mean(frame, values, window, group_col=group_col)
    std = ts_std(frame, values, window, group_col=group_col).replace(0, np.nan)
    ir = mean / std
    return ir.replace([np.inf, -np.inf], np.nan)


def signed_power(values: pd.Series, exponent: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    powered = np.sign(numeric) * np.power(np.abs(numeric), float(exponent))
    return pd.Series(powered, index=values.index).replace([np.inf, -np.inf], np.nan)


def cs_regression_residual(frame: pd.DataFrame, y: pd.Series, x: pd.Series, *, date_col: str = "date") -> pd.Series:
    residual = pd.Series(np.nan, index=frame.index, dtype=float)

    temp = pd.DataFrame({"date": frame[date_col], "y": pd.to_numeric(y, errors="coerce"), "x": pd.to_numeric(x, errors="coerce")})
    for _, idx in temp.groupby("date", sort=False).groups.items():
        group = temp.loc[idx, ["y", "x"]].dropna()
        if group.empty:
            continue
        x_vals = group["x"].to_numpy(dtype=float)
        y_vals = group["y"].to_numpy(dtype=float)
        x_matrix = np.column_stack([np.ones(len(x_vals), dtype=float), x_vals])
        if np.linalg.matrix_rank(x_matrix) < 2:
            residual.loc[group.index] = 0.0
            continue
        beta, *_ = np.linalg.lstsq(x_matrix, y_vals, rcond=None)
        fitted = x_matrix @ beta
        residual.loc[group.index] = y_vals - fitted

    residual.index = y.index
    return residual


def rolling_max_drawdown(frame: pd.DataFrame, values: pd.Series, window: int, *, group_col: str = "symbol") -> pd.Series:
    def _max_drawdown(arr: np.ndarray) -> float:
        numeric = pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(dtype=float)
        valid = numeric[np.isfinite(numeric)]
        if valid.size == 0:
            return np.nan
        running_max = np.maximum.accumulate(valid)
        running_max = np.where(running_max == 0.0, np.nan, running_max)
        drawdown = valid / running_max - 1.0
        if np.isnan(drawdown).all():
            return np.nan
        return float(np.nanmin(drawdown))

    return _group_transform(frame, values, group_col=group_col, func=lambda s: s.rolling(int(window)).apply(_max_drawdown, raw=True))
