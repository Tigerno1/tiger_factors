from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd


def _to_numeric(values: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return values.apply(pd.to_numeric, errors="coerce")


def _apply_serieswise(
    values: pd.Series | pd.DataFrame,
    func,
) -> pd.Series | pd.DataFrame:
    numeric = _to_numeric(values)
    if isinstance(numeric, pd.Series):
        return func(numeric)
    return numeric.apply(func, axis=0)


def lag(values: pd.Series | pd.DataFrame, periods: int = 1):
    return _to_numeric(values).shift(periods)


def ts_delay(values: pd.Series | pd.DataFrame, d: int):
    return lag(values, periods=d)


def diff(values: pd.Series | pd.DataFrame, periods: int = 1):
    return _to_numeric(values).diff(periods)


def ts_delta(values: pd.Series | pd.DataFrame, d: int):
    return diff(values, periods=d)


def pct_change(values: pd.Series | pd.DataFrame, periods: int = 1):
    return _to_numeric(values).pct_change(periods)


def log_return(values: pd.Series | pd.DataFrame, periods: int = 1):
    numeric = _to_numeric(values)
    return np.log(numeric / numeric.shift(periods))


def cumulative_return(returns: pd.Series | pd.DataFrame):
    numeric = _to_numeric(returns)
    return (1.0 + numeric).cumprod() - 1.0


def rolling_mean(values: pd.Series | pd.DataFrame, window: int, min_periods: int | None = None):
    return _to_numeric(values).rolling(window, min_periods=min_periods).mean()


def ts_mean(values: pd.Series | pd.DataFrame, d: int):
    return rolling_mean(values, window=d)


def rolling_std(values: pd.Series | pd.DataFrame, window: int, min_periods: int | None = None, ddof: int = 0):
    return _to_numeric(values).rolling(window, min_periods=min_periods).std(ddof=ddof)


def ts_std_dev(values: pd.Series | pd.DataFrame, d: int):
    return rolling_std(values, window=d, ddof=1)


def ts_var(values: pd.Series | pd.DataFrame, d: int):
    return _to_numeric(values).rolling(d).var(ddof=0)


def rolling_sum(values: pd.Series | pd.DataFrame, window: int, min_periods: int | None = None):
    return _to_numeric(values).rolling(window, min_periods=min_periods).sum()


def ts_sum(values: pd.Series | pd.DataFrame, d: int):
    return rolling_sum(values, window=d)


def ts_wma(values: pd.Series | pd.DataFrame, d: int):
    weights = np.arange(1, int(d) + 1, dtype=float)

    def _weighted(series: pd.Series) -> pd.Series:
        window = len(weights)

        def _apply(arr: np.ndarray) -> float:
            local_weights = weights[-len(arr) :]
            return float(np.dot(arr, local_weights) / local_weights.sum())

        return series.rolling(window).apply(_apply, raw=True)

    return _apply_serieswise(values, _weighted)


def ts_ema(values: pd.Series | pd.DataFrame, d: int):
    return ewm_mean(values, span=d)


def ts_pctchange(values: pd.Series | pd.DataFrame, d: int):
    return _to_numeric(values).pct_change(d)


def rolling_min(values: pd.Series | pd.DataFrame, window: int, min_periods: int | None = None):
    return _to_numeric(values).rolling(window, min_periods=min_periods).min()


def ts_min(values: pd.Series | pd.DataFrame, d: int):
    return rolling_min(values, window=d)


def rolling_max(values: pd.Series | pd.DataFrame, window: int, min_periods: int | None = None):
    return _to_numeric(values).rolling(window, min_periods=min_periods).max()


def ts_max(values: pd.Series | pd.DataFrame, d: int):
    return rolling_max(values, window=d)


def rolling_median(values: pd.Series | pd.DataFrame, window: int, min_periods: int | None = None):
    return _to_numeric(values).rolling(window, min_periods=min_periods).median()


def rolling_skew(values: pd.Series | pd.DataFrame, window: int, min_periods: int | None = None):
    return _to_numeric(values).rolling(window, min_periods=min_periods).skew()


def rolling_kurt(values: pd.Series | pd.DataFrame, window: int, min_periods: int | None = None):
    return _to_numeric(values).rolling(window, min_periods=min_periods).kurt()


def rolling_zscore(values: pd.Series | pd.DataFrame, window: int, min_periods: int | None = None, ddof: int = 0):
    numeric = _to_numeric(values)
    mu = numeric.rolling(window, min_periods=min_periods).mean()
    sigma = numeric.rolling(window, min_periods=min_periods).std(ddof=ddof)
    out = (numeric - mu) / sigma
    return out.replace([np.inf, -np.inf], np.nan)


def ts_zscore(values: pd.Series | pd.DataFrame, d: int):
    return rolling_zscore(values, window=d, ddof=1)


def expanding_zscore(values: pd.Series | pd.DataFrame, min_periods: int = 20, ddof: int = 0):
    numeric = _to_numeric(values)
    mu = numeric.expanding(min_periods=min_periods).mean()
    sigma = numeric.expanding(min_periods=min_periods).std(ddof=ddof)
    out = (numeric - mu) / sigma
    return out.replace([np.inf, -np.inf], np.nan)


def ewm_mean(values: pd.Series | pd.DataFrame, span: int | None = None, halflife: float | None = None, alpha: float | None = None):
    return _to_numeric(values).ewm(span=span, halflife=halflife, alpha=alpha, adjust=False).mean()


def ewm_std(values: pd.Series | pd.DataFrame, span: int | None = None, halflife: float | None = None, alpha: float | None = None):
    return _to_numeric(values).ewm(span=span, halflife=halflife, alpha=alpha, adjust=False).std()


def rolling_rank_pct(values: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Percentile rank of the latest value within each rolling window."""
    numeric = pd.to_numeric(values, errors="coerce")

    def _last_rank(arr: np.ndarray) -> float:
        s = pd.Series(arr)
        return float(s.rank(pct=True).iloc[-1])

    return numeric.rolling(window, min_periods=min_periods).apply(_last_rank, raw=True)


def ts_rank(values: pd.Series | pd.DataFrame, d: int, constant: float = 0):
    def _rank_last(series: pd.Series) -> pd.Series:
        return series.rolling(d).apply(lambda arr: float(pd.Series(arr).rank().iloc[-1] + constant), raw=True)

    return _apply_serieswise(values, _rank_last)


def rolling_corr(x: pd.Series, y: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").rolling(window, min_periods=min_periods).corr(pd.to_numeric(y, errors="coerce"))


def ts_corr(x: pd.Series, y: pd.Series, d: int):
    return rolling_corr(x, y, window=d)


def ts_covariance(y: pd.Series, x: pd.Series, d: int):
    x_numeric = pd.to_numeric(x, errors="coerce")
    y_numeric = pd.to_numeric(y, errors="coerce")
    return x_numeric.rolling(d).cov(y_numeric)


def rolling_beta(y: pd.Series, x: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    yv = pd.to_numeric(y, errors="coerce")
    xv = pd.to_numeric(x, errors="coerce")
    cov = yv.rolling(window, min_periods=min_periods).cov(xv)
    var = xv.rolling(window, min_periods=min_periods).var()
    out = cov / var
    return out.replace([np.inf, -np.inf], np.nan)


def volatility(returns: pd.Series | pd.DataFrame, window: int, annualization: int = 252, min_periods: int | None = None):
    return rolling_std(returns, window=window, min_periods=min_periods) * np.sqrt(annualization)


def rolling_sharpe(
    returns: pd.Series,
    window: int,
    risk_free_rate: float = 0.0,
    annualization: int = 252,
    min_periods: int | None = None,
) -> pd.Series:
    r = pd.to_numeric(returns, errors="coerce") - (risk_free_rate / annualization)
    mu = r.rolling(window, min_periods=min_periods).mean()
    sigma = r.rolling(window, min_periods=min_periods).std(ddof=0)
    out = (mu / sigma) * np.sqrt(annualization)
    return out.replace([np.inf, -np.inf], np.nan)


def rolling_information_ratio(active_returns: pd.Series, window: int, annualization: int = 252, min_periods: int | None = None) -> pd.Series:
    r = pd.to_numeric(active_returns, errors="coerce")
    mu = r.rolling(window, min_periods=min_periods).mean()
    sigma = r.rolling(window, min_periods=min_periods).std(ddof=0)
    out = (mu / sigma) * np.sqrt(annualization)
    return out.replace([np.inf, -np.inf], np.nan)


def drawdown(cumulative_curve: pd.Series | pd.DataFrame):
    curve = _to_numeric(cumulative_curve)
    running_max = curve.cummax()
    out = curve / running_max - 1.0
    return out.replace([np.inf, -np.inf], np.nan)


def max_drawdown(cumulative_curve: pd.Series | pd.DataFrame):
    dd = drawdown(cumulative_curve)
    if isinstance(dd, pd.Series):
        return float(dd.min())
    return dd.min()


def decay_linear(values: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    weights = np.arange(1, window + 1, dtype=float)
    denom = weights.sum()

    def _weighted(arr: np.ndarray) -> float:
        w = weights[-len(arr):]
        return float(np.dot(arr, w) / w.sum())

    return numeric.rolling(window, min_periods=min_periods).apply(_weighted, raw=True)


def decay_exponential(values: pd.Series, halflife: float) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").ewm(halflife=halflife, adjust=False).mean()


def ts_momentum(values: pd.Series, long_window: int = 252, short_window: int = 21) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if short_window <= 0:
        return numeric.pct_change(long_window)
    return numeric.pct_change(long_window) - numeric.pct_change(short_window)


def ts_backfill(values: pd.Series | pd.DataFrame, lookback: int, k: int = 1, ignore: str = "NAN"):
    del lookback, ignore
    return _to_numeric(values).bfill(limit=k)


def ts_quantile(values: pd.Series | pd.DataFrame, d: int, driver: str = "gaussian"):
    normal = NormalDist()

    def _quantile_last(series: pd.Series) -> pd.Series:
        def _convert(arr: np.ndarray) -> float:
            rank = float(pd.Series(arr).rank(pct=True).iloc[-1])
            if driver == "gaussian":
                rank = min(max(rank, 1e-12), 1 - 1e-12)
                return float(normal.inv_cdf(rank))
            return rank

        return series.rolling(d).apply(_convert, raw=True)

    return _apply_serieswise(values, _quantile_last)


def ts_product(values: pd.Series | pd.DataFrame, d: int):
    def _rolling_product(series: pd.Series) -> pd.Series:
        return series.rolling(d).apply(np.prod, raw=True)

    return _apply_serieswise(values, _rolling_product)


def ts_av_diff(values: pd.Series | pd.DataFrame, d: int):
    numeric = _to_numeric(values)
    return numeric - rolling_mean(numeric, window=d)


def ts_scale(values: pd.Series | pd.DataFrame, d: int, constant: float = 0):
    numeric = _to_numeric(values)
    denom = numeric.abs().rolling(d).sum()
    out = numeric * d / (denom + constant)
    return out.replace([np.inf, -np.inf], np.nan)


def ts_arg_max(values: pd.Series | pd.DataFrame, d: int):
    def _argmax(series: pd.Series) -> pd.Series:
        return series.rolling(d).apply(lambda arr: float(d - int(np.argmax(arr)) - 1), raw=True)

    return _apply_serieswise(values, _argmax)


def ts_arg_min(values: pd.Series | pd.DataFrame, d: int):
    def _argmin(series: pd.Series) -> pd.Series:
        return series.rolling(d).apply(lambda arr: float(d - int(np.argmin(arr)) - 1), raw=True)

    return _apply_serieswise(values, _argmin)


def ts_winsorize(values: pd.Series | pd.DataFrame, low: float = 0.01, high: float = 0.99, d: int | None = None):
    def _winsorize(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        if d is None:
            lo = numeric.quantile(low)
            hi = numeric.quantile(high)
        else:
            lo = numeric.rolling(d).quantile(low)
            hi = numeric.rolling(d).quantile(high)
        return numeric.clip(lower=lo, upper=hi)

    return _apply_serieswise(values, _winsorize)


def ts_winsorize_mad(values: pd.Series | pd.DataFrame, k: float = 5.0, d: int | None = None):
    def _winsorize_mad(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        if d is None:
            med = numeric.median()
            mad = (numeric - med).abs().median()
            band = 1.4826 * mad
        else:
            med = numeric.rolling(d).median()
            mad = (numeric - med).abs().rolling(d).median()
            band = 1.4826 * mad
        return numeric.clip(lower=med - k * band, upper=med + k * band)

    return _apply_serieswise(values, _winsorize_mad)
