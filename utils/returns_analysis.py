from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


ANNUALIZATION_DEFAULT = 252


def clean_returns(values: pd.Series | pd.Index | list[Any] | np.ndarray) -> pd.Series:
    series = pd.Series(values, copy=False)
    series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index, errors="coerce")
    series = series.loc[~series.index.isna()].sort_index()
    return series.dropna()


def align_returns(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None,
) -> tuple[pd.Series, pd.Series | None]:
    portfolio = clean_returns(portfolio_returns)
    benchmark = clean_returns(benchmark_returns) if benchmark_returns is not None else None
    if benchmark is None:
        return portfolio, None
    common_index = portfolio.index.intersection(benchmark.index)
    return portfolio.loc[common_index], benchmark.loc[common_index]


def cumulative_returns(returns: pd.Series) -> pd.Series:
    clean = clean_returns(returns).fillna(0.0)
    cumulative = (1.0 + clean).cumprod() - 1.0
    cumulative.index = pd.DatetimeIndex(cumulative.index, name="date_")
    return cumulative


def annualized_return(returns: pd.Series, annualization: int = ANNUALIZATION_DEFAULT) -> float:
    clean = clean_returns(returns)
    if clean.empty:
        return float("nan")
    growth = float((1.0 + clean.fillna(0.0)).prod())
    periods = float(clean.shape[0])
    if growth <= 0.0 or periods <= 0.0:
        return float("nan")
    return growth ** (annualization / periods) - 1.0


def drawdown_series(returns: pd.Series) -> pd.Series:
    cumulative = (1.0 + clean_returns(returns).fillna(0.0)).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1.0
    drawdown.index = pd.DatetimeIndex(drawdown.index, name="date_")
    return drawdown.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def max_drawdown(returns: pd.Series) -> float:
    drawdown = drawdown_series(returns)
    if drawdown.empty:
        return float("nan")
    return float(drawdown.min())


def win_rate(returns: pd.Series) -> float:
    clean = clean_returns(returns)
    if clean.empty:
        return float("nan")
    return float((clean > 0.0).mean())


def sharpe_ratio(returns: pd.Series, annualization: int = ANNUALIZATION_DEFAULT) -> float:
    clean = clean_returns(returns)
    if clean.empty:
        return float("nan")
    vol = float(clean.std(ddof=0))
    if not np.isfinite(vol) or vol == 0.0:
        return float("nan")
    return float(clean.mean() / vol * np.sqrt(annualization))


def annualized_sortino(returns: pd.Series, annualization: int = ANNUALIZATION_DEFAULT) -> float:
    clean = clean_returns(returns)
    if clean.empty:
        return float("nan")
    downside = clean[clean < 0.0]
    downside_dev = float(np.sqrt(np.mean(np.square(downside)))) if not downside.empty else 0.0
    if downside_dev == 0.0 or not np.isfinite(downside_dev):
        return float("nan")
    return float(clean.mean() / downside_dev * np.sqrt(annualization))


def annualized_sharpe(returns: pd.Series, annualization: int = ANNUALIZATION_DEFAULT) -> pd.Series:
    clean = clean_returns(returns).fillna(0.0)
    rolling_mean = clean.rolling(annualization).mean()
    rolling_std = clean.rolling(annualization).std(ddof=0)
    sharpe = (rolling_mean / rolling_std) * np.sqrt(annualization)
    return sharpe.replace([np.inf, -np.inf], np.nan)


def rolling_mean_return(
    returns: pd.Series,
    window: int,
    *,
    min_periods: int | None = None,
) -> pd.Series:
    clean = clean_returns(returns).fillna(0.0)
    mean = clean.rolling(window, min_periods=min_periods).mean()
    return mean.replace([np.inf, -np.inf], np.nan)


def annualized_volatility(returns: pd.Series, annualization: int = ANNUALIZATION_DEFAULT) -> pd.Series:
    clean = clean_returns(returns).fillna(0.0)
    vol = clean.rolling(annualization).std(ddof=0) * np.sqrt(annualization)
    return vol.replace([np.inf, -np.inf], np.nan)


def annualized_volatility_value(returns: pd.Series, annualization: int = ANNUALIZATION_DEFAULT) -> float:
    clean = clean_returns(returns)
    if clean.empty:
        return float("nan")
    return float(clean.std(ddof=0) * np.sqrt(annualization))


def rolling_volatility(
    returns: pd.Series,
    window: int,
    *,
    annualization: int = ANNUALIZATION_DEFAULT,
    min_periods: int | None = None,
) -> pd.Series:
    clean = clean_returns(returns).fillna(0.0)
    vol = clean.rolling(window, min_periods=min_periods).std(ddof=0) * np.sqrt(annualization)
    return vol.replace([np.inf, -np.inf], np.nan)


def rolling_sharpe(
    returns: pd.Series,
    window: int,
    *,
    rf: float = 0.0,
    annualization: int = ANNUALIZATION_DEFAULT,
    min_periods: int | None = None,
) -> pd.Series:
    clean = clean_returns(returns).fillna(0.0) - rf / annualization
    mu = clean.rolling(window, min_periods=min_periods).mean()
    sigma = clean.rolling(window, min_periods=min_periods).std(ddof=0)
    out = (mu / sigma) * np.sqrt(annualization)
    return out.replace([np.inf, -np.inf], np.nan)


def rolling_sortino(
    returns: pd.Series,
    window: int,
    *,
    rf: float = 0.0,
    annualization: int = ANNUALIZATION_DEFAULT,
    min_periods: int | None = None,
) -> pd.Series:
    clean = clean_returns(returns).fillna(0.0) - rf / annualization
    mu = clean.rolling(window, min_periods=min_periods).mean()
    downside = clean.rolling(window, min_periods=min_periods).apply(
        lambda x: float(np.sqrt(np.sum(np.square(x[x < 0])) / len(x))) if len(x) else np.nan,
        raw=False,
    )
    out = (mu / downside.replace(0.0, np.nan)) * np.sqrt(annualization)
    return out.replace([np.inf, -np.inf], np.nan)


def rolling_beta(
    returns: pd.Series,
    benchmark: pd.Series,
    window: int,
    *,
    min_periods: int | None = None,
) -> pd.Series:
    portfolio, benchmark = align_returns(returns, benchmark)
    if benchmark is None or portfolio.empty:
        return pd.Series(dtype=float)
    portfolio = portfolio.fillna(0.0)
    benchmark = benchmark.fillna(0.0)
    cov = portfolio.rolling(window, min_periods=min_periods).cov(benchmark)
    var = benchmark.rolling(window, min_periods=min_periods).var()
    beta = cov / var.replace(0.0, np.nan)
    return beta.replace([np.inf, -np.inf], np.nan)


def monthly_returns_heatmap(returns: pd.Series) -> pd.DataFrame:
    clean = clean_returns(returns)
    if clean.empty:
        return pd.DataFrame()
    monthly = clean.resample("ME").apply(lambda x: float((1.0 + x.fillna(0.0)).prod() - 1.0))
    table = monthly.to_frame("return")
    table["year"] = table.index.year
    table["month"] = table.index.month
    heatmap = table.pivot(index="year", columns="month", values="return").sort_index()
    heatmap.columns = [pd.Timestamp(2000, int(col), 1).strftime("%b") for col in heatmap.columns]
    return heatmap


__all__ = [
    "ANNUALIZATION_DEFAULT",
    "align_returns",
    "annualized_return",
    "annualized_sharpe",
    "annualized_volatility",
    "annualized_volatility_value",
    "annualized_sortino",
    "clean_returns",
    "cumulative_returns",
    "drawdown_series",
    "monthly_returns_heatmap",
    "max_drawdown",
    "rolling_mean_return",
    "rolling_beta",
    "rolling_sharpe",
    "rolling_sortino",
    "rolling_volatility",
    "sharpe_ratio",
    "win_rate",
]
