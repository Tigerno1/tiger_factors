from __future__ import annotations

import json
import math
import warnings
import tempfile
import webbrowser
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from tiger_factors.multifactor_evaluation.common.matplotlib_config import configure_matplotlib
from tiger_factors.multifactor_evaluation.common.parquet_utils import to_parquet_clean
from tiger_factors.multifactor_evaluation.common.plotting import finalize_quantstats_axis
from tiger_factors.multifactor_evaluation.common.plotting import save_quantstats_figure
from tiger_factors.multifactor_evaluation.common.results import TearSheetResultMixin
from tiger_factors.multifactor_evaluation.reporting.html_report import render_summary_report_html
from tiger_factors.utils.returns_analysis import align_returns as shared_align_returns
from tiger_factors.utils.returns_analysis import annualized_return as shared_annualized_return
from tiger_factors.utils.returns_analysis import max_drawdown as shared_max_drawdown
from tiger_factors.utils.returns_analysis import sharpe_ratio as shared_sharpe_ratio
from tiger_factors.utils.returns_analysis import annualized_volatility_value as shared_annualized_volatility_value
from tiger_factors.utils.returns_analysis import win_rate as shared_win_rate
from tiger_factors.utils.returns_analysis import clean_returns as shared_clean_returns
from tiger_factors.utils.returns_analysis import cumulative_returns as shared_cumulative_returns
from tiger_factors.utils.returns_analysis import drawdown_series as shared_drawdown_series
from tiger_factors.utils.returns_analysis import monthly_returns_heatmap as shared_monthly_returns_heatmap
from tiger_factors.utils.returns_analysis import rolling_beta as shared_rolling_beta
from tiger_factors.utils.returns_analysis import rolling_sharpe as shared_rolling_sharpe
from tiger_factors.utils.returns_analysis import rolling_sortino as shared_rolling_sortino
from tiger_factors.utils.returns_analysis import rolling_volatility as shared_rolling_volatility

configure_matplotlib()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter


ANNUALIZATION_DEFAULT = 252

_RESAMPLE_ALIAS_MAP = {
    "M": "ME",
    "Q": "QE",
    "A": "YE",
    "Y": "YE",
}


@dataclass(frozen=True)
class MultifactorSummaryReportResult(TearSheetResultMixin):
    output_dir: Path
    report_name: str
    html_path: Path
    summary_table_path: Path
    figure_paths: list[Path]
    portfolio_returns_path: Path | None
    benchmark_returns_path: Path | None
    manifest_path: Path
    comparison_table_path: Path | None = None
    drawdown_table_path: Path | None = None
    monthly_returns_table_path: Path | None = None
    montecarlo_summary_path: Path | None = None
    montecarlo_plot_path: Path | None = None
    compare_table_paths: dict[str, Path] | None = None

    def _available_table_paths(self) -> dict[str, Path]:
        paths: dict[str, Path] = {
            "summary": self.summary_table_path,
        }
        if self.comparison_table_path is not None:
            paths["comparison"] = self.comparison_table_path
        if self.drawdown_table_path is not None:
            paths["drawdown"] = self.drawdown_table_path
        if self.monthly_returns_table_path is not None:
            paths["monthly_returns"] = self.monthly_returns_table_path
        if self.montecarlo_summary_path is not None:
            paths["montecarlo"] = self.montecarlo_summary_path
        if self.compare_table_paths:
            paths.update(self.compare_table_paths)
            paths.update({f"compare_{name}": path for name, path in self.compare_table_paths.items()})
        return paths

    def report(self) -> str | None:
        return self.html_path.stem if self.html_path.exists() else None

    def preferred_table_order(self) -> list[str]:
        return [
            "summary",
            "comparison",
            "drawdown",
            "monthly_returns",
            "montecarlo",
        ]

    def get_report(self, *, open_browser: bool = True) -> Path:
        if not self.html_path.exists():
            raise FileNotFoundError(f"report not found: {self.html_path}")
        if open_browser:
            webbrowser.open(self.html_path.as_uri())
        return self.html_path

    def to_summary(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "report_name": self.report_name,
            "html_path": str(self.html_path),
            "summary_table_path": str(self.summary_table_path),
            "figure_paths": [str(path) for path in self.figure_paths],
            "portfolio_returns_path": str(self.portfolio_returns_path) if self.portfolio_returns_path else None,
            "benchmark_returns_path": str(self.benchmark_returns_path) if self.benchmark_returns_path else None,
            "manifest_path": str(self.manifest_path),
            "comparison_table_path": str(self.comparison_table_path) if self.comparison_table_path else None,
            "drawdown_table_path": str(self.drawdown_table_path) if self.drawdown_table_path else None,
            "monthly_returns_table_path": str(self.monthly_returns_table_path) if self.monthly_returns_table_path else None,
            "montecarlo_summary_path": str(self.montecarlo_summary_path) if self.montecarlo_summary_path else None,
            "montecarlo_plot_path": str(self.montecarlo_plot_path) if self.montecarlo_plot_path else None,
            "compare_table_paths": {k: str(v) for k, v in self.compare_table_paths.items()} if self.compare_table_paths else None,
        }

    def _table_paths(self) -> dict[str, Path]:
        return self._available_table_paths()


def _clean_series(values: pd.Series | pd.Index | list[Any] | np.ndarray) -> pd.Series:
    return shared_clean_returns(values)


def _as_series(values: pd.Series | pd.DataFrame | pd.Index | list[Any] | np.ndarray) -> pd.Series:
    if isinstance(values, pd.DataFrame):
        if values.empty:
            return pd.Series(dtype=float)
        values = values.iloc[:, 0]
    if isinstance(values, pd.Series):
        return values
    return pd.Series(values)


def _align_series(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None,
) -> tuple[pd.Series, pd.Series | None]:
    return shared_align_returns(portfolio_returns, benchmark_returns)


def _growth_curve(returns: pd.Series) -> pd.Series:
    return shared_cumulative_returns(returns) + 1.0


def _drawdown_series(returns: pd.Series) -> pd.Series:
    return shared_drawdown_series(returns)


def _normalize_resample_period(period: str | None) -> str | None:
    if period is None:
        return None
    return _RESAMPLE_ALIAS_MAP.get(period, period)


def _aggregate_returns(returns: pd.Series, aggregate: str | None, compounded: bool = True) -> pd.Series:
    clean = _clean_series(returns).fillna(0.0)
    if clean.empty:
        return pd.Series(dtype=float)
    aggregate = _normalize_resample_period(aggregate)
    if aggregate is None:
        return clean
    if compounded:
        return clean.resample(aggregate).apply(lambda x: float((1.0 + x.fillna(0.0)).prod() - 1.0))
    return clean.resample(aggregate).sum()


def _comp(returns: pd.Series) -> float:
    clean = _clean_series(returns).fillna(0.0)
    if clean.empty:
        return float("nan")
    return float((1.0 + clean).prod() - 1.0)


def comp(returns: pd.Series) -> float:
    return _comp(returns)


def compsum(returns: pd.Series) -> pd.Series:
    """Cumulative compounded returns series."""
    return _growth_curve(returns) - 1.0


def to_drawdown_series(returns: pd.Series) -> pd.Series:
    """Public alias matching quantstats.stats."""
    return _drawdown_series(returns)


def expected_shortfall(returns: pd.Series, sigma: float = 1.0, confidence: float = 0.95) -> float:
    return _conditional_value_at_risk(returns, sigma=sigma, confidence=confidence)


def outliers(
    returns: pd.Series | pd.DataFrame,
    quantile: float = 0.95,
    sigma: float | None = None,
) -> pd.Series:
    clean = _clean_series(returns.iloc[:, 0] if isinstance(returns, pd.DataFrame) else returns)
    if clean.empty:
        return pd.Series(dtype=float)
    if sigma is not None:
        threshold = sigma * clean.std(ddof=1)
        if not np.isfinite(threshold) or threshold == 0:
            return pd.Series(dtype=float)
        mean = clean.mean()
        return clean.loc[(clean - mean).abs() > threshold]
    return clean.loc[clean > clean.quantile(quantile)].dropna()


def remove_outliers(
    returns: pd.Series | pd.DataFrame,
    quantile: float = 0.95,
    sigma: float | None = None,
) -> pd.Series:
    clean = _clean_series(returns.iloc[:, 0] if isinstance(returns, pd.DataFrame) else returns)
    if clean.empty:
        return clean
    if sigma is not None:
        threshold = sigma * clean.std(ddof=1)
        if not np.isfinite(threshold) or threshold == 0:
            return clean
        mean = clean.mean()
        return clean.loc[(clean - mean).abs() <= threshold]
    return clean.loc[clean < clean.quantile(quantile)].dropna()


def outlier_win_ratio(returns: pd.Series, sigma: float = 3.0) -> float:
    clean = outliers(returns, sigma=sigma)
    if clean.empty:
        return float("nan")
    total = len(clean)
    if total == 0:
        return float("nan")
    return float((clean > 0).sum() / total)


def outlier_loss_ratio(returns: pd.Series, sigma: float = 3.0) -> float:
    clean = outliers(returns, sigma=sigma)
    if clean.empty:
        return float("nan")
    total = len(clean)
    if total == 0:
        return float("nan")
    return float((clean < 0).sum() / total)


def _probability_of_beating_threshold(
    returns: pd.Series,
    threshold: float = 0.0,
    annualization: int = ANNUALIZATION_DEFAULT,
    downside: bool = False,
) -> float:
    clean = _clean_series(returns).fillna(0.0)
    if clean.empty:
        return float("nan")
    observed = _sharpe(clean, rf=threshold, annualization=annualization)
    if downside:
        observed = _sortino(clean, rf=threshold, annualization=annualization)
    if not np.isfinite(observed):
        return float("nan")
    n = len(clean)
    if n < 2:
        return float("nan")
    skew = float(clean.skew())
    kurt = float(clean.kurt())
    denom = 1.0 - skew * observed + ((kurt - 1.0) / 4.0) * (observed ** 2)
    if denom <= 0:
        return float("nan")
    z = (observed * math.sqrt(n - 1.0)) / math.sqrt(denom)
    return float(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def probabilistic_sharpe_ratio(
    returns: pd.Series,
    rf: float = 0.0,
    periods: int = ANNUALIZATION_DEFAULT,
    annualize: bool = False,
    smart: bool = False,
) -> float:
    return probabilistic_ratio(returns, rf=rf, base="sharpe", periods=periods, annualize=annualize, smart=smart)


def probabilistic_sortino_ratio(
    returns: pd.Series,
    rf: float = 0.0,
    periods: int = ANNUALIZATION_DEFAULT,
    annualize: bool = False,
    smart: bool = False,
) -> float:
    return probabilistic_ratio(returns, rf=rf, base="sortino", periods=periods, annualize=annualize, smart=smart)


def probabilistic_adjusted_sortino_ratio(
    returns: pd.Series,
    rf: float = 0.0,
    periods: int = ANNUALIZATION_DEFAULT,
    annualize: bool = False,
    smart: bool = False,
) -> float:
    return probabilistic_ratio(
        returns,
        rf=rf,
        base="adjusted_sortino",
        periods=periods,
        annualize=annualize,
        smart=smart,
    )


def probabilistic_ratio(
    returns: pd.Series,
    rf: float = 0.0,
    base: str = "sharpe",
    periods: int = ANNUALIZATION_DEFAULT,
    annualize: bool = False,
    smart: bool = False,
) -> float:
    if annualize:
        rf = rf / periods
    if base.lower() == "sharpe":
        metric = _smart_sharpe(returns, rf=rf, annualization=periods) if smart else _sharpe(returns, rf=rf, annualization=periods)
    elif base.lower() == "sortino":
        metric = _smart_sortino(returns, rf=rf, annualization=periods) if smart else _sortino(returns, rf=rf, annualization=periods)
    elif base.lower() == "adjusted_sortino":
        metric = adjusted_sortino(returns, rf=rf, annualization=periods)
    else:
        raise ValueError("base must be one of: sharpe, sortino, adjusted_sortino")
    if not np.isfinite(metric):
        return float("nan")
    n = len(_clean_series(returns))
    if n < 2:
        return float("nan")
    skew = float(_clean_series(returns).skew())
    kurt = float(_clean_series(returns).kurt())
    denom = 1.0 - skew * metric + ((kurt - 1.0) / 4.0) * (metric**2)
    if denom <= 0:
        return float("nan")
    z = (metric * math.sqrt(n - 1.0)) / math.sqrt(denom)
    return float(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def safe_concat(
    objs: list[pd.Series | pd.DataFrame] | tuple[pd.Series | pd.DataFrame, ...] | pd.Series | pd.DataFrame,
    axis: int = 0,
    ignore_index: bool = False,
    sort: bool = False,
    **kwargs: Any,
) -> pd.Series | pd.DataFrame:
    if isinstance(objs, (pd.Series, pd.DataFrame)):
        objs = [objs]
    else:
        objs = list(objs)
    objs = [obj for obj in objs if obj is not None]
    if not objs:
        return pd.DataFrame()
    return pd.concat(objs, axis=axis, ignore_index=ignore_index, sort=sort, **kwargs)


def best(
    returns: pd.Series | pd.DataFrame,
    aggregate: str | None = None,
    compounded: bool = True,
    prepare_returns: bool = True,
) -> float:
    if prepare_returns:
        returns = _clean_series(_as_series(returns))
    return float(_aggregate_returns(returns, aggregate, compounded).max())


def worst(
    returns: pd.Series | pd.DataFrame,
    aggregate: str | None = None,
    compounded: bool = True,
    prepare_returns: bool = True,
) -> float:
    if prepare_returns:
        returns = _clean_series(_as_series(returns))
    return float(_aggregate_returns(returns, aggregate, compounded).min())


def consecutive_wins(
    returns: pd.Series | pd.DataFrame,
    aggregate: str | None = None,
    compounded: bool = True,
    prepare_returns: bool = True,
) -> int:
    if prepare_returns:
        returns = _clean_series(_as_series(returns))
    flags = _aggregate_returns(returns, aggregate, compounded) > 0
    return int(_consecutive_runs(flags, positive=True))


def consecutive_losses(
    returns: pd.Series | pd.DataFrame,
    aggregate: str | None = None,
    compounded: bool = True,
    prepare_returns: bool = True,
) -> int:
    if prepare_returns:
        returns = _clean_series(_as_series(returns))
    flags = _aggregate_returns(returns, aggregate, compounded) < 0
    return int(_consecutive_runs(flags, positive=True))


def skew(returns: pd.Series | pd.DataFrame, prepare_returns: bool = True) -> float:
    if prepare_returns:
        returns = _clean_series(_as_series(returns))
    clean = pd.Series(returns)
    return float(clean.skew())


def kurtosis(returns: pd.Series | pd.DataFrame, prepare_returns: bool = True) -> float:
    if prepare_returns:
        returns = _clean_series(_as_series(returns))
    clean = pd.Series(returns)
    return float(clean.kurtosis())


def implied_volatility(
    returns: pd.Series | pd.DataFrame,
    periods: int = ANNUALIZATION_DEFAULT,
    annualize: bool = True,
) -> float | pd.Series:
    if isinstance(returns, pd.DataFrame):
        return returns.apply(lambda col: implied_volatility(col, periods=periods, annualize=annualize))
    clean = _clean_series(_as_series(returns))
    if clean.empty:
        return float("nan")
    logret = np.log1p(clean)
    if annualize:
        return logret.rolling(periods).std(ddof=1) * np.sqrt(periods)
    return float(logret.std(ddof=1))


def ror(returns: pd.Series | pd.DataFrame) -> float:
    return risk_of_ruin(_as_series(returns))


def ghpr(
    returns: pd.Series | pd.DataFrame,
    aggregate: str | None = None,
    compounded: bool = True,
) -> float:
    return expected_return(_as_series(returns), aggregate=aggregate, compounded=compounded)


def rar(returns: pd.Series | pd.DataFrame, rf: float = 0.0) -> float:
    clean = _clean_series(_as_series(returns)).fillna(0.0) - rf / ANNUALIZATION_DEFAULT
    return cagr(clean) / exposure(clean)


def validate_input(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if isinstance(returns, pd.DataFrame):
        if returns.empty:
            raise ValueError("returns dataframe is empty")
        return returns
    clean = _clean_series(returns)
    if clean.empty:
        raise ValueError("returns series is empty")
    return clean


def warn(message: str) -> None:
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def _exposure(returns: pd.Series) -> float:
    clean = _clean_series(returns)
    if clean.empty:
        return float("nan")
    non_zero = clean[clean != 0]
    return float(len(non_zero) / len(clean))


def exposure(returns: pd.Series) -> float:
    return _exposure(returns)


def _win_rate(returns: pd.Series, aggregate: str | None = None, compounded: bool = True) -> float:
    del aggregate, compounded
    return shared_win_rate(returns)


def win_rate(returns: pd.Series, aggregate: str | None = None, compounded: bool = True) -> float:
    return _win_rate(returns, aggregate=aggregate, compounded=compounded)


def _avg_return(returns: pd.Series, aggregate: str | None = None, compounded: bool = True) -> float:
    series = _aggregate_returns(returns, aggregate, compounded)
    if series.empty:
        return float("nan")
    non_zero = series[series != 0].dropna()
    return float(non_zero.mean()) if len(non_zero) else float("nan")


def avg_return(returns: pd.Series, aggregate: str | None = None, compounded: bool = True) -> float:
    return _avg_return(returns, aggregate=aggregate, compounded=compounded)


def _avg_win(returns: pd.Series, aggregate: str | None = None, compounded: bool = True) -> float:
    series = _aggregate_returns(returns, aggregate, compounded)
    wins = series[series > 0].dropna()
    return float(wins.mean()) if len(wins) else float("nan")


def avg_win(returns: pd.Series, aggregate: str | None = None, compounded: bool = True) -> float:
    return _avg_win(returns, aggregate=aggregate, compounded=compounded)


def _avg_loss(returns: pd.Series, aggregate: str | None = None, compounded: bool = True) -> float:
    series = _aggregate_returns(returns, aggregate, compounded)
    losses = series[series < 0].dropna()
    return float(losses.mean()) if len(losses) else float("nan")


def avg_loss(returns: pd.Series, aggregate: str | None = None, compounded: bool = True) -> float:
    return _avg_loss(returns, aggregate=aggregate, compounded=compounded)


def _annualized_return(returns: pd.Series, annualization: int = ANNUALIZATION_DEFAULT) -> float:
    return shared_annualized_return(returns, annualization=annualization)


def cagr(returns: pd.Series, annualization: int = ANNUALIZATION_DEFAULT) -> float:
    return _cagr(returns, annualization=annualization)


def _annualized_volatility(returns: pd.Series, annualization: int = ANNUALIZATION_DEFAULT) -> float:
    return float(shared_annualized_volatility_value(returns, annualization=annualization))


def volatility(returns: pd.Series, annualization: int = ANNUALIZATION_DEFAULT) -> float:
    return _volatility(returns, annualization=annualization)


def _sharpe(returns: pd.Series, rf: float = 0.0, annualization: int = ANNUALIZATION_DEFAULT) -> float:
    clean = _clean_series(returns).fillna(0.0)
    if clean.empty:
        return float("nan")
    return float(shared_sharpe_ratio(clean - rf / annualization, annualization=annualization))


def _autocorr_penalty(returns: pd.Series) -> float:
    clean = _clean_series(returns).fillna(0.0)
    if len(clean) < 2:
        return float("nan")
    coef = abs(np.corrcoef(clean[:-1], clean[1:])[0, 1])
    if not np.isfinite(coef):
        return float("nan")
    x = np.arange(1, len(clean))
    corr = ((len(clean) - x) / len(clean)) * (coef**x)
    return float(np.sqrt(1.0 + 2.0 * np.sum(corr)))


def autocorr_penalty(returns: pd.Series | pd.DataFrame, prepare_returns: bool = False) -> float:
    if isinstance(returns, pd.DataFrame):
        if returns.empty:
            return float("nan")
        returns = returns.iloc[:, 0]
    if prepare_returns:
        returns = _clean_series(returns)
    return _autocorr_penalty(pd.Series(returns))


def _smart_sharpe(returns: pd.Series, rf: float = 0.0, annualization: int = ANNUALIZATION_DEFAULT) -> float:
    clean = _clean_series(returns)
    base = _sharpe(clean, rf=rf, annualization=annualization)
    penalty = _autocorr_penalty(clean)
    if not np.isfinite(base) or not np.isfinite(penalty) or penalty == 0:
        return float("nan")
    return float(base / penalty)


def _sortino(returns: pd.Series, rf: float = 0.0, annualization: int = ANNUALIZATION_DEFAULT) -> float:
    clean = _clean_series(returns).fillna(0.0)
    if clean.empty:
        return float("nan")
    excess = clean - rf / annualization
    downside = np.sqrt((excess[excess < 0] ** 2).sum() / len(excess))
    if not np.isfinite(downside) or downside <= 1e-12:
        return float("nan")
    return float(excess.mean() / downside * np.sqrt(annualization))


def _smart_sortino(returns: pd.Series, rf: float = 0.0, annualization: int = ANNUALIZATION_DEFAULT) -> float:
    clean = _clean_series(returns)
    base = _sortino(clean, rf=rf, annualization=annualization)
    penalty = _autocorr_penalty(clean)
    if not np.isfinite(base) or not np.isfinite(penalty) or penalty == 0:
        return float("nan")
    return float(base / penalty)


def adjusted_sortino(
    returns: pd.Series,
    rf: float = 0.0,
    annualization: int = ANNUALIZATION_DEFAULT,
) -> float:
    return _smart_sortino(returns, rf=rf, annualization=annualization)


def sharpe(returns: pd.Series, rf: float = 0.0, annualization: int = ANNUALIZATION_DEFAULT) -> float:
    return _sharpe(returns, rf=rf, annualization=annualization)


def smart_sharpe(returns: pd.Series, rf: float = 0.0, annualization: int = ANNUALIZATION_DEFAULT) -> float:
    return _smart_sharpe(returns, rf=rf, annualization=annualization)


def sortino(returns: pd.Series, rf: float = 0.0, annualization: int = ANNUALIZATION_DEFAULT) -> float:
    return _sortino(returns, rf=rf, annualization=annualization)


def smart_sortino(returns: pd.Series, rf: float = 0.0, annualization: int = ANNUALIZATION_DEFAULT) -> float:
    return _smart_sortino(returns, rf=rf, annualization=annualization)


def _volatility(returns: pd.Series, annualization: int = ANNUALIZATION_DEFAULT) -> float:
    return _annualized_volatility(returns, annualization=annualization)


def calmar(returns: pd.Series, annualization: int = ANNUALIZATION_DEFAULT) -> float:
    cagr_value = _cagr(returns, annualization=annualization)
    max_dd = _max_drawdown(returns)
    if not np.isfinite(cagr_value) or not np.isfinite(max_dd) or max_dd == 0:
        return float("nan")
    return float(cagr_value / abs(max_dd))


def cvar(returns: pd.Series, sigma: float = 1.0, confidence: float = 0.95) -> float:
    return _conditional_value_at_risk(returns, sigma=sigma, confidence=confidence)


def var(returns: pd.Series, sigma: float = 1.0, confidence: float = 0.95) -> float:
    return _value_at_risk(returns, sigma=sigma, confidence=confidence)


def r_squared(returns: pd.Series, benchmark: pd.Series) -> float:
    return _r_squared(returns, benchmark)


def r2(returns: pd.Series, benchmark: pd.Series) -> float:
    return _r_squared(returns, benchmark)


def upi(returns: pd.Series, rf: float = 0.0) -> float:
    return _ulcer_performance_index(returns, rf=rf)


def ulcer_index(returns: pd.Series) -> float:
    return _ulcer_index(returns)


def ulcer_performance_index(returns: pd.Series, rf: float = 0.0) -> float:
    return _ulcer_performance_index(returns, rf=rf)


def common_sense_ratio(returns: pd.Series) -> float:
    return _common_sense_ratio(returns)


def cpc_index(returns: pd.Series) -> float:
    return _cpc_index(returns)


def gain_to_pain_ratio(returns: pd.Series, resolution: str = "D") -> float:
    return _gain_to_pain_ratio(returns, resolution=resolution)


def payoff_ratio(returns: pd.Series) -> float:
    return _payoff_ratio(returns)


def win_loss_ratio(returns: pd.Series) -> float:
    return _win_loss_ratio(returns)


def profit_ratio(returns: pd.Series) -> float:
    return _profit_ratio(returns)


def recovery_factor(returns: pd.Series, rf: float = 0.0) -> float:
    return _recovery_factor(returns, rf=rf)


def serenity_index(returns: pd.Series, rf: float = 0.0) -> float:
    return _serenity_index(returns, rf=rf)


def risk_return_ratio(returns: pd.Series) -> float:
    return _risk_return_ratio(returns)


def expected_return(returns: pd.Series, aggregate: str | None = None, compounded: bool = True) -> float:
    return _expected_return(returns, aggregate=aggregate, compounded=compounded)


def geometric_mean(returns: pd.Series) -> float:
    clean = _clean_series(returns).fillna(0.0)
    if clean.empty:
        return float("nan")
    growth = (1.0 + clean).prod()
    if growth <= 0:
        return float("nan")
    return float(growth ** (1.0 / len(clean)) - 1.0)


def omega(returns: pd.Series, required_return: float = 0.0) -> float:
    clean = _clean_series(returns).fillna(0.0)
    if clean.empty:
        return float("nan")
    excess = clean - required_return
    gains = excess[excess > 0].sum()
    losses = -excess[excess < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def _cagr(returns: pd.Series, annualization: int = ANNUALIZATION_DEFAULT) -> float:
    clean = _clean_series(returns).fillna(0.0)
    if clean.empty:
        return float("nan")
    total = float((1.0 + clean).prod() - 1.0)
    years = len(clean) / annualization
    if years <= 0:
        return float("nan")
    return float(abs(total + 1.0) ** (1.0 / years) - 1.0)


def _max_drawdown(returns: pd.Series) -> float:
    return shared_max_drawdown(returns)


def max_drawdown(returns: pd.Series) -> float:
    return _max_drawdown(returns)


def _drawdown_details(drawdown: pd.Series) -> pd.DataFrame:
    dd = _clean_series(drawdown).fillna(0.0)
    no_dd = dd == 0
    starts = (~no_dd & no_dd.shift(1, fill_value=True))
    starts = list(starts[starts.values].index)
    ends = no_dd & (~no_dd).shift(1, fill_value=False)
    ends = ends.shift(-1, fill_value=False)
    ends = list(ends[ends.values].index)
    if not starts:
        return pd.DataFrame(columns=("start", "valley", "end", "days", "max drawdown", "99% max drawdown"))
    if ends and starts[0] > ends[0]:
        starts.insert(0, dd.index[0])
    if not ends or starts[-1] > ends[-1]:
        ends.append(dd.index[-1])
    data: list[tuple[Any, ...]] = []
    for i, _ in enumerate(starts):
        period = dd.loc[starts[i]:ends[i]]
        if period.empty:
            continue
        clean_dd = -_remove_outliers(-period, 0.99)
        data.append(
            (
                starts[i],
                period.idxmin(),
                ends[i],
                (ends[i] - starts[i]).days + 1,
                period.min() * 100.0,
                clean_dd.min() * 100.0 if len(clean_dd) else float("nan"),
            )
        )
    frame = pd.DataFrame(
        data=data,
        columns=("start", "valley", "end", "days", "max drawdown", "99% max drawdown"),
    )
    if not frame.empty:
        frame["days"] = frame["days"].astype(int)
        frame["start"] = pd.to_datetime(frame["start"]).dt.strftime("%Y-%m-%d")
        frame["valley"] = pd.to_datetime(frame["valley"]).dt.strftime("%Y-%m-%d")
        frame["end"] = pd.to_datetime(frame["end"]).dt.strftime("%Y-%m-%d")
    return frame


def _remove_outliers(returns: pd.Series, quantile: float = 0.95) -> pd.Series:
    clean = _clean_series(returns)
    if clean.empty:
        return clean
    return clean[clean < clean.quantile(quantile)]


def _best(returns: pd.Series, aggregate: str | None = None, compounded: bool = True) -> float:
    return float(_aggregate_returns(returns, aggregate, compounded).max())


def _worst(returns: pd.Series, aggregate: str | None = None, compounded: bool = True) -> float:
    return float(_aggregate_returns(returns, aggregate, compounded).min())


def _consecutive_runs(returns: pd.Series, positive: bool = True, aggregate: str | None = None, compounded: bool = True) -> int:
    series = _aggregate_returns(returns, aggregate, compounded)
    if series.empty:
        return 0
    flags = (series > 0) if positive else (series < 0)
    best = cur = 0
    for value in flags.fillna(False).tolist():
        if value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _profit_factor(returns: pd.Series) -> float:
    clean = _clean_series(returns)
    if clean.empty:
        return float("nan")
    wins = float(clean[clean >= 0].sum())
    losses = float(abs(clean[clean < 0].sum()))
    if losses == 0:
        return float("inf") if wins > 0 else float("nan")
    return float(wins / losses)


def profit_factor(returns: pd.Series) -> float:
    return _profit_factor(returns)


def _payoff_ratio(returns: pd.Series) -> float:
    win = _avg_win(returns)
    loss = _avg_loss(returns)
    if not np.isfinite(win) or not np.isfinite(loss) or loss == 0:
        return float("nan")
    return float(win / abs(loss))


def _win_loss_ratio(returns: pd.Series) -> float:
    return _payoff_ratio(returns)


def _profit_ratio(returns: pd.Series) -> float:
    clean = _clean_series(returns)
    wins = clean[clean >= 0]
    losses = clean[clean < 0]
    if len(wins) == 0 or len(losses) == 0:
        return float("nan")
    win_ratio = abs(wins.mean() / len(wins))
    loss_ratio = abs(losses.mean() / len(losses))
    if loss_ratio == 0:
        return float("nan")
    return float(win_ratio / loss_ratio)


def _gain_to_pain_ratio(returns: pd.Series, resolution: str = "D") -> float:
    series = _clean_series(returns).fillna(0.0).resample(_normalize_resample_period(resolution) or resolution).sum()
    if series.empty:
        return float("nan")
    pain = abs(series[series < 0].sum())
    if pain == 0:
        return float("nan")
    return float(series.sum() / pain)


def _tail_ratio(returns: pd.Series, cutoff: float = 0.95) -> float:
    clean = _clean_series(returns)
    if clean.empty:
        return float("nan")
    upper = float(clean.quantile(cutoff))
    lower = float(abs(clean.quantile(1 - cutoff)))
    if lower <= 1e-12:
        return float("nan")
    return float(abs(upper / lower))


def tail_ratio(returns: pd.Series, cutoff: float = 0.95) -> float:
    return _tail_ratio(returns, cutoff=cutoff)


def _common_sense_ratio(returns: pd.Series) -> float:
    return float(_profit_factor(returns) * _tail_ratio(returns))


def _cpc_index(returns: pd.Series) -> float:
    return float(_profit_factor(returns) * _win_rate(returns) * _win_loss_ratio(returns))


def _avg_drawdown_days(returns: pd.Series) -> float:
    details = _drawdown_details(_drawdown_series(returns))
    if details.empty:
        return float("nan")
    return float(details["days"].mean())


def _ulcer_index(returns: pd.Series) -> float:
    dd = _drawdown_series(returns)
    clean = _clean_series(dd)
    if clean.empty or len(clean) < 2:
        return float("nan")
    return float(np.sqrt((clean**2).sum() / (len(clean) - 1)))


def _ulcer_performance_index(returns: pd.Series, rf: float = 0.0) -> float:
    ui = _ulcer_index(returns)
    if not np.isfinite(ui) or ui == 0:
        return float("nan")
    return float((_comp(returns) - rf) / ui)


def _recovery_factor(returns: pd.Series, rf: float = 0.0) -> float:
    max_dd = _max_drawdown(returns)
    if not np.isfinite(max_dd) or max_dd == 0:
        return float("nan")
    return float(abs(returns.sum() - rf) / abs(max_dd))


def _risk_return_ratio(returns: pd.Series) -> float:
    clean = _clean_series(returns)
    std = float(clean.std(ddof=1))
    if std == 0 or not np.isfinite(std):
        return float("nan")
    return float(clean.mean() / std)


def _serenity_index(returns: pd.Series, rf: float = 0.0) -> float:
    dd = _drawdown_series(returns)
    std_returns = float(_clean_series(returns).std(ddof=1))
    if std_returns == 0 or not np.isfinite(std_returns):
        return float("nan")
    cvar_val = _conditional_value_at_risk(dd)
    ulcer_val = _ulcer_index(returns)
    if not np.isfinite(cvar_val) or not np.isfinite(ulcer_val) or ulcer_val == 0:
        return float("nan")
    pitfall = -cvar_val / std_returns
    denom = ulcer_val * pitfall
    if denom == 0 or not np.isfinite(denom):
        return float("nan")
    return float((returns.sum() - rf) / denom)


def _kelly_criterion(returns: pd.Series) -> float:
    payoff = _payoff_ratio(returns)
    win_prob = _win_rate(returns)
    lose_prob = 1.0 - win_prob
    if not np.isfinite(payoff) or payoff == 0:
        return float("nan")
    return float(((payoff * win_prob) - lose_prob) / payoff)


def kelly_criterion(returns: pd.Series) -> float:
    return _kelly_criterion(returns)


def _risk_of_ruin(returns: pd.Series) -> float:
    win_prob = _win_rate(returns)
    if not np.isfinite(win_prob):
        return float("nan")
    return float(((1.0 - win_prob) / (1.0 + win_prob)) ** len(_clean_series(returns)))


def risk_of_ruin(returns: pd.Series) -> float:
    return _risk_of_ruin(_as_series(returns))


def _value_at_risk(returns: pd.Series, sigma: float = 1.0, confidence: float = 0.95) -> float:
    clean = _clean_series(returns)
    if clean.empty:
        return float("nan")
    if confidence > 1:
        confidence = confidence / 100.0
    mu = float(clean.mean())
    sigma = float(clean.std(ddof=1) * sigma)
    if not np.isfinite(sigma):
        return float("nan")
    # Normal approximation without scipy dependency.
    z = 1.6448536269514722 if confidence == 0.95 else 1.959963984540054 if confidence == 0.975 else 1.2815515655446004
    return float(mu - z * sigma)


def value_at_risk(returns: pd.Series, sigma: float = 1.0, confidence: float = 0.95) -> float:
    return _value_at_risk(returns, sigma=sigma, confidence=confidence)


def _conditional_value_at_risk(returns: pd.Series, sigma: float = 1.0, confidence: float = 0.95) -> float:
    clean = _clean_series(returns)
    if clean.empty:
        return float("nan")
    threshold = _value_at_risk(clean, sigma=sigma, confidence=confidence)
    below = clean[clean < threshold]
    if below.empty:
        return float(threshold)
    return float(below.mean())


def conditional_value_at_risk(returns: pd.Series, sigma: float = 1.0, confidence: float = 0.95) -> float:
    return _conditional_value_at_risk(returns, sigma=sigma, confidence=confidence)


def _expected_return(returns: pd.Series, aggregate: str | None = None, compounded: bool = True) -> float:
    series = _aggregate_returns(returns, aggregate, compounded)
    if series.empty:
        return float("nan")
    if len(series) == 0:
        return float("nan")
    return float((np.prod(1.0 + series) ** (1.0 / len(series))) - 1.0)


def information_ratio(returns: pd.Series, benchmark: pd.Series) -> float:
    return _information_ratio(returns, benchmark)


def beta_alpha(returns: pd.Series, benchmark: pd.Series, annualization: int = ANNUALIZATION_DEFAULT) -> tuple[float, float]:
    return _beta_alpha(returns, benchmark, annualization=annualization)


def treynor_ratio(returns: pd.Series, benchmark: pd.Series, rf: float = 0.0, annualization: int = ANNUALIZATION_DEFAULT) -> float:
    return _treynor_ratio(returns, benchmark, rf=rf, annualization=annualization)


def _period_return(returns: pd.Series) -> float:
    clean = _clean_series(returns).fillna(0.0)
    if clean.empty:
        return float("nan")
    return float((1.0 + clean).prod() - 1.0)


def _r_squared(returns: pd.Series, benchmark: pd.Series) -> float:
    aligned_returns, aligned_benchmark = _align_series(returns, benchmark)
    if aligned_benchmark is None or aligned_returns.empty:
        return float("nan")
    corr = aligned_returns.corr(aligned_benchmark)
    if not np.isfinite(corr):
        return float("nan")
    return float(corr**2)


def _information_ratio(returns: pd.Series, benchmark: pd.Series) -> float:
    aligned_returns, aligned_benchmark = _align_series(returns, benchmark)
    if aligned_benchmark is None or aligned_returns.empty:
        return float("nan")
    active = aligned_returns - aligned_benchmark
    std = float(active.std(ddof=1))
    if std == 0 or not np.isfinite(std):
        return float("nan")
    return float(active.mean() / std)


def _beta_alpha(returns: pd.Series, benchmark: pd.Series, annualization: int = ANNUALIZATION_DEFAULT) -> tuple[float, float]:
    aligned_returns, aligned_benchmark = _align_series(returns, benchmark)
    if aligned_benchmark is None or aligned_returns.empty:
        return float("nan"), float("nan")
    cov = np.cov(aligned_returns, aligned_benchmark)
    if cov[1, 1] == 0:
        beta = float("nan")
    else:
        beta = float(cov[0, 1] / cov[1, 1])
    alpha = float((aligned_returns.mean() - beta * aligned_benchmark.mean()) * annualization)
    return beta, alpha


def _treynor_ratio(returns: pd.Series, benchmark: pd.Series, rf: float = 0.0, annualization: int = ANNUALIZATION_DEFAULT) -> float:
    beta, _ = _beta_alpha(returns, benchmark, annualization=annualization)
    if not np.isfinite(beta) or beta == 0:
        return float("nan")
    clean = _clean_series(returns)
    if clean.empty:
        return float("nan")
    excess = clean.mean() - rf / annualization
    return float(excess * annualization / beta)


def _monthly_returns(returns: pd.Series, compounded: bool = True) -> pd.Series:
    clean = _clean_series(returns).fillna(0.0)
    if clean.empty:
        return pd.Series(dtype=float)
    if compounded:
        monthly = clean.resample("ME").apply(lambda x: float((1.0 + x.fillna(0.0)).prod() - 1.0))
    else:
        monthly = clean.resample("ME").sum()
    monthly.index = pd.DatetimeIndex(monthly.index, name="date_")
    return monthly


def _annual_returns(returns: pd.Series, compounded: bool = True) -> pd.Series:
    clean = _clean_series(returns).fillna(0.0)
    if clean.empty:
        return pd.Series(dtype=float)
    if compounded:
        annual = clean.resample("YE").apply(lambda x: float((1.0 + x.fillna(0.0)).prod() - 1.0))
    else:
        annual = clean.resample("YE").sum()
    annual.index = annual.index.year.astype(str)
    annual.name = "return"
    return annual


def _monthly_heatmap(returns: pd.Series, compounded: bool = True) -> pd.DataFrame:
    del compounded
    return shared_monthly_returns_heatmap(returns)


def _monthly_returns_table(returns: pd.Series, compounded: bool = True) -> pd.DataFrame:
    monthly = _monthly_returns(returns, compounded=compounded)
    if monthly.empty:
        return pd.DataFrame()
    frame = monthly.to_frame("Returns")
    frame["Year"] = frame.index.year.astype(str)
    frame["Month"] = frame.index.strftime("%b")
    table = frame.pivot(index="Year", columns="Month", values="Returns").fillna(0.0)
    for month in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]:
        if month not in table.columns:
            table[month] = 0.0
    table = table[["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]]
    table["EOY"] = _annual_returns(returns, compounded=compounded)
    table.index.name = None
    return table


def _compare_table(
    returns: pd.Series,
    benchmark: pd.Series | None,
    aggregate: str | None = None,
    compounded: bool = True,
) -> pd.DataFrame:
    if benchmark is None:
        return pd.DataFrame()
    aligned_returns, aligned_benchmark = _align_series(returns, benchmark)
    if aligned_benchmark is None or aligned_returns.empty:
        return pd.DataFrame()
    if aggregate is None:
        returns_agg = aligned_returns * 100.0
        benchmark_agg = aligned_benchmark * 100.0
        index = aligned_returns.index
    else:
        returns_agg = _aggregate_returns(aligned_returns, aggregate, compounded=compounded) * 100.0
        benchmark_agg = _aggregate_returns(aligned_benchmark, aggregate, compounded=compounded) * 100.0
        index = returns_agg.index
    data = pd.DataFrame({"Benchmark": benchmark_agg, "Returns": returns_agg}, index=index)
    data["Multiplier"] = data["Returns"] / data["Benchmark"].replace(0.0, np.nan)
    data["Won"] = np.where(data["Returns"] >= data["Benchmark"], "+", "-")
    return data


def compare(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    aggregate: str | None = None,
    compounded: bool = True,
) -> pd.DataFrame:
    return _compare_table(returns, benchmark, aggregate=aggregate, compounded=compounded)


def monthly_returns(returns: pd.Series, compounded: bool = True) -> pd.Series:
    return _monthly_returns(returns, compounded=compounded)


def drawdown_details(returns: pd.Series) -> pd.DataFrame:
    return _drawdown_details(_drawdown_series(returns))


def montecarlo(
    returns: pd.Series,
    sims: int = 1000,
    seed: int | None = None,
) -> pd.DataFrame:
    return _montecarlo_paths(returns, sims=sims, seed=seed)


def montecarlo_sharpe(
    returns: pd.Series,
    sims: int = 1000,
    seed: int | None = None,
    rf: float = 0.0,
    annualization: int = ANNUALIZATION_DEFAULT,
) -> pd.Series:
    paths = _montecarlo_paths(returns, sims=sims, seed=seed)
    if paths.empty:
        return pd.Series(dtype=float)
    clean = _clean_series(returns).fillna(0.0)
    if clean.empty:
        return pd.Series(dtype=float)
    sharpe = []
    for _, path in paths.items():
        sharpe.append(_sharpe(path.diff().fillna(path.iloc[0]), rf=rf, annualization=annualization))
    return pd.Series(sharpe, name="sharpe")


def montecarlo_drawdown(
    returns: pd.Series,
    sims: int = 1000,
    seed: int | None = None,
) -> pd.Series:
    paths = _montecarlo_paths(returns, sims=sims, seed=seed)
    if paths.empty:
        return pd.Series(dtype=float)
    drawdowns = paths.apply(lambda col: float(col.min()), axis=0)
    drawdowns.name = "drawdown"
    return drawdowns


def montecarlo_cagr(
    returns: pd.Series,
    sims: int = 1000,
    seed: int | None = None,
    annualization: int = ANNUALIZATION_DEFAULT,
) -> pd.Series:
    paths = _montecarlo_paths(returns, sims=sims, seed=seed)
    if paths.empty:
        return pd.Series(dtype=float)
    cagr_values = []
    for _, path in paths.items():
        growth = 1.0 + path.iloc[-1]
        years = len(path) / annualization
        cagr_values.append(float(growth ** (1.0 / years) - 1.0) if years > 0 and growth > 0 else np.nan)
    return pd.Series(cagr_values, name="cagr")


def _montecarlo_paths(
    returns: pd.Series,
    sims: int = 1000,
    seed: int | None = None,
) -> pd.DataFrame:
    clean = _clean_series(returns).fillna(0.0)
    if clean.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    values = clean.to_numpy()
    paths: dict[str, pd.Series] = {}
    for i in range(sims):
        sample = rng.choice(values, size=len(values), replace=True)
        path = pd.Series((1.0 + sample).cumprod() - 1.0, index=clean.index)
        paths[f"sim_{i + 1}"] = path
    return pd.DataFrame(paths)


def _montecarlo_summary(paths: pd.DataFrame) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame()
    terminal = (paths.iloc[-1] + 1.0).astype(float)
    max_dd = paths.apply(lambda col: float(col.min()), axis=0)
    summary = pd.DataFrame(
        {
            "min": [float(terminal.min())],
            "max": [float(terminal.max())],
            "mean": [float(terminal.mean())],
            "median": [float(terminal.median())],
            "std": [float(terminal.std(ddof=1))],
            "percentile_5": [float(terminal.quantile(0.05))],
            "percentile_95": [float(terminal.quantile(0.95))],
            "max_drawdown_min": [float(max_dd.min())],
            "max_drawdown_mean": [float(max_dd.mean())],
            "max_drawdown_median": [float(max_dd.median())],
            "max_drawdown_percentile_5": [float(max_dd.quantile(0.05))],
            "max_drawdown_percentile_95": [float(max_dd.quantile(0.95))],
        }
    )
    return summary


def _plot_montecarlo(paths: pd.DataFrame, *, output_dir: Path, report_name: str) -> Path | None:
    if paths.empty:
        return None
    path = output_dir / f"{report_name}_montecarlo.png"
    plt.figure(figsize=(12, 5))
    for _, col in paths.items():
        plt.plot(col.index, col.values, color="#9ca3af", alpha=0.08, linewidth=1.0)
    median = paths.median(axis=1)
    plt.plot(median.index, median.values, color="#2a6fdb", linewidth=2.0, label="median")
    finalize_quantstats_axis(plt.gca(), title="Monte Carlo equity paths", ylabel="Cumulative return", legend=True)
    save_quantstats_figure(path)
    return path


def distribution(
    returns: pd.Series | pd.DataFrame,
    fontname: str = "Arial",
    grayscale: bool = False,
    ylabel: bool = True,
    figsize: tuple[float, float] = (10, 6),
    subtitle: bool = True,
    compounded: bool = True,
    savefig: str | dict | None = None,
    show: bool = True,
    title: str | None = None,
    prepare_returns: bool = True,
) -> Path | None:
    del fontname, ylabel, subtitle, show, title
    series = _as_series(returns)
    if prepare_returns:
        series = _clean_series(series)
    if isinstance(returns, pd.DataFrame):
        series = _clean_series(series)
    if series.empty:
        return None
    values = _aggregate_returns(series, None, compounded=compounded)
    color = "#444444" if grayscale else "#2a6fdb"
    path: Path | None = None
    if isinstance(savefig, str):
        path = Path(savefig)
    elif isinstance(savefig, dict) and isinstance(savefig.get("fname"), str):
        path = Path(savefig["fname"])
    else:
        path = Path(tempfile.gettempdir()) / "quantstats_distribution.png"
    plt.figure(figsize=figsize)
    plt.hist(values.dropna().values, bins=30, color=color, alpha=0.8, density=True)
    finalize_quantstats_axis(plt.gca(), title="Distribution")
    save_quantstats_figure(path)
    return path


def rolling_sharpe(
    returns: pd.Series | pd.DataFrame,
    benchmark: pd.Series | pd.DataFrame | str | None = None,
    rf: float = 0.0,
    period: int = 126,
    period_label: str = "6-Months",
    periods_per_year: int = ANNUALIZATION_DEFAULT,
    lw: float = 1.25,
    fontname: str = "Arial",
    grayscale: bool = False,
    figsize: tuple[float, float] = (10, 3),
    ylabel: str = "Sharpe",
    subtitle: bool = True,
    savefig: str | dict | None = None,
    show: bool = True,
) -> Path | None:
    del fontname, period_label, subtitle, show
    series = _as_series(returns)
    if isinstance(returns, pd.DataFrame):
        series = _clean_series(series)
    if series.empty:
        return None
    color = "#444444" if grayscale else "#1d4ed8"
    path = Path(savefig) if isinstance(savefig, str) else Path(tempfile.gettempdir()) / "quantstats_rolling_sharpe.png"
    plt.figure(figsize=figsize)
    _rolling_sharpe(series, window=period, annualization=periods_per_year, rf=rf).plot(color=color, linewidth=lw)
    if benchmark is not None and not isinstance(benchmark, str):
        bseries = _as_series(benchmark)
        bseries = _clean_series(bseries)
        if not bseries.empty:
            _rolling_sharpe(bseries, window=period, annualization=periods_per_year, rf=rf).plot(
                color="#9ca3af" if not grayscale else "#777777",
                linewidth=lw,
            )
    finalize_quantstats_axis(plt.gca(), ylabel=ylabel)
    save_quantstats_figure(path)
    return path


def rolling_sortino(
    returns: pd.Series | pd.DataFrame,
    benchmark: pd.Series | pd.DataFrame | str | None = None,
    rf: float = 0.0,
    period: int = 126,
    period_label: str = "6-Months",
    periods_per_year: int = ANNUALIZATION_DEFAULT,
    lw: float = 1.25,
    fontname: str = "Arial",
    grayscale: bool = False,
    figsize: tuple[float, float] = (10, 3),
    ylabel: str = "Sortino",
    subtitle: bool = True,
    savefig: str | dict | None = None,
    show: bool = True,
) -> Path | None:
    del fontname, period_label, subtitle, show
    series = _as_series(returns)
    if isinstance(returns, pd.DataFrame):
        series = _clean_series(series)
    if series.empty:
        return None
    color = "#444444" if grayscale else "#7c3aed"
    path = Path(savefig) if isinstance(savefig, str) else Path(tempfile.gettempdir()) / "quantstats_rolling_sortino.png"
    plt.figure(figsize=figsize)
    _rolling_sortino(series, window=period, annualization=periods_per_year, rf=rf).plot(color=color, linewidth=lw)
    if benchmark is not None and not isinstance(benchmark, str):
        bseries = _as_series(benchmark)
        bseries = _clean_series(bseries)
        if not bseries.empty:
            _rolling_sortino(bseries, window=period, annualization=periods_per_year, rf=rf).plot(
                color="#9ca3af" if not grayscale else "#777777",
                linewidth=lw,
            )
    finalize_quantstats_axis(plt.gca(), ylabel=ylabel)
    save_quantstats_figure(path)
    return path


def rolling_volatility(
    returns: pd.Series | pd.DataFrame,
    benchmark: pd.Series | pd.DataFrame | str | None = None,
    period: int = 126,
    period_label: str = "6-Months",
    periods_per_year: int = ANNUALIZATION_DEFAULT,
    lw: float = 1.5,
    fontname: str = "Arial",
    grayscale: bool = False,
    figsize: tuple[float, float] = (10, 3),
    ylabel: str = "Volatility",
    subtitle: bool = True,
    savefig: str | dict | None = None,
    show: bool = True,
) -> Path | None:
    del fontname, period_label, subtitle, show
    series = _as_series(returns)
    if isinstance(returns, pd.DataFrame):
        series = _clean_series(series)
    if series.empty:
        return None
    color = "#444444" if grayscale else "#1f7a8c"
    path = Path(savefig) if isinstance(savefig, str) else Path(tempfile.gettempdir()) / "quantstats_rolling_volatility.png"
    plt.figure(figsize=figsize)
    _rolling_volatility(series, window=period, annualization=periods_per_year).plot(color=color, linewidth=lw)
    if benchmark is not None and not isinstance(benchmark, str):
        bseries = _as_series(benchmark)
        bseries = _clean_series(bseries)
        if not bseries.empty:
            _rolling_volatility(bseries, window=period, annualization=periods_per_year).plot(
                color="#9ca3af" if not grayscale else "#777777",
                linewidth=lw,
            )
    finalize_quantstats_axis(plt.gca(), ylabel=ylabel)
    save_quantstats_figure(path)
    return path


def rolling_beta(
    returns: pd.Series | pd.DataFrame,
    benchmark: pd.Series | pd.DataFrame | str | None = None,
    period: int = 126,
    period_label: str = "6-Months",
    periods_per_year: int = ANNUALIZATION_DEFAULT,
    lw: float = 1.25,
    fontname: str = "Arial",
    grayscale: bool = False,
    figsize: tuple[float, float] = (10, 3),
    ylabel: str = "Beta",
    subtitle: bool = True,
    savefig: str | dict | None = None,
    show: bool = True,
) -> Path | None:
    del fontname, period_label, periods_per_year, subtitle, show
    series = _as_series(returns)
    if isinstance(returns, pd.DataFrame):
        series = _clean_series(series)
    if series.empty or benchmark is None or isinstance(benchmark, str):
        return None
    bseries = _as_series(benchmark)
    bseries = _clean_series(bseries)
    if bseries.empty:
        return None
    color = "#444444" if grayscale else "#457b9d"
    path = Path(savefig) if isinstance(savefig, str) else Path(tempfile.gettempdir()) / "quantstats_rolling_beta.png"
    plt.figure(figsize=figsize)
    _rolling_beta(series, bseries, window=period).plot(color=color, linewidth=lw)
    finalize_quantstats_axis(plt.gca(), ylabel=ylabel)
    save_quantstats_figure(path)
    return path


def safe_resample(series: pd.Series | pd.DataFrame, period: str = "ME", agg: str = "sum") -> pd.Series | pd.DataFrame:
    values = _as_series(series)
    clean = _clean_series(values)
    if clean.empty:
        return pd.Series(dtype=float)
    period = _normalize_resample_period(period) or "ME"
    if agg == "sum":
        return clean.resample(period).sum()
    if agg == "mean":
        return clean.resample(period).mean()
    if agg == "prod":
        return clean.resample(period).apply(lambda x: float((1.0 + x.fillna(0.0)).prod() - 1.0))
    return clean.resample(period).sum()


def to_plotly(*args: Any, **kwargs: Any) -> None:
    del args, kwargs
    return None


def returns(*args: Any, **kwargs: Any) -> Path | None:
    del kwargs
    returns_series = _as_series(args[0]) if args else pd.Series(dtype=float)
    benchmark = args[1] if len(args) > 1 else None
    if isinstance(benchmark, (pd.Series, pd.DataFrame)):
        benchmark = _as_series(benchmark)
    return _plot_cumulative_returns(_as_series(returns_series), benchmark if isinstance(benchmark, pd.Series) else None, output_dir=Path(tempfile.gettempdir()), report_name="quantstats_returns")


def log_returns(*args: Any, **kwargs: Any) -> Path | None:
    del kwargs
    returns_series = _as_series(args[0]) if args else pd.Series(dtype=float)
    benchmark = args[1] if len(args) > 1 else None
    if isinstance(benchmark, (pd.Series, pd.DataFrame)):
        benchmark = _as_series(benchmark)
    return _plot_log_returns(_as_series(returns_series), benchmark if isinstance(benchmark, pd.Series) else None, output_dir=Path(tempfile.gettempdir()), report_name="quantstats_log_returns")


def yearly_returns(*args: Any, **kwargs: Any) -> Path | None:
    del kwargs
    returns_series = _as_series(args[0]) if args else pd.Series(dtype=float)
    benchmark = args[1] if len(args) > 1 else None
    if isinstance(benchmark, (pd.Series, pd.DataFrame)):
        benchmark = _as_series(benchmark)
    return _plot_yearly_returns(_as_series(returns_series), benchmark if isinstance(benchmark, pd.Series) else None, output_dir=Path(tempfile.gettempdir()), report_name="quantstats_yearly_returns")


def daily_returns(*args: Any, **kwargs: Any) -> Path | None:
    del kwargs
    returns_series = _as_series(args[0]) if args else pd.Series(dtype=float)
    benchmark = args[1] if len(args) > 1 else None
    if isinstance(benchmark, (pd.Series, pd.DataFrame)):
        benchmark = _as_series(benchmark)
    return _plot_daily_returns(_as_series(returns_series), benchmark if isinstance(benchmark, pd.Series) else None, output_dir=Path(tempfile.gettempdir()), report_name="quantstats_daily_returns")


def drawdown(*args: Any, **kwargs: Any) -> Path | None:
    del kwargs
    returns_series = _as_series(args[0]) if args else pd.Series(dtype=float)
    return _plot_drawdown(_as_series(returns_series), output_dir=Path(tempfile.gettempdir()), report_name="quantstats_drawdown")


def drawdowns_periods(*args: Any, **kwargs: Any) -> Path | None:
    del kwargs
    returns_series = _as_series(args[0]) if args else pd.Series(dtype=float)
    return _plot_drawdown_periods(_as_series(returns_series), output_dir=Path(tempfile.gettempdir()), report_name="quantstats_drawdowns_periods")


def histogram(*args: Any, **kwargs: Any) -> Path | None:
    del kwargs
    returns_series = _as_series(args[0]) if args else pd.Series(dtype=float)
    return _plot_return_histogram(_as_series(returns_series), output_dir=Path(tempfile.gettempdir()), report_name="quantstats_histogram")


def monthly_heatmap(*args: Any, **kwargs: Any) -> Path | None:
    del kwargs
    returns_series = _as_series(args[0]) if args else pd.Series(dtype=float)
    return _plot_monthly_heatmap(_as_series(returns_series), output_dir=Path(tempfile.gettempdir()), report_name="quantstats_monthly_heatmap")


def snapshot(*args: Any, **kwargs: Any) -> Path | None:
    del kwargs
    returns_series = _as_series(args[0]) if args else pd.Series(dtype=float)
    benchmark = args[1] if len(args) > 1 else None
    if isinstance(benchmark, (pd.Series, pd.DataFrame)):
        benchmark = _as_series(benchmark)
    return _plot_snapshot(_as_series(returns_series), benchmark if isinstance(benchmark, pd.Series) else None, output_dir=Path(tempfile.gettempdir()), report_name="quantstats_snapshot")


def earnings(*args: Any, **kwargs: Any) -> Path | None:
    del kwargs
    returns_series = _as_series(args[0]) if args else pd.Series(dtype=float)
    return _plot_earnings(_as_series(returns_series), output_dir=Path(tempfile.gettempdir()), report_name="quantstats_earnings")


def montecarlo_distribution(*args: Any, **kwargs: Any) -> Path | None:
    del kwargs
    returns_series = _as_series(args[0]) if args else pd.Series(dtype=float)
    paths = _montecarlo_paths(_as_series(returns_series), sims=500)
    if paths.empty:
        return None
    path = Path(tempfile.gettempdir()) / "quantstats_montecarlo_distribution.png"
    plt.figure(figsize=(10, 6))
    plt.hist((paths.iloc[-1] + 1.0).values, bins=30, color="#2a6fdb", alpha=0.8)
    finalize_quantstats_axis(plt.gca(), title="Monte Carlo distribution")
    save_quantstats_figure(path)
    return path


def _rolling_sharpe(returns: pd.Series, window: int = 63, rf: float = 0.0, annualization: int = ANNUALIZATION_DEFAULT) -> pd.Series:
    return shared_rolling_sharpe(
        returns,
        window=window,
        rf=rf,
        annualization=annualization,
        min_periods=max(5, window // 4),
    )


def _rolling_sortino(returns: pd.Series, window: int = 63, rf: float = 0.0, annualization: int = ANNUALIZATION_DEFAULT) -> pd.Series:
    return shared_rolling_sortino(
        returns,
        window=window,
        rf=rf,
        annualization=annualization,
        min_periods=max(5, window // 4),
    )


def _rolling_volatility(returns: pd.Series, window: int = 63, annualization: int = ANNUALIZATION_DEFAULT) -> pd.Series:
    return shared_rolling_volatility(
        returns,
        window=window,
        annualization=annualization,
        min_periods=max(5, window // 4),
    )


def _rolling_beta(returns: pd.Series, benchmark: pd.Series, window: int = 63) -> pd.Series:
    return shared_rolling_beta(returns, benchmark, window=window, min_periods=max(5, window // 4))


def greeks(
    returns: pd.Series,
    benchmark: pd.Series,
    periods: int = ANNUALIZATION_DEFAULT,
    prepare_returns: bool = True,
) -> pd.Series:
    if prepare_returns:
        returns = _clean_series(returns)
    aligned_returns, aligned_benchmark = _align_series(returns, benchmark)
    if aligned_benchmark is None or aligned_returns.empty:
        return pd.Series({"beta": np.nan, "alpha": np.nan})
    beta, alpha = _beta_alpha(aligned_returns, aligned_benchmark, annualization=periods)
    return pd.Series({"beta": beta, "alpha": alpha})


def rolling_greeks(
    returns: pd.Series,
    benchmark: pd.Series,
    periods: int = ANNUALIZATION_DEFAULT,
    prepare_returns: bool = True,
) -> pd.DataFrame:
    if prepare_returns:
        returns = _clean_series(returns)
    aligned_returns, aligned_benchmark = _align_series(returns, benchmark)
    if aligned_benchmark is None or aligned_returns.empty:
        return pd.DataFrame(columns=["alpha", "beta"])
    df = pd.DataFrame({"returns": aligned_returns, "benchmark": aligned_benchmark}).fillna(0.0)
    window = int(periods)
    corr = df.rolling(window).corr().unstack()["returns"]["benchmark"]
    std = df.rolling(window).std()
    beta = corr * std["returns"] / std["benchmark"].replace(0.0, np.nan)
    alpha = df["returns"].rolling(window).mean() - beta * df["benchmark"].rolling(window).mean()
    result = pd.DataFrame({"alpha": alpha, "beta": beta})
    result.index = pd.DatetimeIndex(result.index, name="date_")
    return result.replace([np.inf, -np.inf], np.nan)


def pct_rank(prices: pd.Series, window: int = 60) -> pd.Series:
    clean = _clean_series(prices)
    if clean.empty:
        return pd.Series(dtype=float)
    rank = clean.rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100.0, raw=False)
    rank.index = pd.DatetimeIndex(rank.index, name="date_")
    return rank.replace([np.inf, -np.inf], np.nan)


def _yearly_comparison_table(returns: pd.Series, benchmark: pd.Series, compounded: bool = True) -> pd.DataFrame:
    aligned_returns, aligned_benchmark = _align_series(returns, benchmark)
    if aligned_benchmark is None or aligned_returns.empty:
        return pd.DataFrame()
    returns_agg = _aggregate_returns(aligned_returns, "Y", compounded=compounded) * 100.0
    benchmark_agg = _aggregate_returns(aligned_benchmark, "Y", compounded=compounded) * 100.0
    data = pd.DataFrame({"Benchmark": benchmark_agg, "Returns": returns_agg})
    data["Multiplier"] = data["Returns"] / data["Benchmark"].replace(0.0, np.nan)
    data["Won"] = np.where(data["Returns"] >= data["Benchmark"], "+", "-")
    data.index = data.index.year.astype(str)
    data.index.name = "Year"
    return data


def _metric_row(
    series: pd.Series,
    *,
    rf: float = 0.0,
    annualization: int = ANNUALIZATION_DEFAULT,
    benchmark: pd.Series | None = None,
    compounded: bool = True,
) -> dict[str, Any]:
    clean = _clean_series(series)
    exposure = _exposure(clean) if not clean.empty else float("nan")
    cumulative_return = _comp(clean)
    cagr = _cagr(clean, annualization=annualization)
    max_dd = _max_drawdown(clean)
    vol_ann = _volatility(clean, annualization=annualization)
    row: dict[str, Any] = {
        "Start Period": clean.index.strftime("%Y-%m-%d")[0] if not clean.empty else "-",
        "End Period": clean.index.strftime("%Y-%m-%d")[-1] if not clean.empty else "-",
        "Risk-Free Rate %": rf * 100.0,
        "Time in Market %": exposure * 100.0 if np.isfinite(exposure) else float("nan"),
        "Cumulative Return %": cumulative_return * 100.0,
        "CAGR %": cagr * 100.0,
        "Sharpe": _sharpe(clean, rf=rf, annualization=annualization),
        "Smart Sharpe": _smart_sharpe(clean, rf=rf, annualization=annualization),
        "Sortino": _sortino(clean, rf=rf, annualization=annualization),
        "Smart Sortino": _smart_sortino(clean, rf=rf, annualization=annualization),
        "Sortino/√2": _sortino(clean, rf=rf, annualization=annualization) / math.sqrt(2) if np.isfinite(_sortino(clean, rf=rf, annualization=annualization)) else float("nan"),
        "Smart Sortino/√2": _smart_sortino(clean, rf=rf, annualization=annualization) / math.sqrt(2) if np.isfinite(_smart_sortino(clean, rf=rf, annualization=annualization)) else float("nan"),
        "Omega": float(clean[clean > 0].sum() / abs(clean[clean < 0].sum())) if not clean[clean < 0].empty and abs(clean[clean < 0].sum()) > 0 else float("nan"),
        "Max Drawdown %": max_dd * 100.0,
        "Max DD Date": _drawdown_series(clean).idxmin().strftime("%Y-%m-%d") if not _drawdown_series(clean).empty else "-",
        "Max DD Period Start": "-",
        "Max DD Period End": "-",
        "Longest DD Days": float("nan"),
        "Volatility (ann.) %": vol_ann * 100.0,
        "Calmar": cagr / abs(max_dd) if np.isfinite(max_dd) and max_dd != 0 else float("nan"),
        "Skew": float(clean.skew()) if not clean.empty else float("nan"),
        "Kurtosis": float(clean.kurtosis()) if not clean.empty else float("nan"),
        "Ulcer Performance Index": _ulcer_performance_index(clean, rf=rf),
        "Risk-Adjusted Return %": (cagr / exposure) * 100.0 if np.isfinite(exposure) and exposure != 0 else float("nan"),
        "Risk-Return Ratio": _risk_return_ratio(clean),
        "Avg. Return %": _avg_return(clean) * 100.0,
        "Avg. Win %": _avg_win(clean) * 100.0,
        "Avg. Loss %": _avg_loss(clean) * 100.0,
        "Win/Loss Ratio": _win_loss_ratio(clean),
        "Profit Ratio": _profit_ratio(clean),
        "Expected Daily %": _expected_return(clean, compounded=compounded) * 100.0,
        "Expected Monthly %": _expected_return(clean, aggregate="ME", compounded=compounded) * 100.0,
        "Expected Yearly %": _expected_return(clean, aggregate="YE", compounded=compounded) * 100.0,
        "Kelly Criterion %": _kelly_criterion(clean) * 100.0,
        "Risk of Ruin %": _risk_of_ruin(clean) * 100.0,
        "Daily Value-at-Risk %": _value_at_risk(clean, confidence=0.95) * 100.0,
        "Expected Shortfall (cVaR) %": _conditional_value_at_risk(clean, confidence=0.95) * 100.0,
        "Max Consecutive Wins": _consecutive_runs(clean, positive=True),
        "Max Consecutive Losses": _consecutive_runs(clean, positive=False),
        "Gain/Pain Ratio": _gain_to_pain_ratio(clean, resolution="D"),
        "Gain/Pain (1M)": _gain_to_pain_ratio(clean, resolution="ME"),
        "Payoff Ratio": _payoff_ratio(clean),
        "Profit Factor": _profit_factor(clean),
        "Common Sense Ratio": _common_sense_ratio(clean),
        "CPC Index": _cpc_index(clean),
        "Tail Ratio": _tail_ratio(clean),
        "Outlier Win Ratio": float(clean[clean >= clean.quantile(0.99)].mean() / clean[clean >= 0].mean()) if not clean[clean >= 0].empty and clean[clean >= 0].mean() != 0 else float("nan"),
        "Outlier Loss Ratio": float(clean[clean <= clean.quantile(0.01)].mean() / clean[clean < 0].mean()) if not clean[clean < 0].empty and clean[clean < 0].mean() != 0 else float("nan"),
        "MTD %": _period_return(clean[clean.index >= pd.Timestamp(clean.index[-1].year, clean.index[-1].month, 1)]) * 100.0 if not clean.empty else float("nan"),
        "3M %": _period_return(clean[clean.index >= (clean.index[-1] - pd.DateOffset(months=3))]) * 100.0 if not clean.empty else float("nan"),
        "6M %": _period_return(clean[clean.index >= (clean.index[-1] - pd.DateOffset(months=6))]) * 100.0 if not clean.empty else float("nan"),
        "YTD %": _period_return(clean[clean.index >= pd.Timestamp(clean.index[-1].year, 1, 1)]) * 100.0 if not clean.empty else float("nan"),
        "1Y %": _period_return(clean[clean.index >= (clean.index[-1] - pd.DateOffset(years=1))]) * 100.0 if not clean.empty else float("nan"),
        "3Y (ann.) %": _cagr(clean[clean.index >= (clean.index[-1] - pd.DateOffset(years=3))], annualization=annualization) * 100.0 if not clean.empty else float("nan"),
        "5Y (ann.) %": _cagr(clean[clean.index >= (clean.index[-1] - pd.DateOffset(years=5))], annualization=annualization) * 100.0 if not clean.empty else float("nan"),
        "10Y (ann.) %": _cagr(clean[clean.index >= (clean.index[-1] - pd.DateOffset(years=10))], annualization=annualization) * 100.0 if not clean.empty else float("nan"),
        "All-time (ann.) %": _cagr(clean, annualization=annualization) * 100.0,
        "Recovery Factor": _recovery_factor(clean, rf=rf),
        "Ulcer Index": _ulcer_index(clean),
        "Serenity Index": _serenity_index(clean, rf=rf),
        "Avg. Up Month %": _avg_win(clean, aggregate="ME", compounded=compounded) * 100.0,
        "Avg. Down Month %": _avg_loss(clean, aggregate="ME", compounded=compounded) * 100.0,
        "Win Days %": _win_rate(clean) * 100.0,
        "Win Month %": _win_rate(clean, aggregate="ME", compounded=compounded) * 100.0,
        "Win Quarter %": _win_rate(clean, aggregate="Q", compounded=compounded) * 100.0,
        "Win Year %": _win_rate(clean, aggregate="YE", compounded=compounded) * 100.0,
        "Best Day %": _best(clean, compounded=compounded) * 100.0,
        "Worst Day %": _worst(clean, compounded=compounded) * 100.0,
        "Best Month %": _best(clean, aggregate="ME", compounded=compounded) * 100.0,
        "Worst Month %": _worst(clean, aggregate="ME", compounded=compounded) * 100.0,
        "Best Year %": _best(clean, aggregate="YE", compounded=compounded) * 100.0,
        "Worst Year %": _worst(clean, aggregate="YE", compounded=compounded) * 100.0,
    }

    dd = _drawdown_details(_drawdown_series(clean))
    if not dd.empty:
        worst = dd.sort_values(by="max drawdown", ascending=True).iloc[0]
        row["Max DD Period Start"] = worst["start"]
        row["Max DD Period End"] = worst["end"]
        row["Longest DD Days"] = int(dd["days"].max())
        row["Avg. Drawdown Days"] = float(dd["days"].mean())
    else:
        row["Avg. Drawdown Days"] = float("nan")

    if benchmark is not None:
        aligned_returns, aligned_benchmark = _align_series(clean, benchmark)
        if aligned_benchmark is not None and not aligned_returns.empty:
            active = aligned_returns - aligned_benchmark
            beta, alpha = _beta_alpha(aligned_returns, aligned_benchmark, annualization=annualization)
            row.update(
                {
                    "Volatility (ann.) %": _volatility(aligned_returns, annualization=annualization) * 100.0,
                    "R^2": _r_squared(aligned_returns, aligned_benchmark),
                    "Information Ratio": _information_ratio(aligned_returns, aligned_benchmark),
                    "Beta": beta,
                    "Alpha": alpha,
                    "Correlation": aligned_returns.corr(aligned_benchmark),
                    "Treynor Ratio": _treynor_ratio(aligned_returns, aligned_benchmark, rf=rf, annualization=annualization),
                }
            )
            row["Active Return %"] = _comp(active) * 100.0
            row["Active Sharpe"] = _sharpe(active, rf=0.0, annualization=annualization)
            row["Active Sortino"] = _sortino(active, rf=0.0, annualization=annualization)
        else:
            row.update(
                {
                    "R^2": float("nan"),
                    "Information Ratio": float("nan"),
                    "Beta": float("nan"),
                    "Alpha": float("nan"),
                    "Correlation": float("nan"),
                    "Treynor Ratio": float("nan"),
                    "Active Return %": float("nan"),
                    "Active Sharpe": float("nan"),
                    "Active Sortino": float("nan"),
                }
            )
    return row


def _summary_table(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None,
    *,
    annualization: int = ANNUALIZATION_DEFAULT,
    rf: float = 0.0,
    compounded: bool = True,
) -> pd.DataFrame:
    portfolio = _clean_series(portfolio_returns)
    benchmark = _clean_series(benchmark_returns) if benchmark_returns is not None else None
    rows = {"portfolio": _metric_row(portfolio, rf=rf, annualization=annualization, benchmark=benchmark, compounded=compounded)}
    if benchmark is not None:
        rows["benchmark"] = _metric_row(benchmark, rf=rf, annualization=annualization, benchmark=None, compounded=compounded)
        active_portfolio, active_benchmark = _align_series(portfolio, benchmark)
        if active_benchmark is not None and not active_portfolio.empty:
            rows["active"] = _metric_row(active_portfolio - active_benchmark, rf=0.0, annualization=annualization, benchmark=None, compounded=compounded)
    frame = pd.DataFrame(rows)
    frame.index.name = "metric"
    return frame


def _plot_cumulative_returns(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None,
    *,
    output_dir: Path,
    report_name: str,
) -> Path:
    path = output_dir / f"{report_name}_equity_curve.png"
    plt.figure(figsize=(12, 5))
    equity = _growth_curve(portfolio_returns) - 1.0
    equity.plot(color="#2a6fdb", linewidth=2.0, label="portfolio")
    if benchmark_returns is not None:
        _equity = _growth_curve(benchmark_returns) - 1.0
        _equity.plot(color="#6c757d", linewidth=1.8, label="benchmark")
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    finalize_quantstats_axis(plt.gca(), title="Cumulative returns", ylabel="Cumulative return", legend=True)
    save_quantstats_figure(path)
    return path


def _plot_snapshot(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None,
    *,
    output_dir: Path,
    report_name: str,
) -> Path:
    path = output_dir / f"{report_name}_snapshot.png"
    clean = _clean_series(portfolio_returns).fillna(0.0)
    benchmark_clean = _clean_series(benchmark_returns).fillna(0.0) if benchmark_returns is not None else None
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 7.4), gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle("Portfolio Snapshot", fontsize=13, y=0.975, fontweight="bold", color="black")
    fig.text(
        0.5,
        0.942,
        "Cumulative returns, drawdown, and daily return profile",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#6b7280",
    )
    axes[0].plot((1.0 + clean).cumprod() - 1.0, color="#2a6fdb", lw=1.5, label="portfolio")
    if benchmark_clean is not None and not benchmark_clean.empty:
        axes[0].plot((1.0 + benchmark_clean).cumprod() - 1.0, color="#6c757d", lw=1.2, label="benchmark")
    axes[0].set_title(
        f"{clean.index[0].strftime('%e %b %y')} - {clean.index[-1].strftime('%e %b %y')}  |  Sharpe: {_sharpe(clean):.2f}",
        fontsize=11.5,
        color="gray",
    )
    axes[0].axhline(0, color="silver", lw=1)
    finalize_quantstats_axis(axes[0], ylabel="Cumulative Return", legend=True)

    dd = _drawdown_series(clean) * 100.0
    axes[1].plot(dd, color="#6a4c93", lw=1.0)
    axes[1].fill_between(dd.index, 0, dd, color="#6a4c93", alpha=0.25)
    axes[1].axhline(0, color="silver", lw=1)
    finalize_quantstats_axis(axes[1], ylabel="Drawdown")

    axes[2].plot(clean * 100.0, color="#2a6fdb", lw=0.5)
    axes[2].axhline(0, color="silver", lw=1)
    finalize_quantstats_axis(axes[2], ylabel="Daily Return")
    for ax in axes:
        ax.set_facecolor("white")
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}%"))
    fig.set_facecolor("white")
    fig.autofmt_xdate()
    fig.subplots_adjust(top=0.88, hspace=0.22)
    save_quantstats_figure(path)
    return path


def _plot_earnings(
    portfolio_returns: pd.Series,
    *,
    output_dir: Path,
    report_name: str,
    start_balance: float = 100000.0,
) -> Path | None:
    clean = _clean_series(portfolio_returns).fillna(0.0)
    if clean.empty:
        return None
    path = output_dir / f"{report_name}_earnings.png"
    value = start_balance * (1.0 + clean).cumprod()
    max_ix = value.idxmax()
    max_val = value.max()
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Portfolio Earnings", fontsize=12.5, y=0.962, fontweight="bold", color="black")
    ax.plot(value.index, np.where(value.index == max_ix, max_val, np.nan), marker="o", lw=0, markersize=12, color="#2a6fdb")
    ax.plot(value.index, value, color="#6c757d", lw=1.4)
    ax.set_title(
        f"{clean.index[0].strftime('%e %b %y')} - {clean.index[-1].strftime('%e %b %y')} ;  P&L: ${value.iloc[-1] - value.iloc[0]:,.2f} ({(value.iloc[-1] / value.iloc[0] - 1) * 100:,.2f}%)",
        fontsize=10,
        color="gray",
    )
    ax.set_ylabel(f"Value of  ${start_balance:,.0f}", fontsize=11)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.set_facecolor("white")
    ax.set_facecolor("white")
    fig.autofmt_xdate()
    fig.subplots_adjust(top=0.88)
    finalize_quantstats_axis(ax)
    save_quantstats_figure(path)
    return path


def _plot_log_returns(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None,
    *,
    output_dir: Path,
    report_name: str,
) -> Path:
    path = output_dir / f"{report_name}_log_returns.png"
    plt.figure(figsize=(12, 4))
    portfolio = np.log1p(_clean_series(portfolio_returns).fillna(0.0)).cumsum()
    portfolio.plot(color="#1d3557", linewidth=2.0, label="portfolio")
    if benchmark_returns is not None:
        benchmark = np.log1p(_clean_series(benchmark_returns).fillna(0.0)).cumsum()
        benchmark.plot(color="#adb5bd", linewidth=1.6, label="benchmark")
    finalize_quantstats_axis(plt.gca(), title="Log cumulative returns", ylabel="Log cumulative return", legend=True)
    save_quantstats_figure(path)
    return path


def _plot_volatility_matched_returns(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None,
    *,
    output_dir: Path,
    report_name: str,
) -> Path | None:
    if benchmark_returns is None:
        return None
    path = output_dir / f"{report_name}_vol_returns.png"
    aligned_portfolio, aligned_benchmark = _align_series(portfolio_returns, benchmark_returns)
    if aligned_benchmark is None or aligned_portfolio.empty:
        return None
    portfolio_std = float(aligned_portfolio.std(ddof=1))
    benchmark_std = float(aligned_benchmark.std(ddof=1))
    scale = portfolio_std / benchmark_std if np.isfinite(benchmark_std) and benchmark_std != 0.0 else 1.0
    scaled_benchmark = aligned_benchmark * scale
    plt.figure(figsize=(12, 4))
    (_growth_curve(aligned_portfolio) - 1.0).plot(color="#2a6fdb", linewidth=2.0, label="portfolio")
    (_growth_curve(scaled_benchmark) - 1.0).plot(color="#6c757d", linewidth=1.8, label="benchmark (vol matched)")
    finalize_quantstats_axis(plt.gca(), title="Volatility matched returns", ylabel="Cumulative return", legend=True)
    save_quantstats_figure(path)
    return path


def _plot_yearly_returns(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None,
    *,
    output_dir: Path,
    report_name: str,
) -> Path | None:
    annual = _annual_returns(portfolio_returns)
    if annual.empty:
        return None
    path = output_dir / f"{report_name}_annual_returns.png"
    plt.figure(figsize=(12, 4))
    annual.plot(kind="bar", color="#2a6fdb", label="portfolio")
    if benchmark_returns is not None:
        benchmark_annual = _annual_returns(benchmark_returns)
        if not benchmark_annual.empty:
            benchmark_annual.reindex(annual.index).plot(kind="bar", color="#6c757d", alpha=0.45, label="benchmark")
    finalize_quantstats_axis(plt.gca(), title="Annual returns", ylabel="Return", legend=True)
    save_quantstats_figure(path)
    return path


def _plot_return_histogram(
    portfolio_returns: pd.Series,
    *,
    output_dir: Path,
    report_name: str,
) -> Path | None:
    clean = _clean_series(portfolio_returns)
    if clean.empty:
        return None
    path = output_dir / f"{report_name}_return_histogram.png"
    plt.figure(figsize=(10, 4))
    sns.histplot(clean, bins=30, color="#2a6fdb", kde=True)
    plt.axvline(clean.mean(), color="black", linestyle="--", linewidth=1.1, label="mean")
    finalize_quantstats_axis(plt.gca(), title="Return distribution", legend=True)
    save_quantstats_figure(path)
    return path


def _plot_distribution(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None,
    *,
    output_dir: Path,
    report_name: str,
) -> Path | None:
    clean = _clean_series(portfolio_returns)
    if clean.empty:
        return None
    path = output_dir / f"{report_name}_returns_dist.png"
    plt.figure(figsize=(10, 4))
    sns.histplot(clean, bins=40, color="#2a6fdb", kde=True, stat="density")
    if benchmark_returns is not None:
        benchmark_clean = _clean_series(benchmark_returns)
        if not benchmark_clean.empty:
            sns.kdeplot(benchmark_clean, color="#6c757d", linewidth=1.6, label="benchmark")
    finalize_quantstats_axis(plt.gca(), title="Distribution of Returns", xlabel="Return", legend=True)
    save_quantstats_figure(path)
    return path


def _plot_daily_returns(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None,
    *,
    output_dir: Path,
    report_name: str,
) -> Path | None:
    clean = _clean_series(portfolio_returns)
    if clean.empty:
        return None
    path = output_dir / f"{report_name}_daily_returns.png"
    plt.figure(figsize=(12, 4))
    plt.scatter(clean.index, clean.values, s=8, alpha=0.45, color="#2a6fdb", label="portfolio")
    if benchmark_returns is not None:
        aligned_portfolio, aligned_benchmark = _align_series(portfolio_returns, benchmark_returns)
        if aligned_benchmark is not None and not aligned_benchmark.empty:
            plt.scatter(aligned_benchmark.index, aligned_benchmark.values, s=8, alpha=0.25, color="#6c757d", label="benchmark")
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    finalize_quantstats_axis(plt.gca(), title="Daily returns", ylabel="Return", legend=True)
    save_quantstats_figure(path)
    return path


def _plot_rolling_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None,
    *,
    output_dir: Path,
    report_name: str,
    window: int = 126,
) -> Path | None:
    if benchmark_returns is None:
        return None
    beta = _rolling_beta(portfolio_returns, benchmark_returns, window=window)
    if beta.empty:
        return None
    path = output_dir / f"{report_name}_rolling_beta.png"
    plt.figure(figsize=(12, 4))
    beta.plot(color="#457b9d", linewidth=1.8)
    plt.axhline(1.0, color="black", linewidth=0.8, alpha=0.7)
    finalize_quantstats_axis(plt.gca(), title="Rolling beta", ylabel="Beta")
    save_quantstats_figure(path)
    return path


def _plot_rolling_volatility(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None,
    *,
    output_dir: Path,
    report_name: str,
    window: int = 63,
) -> Path | None:
    vol = _rolling_volatility(portfolio_returns, window=window)
    if vol.empty:
        return None
    path = output_dir / f"{report_name}_rolling_vol.png"
    plt.figure(figsize=(12, 4))
    vol.plot(color="#1f7a8c", linewidth=1.8, label="portfolio")
    if benchmark_returns is not None:
        bvol = _rolling_volatility(benchmark_returns, window=window)
        if not bvol.empty:
            bvol.plot(color="#adb5bd", linewidth=1.5, label="benchmark")
    finalize_quantstats_axis(plt.gca(), title="Rolling volatility", ylabel="Volatility", legend=True)
    save_quantstats_figure(path)
    return path


def _plot_rolling_sharpe(
    portfolio_returns: pd.Series,
    *,
    output_dir: Path,
    report_name: str,
    window: int = 63,
) -> Path | None:
    sharpe = _rolling_sharpe(portfolio_returns, window=window)
    if sharpe.empty:
        return None
    path = output_dir / f"{report_name}_rolling_sharpe.png"
    plt.figure(figsize=(12, 4))
    sharpe.plot(color="#1d4ed8", linewidth=1.8)
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    finalize_quantstats_axis(plt.gca(), title="Rolling Sharpe", ylabel="Sharpe")
    save_quantstats_figure(path)
    return path


def _plot_rolling_sortino(
    portfolio_returns: pd.Series,
    *,
    output_dir: Path,
    report_name: str,
    window: int = 63,
) -> Path | None:
    sortino = _rolling_sortino(portfolio_returns, window=window)
    if sortino.empty:
        return None
    path = output_dir / f"{report_name}_rolling_sortino.png"
    plt.figure(figsize=(12, 4))
    sortino.plot(color="#7c3aed", linewidth=1.8)
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    finalize_quantstats_axis(plt.gca(), title="Rolling Sortino", ylabel="Sortino")
    save_quantstats_figure(path)
    return path


def _plot_drawdown_periods(
    portfolio_returns: pd.Series,
    *,
    output_dir: Path,
    report_name: str,
) -> Path | None:
    details = _drawdown_details(_drawdown_series(portfolio_returns))
    if details.empty:
        return None
    top = details.sort_values(by="max drawdown", ascending=True).head(10)
    if top.empty:
        return None
    path = output_dir / f"{report_name}_drawdown_periods.png"
    plt.figure(figsize=(12, max(4, 0.45 * len(top))))
    y = np.arange(len(top))
    widths = top["days"].astype(float).to_numpy()
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(top)))
    plt.barh(y, widths, color=colors)
    plt.yticks(y, [f"{s} -> {e}" for s, e in zip(top["start"], top["end"])], fontsize=9)
    finalize_quantstats_axis(plt.gca(), title="Worst drawdown periods", xlabel="Days")
    save_quantstats_figure(path)
    return path


def _plot_drawdown(
    portfolio_returns: pd.Series,
    *,
    output_dir: Path,
    report_name: str,
) -> Path:
    path = output_dir / f"{report_name}_drawdown.png"
    plt.figure(figsize=(12, 4))
    _drawdown_series(portfolio_returns).plot(color="#6a4c93", linewidth=1.8)
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    finalize_quantstats_axis(plt.gca(), title="Drawdown", ylabel="Drawdown")
    save_quantstats_figure(path)
    return path


def _plot_monthly_heatmap(
    portfolio_returns: pd.Series,
    *,
    output_dir: Path,
    report_name: str,
) -> Path | None:
    heatmap = _monthly_heatmap(portfolio_returns)
    if heatmap.empty:
        return None
    path = output_dir / f"{report_name}_monthly_heatmap.png"
    plt.figure(figsize=(12, max(4, 0.35 * len(heatmap))))
    sns.heatmap(heatmap, annot=True, fmt=".1%", cmap="RdYlGn", center=0.0, cbar_kws={"label": "Return"})
    finalize_quantstats_axis(plt.gca(), title="Monthly returns")
    save_quantstats_figure(path)
    return path


def _render_html(
    summary_table: pd.DataFrame,
    figure_paths: list[Path],
    report_name: str,
    *,
    comparison_table: pd.DataFrame | None = None,
    compare_tables: dict[str, pd.DataFrame] | None = None,
    drawdown_table: pd.DataFrame | None = None,
    monthly_returns_table: pd.DataFrame | None = None,
    montecarlo_summary: pd.DataFrame | None = None,
    title: str = "Tearsheet",
    date_range: str = "",
    params_text: str = "",
    matched_dates_text: str = "",
) -> str:
    def _figure(name: str) -> str:
        for path in figure_paths:
            if path.stem == name or path.name == name:
                return f"<img src='{escape(path.name)}' alt='{escape(path.stem)}' />"
        return ""

    metrics_html = summary_table.to_html(classes="summary-table", border=0, escape=False)
    eoy_html = ""
    eoy_title = ""
    if comparison_table is not None and not comparison_table.empty:
        eoy_title = "<h3>EOY Returns vs Benchmark</h3>"
        eoy_html = comparison_table.to_html(classes="summary-table", border=0, escape=False)
    elif comparison_table is None:
        eoy_title = "<h3>EOY Returns</h3>"
        eoy_html = ""

    compare_html = ""
    if compare_tables:
        compare_chunks: list[str] = ["<h3>Compare Tables</h3>"]
        compare_order = ["daily", "weekly", "monthly", "quarterly", "yearly"]
        compare_titles = {
            "daily": "Daily Compare",
            "weekly": "Weekly Compare",
            "monthly": "Monthly Compare",
            "quarterly": "Quarterly Compare",
            "yearly": "Yearly Compare",
        }
        for key in compare_order:
            table = compare_tables.get(key)
            if table is None or table.empty:
                continue
            compare_chunks.append(f"<h4>{escape(compare_titles[key])}</h4>")
            compare_chunks.append(table.to_html(classes='summary-table', border=0, escape=False))
        compare_html = "".join(compare_chunks)

    dd_html = ""
    if drawdown_table is not None and not drawdown_table.empty:
        dd_html = drawdown_table.to_html(classes="summary-table", border=0, escape=False)

    monthly_html = ""
    if monthly_returns_table is not None and not monthly_returns_table.empty:
        monthly_html = "<h3>Monthly Returns</h3>" + monthly_returns_table.to_html(classes="summary-table", border=0, escape=False)

    mc_html = ""
    if montecarlo_summary is not None and not montecarlo_summary.empty:
        mc_html = montecarlo_summary.to_html(classes="summary-table", border=0, escape=False)

    sections = {
        "snapshot": _figure(f"{report_name}_snapshot"),
        "earnings": _figure(f"{report_name}_earnings"),
        "returns": _figure(f"{report_name}_equity_curve"),
        "log_returns": _figure(f"{report_name}_log_returns"),
        "vol_returns": _figure(f"{report_name}_vol_returns"),
        "eoy_returns": _figure(f"{report_name}_annual_returns"),
        "monthly_dist": _figure(f"{report_name}_return_histogram"),
        "daily_returns": _figure(f"{report_name}_daily_returns"),
        "rolling_beta": _figure(f"{report_name}_rolling_beta"),
        "rolling_vol": _figure(f"{report_name}_rolling_vol"),
        "rolling_sharpe": _figure(f"{report_name}_rolling_sharpe"),
        "rolling_sortino": _figure(f"{report_name}_rolling_sortino"),
        "dd_periods": _figure(f"{report_name}_drawdown_periods"),
        "dd_plot": _figure(f"{report_name}_drawdown"),
        "monthly_heatmap": _figure(f"{report_name}_monthly_heatmap"),
        "returns_dist": _figure(f"{report_name}_returns_dist"),
        "montecarlo_plot": _figure(f"{report_name}_montecarlo"),
    }
    return (
        "<!-- generated by TigerQuant for multifactor evaluation -->"
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta http-equiv='X-UA-Compatible' content='IE=edge,chrome=1'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1, shrink-to-fit=no'>"
        f"<title>{escape(title)}</title>"
        "<meta name='robots' content='noindex, nofollow'>"
        "<style>"
        "body{-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale;margin:30px;background:#fff;color:#000}"
        "body,p,table,td,th{font:13px/1.4 Arial,sans-serif}.container{max-width:960px;margin:auto}"
        "img,svg{width:100%}h1,h2,h3,h4{font-weight:400;margin:0}h1 dt{display:inline;margin-left:10px;font-size:14px}h3{margin-bottom:10px;font-weight:700}h4{color:grey}h4 a{color:#09c;text-decoration:none}h4 a:hover{color:#069;text-decoration:underline}hr{margin:25px 0 40px;height:0;border:0;border-top:1px solid #ccc}#left{width:620px;margin-right:18px;margin-top:-1.2rem;float:left}#right{width:320px;float:right}#left svg{margin:-1.5rem 0}#monthly_heatmap{overflow:hidden}#monthly_heatmap svg{margin:-1.5rem 0}table{margin:0 0 40px;border:0;border-spacing:0;width:100%}table td,table th{text-align:right;padding:4px 5px 3px 5px}table th{text-align:right;padding:6px 5px 5px 5px}table td:first-of-type,table th:first-of-type{text-align:left;padding-left:2px}table td:last-of-type,table th:last-of-type{text-align:right;padding-right:2px}td hr{margin:5px 0}table th{font-weight:400}table thead th{font-weight:700;background:#eee}#eoy table td:after{content:\"%\"}#eoy table td:first-of-type:after,#eoy table td:last-of-type:after,#eoy table td:nth-of-type(4):after{content:\"\"}#eoy table th{text-align:right}#eoy table th:first-of-type{text-align:left}#eoy table td:after{content:\"%\"}#eoy table td:first-of-type:after,#eoy table td:last-of-type:after{content:\"\"}#ddinfo table td:nth-of-type(3):after{content:\"%\"}#ddinfo table th{text-align:right}#ddinfo table td:first-of-type,#ddinfo table td:nth-of-type(2),#ddinfo table th:first-of-type,#ddinfo table th:nth-of-type(2){text-align:left}#ddinfo table td:nth-of-type(3):after{content:\"%\"}"
        "@media print{hr{margin:25px 0}body{margin:0}.container{max-width:100%;margin:0}#left{width:55%;margin:0}#left svg{margin:0 0 -10%}#left svg:first-of-type{margin-top:-30%}#right{margin:0;width:45%}}"
        "</style></head><body>"
        "<div class='container'>"
        f"<h1>{escape(title)} <dt>{escape(date_range)}{escape(matched_dates_text)}</dt></h1>"
        f"<h4>{escape(params_text)} Generated by <a href='http://quantstats.io' target='quantstats'>QuantStats</a> style report (TigerQuant)</h4>"
        "<hr>"
        "<div id='left'>"
        f"<div>{sections['snapshot']}</div>"
        f"<div>{sections['earnings']}</div>"
        f"<div>{sections['returns']}</div>"
        f"<div id='log_returns'>{sections['log_returns']}</div>"
        f"<div id='vol_returns'>{sections['vol_returns']}</div>"
        f"<div id='eoy_returns'>{sections['eoy_returns']}</div>"
        f"<div id='monthly_dist'>{sections['monthly_dist']}</div>"
        f"<div id='daily_returns'>{sections['daily_returns']}</div>"
        f"<div id='rolling_beta'>{sections['rolling_beta']}</div>"
        f"<div id='rolling_vol'>{sections['rolling_vol']}</div>"
        f"<div id='rolling_sharpe'>{sections['rolling_sharpe']}</div>"
        f"<div id='rolling_sortino'>{sections['rolling_sortino']}</div>"
        f"<div id='dd_periods'>{sections['dd_periods']}</div>"
        f"<div id='dd_plot'>{sections['dd_plot']}</div>"
        f"<div id='monthly_heatmap'>{sections['monthly_heatmap']}</div>"
        f"<div id='returns_dist'>{sections['returns_dist']}</div>"
        "</div>"
        "<div id='right'>"
        "<h3>Summary Metrics / Key Performance Metrics</h3>"
        f"{metrics_html}"
        f"<div id='eoy'>{eoy_title}{eoy_html}</div>"
        f"<div id='compare'>{compare_html}</div>"
        f"<div id='monthly_returns'>{monthly_html}</div>"
        "<div id='ddinfo'><h3>Worst 10 Drawdowns</h3>"
        f"{dd_html}</div>"
        f"<div id='montecarlo'><h3>Monte Carlo</h3>{mc_html}{sections['montecarlo_plot']}</div>"
        "</div></div><style>*{white-space:auto !important;}</style></body></html>"
    )


def create_multifactor_summary_report(
    backtest: pd.DataFrame,
    *,
    output_dir: str | Path,
    report_name: str = "summary_report",
    portfolio_column: str = "portfolio",
    benchmark_column: str = "benchmark",
    annualization: int = ANNUALIZATION_DEFAULT,
    format: str = "all",
    open_browser: bool = False,
) -> MultifactorSummaryReportResult | None:
    if portfolio_column not in backtest.columns:
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    portfolio = backtest[portfolio_column]
    benchmark = backtest[benchmark_column] if benchmark_column in backtest.columns else None
    portfolio, benchmark = _align_series(portfolio, benchmark)
    if portfolio.empty:
        return None

    summary_table = _summary_table(
        portfolio,
        benchmark,
        annualization=annualization,
        compounded=True,
    )
    summary_table_path = output_path / f"{report_name}_summary.parquet"
    summary_export = summary_table.reset_index().rename(columns={"index": "metric"})
    to_parquet_clean(summary_export.astype(str), summary_table_path)
    summary_export.to_csv(output_path / f"{report_name}_summary.csv", index=False)

    portfolio_returns_path = output_path / f"{report_name}_portfolio_returns.csv"
    portfolio.rename("portfolio").to_csv(portfolio_returns_path)
    benchmark_returns_path = None
    if benchmark is not None:
        benchmark_returns_path = output_path / f"{report_name}_benchmark_returns.csv"
        benchmark.rename("benchmark").to_csv(benchmark_returns_path)

    normalized_format = str(format).strip().lower()
    valid_formats = {"img", "table", "img_table", "report", "all"}
    if normalized_format not in valid_formats:
        raise ValueError(f"format must be one of: {', '.join(sorted(valid_formats))}")
    write_figures = normalized_format in {"img", "img_table", "report", "all"}
    write_tables = normalized_format in {"table", "img_table", "report", "all"}
    write_html = normalized_format in {"report", "all"}

    comparison_table = _yearly_comparison_table(portfolio, benchmark) if benchmark is not None else None
    comparison_table_path = None
    if write_tables and comparison_table is not None and not comparison_table.empty:
        comparison_table_path = output_path / f"{report_name}_eoy_comparison.parquet"
        to_parquet_clean(comparison_table.reset_index().astype(str), comparison_table_path)
        comparison_table.reset_index().to_csv(output_path / f"{report_name}_eoy_comparison.csv", index=False)

    drawdown_table = _drawdown_details(_drawdown_series(portfolio)).sort_values(by="max drawdown", ascending=True)
    drawdown_table_path = None
    if write_tables and not drawdown_table.empty:
        drawdown_table_path = output_path / f"{report_name}_drawdown_details.parquet"
        to_parquet_clean(drawdown_table.astype(str), drawdown_table_path)
        drawdown_table.to_csv(output_path / f"{report_name}_drawdown_details.csv", index=False)

    monthly_returns_table = _monthly_returns_table(portfolio)
    monthly_returns_table_path = None
    if write_tables and not monthly_returns_table.empty:
        monthly_returns_table_path = output_path / f"{report_name}_monthly_returns.parquet"
        to_parquet_clean(monthly_returns_table.astype(str), monthly_returns_table_path)
        monthly_returns_table.to_csv(output_path / f"{report_name}_monthly_returns.csv")

    compare_table_paths: dict[str, Path] = {}
    compare_tables: dict[str, pd.DataFrame] = {}
    if benchmark is not None:
        compare_specs = {
            "daily": None,
            "weekly": "W",
            "monthly": "ME",
            "quarterly": "QE",
            "yearly": "YE",
        }
        for label, aggregate in compare_specs.items():
            table = _compare_table(portfolio, benchmark, aggregate=aggregate, compounded=True)
            if table.empty:
                continue
            compare_tables[label] = table
            path = output_path / f"{report_name}_{label}_compare.parquet"
            to_parquet_clean(table.reset_index().astype(str), path)
            table.reset_index().to_csv(output_path / f"{report_name}_{label}_compare.csv", index=False)
            compare_table_paths[label] = path

    figure_paths: list[Path] = []
    if write_figures or write_html:
        for path in (
            _plot_snapshot(portfolio, benchmark, output_dir=output_path, report_name=report_name),
            _plot_earnings(portfolio, output_dir=output_path, report_name=report_name),
            _plot_cumulative_returns(portfolio, benchmark, output_dir=output_path, report_name=report_name),
            _plot_log_returns(portfolio, benchmark, output_dir=output_path, report_name=report_name),
            _plot_volatility_matched_returns(portfolio, benchmark, output_dir=output_path, report_name=report_name),
            _plot_yearly_returns(portfolio, benchmark, output_dir=output_path, report_name=report_name),
            _plot_return_histogram(portfolio, output_dir=output_path, report_name=report_name),
            _plot_distribution(portfolio, benchmark, output_dir=output_path, report_name=report_name),
            _plot_daily_returns(portfolio, benchmark, output_dir=output_path, report_name=report_name),
            _plot_rolling_beta(portfolio, benchmark, output_dir=output_path, report_name=report_name),
            _plot_rolling_volatility(portfolio, benchmark, output_dir=output_path, report_name=report_name),
            _plot_rolling_sharpe(portfolio, output_dir=output_path, report_name=report_name),
            _plot_rolling_sortino(portfolio, output_dir=output_path, report_name=report_name),
            _plot_drawdown_periods(portfolio, output_dir=output_path, report_name=report_name),
            _plot_drawdown(portfolio, output_dir=output_path, report_name=report_name),
            _plot_monthly_heatmap(portfolio, output_dir=output_path, report_name=report_name),
        ):
            if path is not None:
                figure_paths.append(path)

    montecarlo_paths = _montecarlo_paths(portfolio, sims=500)
    montecarlo_summary = None
    montecarlo_summary_path = None
    montecarlo_plot_path = None
    if write_tables and not montecarlo_paths.empty:
        montecarlo_summary = _montecarlo_summary(montecarlo_paths)
        if not montecarlo_summary.empty:
            montecarlo_summary_path = output_path / f"{report_name}_montecarlo_summary.parquet"
            to_parquet_clean(montecarlo_summary.astype(str), montecarlo_summary_path)
            montecarlo_summary.to_csv(output_path / f"{report_name}_montecarlo_summary.csv", index=False)
        if write_figures or write_html:
            montecarlo_plot_path = _plot_montecarlo(montecarlo_paths, output_dir=output_path, report_name=report_name)
            if montecarlo_plot_path is not None:
                figure_paths.append(montecarlo_plot_path)

    html_path = output_path / f"{report_name}.html"
    if write_html:
        html = render_summary_report_html(
            summary_table,
            figure_paths,
            report_name,
            comparison_table=comparison_table,
            compare_tables=compare_tables,
            drawdown_table=drawdown_table,
            monthly_returns_table=monthly_returns_table,
            montecarlo_summary=montecarlo_summary,
            title="Multifactor Tearsheet",
            date_range=f"{portfolio.index.strftime('%e %b, %Y')[0]} - {portfolio.index.strftime('%e %b, %Y')[-1]}",
            params_text=f"Benchmark: {benchmark_column.upper()} • Periods/Year: {annualization} • RF: 0.0%",
            matched_dates_text=" (matched dates)" if benchmark is not None else "",
        )
        html_path.write_text(html, encoding="utf-8")

    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "report_name": report_name,
                "portfolio_column": portfolio_column,
                "benchmark_column": benchmark_column,
                "annualization": annualization,
                "summary_table_path": str(summary_table_path),
                "comparison_table_path": str(comparison_table_path) if comparison_table_path else None,
                "drawdown_table_path": str(drawdown_table_path) if drawdown_table_path else None,
                "monthly_returns_table_path": str(monthly_returns_table_path) if monthly_returns_table_path else None,
                "compare_table_paths": {k: str(v) for k, v in compare_table_paths.items()} if compare_table_paths else None,
                "montecarlo_summary_path": str(montecarlo_summary_path) if montecarlo_summary_path else None,
                "montecarlo_plot_path": str(montecarlo_plot_path) if montecarlo_plot_path else None,
                "html_path": str(html_path),
                "figure_paths": [str(path) for path in figure_paths],
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    if open_browser and html_path.exists():
        webbrowser.open(html_path.as_uri())

    return MultifactorSummaryReportResult(
        output_dir=output_path,
        report_name=report_name,
        html_path=html_path,
        summary_table_path=summary_table_path,
        figure_paths=figure_paths,
        portfolio_returns_path=portfolio_returns_path,
        benchmark_returns_path=benchmark_returns_path,
        manifest_path=manifest_path,
        comparison_table_path=comparison_table_path,
        drawdown_table_path=drawdown_table_path,
        monthly_returns_table_path=monthly_returns_table_path,
        montecarlo_summary_path=montecarlo_summary_path,
        montecarlo_plot_path=montecarlo_plot_path,
        compare_table_paths=compare_table_paths or None,
    )


def create_summary_tear_sheet(*args, **kwargs):
    return create_multifactor_summary_report(*args, **kwargs)


__all__ = [
    "adjusted_sortino",
    "best",
    "autocorr_penalty",
    "avg_loss",
    "avg_return",
    "avg_win",
    "beta_alpha",
    "calmar",
    "comp",
    "compsum",
    "common_sense_ratio",
    "conditional_value_at_risk",
    "cpc_index",
    "consecutive_losses",
    "consecutive_wins",
    "create_multifactor_summary_report",
    "create_summary_tear_sheet",
    "compare",
    "cvar",
    "daily_returns",
    "drawdown_details",
    "drawdown",
    "drawdowns_periods",
    "distribution",
    "earnings",
    "expected_return",
    "expected_shortfall",
    "exposure",
    "ghpr",
    "geometric_mean",
    "gain_to_pain_ratio",
    "greeks",
    "histogram",
    "implied_volatility",
    "information_ratio",
    "kelly_criterion",
    "kurtosis",
    "max_drawdown",
    "montecarlo",
    "montecarlo_cagr",
    "montecarlo_drawdown",
    "montecarlo_sharpe",
    "monthly_returns",
    "monthly_heatmap",
    "omega",
    "outliers",
    "payoff_ratio",
    "pct_rank",
    "probabilistic_adjusted_sortino_ratio",
    "probabilistic_ratio",
    "probabilistic_sharpe_ratio",
    "probabilistic_sortino_ratio",
    "profit_factor",
    "profit_ratio",
    "rar",
    "r2",
    "r_squared",
    "ror",
    "recovery_factor",
    "remove_outliers",
    "risk_of_ruin",
    "risk_return_ratio",
    "rolling_sharpe",
    "rolling_sortino",
    "rolling_volatility",
    "rolling_greeks",
    "serenity_index",
    "sharpe",
    "safe_concat",
    "safe_resample",
    "smart_sharpe",
    "smart_sortino",
    "skew",
    "snapshot",
    "sortino",
    "worst",
    "tail_ratio",
    "to_drawdown_series",
    "to_plotly",
    "treynor_ratio",
    "upi",
    "ulcer_index",
    "ulcer_performance_index",
    "var",
    "value_at_risk",
    "volatility",
    "win_loss_ratio",
    "win_rate",
    "yearly_returns",
    "warn",
    "MultifactorSummaryReportResult",
]
