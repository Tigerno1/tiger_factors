from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Mapping

import numpy as np
import pandas as pd

from tiger_factors.multifactor_evaluation._inputs import coerce_factor_series
from tiger_factors.multifactor_evaluation.reporting.portfolio import run_portfolio_from_backtest
from tiger_factors.utils.weighting import normalize_weights


def _ensure_datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.index = pd.to_datetime(result.index, errors="coerce")
    result = result.loc[~result.index.isna()].sort_index()
    result.index.name = result.index.name or "date_"
    result.columns = result.columns.astype(str)
    return result


def coerce_factor_panel(factor: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Convert a factor input into a wide date x code panel.

    Accepted inputs:
    - MultiIndex Series indexed by ``(date_, code)``
    - long DataFrame with ``date_`` and ``code`` columns
    - wide DataFrame already shaped like date x code
    """

    if isinstance(factor, pd.Series):
        series = coerce_factor_series(factor)
        panel = series.unstack("code").sort_index()
        return _ensure_datetime_index(panel)

    if not isinstance(factor, pd.DataFrame):
        raise TypeError("factor must be a pandas Series or DataFrame")

    if {"date_", "code"}.issubset(factor.columns):
        series = coerce_factor_series(factor)
        panel = series.unstack("code").sort_index()
        return _ensure_datetime_index(panel)

    panel = factor.copy()
    return _ensure_datetime_index(panel)


def standardize_cross_section(panel: pd.DataFrame) -> pd.DataFrame:
    """Z-score each date row across stocks."""

    frame = coerce_factor_panel(panel)
    mean = frame.mean(axis=1)
    std = frame.std(axis=1, ddof=0).replace(0.0, np.nan)
    standardized = frame.sub(mean, axis=0).div(std, axis=0)
    return standardized.replace([np.inf, -np.inf], np.nan)


def _row_normalize(signal: pd.DataFrame, *, long_only: bool, gross_exposure: float) -> pd.DataFrame:
    frame = signal.copy().replace([np.inf, -np.inf], np.nan)
    gross_exposure = float(gross_exposure)
    if gross_exposure < 0:
        raise ValueError("gross_exposure must be non-negative")

    if long_only:
        frame = frame.clip(lower=0.0)
        totals = frame.sum(axis=1).replace(0.0, np.nan)
        weights = frame.div(totals, axis=0) * gross_exposure
    else:
        frame = frame.sub(frame.mean(axis=1), axis=0)
        totals = frame.abs().sum(axis=1).replace(0.0, np.nan)
        weights = frame.div(totals, axis=0) * gross_exposure

    return weights.fillna(0.0)


def _annualized_return_stats(returns: pd.Series, annual_trading_days: int = 252) -> dict[str, float]:
    series = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return {
            "ann_return": 0.0,
            "ann_volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "win_rate": 0.0,
            "total_return": 0.0,
        }
    cumulative = (1.0 + series).cumprod()
    total_return = float(cumulative.iloc[-1] - 1.0)
    ann_return = float((1.0 + total_return) ** (annual_trading_days / max(len(series), 1)) - 1.0)
    ann_vol = float(series.std(ddof=0) * np.sqrt(annual_trading_days))
    sharpe = float(ann_return / ann_vol) if ann_vol > 1e-12 else 0.0
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    calmar = float(ann_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0
    win_rate = float((series > 0).mean())
    return {
        "ann_return": ann_return,
        "ann_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "win_rate": win_rate,
        "total_return": total_return,
    }


@dataclass(frozen=True)
class FactorPortfolioResult:
    signal: pd.DataFrame
    weights: pd.DataFrame
    long_only: bool
    gross_exposure: float
    standardize: bool
    factor_weights: dict[str, float] | None = None
    factor_names: tuple[str, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal": self.signal.to_dict(orient="split"),
            "weights": self.weights.to_dict(orient="split"),
            "long_only": self.long_only,
            "gross_exposure": self.gross_exposure,
            "standardize": self.standardize,
            "factor_weights": self.factor_weights,
            "factor_names": list(self.factor_names) if self.factor_names is not None else None,
        }


@dataclass(frozen=True)
class FactorPortfolioWorkflowResult:
    portfolio: FactorPortfolioResult
    backtest: pd.DataFrame
    stats: dict[str, dict[str, float]]
    positions: pd.DataFrame
    report: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "portfolio": self.portfolio.to_dict(),
            "backtest": self.backtest.to_dict(orient="split"),
            "stats": self.stats,
            "positions": self.positions.to_dict(orient="split"),
            "report": None if self.report is None else getattr(self.report, "to_summary", lambda: self.report)(),
        }


def factor_to_stock_portfolio(
    factor: pd.Series | pd.DataFrame,
    *,
    long_only: bool = False,
    gross_exposure: float = 1.0,
    standardize: bool = True,
) -> FactorPortfolioResult:
    """Convert one factor panel into stock weights.

    The function first standardizes each cross-section when requested, then
    converts the signal into a portfolio:
    - long-only: positive scores are normalized to sum to ``gross_exposure``
    - long-short: the cross-section is demeaned and normalized by absolute sum
    """

    panel = coerce_factor_panel(factor)
    signal = standardize_cross_section(panel) if standardize else panel.copy()
    weights = _row_normalize(signal, long_only=long_only, gross_exposure=gross_exposure)
    return FactorPortfolioResult(
        signal=signal,
        weights=weights,
        long_only=bool(long_only),
        gross_exposure=float(gross_exposure),
        standardize=bool(standardize),
    )


def _blend_factor_panels(
    factor_panels: Mapping[str, pd.Series | pd.DataFrame],
    factor_weights: Mapping[str, float],
    *,
    standardize: bool,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if not factor_panels:
        raise ValueError("factor_panels must not be empty")

    panels = {str(name): panel for name, panel in factor_panels.items()}
    selected = [str(name) for name in factor_weights.keys() if str(name) in panels]
    if not selected:
        raise ValueError("factor_weights does not overlap with factor_panels")

    normalized_weights = normalize_weights({name: float(factor_weights[name]) for name in selected})

    signal_frames: list[pd.DataFrame] = []
    for name in selected:
        panel = coerce_factor_panel(panels[name])
        signal_frames.append((standardize_cross_section(panel) if standardize else panel) * float(normalized_weights[name]))

    combined = pd.concat(signal_frames).groupby(level=0).sum().sort_index()
    combined.columns = combined.columns.astype(str)
    return combined, normalized_weights


def multi_factor_to_stock_portfolio(
    factor_panels: Mapping[str, pd.Series | pd.DataFrame],
    *,
    factor_weights: Mapping[str, float] | None = None,
    long_only: bool = False,
    gross_exposure: float = 1.0,
    standardize: bool = True,
) -> FactorPortfolioResult:
    """Blend several factors into one stock portfolio.

    If ``factor_weights`` is omitted, the factors are equally weighted.
    The blended signal is then converted into stock weights using the same
    single-factor rules as :func:`factor_to_stock_portfolio`.
    """

    normalized_panels = {str(name): panel for name, panel in factor_panels.items()}
    factor_names = tuple(normalized_panels.keys())
    if not factor_names:
        raise ValueError("factor_panels must not be empty")

    weights = normalize_weights(
        {name: 1.0 for name in factor_names} if factor_weights is None else {str(k): float(v) for k, v in factor_weights.items()}
    )
    signal, normalized_factor_weights = _blend_factor_panels(
        normalized_panels,
        weights,
        standardize=standardize,
    )
    stock_weights = _row_normalize(signal, long_only=long_only, gross_exposure=gross_exposure)
    return FactorPortfolioResult(
        signal=signal,
        weights=stock_weights,
        long_only=bool(long_only),
        gross_exposure=float(gross_exposure),
        standardize=bool(standardize),
        factor_weights=normalized_factor_weights,
        factor_names=factor_names,
    )


def weights_to_positions_frame(
    weights: pd.DataFrame | pd.Series,
    *,
    date_column: str = "date_",
    code_column: str = "code",
    weight_column: str = "weight",
) -> pd.DataFrame:
    """Convert a stock weight panel into a long positions frame."""

    if isinstance(weights, pd.Series):
        if not isinstance(weights.index, pd.MultiIndex) or weights.index.nlevels != 2:
            raise ValueError("weight series must use a (date_, code) MultiIndex.")
        frame = weights.rename(weight_column).reset_index()
        frame.columns = [date_column, code_column, weight_column]
        frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
        frame = frame.dropna(subset=[date_column, weight_column]).sort_values([date_column, code_column]).reset_index(drop=True)
        return frame

    if not isinstance(weights, pd.DataFrame):
        raise TypeError("weights must be a pandas DataFrame or Series")

    frame = weights.copy()
    if {date_column, code_column, weight_column}.issubset(frame.columns):
        frame = frame[[date_column, code_column, weight_column]].copy()
        frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
        frame[weight_column] = pd.to_numeric(frame[weight_column], errors="coerce")
        frame = frame.dropna(subset=[date_column, code_column, weight_column]).sort_values([date_column, code_column]).reset_index(drop=True)
        frame[code_column] = frame[code_column].astype(str)
        return frame

    if {"date_", "code"}.issubset(frame.columns):
        value_columns = [col for col in frame.columns if col not in {"date_", "code"}]
        if len(value_columns) != 1:
            raise ValueError("long weight frame must contain exactly one value column besides date_ and code.")
        frame = frame.rename(columns={value_columns[0]: weight_column})
        return weights_to_positions_frame(frame, date_column=date_column, code_column=code_column, weight_column=weight_column)

    wide = coerce_factor_panel(frame)
    long = wide.stack(future_stack=True).rename(weight_column).reset_index()
    long.columns = [date_column, code_column, weight_column]
    long[date_column] = pd.to_datetime(long[date_column], errors="coerce")
    long[weight_column] = pd.to_numeric(long[weight_column], errors="coerce")
    long = long.dropna(subset=[date_column, weight_column]).sort_values([date_column, code_column]).reset_index(drop=True)
    long[code_column] = long[code_column].astype(str)
    return long


def run_weight_panel_backtest(
    weight_panel: pd.DataFrame,
    close_panel: pd.DataFrame,
    *,
    rebalance_freq: str = "ME",
    annual_trading_days: int = 252,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Backtest a precomputed stock-weight panel against a close panel.

    The input weights are treated as target holdings at each rebalance date.
    """

    weights = coerce_factor_panel(weight_panel).ffill().sort_index()
    close = close_panel.ffill().sort_index()
    daily_returns = close.pct_change(fill_method=None)

    weights_eom = weights.resample(rebalance_freq).last()
    if start is not None:
        weights_eom = weights_eom.loc[weights_eom.index >= pd.Timestamp(start)]
    if end is not None:
        weights_eom = weights_eom.loc[weights_eom.index <= pd.Timestamp(end)]

    rebalance_dates = weights_eom.index
    records: list[dict[str, float | int | pd.Timestamp]] = []
    position_records: list[dict[str, float | pd.Timestamp]] = []
    previous_weights: dict[str, float] = {}
    friction_bps = max(float(transaction_cost_bps), 0.0) + max(float(slippage_bps), 0.0)

    for i, rebalance_date in enumerate(rebalance_dates[:-1]):
        next_rebalance = rebalance_dates[i + 1]
        target = pd.to_numeric(weights_eom.loc[rebalance_date], errors="coerce").dropna()
        if target.empty:
            continue

        total = float(target.abs().sum())
        if total > 1e-12:
            target = target / total
        target_weights = {str(code): float(weight) for code, weight in target.items()}

        all_codes = sorted(set(previous_weights) | set(target_weights))
        turnover = 0.5 * sum(abs(float(target_weights.get(code, 0.0)) - float(previous_weights.get(code, 0.0))) for code in all_codes)
        rebalance_cost = turnover * friction_bps / 10000.0
        previous_weights = target_weights

        holding_mask = (daily_returns.index > rebalance_date) & (daily_returns.index <= next_rebalance)
        holding_slice = daily_returns.loc[holding_mask]
        first_day = True
        for date, row in holding_slice.iterrows():
            portfolio_return = float(
                sum(float(row.get(code, 0.0)) * weight for code, weight in target_weights.items() if pd.notna(row.get(code, np.nan)))
            )
            cost = rebalance_cost if first_day else 0.0
            portfolio_return -= cost
            first_day = False
            benchmark_return = float(row.mean(skipna=True))
            records.append(
                {
                    "date": date,
                    "portfolio": portfolio_return,
                    "benchmark": benchmark_return,
                    "turnover": turnover if cost > 0 else 0.0,
                    "cost": cost,
                }
            )
            position_record: dict[str, float | pd.Timestamp] = {"date": date}
            position_record.update(target_weights)
            position_record["cash"] = float(1.0 - sum(target_weights.values()))
            position_records.append(position_record)

    if not records:
        backtest = pd.DataFrame(columns=["portfolio", "benchmark", "turnover", "cost"])
        backtest.attrs["positions"] = pd.DataFrame()
        backtest.attrs["close_panel"] = close
        return backtest, {
            "portfolio": {"ann_return": 0.0, "ann_volatility": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "calmar": 0.0, "win_rate": 0.0, "total_return": 0.0},
            "benchmark": {"ann_return": 0.0, "ann_volatility": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "calmar": 0.0, "win_rate": 0.0, "total_return": 0.0},
        }

    backtest = pd.DataFrame(records).set_index("date").sort_index()
    backtest.index = pd.DatetimeIndex(backtest.index)
    positions_frame = pd.DataFrame.from_records(position_records)
    if not positions_frame.empty:
        positions_frame["date"] = pd.to_datetime(positions_frame["date"], errors="coerce")
        positions_frame = positions_frame.dropna(subset=["date"]).set_index("date").sort_index()
        positions_frame.index = pd.DatetimeIndex(positions_frame.index, name="date_")
        positions_frame = positions_frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    positions_frame.columns = positions_frame.columns.astype(str)
    backtest.attrs["positions"] = positions_frame
    backtest.attrs["close_panel"] = close
    stats = {
        "portfolio": _annualized_return_stats(backtest["portfolio"], annual_trading_days),
        "benchmark": _annualized_return_stats(backtest["benchmark"], annual_trading_days),
    }
    return backtest, stats


def run_factor_portfolio_workflow(
    factor: pd.Series | pd.DataFrame | Mapping[str, pd.Series | pd.DataFrame],
    close_panel: pd.DataFrame,
    *,
    factor_weights: Mapping[str, float] | None = None,
    long_only: bool = False,
    gross_exposure: float = 1.0,
    standardize: bool = True,
    rebalance_freq: str = "ME",
    annual_trading_days: int = 252,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    output_dir: str | os.PathLike[str] | None = None,
    report_name: str = "factor_portfolio",
    open_report: bool = False,
) -> FactorPortfolioWorkflowResult:
    """Build a factor portfolio, backtest it, and optionally render a report."""

    if isinstance(factor, Mapping):
        portfolio = multi_factor_to_stock_portfolio(
            factor,
            factor_weights=factor_weights,
            long_only=long_only,
            gross_exposure=gross_exposure,
            standardize=standardize,
        )
    else:
        portfolio = factor_to_stock_portfolio(
            factor,
            long_only=long_only,
            gross_exposure=gross_exposure,
            standardize=standardize,
        )

    backtest, stats = run_weight_panel_backtest(
        portfolio.weights,
        close_panel,
        rebalance_freq=rebalance_freq,
        annual_trading_days=annual_trading_days,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
    )
    positions = backtest.attrs.get("positions", pd.DataFrame()).copy()

    report = None
    if output_dir is not None:
        report = run_portfolio_from_backtest(
            backtest,
            output_dir=output_dir,
            report_name=report_name,
        )
        if report is not None and open_report:
            report.get_report(open_browser=False)

    return FactorPortfolioWorkflowResult(
        portfolio=portfolio,
        backtest=backtest,
        stats=stats,
        positions=positions,
        report=report,
    )


def summarize_factor_portfolio_holdings(
    factor: pd.Series | pd.DataFrame | Mapping[str, pd.Series | pd.DataFrame],
    *,
    factor_weights: Mapping[str, float] | None = None,
    long_only: bool = False,
    gross_exposure: float = 1.0,
    standardize: bool = True,
    top_n: int = 20,
) -> dict[str, Any]:
    """Return the factor weights plus the underlying stock holdings.

    This is the compact helper for "what stocks and what proportions did my
    selected factors turn into?".
    """

    if isinstance(factor, Mapping):
        portfolio = multi_factor_to_stock_portfolio(
            factor,
            factor_weights=factor_weights,
            long_only=long_only,
            gross_exposure=gross_exposure,
            standardize=standardize,
        )
    else:
        portfolio = factor_to_stock_portfolio(
            factor,
            long_only=long_only,
            gross_exposure=gross_exposure,
            standardize=standardize,
        )

    positions = weights_to_positions_frame(portfolio.weights)
    if positions.empty:
        latest_date = None
        latest_holdings = positions.copy()
    else:
        latest_date = pd.Timestamp(positions["date_"].max())
        latest_holdings = positions.loc[positions["date_"] == latest_date].copy()
        latest_holdings = latest_holdings.sort_values(
            by="weight",
            key=lambda series: series.abs(),
            ascending=False,
        ).head(int(top_n))

    return {
        "factor_weights": portfolio.factor_weights,
        "stock_weights": portfolio.weights,
        "positions": positions,
        "latest_date": latest_date,
        "latest_holdings": latest_holdings.reset_index(drop=True),
        "portfolio": portfolio,
    }


__all__ = [
    "FactorPortfolioResult",
    "FactorPortfolioWorkflowResult",
    "coerce_factor_panel",
    "factor_to_stock_portfolio",
    "multi_factor_to_stock_portfolio",
    "summarize_factor_portfolio_holdings",
    "run_factor_portfolio_workflow",
    "run_weight_panel_backtest",
    "weights_to_positions_frame",
    "standardize_cross_section",
]
