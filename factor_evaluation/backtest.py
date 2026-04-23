from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tiger_factors.factor_evaluation.core import evaluate_factor_panel
from tiger_factors.factor_evaluation.input import build_tiger_evaluation_input
from tiger_factors.factor_evaluation.performance import factor_cumulative_returns
from tiger_factors.factor_evaluation.performance import factor_positions
from tiger_factors.factor_evaluation.performance import factor_returns


@dataclass(frozen=True)
class SingleFactorBacktestConfig:
    factor_column: str
    price_column: str = "close"
    date_column: str = "date_"
    code_column: str = "code"
    forward_days: int = 1
    n_quantiles: int = 5
    long_short: bool = True
    direction: str = "long"
    initial_capital: float = 1_000_000.0
    annual_trading_days: int = 252


def _ensure_frame(source: pd.DataFrame | str | Path, name: str = "frame") -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    path = Path(source)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported {name} format: {path.suffix}")


def _assign_quantiles(values: pd.Series, quantiles: int) -> pd.Series:
    series = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    clean = series.dropna()
    out = pd.Series(np.nan, index=series.index, dtype=float)
    if clean.empty:
        return out
    unique = max(2, min(int(quantiles), int(clean.size)))
    try:
        ranked = clean.rank(method="first")
        labels = pd.qcut(ranked, q=unique, labels=False, duplicates="drop")
        out.loc[labels.index] = labels.astype(float) + 1.0
    except Exception:
        ranks = clean.rank(method="first", pct=True)
        labels = np.ceil(ranks * unique).clip(1, unique)
        out.loc[labels.index] = labels.astype(float)
    return out


def calculate_drawdown(equity_curve: pd.Series) -> pd.Series:
    series = pd.to_numeric(equity_curve, errors="coerce").replace([np.inf, -np.inf], np.nan)
    peak = series.cummax()
    return (series / peak).sub(1.0)


def calculate_monthly_returns(returns: pd.Series) -> pd.DataFrame:
    series = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if series.empty:
        return pd.DataFrame()
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index, errors="coerce")
    series = series[~series.index.isna()]
    monthly_returns = (1.0 + series).resample("ME").prod() - 1.0
    monthly_df = monthly_returns.to_frame(name="return")
    monthly_df["year"] = monthly_df.index.year
    monthly_df["month"] = monthly_df.index.month
    return monthly_df.pivot(index="year", columns="month", values="return")


def calculate_benchmark_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    annual_trading_days: int = 252,
) -> dict[str, float]:
    strategy = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan)
    benchmark = pd.to_numeric(benchmark_returns, errors="coerce").replace([np.inf, -np.inf], np.nan)
    aligned = pd.concat([strategy.rename("strategy"), benchmark.rename("benchmark")], axis=1).dropna()
    if aligned.empty:
        return {
            "excess_return": 0.0,
            "tracking_error": 0.0,
            "information_ratio": 0.0,
            "correlation": 0.0,
            "beta": 0.0,
        }

    excess = aligned["strategy"] - aligned["benchmark"]
    excess_return = float(excess.mean() * annual_trading_days)
    tracking_error = float(excess.std(ddof=0) * np.sqrt(annual_trading_days))
    information_ratio = float(excess_return / tracking_error) if tracking_error > 1e-12 else 0.0
    correlation = float(aligned["strategy"].corr(aligned["benchmark"]))
    benchmark_var = float(aligned["benchmark"].var(ddof=0))
    beta = float(aligned["strategy"].cov(aligned["benchmark"]) / benchmark_var) if benchmark_var > 1e-12 else 0.0
    return {
        "excess_return": excess_return,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "correlation": correlation,
        "beta": beta,
    }


def generate_signals(
    data: pd.DataFrame | str | Path,
    *,
    factor_column: str,
    date_column: str = "date_",
    code_column: str = "code",
    method: str = "percentile",
    threshold: float = 0.5,
    direction: str = "long",
    quantiles: int = 5,
) -> pd.DataFrame:
    frame = _ensure_frame(data, "data")
    if date_column not in frame.columns or code_column not in frame.columns or factor_column not in frame.columns:
        raise ValueError("data must contain date, code and factor columns")
    frame = frame[[date_column, code_column, factor_column]].copy()
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    frame = frame.dropna(subset=[date_column, code_column]).copy()
    frame[code_column] = frame[code_column].astype(str)
    frame[factor_column] = pd.to_numeric(frame[factor_column], errors="coerce")
    frame = frame.dropna(subset=[factor_column]).copy()

    if method == "threshold":
        if direction == "long":
            frame["signal"] = frame[factor_column] >= threshold
        else:
            frame["signal"] = frame[factor_column] <= threshold
        return frame[[date_column, code_column, "signal"]].reset_index(drop=True)

    if method != "percentile":
        raise ValueError("method must be either 'percentile' or 'threshold'")

    signals: list[pd.DataFrame] = []
    percentile = float(threshold)
    for date, group in frame.groupby(date_column, observed=True):
        ranked = group[factor_column].rank(pct=True, method="first")
        if direction == "long":
            mask = ranked >= percentile
        else:
            mask = ranked <= percentile
        signals.append(
            pd.DataFrame(
                {
                    date_column: date,
                    code_column: group[code_column].to_numpy(),
                    "signal": mask.to_numpy(),
                }
            )
        )
    return pd.concat(signals, ignore_index=True) if signals else pd.DataFrame(columns=[date_column, code_column, "signal"])


def single_factor_backtest(
    data: pd.DataFrame | str | Path,
    *,
    factor_column: str,
    price_column: str = "close",
    date_column: str = "date_",
    code_column: str = "code",
    forward_days: int = 1,
    n_quantiles: int = 5,
    long_short: bool = True,
    direction: str = "long",
    initial_capital: float = 1_000_000.0,
    annual_trading_days: int = 252,
) -> dict[str, Any]:
    frame = _ensure_frame(data, "data")
    if factor_column not in frame.columns or price_column not in frame.columns:
        raise ValueError("data must contain factor and price columns")

    evaluation_input = build_tiger_evaluation_input(
        factor_frame=frame,
        price_frame=frame,
        factor_column=factor_column,
        date_column=date_column,
        code_column=code_column,
        price_column=price_column,
        forward_days=forward_days,
    )
    factor_panel = evaluation_input.factor_panel.copy()
    forward_returns = evaluation_input.forward_returns.copy()
    factor_panel = factor_panel.reindex(index=forward_returns.index, columns=forward_returns.columns)

    stacked = factor_panel.stack(future_stack=True)
    factor_data = stacked.rename("factor").dropna().to_frame()
    factor_data["factor_quantile"] = np.nan
    factor_eval = evaluate_factor_panel(factor_panel, forward_returns)

    quantile_returns: dict[str, pd.Series] = {}
    long_short_returns: list[float] = []
    signal_rows: list[dict[str, Any]] = []
    rank_panel = factor_panel.rank(axis=1, pct=True, method="average")

    quantile_membership = pd.DataFrame(index=factor_panel.index, columns=factor_panel.columns, dtype=float)
    for dt in factor_panel.index:
        factor_row = factor_panel.loc[dt]
        return_row = forward_returns.loc[dt] if dt in forward_returns.index else pd.Series(dtype=float)
        quantiles = _assign_quantiles(factor_row, n_quantiles)
        quantile_membership.loc[dt] = quantiles
        aligned = pd.concat([quantiles.rename("quantile"), return_row.rename("return")], axis=1).dropna()
        if aligned.empty:
            long_short_returns.append(float("nan"))
            continue
        for q in range(1, int(max(2, min(n_quantiles, len(aligned)))) + 1):
            q_returns = aligned.loc[aligned["quantile"] == q, "return"]
            if q_returns.empty:
                continue
            quantile_returns.setdefault(f"Q{q}", pd.Series(dtype=float))
            quantile_returns[f"Q{q}"].loc[pd.Timestamp(dt)] = float(q_returns.mean())
        top_q = int(np.nanmax(aligned["quantile"].to_numpy(dtype=float)))
        bottom_q = int(np.nanmin(aligned["quantile"].to_numpy(dtype=float)))
        top_ret = aligned.loc[aligned["quantile"] == top_q, "return"].mean()
        bottom_ret = aligned.loc[aligned["quantile"] == bottom_q, "return"].mean()
        if long_short:
            portfolio_ret = float(top_ret - bottom_ret)
        else:
            portfolio_ret = float(top_ret if direction == "long" else bottom_ret)
        long_short_returns.append(portfolio_ret)
        signal_rows.append(
            {
                date_column: pd.Timestamp(dt),
                "top_quantile": top_q,
                "bottom_quantile": bottom_q,
                "top_count": int((aligned["quantile"] == top_q).sum()),
                "bottom_count": int((aligned["quantile"] == bottom_q).sum()),
            }
        )

    portfolio_returns = pd.Series(long_short_returns, index=factor_panel.index, name="portfolio_returns")
    portfolio_returns = portfolio_returns.replace([np.inf, -np.inf], np.nan).dropna()
    equity_curve = (1.0 + portfolio_returns.fillna(0.0)).cumprod() * float(initial_capital)
    trades_count = int(max(len(signal_rows) - 1, 0))

    benchmark_returns = forward_returns.mean(axis=1).reindex(portfolio_returns.index)
    metrics = {
        "factor": factor_eval.__dict__,
        "portfolio": {
            "annual_return": float((1.0 + portfolio_returns).prod() ** (annual_trading_days / max(len(portfolio_returns), 1)) - 1.0) if not portfolio_returns.empty else 0.0,
            "sharpe_ratio": float(portfolio_returns.mean() / portfolio_returns.std(ddof=0) * np.sqrt(annual_trading_days)) if portfolio_returns.std(ddof=0) > 1e-12 else 0.0,
            "max_drawdown": float(calculate_drawdown(equity_curve).min()) if not equity_curve.empty else 0.0,
            "turnover": float(quantile_membership.diff().abs().mean().mean()) if not quantile_membership.empty else 0.0,
        },
        "benchmark": calculate_benchmark_metrics(portfolio_returns, benchmark_returns, annual_trading_days=annual_trading_days),
    }

    return {
        "quantile_returns": quantile_returns,
        "portfolio_returns": portfolio_returns,
        "equity_curve": equity_curve,
        "trades_count": trades_count,
        "signal_mask": pd.DataFrame(signal_rows).set_index(date_column) if signal_rows else pd.DataFrame(),
        "factor_rank": rank_panel,
        "quantile_membership": quantile_membership,
        "monthly_returns": calculate_monthly_returns(portfolio_returns),
        "metrics": metrics,
    }


def cross_sectional_backtest(
    data: pd.DataFrame | str | Path,
    *,
    factor_column: str,
    price_column: str = "close",
    date_column: str = "date_",
    code_column: str = "code",
    top_percentile: float = 0.2,
    direction: str = "long",
    annual_trading_days: int = 252,
) -> dict[str, Any]:
    return single_factor_backtest(
        data,
        factor_column=factor_column,
        price_column=price_column,
        date_column=date_column,
        code_column=code_column,
        forward_days=1,
        n_quantiles=max(int(round(1.0 / max(top_percentile, 1e-6))), 2),
        long_short=True,
        direction=direction,
        annual_trading_days=annual_trading_days,
    )


__all__ = [
    "SingleFactorBacktestConfig",
    "calculate_benchmark_metrics",
    "calculate_drawdown",
    "calculate_monthly_returns",
    "cross_sectional_backtest",
    "generate_signals",
    "single_factor_backtest",
]
