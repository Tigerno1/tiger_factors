from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from tiger_factors.multifactor_evaluation.allocation import LongShortReturnConfig
from tiger_factors.multifactor_evaluation.pipeline import blend_factor_panels
from tiger_factors.multifactor_evaluation.pipeline import run_factor_backtest
from tiger_factors.factor_store import TigerFactorLibrary


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


def multi_factor_backtest(
    factor_panels: Mapping[str, pd.DataFrame],
    close_panel: pd.DataFrame | str | Path,
    *,
    weights: Mapping[str, float] | None = None,
    standardize: bool = True,
    long_short_config: LongShortReturnConfig | None = None,
    rebalance_freq: str = "ME",
    long_pct: float = 0.2,
    long_short: bool = True,
    annual_trading_days: int = 252,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> dict[str, Any]:
    if isinstance(close_panel, (str, Path)):
        suffix = Path(close_panel).suffix.lower()
        if suffix in {".parquet", ".pq"}:
            close = pd.read_parquet(close_panel)
        elif suffix == ".csv":
            close = pd.read_csv(close_panel)
        else:
            raise ValueError(f"Unsupported close_panel format: {suffix}")
    else:
        close = close_panel.copy()

    if not factor_panels:
        raise ValueError("factor_panels must not be empty")

    if weights is None:
        equal = 1.0 / len(factor_panels)
        weights = {name: equal for name in factor_panels}

    composite_factor = blend_factor_panels(
        dict(factor_panels),
        dict(weights),
        standardize=standardize,
    )
    backtest, stats = run_factor_backtest(
        composite_factor,
        close,
        long_pct=long_pct,
        rebalance_freq=rebalance_freq,
        long_short=long_short,
        annual_trading_days=annual_trading_days,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
    )
    return {
        "composite_factor": composite_factor,
        "weights": dict(weights),
        "backtest": backtest,
        "stats": stats,
        "portfolio_returns": backtest["portfolio"],
        "benchmark_returns": backtest["benchmark"],
    }


def run_return_backtest(
    returns: pd.Series | pd.DataFrame,
    *,
    weights: Mapping[str, float] | None = None,
    benchmark_returns: pd.Series | None = None,
    annual_trading_days: int = 252,
) -> dict[str, Any]:
    """Run a backtest directly from return series or a return panel.

    This is the return-only entrypoint. It does not require factor panels or
    prices; it only needs the return stream you want to evaluate.
    """
    if isinstance(returns, pd.Series):
        portfolio = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().sort_index()
        component_returns = portfolio.to_frame(name=portfolio.name or "portfolio")
        weight_map = {str(component_returns.columns[0]): 1.0}
    else:
        component_returns = returns.copy().sort_index()
        if component_returns.empty:
            portfolio = pd.Series(dtype=float)
            weight_map = {}
        else:
            if weights is None:
                equal = 1.0 / len(component_returns.columns)
                weight_map = {str(name): equal for name in component_returns.columns}
            else:
                weight_map = {str(name): float(value) for name, value in weights.items() if str(name) in component_returns.columns}
                if not weight_map:
                    raise ValueError("weights must overlap with the return panel columns")
            weight_series = pd.Series(weight_map, dtype=float)
            weight_series = weight_series / weight_series.sum() if float(weight_series.sum()) != 0.0 else weight_series
            component_returns = component_returns.loc[:, weight_series.index]
            portfolio = component_returns.mul(weight_series, axis=1).sum(axis=1)

    backtest = pd.DataFrame({"portfolio": portfolio}).sort_index()
    backtest.index = pd.DatetimeIndex(backtest.index)
    if benchmark_returns is not None:
        benchmark = pd.to_numeric(benchmark_returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().sort_index()
        backtest["benchmark"] = benchmark.reindex(backtest.index)
    else:
        benchmark = pd.Series(dtype=float)

    backtest.attrs["component_returns"] = component_returns
    backtest.attrs["weights"] = weight_map
    backtest.attrs["benchmark_returns"] = benchmark
    stats = {
        "portfolio": _annualized_return_stats(backtest["portfolio"], annual_trading_days),
        "benchmark": _annualized_return_stats(backtest["benchmark"], annual_trading_days) if "benchmark" in backtest.columns else _annualized_return_stats(pd.Series(dtype=float), annual_trading_days),
    }
    return {
        "backtest": backtest,
        "stats": stats,
        "portfolio_returns": backtest["portfolio"],
        "benchmark_returns": backtest["benchmark"] if "benchmark" in backtest.columns else benchmark,
        "component_returns": component_returns,
        "weights": weight_map,
    }


def multi_factor_backtest_from_store(
    library: TigerFactorLibrary,
    factor_names: list[str] | tuple[str, ...],
    *,
    provider: str = "tiger",
    freq: str = "1d",
    variant: str | None = None,
    codes: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    price_provider: str | None = None,
    weights: Mapping[str, float] | None = None,
    standardize: bool = True,
    long_short_config: LongShortReturnConfig | None = None,
    rebalance_freq: str = "ME",
    long_pct: float = 0.2,
    long_short: bool = True,
    annual_trading_days: int = 252,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> dict[str, Any]:
    """Load stored factor panels from a TigerFactorLibrary and backtest them.

    This is the compact factor-store companion to ``multi_factor_backtest``.
    It loads each factor as a wide panel, fetches the matching close panel,
    and then runs the standard composite backtest.
    """
    factor_names = [str(name) for name in factor_names]
    if not factor_names:
        raise ValueError("factor_names must not be empty.")

    factor_panels = library.load_factor_panels(
        factor_names=factor_names,
        provider=provider,
        freq=freq,
        variant=variant,
        codes=codes,
        start=start,
        end=end,
    )
    if not factor_panels:
        raise ValueError("No stored factor panels could be loaded.")

    if codes is not None:
        universe = [str(code) for code in codes]
    else:
        universe = sorted({code for panel in factor_panels.values() for code in panel.columns.astype(str)})
    if not universe:
        raise ValueError("Could not infer any overlapping asset universe from the loaded factor panels.")

    if start is None or end is None:
        date_bounds: list[pd.Timestamp] = []
        for panel in factor_panels.values():
            if panel.empty:
                continue
            date_bounds.append(pd.Timestamp(panel.index.min()))
            date_bounds.append(pd.Timestamp(panel.index.max()))
        if not date_bounds:
            raise ValueError("Could not infer a date range from the loaded factor panels.")
        start = start or str(min(date_bounds).date())
        end = end or str(max(date_bounds).date())

    close = library.price_panel(
        codes=universe,
        start=start,
        end=end,
        provider=price_provider or library.price_provider,
        field="close",
    )
    if close.empty:
        raise ValueError("Could not load a matching close panel from the factor store library.")

    result = multi_factor_backtest(
        factor_panels,
        close,
        weights=weights,
        standardize=standardize,
        long_short_config=long_short_config,
        rebalance_freq=rebalance_freq,
        long_pct=long_pct,
        long_short=long_short,
        annual_trading_days=annual_trading_days,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
    )
    result["factor_panels"] = factor_panels
    result["close_panel"] = close
    result["factor_frame"] = library.load_factor_frame(
        factor_names=factor_names,
        provider=provider,
        freq=freq,
        variant=variant,
        codes=codes,
        start=start,
        end=end,
    )
    return result


__all__ = [
    "multi_factor_backtest",
    "multi_factor_backtest_from_store",
    "run_return_backtest",
]
