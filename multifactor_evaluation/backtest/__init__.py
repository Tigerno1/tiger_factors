from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from tiger_factors.multifactor_evaluation.allocation import LongShortReturnConfig
from tiger_factors.multifactor_evaluation.pipeline import blend_factor_panels
from tiger_factors.multifactor_evaluation.pipeline import run_factor_backtest
from tiger_factors.factor_store import TigerFactorLibrary


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
]
