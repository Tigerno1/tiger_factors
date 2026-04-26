from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Iterable

import numpy as np
import pandas as pd

from tiger_factors.factor_evaluation.performance import factor_returns as tiger_factor_returns
from tiger_factors.factor_evaluation.utils import get_clean_factor_and_forward_returns as tiger_clean_factor_and_forward_returns
from tiger_factors.factor_evaluation.utils import period_to_label
from tiger_factors.multifactor_evaluation._inputs import coerce_factor_series
from tiger_factors.multifactor_evaluation._inputs import coerce_price_panel


@dataclass(frozen=True)
class LongShortReturnConfig:
    periods: tuple[int | str | pd.Timedelta, ...] = (1, 5, 10)
    quantiles: int = 5
    long_short: bool = True
    group_neutral: bool = False
    equal_weight: bool = False
    binning_by_group: bool = False
    filter_zscore: float | None = 20
    max_loss: float | None = 0.35
    zero_aware: bool = False
    cumulative_returns: bool = True
    selected_period: int | str | pd.Timedelta | None = None
    source: str = "tiger"


@dataclass(frozen=True)
class RiskfolioConfig:
    model: str = "Classic"
    rm: str = "MV"
    obj: str = "Sharpe"
    rf: float = 0.0
    l: float = 2.0
    hist: bool = True
    method_mu: str = "hist"
    method_cov: str = "hist"
    max_kelly: bool = False
    weight_bounds: tuple[float, float] | None = (0.0, 1.0)
    portfolio_kwargs: dict[str, Any] = field(default_factory=dict)
    assets_stats_kwargs: dict[str, Any] = field(default_factory=dict)
    optimization_kwargs: dict[str, Any] = field(default_factory=dict)


def _resolve_period(config: LongShortReturnConfig) -> int | str | pd.Timedelta:
    if config.selected_period is not None:
        return config.selected_period
    return config.periods[0]


def resolve_return_period(preferred_period: int | str | pd.Timedelta | None) -> int | str | pd.Timedelta:
    if preferred_period is not None:
        return preferred_period
    return 1


def _import_riskfolio() -> Any:
    try:
        import riskfolio as rp
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "riskfolio is unavailable. The current environment failed to import it. "
            f"Original error: {exc}"
        ) from exc
    return rp


def compute_factor_long_short_returns(
    factor: pd.Series | pd.DataFrame,
    prices: pd.DataFrame,
    *,
    config: LongShortReturnConfig | None = None,
    groupby: Any = None,
    groupby_labels: dict[Any, str] | None = None,
) -> pd.Series:
    cfg = config if config is not None else LongShortReturnConfig()
    factor_series = coerce_factor_series(factor)
    prices_panel = coerce_price_panel(prices)
    selected_period = _resolve_period(cfg)
    period_label = period_to_label(selected_period)

    if cfg.source in {"tiger", "alphalens"}:
        prepared = tiger_clean_factor_and_forward_returns(
            factor=factor_series,
            prices=prices_panel,
            groupby=groupby,
            binning_by_group=cfg.binning_by_group,
            quantiles=cfg.quantiles,
            periods=cfg.periods,
            filter_zscore=cfg.filter_zscore,
            groupby_labels=groupby_labels,
            max_loss=cfg.max_loss,
            zero_aware=cfg.zero_aware,
            cumulative_returns=cfg.cumulative_returns,
        )
        factor_data = prepared.factor_data
        portfolio_returns = tiger_factor_returns(
            factor_data,
            demeaned=cfg.long_short,
            group_adjust=cfg.group_neutral,
            equal_weight=cfg.equal_weight,
        )
    else:
        raise ValueError("source must be either 'tiger' or 'alphalens'")

    if period_label not in portfolio_returns.columns:
        raise ValueError(f"Period {period_label} was not found in factor returns.")

    series = pd.to_numeric(portfolio_returns[period_label], errors="coerce").dropna()
    series.index = pd.DatetimeIndex(series.index)
    series.name = getattr(factor_series, "name", None) or "factor"
    return series.sort_index()


def build_long_short_return_panel(
    factor_dict: dict[str, pd.Series | pd.DataFrame],
    prices: pd.DataFrame,
    *,
    factor_names: Iterable[str] | None = None,
    config: LongShortReturnConfig | None = None,
    groupby_map: dict[str, Any] | None = None,
    groupby_labels_map: dict[str, dict[Any, str]] | None = None,
) -> pd.DataFrame:
    names = list(factor_names) if factor_names is not None else list(factor_dict.keys())
    panel: dict[str, pd.Series] = {}
    for name in names:
        panel[name] = compute_factor_long_short_returns(
            factor_dict[name],
            prices,
            config=config,
            groupby=None if groupby_map is None else groupby_map.get(name),
            groupby_labels=None if groupby_labels_map is None else groupby_labels_map.get(name),
        )
    frame = pd.DataFrame(panel).sort_index()
    return frame.dropna(how="all")


def optimize_factor_weights_with_riskfolio(
    factor_return_panel: pd.DataFrame,
    *,
    config: RiskfolioConfig | None = None,
) -> pd.Series:
    cfg = config if config is not None else RiskfolioConfig()
    returns = factor_return_panel.copy().dropna(how="all")
    if returns.empty:
        raise ValueError("factor_return_panel is empty")

    rp = _import_riskfolio()
    portfolio = rp.Portfolio(returns=returns, **cfg.portfolio_kwargs)

    if cfg.weight_bounds is not None:
        portfolio.lowerret = None
        portfolio.upperlng = float(cfg.weight_bounds[1])
        portfolio.uppersht = 0.0

    portfolio.assets_stats(
        method_mu=cfg.method_mu,
        method_cov=cfg.method_cov,
        **cfg.assets_stats_kwargs,
    )
    weights = portfolio.optimization(
        model=cfg.model,
        rm=cfg.rm,
        obj=cfg.obj,
        rf=cfg.rf,
        l=cfg.l,
        hist=cfg.hist,
        **cfg.optimization_kwargs,
    )
    if weights is None or len(weights) == 0:
        raise RuntimeError("riskfolio optimization returned empty weights")

    if isinstance(weights, pd.DataFrame):
        if weights.shape[1] == 1:
            series = weights.iloc[:, 0]
        else:
            series = weights.squeeze(axis=1)
    else:
        series = pd.Series(weights)
    series.index = series.index.astype(str)
    series = pd.to_numeric(series, errors="coerce").dropna()
    total = float(series.sum())
    if np.isfinite(total) and abs(total) > 1e-12:
        series = series / total
    series.name = "weight"
    return series.sort_index()


def allocate_from_return_panel(
    factor_return_panel: pd.DataFrame,
    *,
    config: RiskfolioConfig | None = None,
) -> pd.Series:
    """Allocate weights directly from a factor return panel.

    This is the return-only allocation entrypoint. It is intentionally thin
    and reuses the same optimizer as the factor-based path.
    """
    return optimize_factor_weights_with_riskfolio(factor_return_panel, config=config)


def allocate_selected_factors(
    factor_dict: dict[str, pd.Series | pd.DataFrame],
    prices: pd.DataFrame,
    *,
    screening_summary: pd.DataFrame | None = None,
    selected_factors: Iterable[str] | None = None,
    long_short_config: LongShortReturnConfig | None = None,
    riskfolio_config: RiskfolioConfig | None = None,
    groupby_map: dict[str, Any] | None = None,
    groupby_labels_map: dict[str, dict[Any, str]] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    if selected_factors is not None:
        chosen = list(selected_factors)
    elif screening_summary is not None:
        if "factor_name" not in screening_summary.columns:
            raise ValueError("screening_summary must contain factor_name column")
        if "usable" in screening_summary.columns:
            chosen = screening_summary.loc[screening_summary["usable"].fillna(False), "factor_name"].astype(str).tolist()
        else:
            chosen = screening_summary["factor_name"].astype(str).tolist()
    else:
        chosen = list(factor_dict.keys())

    if not chosen:
        raise ValueError("No factors were selected for allocation.")

    return_panel = build_long_short_return_panel(
        factor_dict,
        prices,
        factor_names=chosen,
        config=long_short_config,
        groupby_map=groupby_map,
        groupby_labels_map=groupby_labels_map,
    )
    weights = optimize_factor_weights_with_riskfolio(return_panel, config=riskfolio_config)
    weights = weights.reindex(return_panel.columns).dropna()
    return return_panel, weights


__all__ = [
    "LongShortReturnConfig",
    "RiskfolioConfig",
    "allocate_from_return_panel",
    "allocate_selected_factors",
    "build_long_short_return_panel",
    "compute_factor_long_short_returns",
    "optimize_factor_weights_with_riskfolio",
    "resolve_return_period",
]
