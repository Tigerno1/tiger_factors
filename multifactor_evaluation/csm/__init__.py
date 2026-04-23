from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from tiger_factors.factor_frame import CSMModel
from tiger_factors.factor_frame import build_csm_model
from tiger_factors.factor_frame import build_csm_training_frame
from tiger_factors.factor_frame import infer_csm_feature_columns
from tiger_factors.multifactor_evaluation.pipeline import run_factor_backtest


@dataclass(frozen=True)
class CSMBacktestResult:
    model: CSMModel
    score_panel: pd.DataFrame
    selection_panel: pd.DataFrame | None
    backtest_returns: pd.DataFrame
    backtest_stats: dict[str, dict[str, float]]


def run_csm_backtest(
    frame: pd.DataFrame,
    close_panel: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
    *,
    label_column: str = "forward_return",
    fit_method: str = "rank_ic",
    feature_transform: str = "zscore",
    min_group_size: int = 5,
    winsorize_limits: tuple[float, float] | None = (0.01, 0.99),
    normalize_score_by_date: bool = True,
    score_clip: tuple[float, float] | None = None,
    learning_rate: float = 0.05,
    max_iter: int = 250,
    l2_reg: float = 1e-3,
    temperature: float = 1.0,
    pairwise_max_pairs: int = 256,
    long_pct: float = 0.25,
    rebalance_freq: str = "W-FRI",
    long_short: bool = True,
    annual_trading_days: int = 252,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> CSMBacktestResult:
    inferred_feature_columns = tuple(feature_columns) if feature_columns is not None else infer_csm_feature_columns(
        frame,
        label_column=label_column,
    )
    model = build_csm_model(
        inferred_feature_columns,
        label_column=label_column,
        fit_method=fit_method,
        feature_transform=feature_transform,
        min_group_size=min_group_size,
        winsorize_limits=winsorize_limits,
        normalize_score_by_date=normalize_score_by_date,
        score_clip=score_clip,
        learning_rate=learning_rate,
        max_iter=max_iter,
        l2_reg=l2_reg,
        temperature=temperature,
        pairwise_max_pairs=pairwise_max_pairs,
    )
    model.fit(frame, label_column=label_column)
    score_panel = model.score_panel(frame)

    backtest_returns, backtest_stats = run_factor_backtest(
        score_panel,
        close_panel,
        long_pct=long_pct,
        rebalance_freq=rebalance_freq,
        long_short=long_short,
        annual_trading_days=annual_trading_days,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
    )
    return CSMBacktestResult(
        model=model,
        score_panel=score_panel,
        selection_panel=None,
        backtest_returns=backtest_returns,
        backtest_stats=backtest_stats,
    )


def run_csm_selection_backtest(
    frame: pd.DataFrame,
    close_panel: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
    *,
    label_column: str = "forward_return",
    fit_method: str = "rank_ic",
    feature_transform: str = "zscore",
    min_group_size: int = 5,
    winsorize_limits: tuple[float, float] | None = (0.01, 0.99),
    normalize_score_by_date: bool = True,
    score_clip: tuple[float, float] | None = None,
    learning_rate: float = 0.05,
    max_iter: int = 250,
    l2_reg: float = 1e-3,
    temperature: float = 1.0,
    pairwise_max_pairs: int = 256,
    top_n: int = 10,
    bottom_n: int = 0,
    long_only: bool = True,
    annual_trading_days: int = 252,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> CSMBacktestResult:
    inferred_feature_columns = tuple(feature_columns) if feature_columns is not None else infer_csm_feature_columns(
        frame,
        label_column=label_column,
    )
    model = build_csm_model(
        inferred_feature_columns,
        label_column=label_column,
        fit_method=fit_method,
        feature_transform=feature_transform,
        min_group_size=min_group_size,
        winsorize_limits=winsorize_limits,
        normalize_score_by_date=normalize_score_by_date,
        score_clip=score_clip,
        learning_rate=learning_rate,
        max_iter=max_iter,
        l2_reg=l2_reg,
        temperature=temperature,
        pairwise_max_pairs=pairwise_max_pairs,
    )
    model.fit(frame, label_column=label_column)
    score_panel = model.score_panel(frame)
    selection_panel = model.selection_panel(frame, top_n=top_n, bottom_n=bottom_n, long_only=long_only)
    if selection_panel.empty:
        raise ValueError("selection_panel is empty.")
    backtest_returns, backtest_stats = run_factor_backtest(
        selection_panel,
        close_panel,
        long_pct=0.25,
        rebalance_freq="W-FRI",
        long_short=not long_only,
        annual_trading_days=annual_trading_days,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
    )
    return CSMBacktestResult(
        model=model,
        score_panel=score_panel,
        selection_panel=selection_panel,
        backtest_returns=backtest_returns,
        backtest_stats=backtest_stats,
    )


def run_csm_factor_frame_selection_backtest(
    factor_frame: pd.DataFrame,
    close_panel: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
    *,
    label_column: str = "forward_return",
    date_column: str = "date_",
    code_column: str = "code",
    top_n: int = 10,
    bottom_n: int = 0,
    long_only: bool = True,
    fit_method: str = "rank_ic",
    feature_transform: str = "zscore",
    min_group_size: int = 5,
    winsorize_limits: tuple[float, float] | None = (0.01, 0.99),
    normalize_score_by_date: bool = True,
    score_clip: tuple[float, float] | None = None,
    learning_rate: float = 0.05,
    max_iter: int = 250,
    l2_reg: float = 1e-3,
    temperature: float = 1.0,
    pairwise_max_pairs: int = 256,
    annual_trading_days: int = 252,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> CSMBacktestResult:
    inferred_feature_columns = tuple(feature_columns) if feature_columns is not None else infer_csm_feature_columns(
        factor_frame,
        label_column=label_column,
        date_column=date_column,
        code_column=code_column,
    )
    training_frame = build_csm_training_frame(
        factor_frame,
        inferred_feature_columns,
        label_column=label_column,
        date_column=date_column,
        code_column=code_column,
    )
    return run_csm_selection_backtest(
        training_frame,
        close_panel,
        inferred_feature_columns,
        label_column=label_column,
        fit_method=fit_method,
        feature_transform=feature_transform,
        min_group_size=min_group_size,
        winsorize_limits=winsorize_limits,
        normalize_score_by_date=normalize_score_by_date,
        score_clip=score_clip,
        learning_rate=learning_rate,
        max_iter=max_iter,
        l2_reg=l2_reg,
        temperature=temperature,
        pairwise_max_pairs=pairwise_max_pairs,
        top_n=top_n,
        bottom_n=bottom_n,
        long_only=long_only,
        annual_trading_days=annual_trading_days,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
    )


__all__ = [
    "CSMBacktestResult",
    "run_csm_factor_frame_selection_backtest",
    "run_csm_backtest",
    "run_csm_selection_backtest",
]
