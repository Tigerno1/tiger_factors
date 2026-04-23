from __future__ import annotations

import pandas as pd

from tiger_factors.factor_evaluation import alpha_beta_regression
from tiger_factors.factor_evaluation import evaluate_factor_groups
from tiger_factors.factor_evaluation import evaluate_factor_panel
from tiger_factors.factor_evaluation import factor_autocorrelation
from tiger_factors.factor_evaluation import rank_factor_autocorrelation


def _sample_factor_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    factor = pd.DataFrame(
        {
            "AAA": [1.0, 1.1, 1.2, 1.3, 1.4],
            "BBB": [2.0, 2.1, 2.2, 2.3, 2.4],
            "CCC": [3.0, 3.1, 3.2, 3.3, 3.4],
        },
        index=dates,
    )
    forward_returns = pd.DataFrame(
        {
            "AAA": [0.01, 0.02, 0.01, 0.00, 0.01],
            "BBB": [0.02, 0.03, 0.02, 0.01, 0.02],
            "CCC": [0.03, 0.04, 0.03, 0.02, 0.03],
        },
        index=dates,
    )
    return factor, forward_returns


def test_factor_panel_additional_metrics():
    factor, forward_returns = _sample_factor_panel()
    benchmark = pd.Series([0.005, 0.01, 0.015, 0.02, 0.025], index=factor.index)

    evaluation = evaluate_factor_panel(factor, forward_returns, benchmark_returns=benchmark)

    assert evaluation.factor_autocorr_mean == evaluation.factor_autocorr_mean
    assert evaluation.rank_factor_autocorr_mean == evaluation.rank_factor_autocorr_mean
    assert evaluation.benchmark_n_obs > 0
    assert evaluation.benchmark_r2 == evaluation.benchmark_r2

    autocorr = factor_autocorrelation(factor)
    rank_autocorr = rank_factor_autocorrelation(factor)
    assert len(autocorr) == len(factor.index) - 1
    assert len(rank_autocorr) == len(factor.index) - 1
    assert autocorr.dropna().mean() > 0.9
    assert rank_autocorr.dropna().mean() > 0.9

    regression = alpha_beta_regression(pd.Series([0.01, 0.02, 0.03], index=factor.index[:3]), benchmark.iloc[:3])
    assert regression["n_obs"] == 3


def test_factor_group_summary_works():
    factor, forward_returns = _sample_factor_panel()
    group_labels = pd.Series({"AAA": "tech", "BBB": "tech", "CCC": "fin"})

    summary = evaluate_factor_groups(factor, forward_returns, group_labels)

    assert not summary.empty
    assert set(summary["group"]) == {"tech", "fin"}
    assert {"ic_mean", "rank_ic_mean", "fitness", "observations"}.issubset(summary.columns)
