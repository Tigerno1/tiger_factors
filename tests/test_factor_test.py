from __future__ import annotations

import numpy as np
import pandas as pd

from tiger_factors import factor_convexity_test
from tiger_factors.factor_test import bayesian_fdr
from tiger_factors.factor_test import factor_decay_test
from tiger_factors.factor_test import factor_recent_ic_test
from tiger_factors.factor_test import fit_bayesian_mixture
from tiger_factors.factor_test import factor_stability_test
from tiger_factors.factor_evaluation import TigerFactorData


def _sample_tiger_factor_data(*, recent_weaker: bool = False) -> TigerFactorData:
    dates = pd.bdate_range("2024-01-01", periods=30)
    codes = ["AAA", "BBB", "CCC", "DDD"]
    rng = np.random.default_rng(123)
    rows: list[dict[str, object]] = []
    for date_idx, date in enumerate(dates):
        regime_scale = 1.0 if (not recent_weaker or date_idx < len(dates) - 10) else -0.25
        for code_idx, code in enumerate(codes):
            factor = float(code_idx) + 0.15 * date_idx
            centered = factor - 1.5
            forward_1d = regime_scale * (0.30 * centered) + rng.normal(0.0, 0.02)
            forward_5d = regime_scale * (0.12 * centered) + rng.normal(0.0, 0.02)
            forward_20d = regime_scale * (0.03 * centered) + rng.normal(0.0, 0.02)
            rows.append(
                {
                    "date_": date,
                    "code": code,
                    "factor": factor,
                    "1D": forward_1d,
                    "5D": forward_5d,
                    "20D": forward_20d,
                }
            )
    factor_data = pd.DataFrame(rows).set_index(["date_", "code"]).sort_index()
    factor_frame = factor_data.reset_index()
    factor_series = factor_data["factor"]
    factor_panel = factor_data["factor"].unstack()
    forward_returns = factor_data["1D"].unstack()
    prices = (1.0 + forward_returns.fillna(0.0)).cumprod()
    price_frame = prices.stack().rename("close").reset_index()
    return TigerFactorData(
        factor_data=factor_data,
        factor_frame=factor_frame,
        price_frame=price_frame,
        factor_series=factor_series,
        prices=prices,
        factor_panel=factor_panel,
        forward_returns=forward_returns,
        factor_column="factor",
        date_column="date_",
        code_column="code",
        price_column="close",
        periods=("1D", "5D", "20D"),
        quantiles=5,
    )


def test_factor_convexity_detects_quadratic_loading() -> None:
    rng = np.random.default_rng(42)
    market = pd.Series(rng.normal(0.0, 0.04, 250), name="market")
    factor = 0.01 + 1.2 * market + 3.0 * market**2 + pd.Series(rng.normal(0.0, 0.003, 250), name="factor")

    result = factor_convexity_test(factor, market)

    assert result.n_obs == 250
    assert result.gamma_hat > 0.0
    assert result.gamma_pvalue < 0.05
    assert "market_sq" in set(result.table["term"])


def test_factor_decay_test_reports_slower_long_horizon_signal() -> None:
    tfd = _sample_tiger_factor_data()

    result = factor_decay_test(tfd)

    assert result.n_periods == 3
    assert list(result.table["horizon"]) == sorted(result.table["horizon"].tolist())
    assert result.ic_decay_slope < 0.0
    assert result.rank_ic_decay_slope < 0.0


def test_factor_recent_ic_test_flags_recent_ic_drop() -> None:
    tfd = _sample_tiger_factor_data(recent_weaker=True)

    result = factor_recent_ic_test(
        tfd,
        period="1D",
        recent_window=10,
        min_recent_ic_mean=0.01,
        min_recent_to_history_ratio=0.6,
        min_recent_rank_ic_mean=0.01,
        min_recent_rank_to_history_ratio=0.6,
    )

    assert result.historical_window > 0
    assert result.recent_window == 10
    assert result.historical_ic_mean > result.recent_ic_mean
    assert not result.passed


def test_factor_stability_test_combines_effectiveness_and_recency() -> None:
    tfd = _sample_tiger_factor_data(recent_weaker=True)

    summary = factor_stability_test(
        tfd,
        recent_window=10,
        min_recent_ic_mean=0.01,
        min_recent_to_history_ratio=0.6,
        min_recent_rank_ic_mean=0.01,
        min_recent_rank_to_history_ratio=0.6,
    )

    assert "effectiveness" in summary
    assert "decay" in summary
    assert "recent_ic" in summary
    assert summary["passed"] is False


def test_factor_test_bayesian_aliases_still_work() -> None:
    frame = pd.DataFrame(
        {
            "factor_name": ["alpha_a", "alpha_b", "alpha_c"],
            "p_value": [0.001, 0.02, 0.3],
            "fitness": [1.2, 0.8, 0.1],
        }
    )
    bayes = fit_bayesian_mixture(frame, alpha=0.2)

    assert bayesian_fdr(bayes["result"]) >= 0.0
    assert bayes["table"]["posterior_signal_prob"].between(0.0, 1.0).all()
