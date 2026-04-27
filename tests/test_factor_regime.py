from __future__ import annotations

import numpy as np
import pandas as pd

from tiger_factors.factor_evaluation import TigerFactorData
from tiger_factors.factor_test import factor_regime_decay_test
from tiger_factors.factor_test import factor_regime_ic_test
from tiger_factors.factor_test import factor_regime_report
from tiger_factors.factor_test import factor_regime_stability_test
from tiger_factors.factor_test import factor_regime_turning_point_test


def _sample_regime_factor_data() -> TigerFactorData:
    dates = pd.bdate_range("2024-01-01", periods=60)
    codes = ["AAA", "BBB", "CCC", "DDD"]
    rng = np.random.default_rng(99)
    rows: list[dict[str, object]] = []
    for date_idx, date in enumerate(dates):
        regime_sign = -1.0 if date_idx < len(dates) // 2 else 1.0
        market_drift = regime_sign * 0.01
        for code_idx, code in enumerate(codes):
            centered = code_idx - 1.5
            factor = centered + 0.05 * date_idx + rng.normal(0.0, 0.05)
            forward_1d = regime_sign * (0.20 * centered) + rng.normal(0.0, 0.03)
            price = 100.0 + np.cumsum(np.repeat(market_drift + 0.001 * centered, 1))[0] + 0.5 * date_idx + code_idx
            rows.append(
                {
                    "date_": date,
                    "code": code,
                    "factor": factor,
                    "1D": forward_1d,
                    "close": price,
                }
            )
    frame = pd.DataFrame(rows).set_index(["date_", "code"]).sort_index()
    factor_frame = frame.reset_index()
    factor_series = frame["factor"]
    factor_panel = factor_series.unstack()
    forward_returns = frame["1D"].unstack()
    prices = (1.0 + forward_returns.fillna(0.0)).cumprod()
    price_frame = prices.stack().rename("close").reset_index()
    return TigerFactorData(
        factor_data=frame,
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
        periods=("1D",),
        quantiles=5,
    )


def test_factor_regime_ic_test_detects_state_dependent_signal() -> None:
    tfd = _sample_regime_factor_data()

    result = factor_regime_ic_test(tfd, period="1D", use_hmm=False, n_states=3, min_current_ic_mean=0.01)

    assert result.current_state is not None
    assert not result.table.empty
    assert {"state", "ic_mean", "rank_ic_mean"}.issubset(result.table.columns)
    assert result.ic_spread != 0.0
    assert result.market_state.current_state == result.current_state


def test_factor_regime_stability_test_wraps_market_state_and_ic() -> None:
    tfd = _sample_regime_factor_data()

    result = factor_regime_stability_test(
        tfd,
        period="1D",
        use_hmm=False,
        n_states=3,
        min_current_ic_mean=0.01,
        min_current_rank_ic_mean=0.01,
        min_ic_spread=0.01,
        min_rank_ic_spread=0.01,
    )

    assert result.current_state is not None
    assert result.regime_ic.table.equals(result.historical_table)
    assert "state" in result.regime_ic.table.columns
    assert "market_state" in result.to_dict()


def test_factor_regime_decay_and_turning_point_tests_report_transition_metrics() -> None:
    tfd = _sample_regime_factor_data()

    decay = factor_regime_decay_test(
        tfd,
        period="1D",
        use_hmm=False,
        n_states=3,
        pre_window=3,
        post_window=3,
    )
    turning = factor_regime_turning_point_test(
        tfd,
        period="1D",
        use_hmm=False,
        n_states=3,
        pre_window=3,
        post_window=3,
    )

    assert not decay.table.empty
    assert decay.transition_table.shape[1] >= 10
    assert 0.0 <= decay.decay_score <= 1.0
    assert decay.market_state.current_state is not None
    assert not turning.table.empty
    assert 0.0 <= turning.sign_flip_ratio <= 1.0
    assert turning.market_state.current_state == turning.current_state


def test_factor_regime_report_collects_all_regime_diagnostics() -> None:
    tfd = _sample_regime_factor_data()

    report = factor_regime_report(
        tfd,
        period="1D",
        use_hmm=False,
        n_states=3,
        pre_window=3,
        post_window=3,
    )

    assert report.market_state.current_state is not None
    assert report.regime_ic.current_state == report.market_state.current_state
    assert report.decay.market_state.current_state == report.market_state.current_state
    assert report.turning_point.market_state.current_state == report.market_state.current_state
    assert report.stability.market_state.current_state == report.market_state.current_state
    assert {"regime_ic", "decay", "turning_point", "stability"}.issubset(report.to_dict().keys())
