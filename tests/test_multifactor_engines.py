from __future__ import annotations

import numpy as np
import pandas as pd

from tiger_factors.factor_evaluation.horizon import HoldingPeriodAnalyzer
from tiger_factors.factor_screener import FactorFilterConfig
from tiger_factors.factor_screener import FactorScreeningEngine
from tiger_factors.factor_screener import cluster_factors
from tiger_factors.factor_screener import factor_correlation_matrix
from tiger_factors.factor_screener import ic_correlation_matrix
from tiger_factors.multifactor_evaluation.slicing import AutoSlicingAnalyzer
from tiger_factors.multifactor_evaluation.regime import RegimeAwareAlphaEngine
from tiger_factors.multifactor_evaluation.regime import SimpleRegimeDetector


def _sample_prices_and_factor() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    dates = pd.bdate_range("2024-01-01", periods=14)
    codes = ["A", "B", "C", "D"]
    daily_returns = {
        "A": 0.015,
        "B": 0.010,
        "C": -0.005,
        "D": -0.010,
    }
    prices = pd.DataFrame(index=dates, columns=codes, dtype=float)
    prices.iloc[0] = [100.0, 100.0, 100.0, 100.0]
    for i in range(1, len(dates)):
        for code in codes:
            prices.loc[dates[i], code] = prices.loc[dates[i - 1], code] * (1.0 + daily_returns[code])

    factor_wide = pd.DataFrame(
        np.tile([2.0, 1.0, -1.0, -2.0], (len(dates), 1)),
        index=dates,
        columns=codes,
    )
    reverse_wide = -factor_wide
    factor = factor_wide.stack(future_stack=True).rename("factor")
    reverse_factor = reverse_wide.stack(future_stack=True).rename("reverse_factor")
    return prices, factor, reverse_factor


def _sample_labels() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=14)
    pairs = pd.MultiIndex.from_product([dates, ["A", "B", "C", "D"]], names=["date_", "code"])
    frame = pd.DataFrame(index=pairs)
    frame["sector"] = ["tech", "tech", "finance", "finance"] * len(dates)
    frame["size_bucket"] = ["large", "mid", "mid", "small"] * len(dates)
    return frame


def _sample_regime_series() -> pd.Series:
    dates = pd.bdate_range("2024-01-01", periods=14)
    values = ["bull_low_vol"] * 7 + ["bear_high_vol"] * 7
    return pd.Series(values, index=dates, name="regime")


def test_holding_period_analyzer_accepts_series_input_with_min_names() -> None:
    prices, factor, _ = _sample_prices_and_factor()

    result = HoldingPeriodAnalyzer(
        factor,
        prices,
        quantiles=2,
        long_short_pct=0.25,
        min_names=3,
    ).run([1, 3])

    assert list(result["horizon"]) == [1, 3]
    assert {"mean_ic", "sharpe", "avg_turnover"}.issubset(result.columns)


def test_factor_screening_engine_returns_summary_and_detail_map() -> None:
    prices, factor, reverse_factor = _sample_prices_and_factor()

    summary, detail_map = FactorScreeningEngine(
        {"trend": factor, "reverse": reverse_factor},
        prices,
        horizons=[1, 3],
        quantiles=2,
        long_short_pct=0.25,
        min_names=3,
        filter_config=FactorFilterConfig(min_net_sharpe=-10.0, min_abs_ic_ir=0.0),
    ).run()

    assert set(summary["factor_name"]) == {"trend", "reverse"}
    assert set(detail_map) == {"trend", "reverse"}
    assert "score" in summary.columns


def test_auto_slicing_analyzer_supports_multi_column_slices() -> None:
    prices, factor, _ = _sample_prices_and_factor()
    labels = _sample_labels()

    summary, detail_map = AutoSlicingAnalyzer(
        factor,
        prices,
        labels,
        horizons=[1, 3],
        quantiles=2,
        long_short_pct=0.25,
        min_sample_dates=5,
        min_names=2,
    ).run(["sector", "size_bucket"])

    assert not summary.empty
    assert {"sector", "size_bucket", "n_dates"}.issubset(summary.columns)
    assert all(isinstance(key, tuple) for key in detail_map)


def test_regime_detector_and_regime_aware_engine() -> None:
    prices, factor, _ = _sample_prices_and_factor()
    market_price = prices.mean(axis=1)
    regimes = SimpleRegimeDetector(market_price, ma_window=3, vol_window=3).detect()
    assert regimes.dropna().nunique() >= 1

    manual_regimes = _sample_regime_series()
    summary, detail_map = RegimeAwareAlphaEngine(
        factor,
        prices,
        manual_regimes,
        horizons=[1, 3],
        quantiles=2,
        long_short_pct=0.25,
        min_names=2,
        min_dates_per_regime=3,
    ).run()
    advice = RegimeAwareAlphaEngine(
        factor,
        prices,
        manual_regimes,
        horizons=[1, 3],
        quantiles=2,
        long_short_pct=0.25,
        min_names=2,
        min_dates_per_regime=3,
    ).recommend_for_current_regime()

    assert not summary.empty
    assert set(detail_map) == {"bear_high_vol", "bull_low_vol"}
    assert advice["current_regime"] == "bear_high_vol"


def test_redundancy_tools_build_correlation_and_clusters() -> None:
    prices, factor, reverse_factor = _sample_prices_and_factor()
    factors = {"trend": factor, "reverse": reverse_factor}

    factor_corr = factor_correlation_matrix(factors)
    ic_corr = ic_correlation_matrix(factors, prices, horizon=1, min_names=2)
    clusters = cluster_factors(factor_corr, threshold=0.5)

    assert factor_corr.shape == (2, 2)
    assert ic_corr.shape == (2, 2)
    assert set(clusters) == {"trend", "reverse"}
