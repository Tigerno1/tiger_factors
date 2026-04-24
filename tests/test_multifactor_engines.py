from __future__ import annotations

import numpy as np
import pandas as pd

from tiger_factors.factor_evaluation.horizon import HoldingPeriodAnalyzer
from tiger_factors.factor_screener import FactorFilterConfig
from tiger_factors.factor_screener import cluster_factors
from tiger_factors.factor_screener import factor_correlation_matrix
from tiger_factors.factor_screener import ic_correlation_matrix
from tiger_factors.factor_screener import select_by_average_correlation
from tiger_factors.factor_screener import select_by_graph_independent_set
from tiger_factors.factor_screener import run_correlation_screening
from tiger_factors.factor_screener import run_ic_correlation_screening
from tiger_factors.factor_evaluation.factor_screening import build_single_factor_return_long_frame
from tiger_factors.multifactor_evaluation.batch import FactorScreeningEngine
from tiger_factors.factor_allocation import allocate_from_return_panel
from tiger_factors.factor_backtest import run_return_backtest
from tiger_factors.multifactor_evaluation import analyze_multifactor
from tiger_factors.multifactor_evaluation import analyze_positions
from tiger_factors.multifactor_evaluation import analyze_returns
from tiger_factors.multifactor_evaluation import analyze_transactions
from tiger_factors.multifactor_evaluation import create_analysis_report
from tiger_factors.multifactor_evaluation import MultifactorAnalysisReportSpec
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
    orthogonal_factor = pd.Series(
        np.tile([1.0, -1.0, 1.0, -1.0], len(prices.index)),
        index=factor.index,
        name="orthogonal",
    )
    factors = {"trend": factor, "reverse": reverse_factor, "orthogonal": orthogonal_factor}

    factor_corr = factor_correlation_matrix(factors)
    ic_corr = ic_correlation_matrix(factors, prices, horizon=1, min_names=2)
    clusters = cluster_factors(factor_corr, threshold=0.5)
    average_selected = select_by_average_correlation(factors, {"trend": 3.0, "reverse": 2.0, "orthogonal": 1.0}, threshold=0.5)
    graph_selected = select_by_graph_independent_set(factors, {"trend": 3.0, "reverse": 2.0, "orthogonal": 1.0}, threshold=0.5)
    average_via_wrapper = run_correlation_screening(
        factors,
        {"trend": 3.0, "reverse": 2.0, "orthogonal": 1.0},
        method="average",
        threshold=0.5,
    )
    graph_via_wrapper = run_correlation_screening(
        factors,
        {"trend": 3.0, "reverse": 2.0, "orthogonal": 1.0},
        method="graph",
        threshold=0.5,
    )
    ic_average_via_wrapper = run_ic_correlation_screening(
        factors,
        prices,
        method="average",
        threshold=0.5,
        scores={"trend": 3.0, "reverse": 2.0, "orthogonal": 1.0},
    )

    assert factor_corr.shape == (3, 3)
    assert ic_corr.shape == (3, 3)
    assert set(clusters) == {"trend", "reverse", "orthogonal"}
    assert set(average_selected).issubset(set(factors))
    assert set(graph_selected).issubset(set(factors))
    assert average_via_wrapper == average_selected
    assert graph_via_wrapper == graph_selected
    assert set(ic_average_via_wrapper).issubset(set(factors))


def test_return_only_allocation_and_backtest() -> None:
    returns = pd.DataFrame(
        {
            "factor_a": [0.01, 0.02, -0.01, 0.03],
            "factor_b": [0.00, 0.01, 0.02, -0.02],
        },
        index=pd.bdate_range("2024-01-01", periods=4),
    )

    backtest = run_return_backtest(returns)
    assert "portfolio" in backtest["backtest"].columns
    assert not backtest["portfolio_returns"].empty
    assert "portfolio" in backtest["stats"]

    try:
        import riskfolio  # noqa: F401
    except Exception:
        return

    weights = allocate_from_return_panel(returns)
    assert set(weights.index) == {"factor_a", "factor_b"}
    assert abs(float(weights.sum()) - 1.0) < 1e-9


def test_single_factor_return_long_frame_supports_modes() -> None:
    prices, factor, _ = _sample_prices_and_factor()

    long_short_frame = build_single_factor_return_long_frame(
        factor,
        prices,
        factor_name="trend",
        return_modes=("long_short",),
        preferred_return_period=1,
    )
    long_only_frame = build_single_factor_return_long_frame(
        factor,
        prices,
        factor_name="trend",
        return_modes=("long_only",),
        preferred_return_period=1,
    )

    assert set(long_short_frame.columns) == {"date_", "factor", "return", "return_mode"}
    assert set(long_only_frame.columns) == {"date_", "factor", "return", "return_mode"}
    assert set(long_short_frame["return_mode"].unique()) == {"long_short"}
    assert set(long_only_frame["return_mode"].unique()) == {"long_only"}
    assert set(long_short_frame["factor"].unique()) == {"trend"}
    assert set(long_only_frame["factor"].unique()) == {"trend"}


def test_multifactor_analysis_facade_covers_common_pyfolio_views() -> None:
    idx = pd.bdate_range("2024-01-01", periods=6)
    returns = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01, -0.01], index=idx)
    benchmark = pd.Series([0.00, 0.01, -0.01, 0.02, 0.0, 0.01], index=idx)
    positions = pd.DataFrame({"AAA": [0.5, 0.4, 0.3, 0.2, 0.1, 0.0], "BBB": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}, index=idx)
    transactions = pd.DataFrame(
        {
            "dt": [idx[0], idx[1], idx[2]],
            "symbol": ["AAA", "BBB", "AAA"],
            "amount": [10, -4, 5],
            "price": [1.0, 2.0, 1.5],
        }
    )

    return_result = analyze_returns(returns, benchmark_returns=benchmark, live_start_date=idx[3])
    position_result = analyze_positions(positions, returns=returns, benchmark_returns=benchmark)
    transaction_result = analyze_transactions(transactions, positions=positions)
    multifactor_result = analyze_multifactor(
        returns=returns,
        positions=positions,
        transactions=transactions,
        benchmark_returns=benchmark,
        live_start_date=idx[3],
    )

    assert not return_result.metric_table.empty
    assert "annual_return" in return_result.metric_table.index
    assert not return_result.monthly_returns_heatmap.empty
    assert return_result.compare_table.shape[1] >= 4
    assert position_result.latest_holdings.shape[0] == 2
    assert position_result.gross_leverage.iloc[-1] == 1.0
    assert transaction_result.round_trip_summary is not None
    assert multifactor_result.returns is not None
    assert multifactor_result.positions is not None
    assert multifactor_result.transactions is not None


def test_create_analysis_report_exports_tables_and_figures(tmp_path: Path) -> None:
    idx = pd.bdate_range("2024-01-01", periods=8)
    returns = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01, -0.01, 0.02, 0.01], index=idx)
    benchmark = pd.Series([0.00, 0.01, -0.01, 0.02, 0.0, 0.01, 0.0, -0.01], index=idx)
    positions = pd.DataFrame({"AAA": [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0], "BBB": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]}, index=idx)
    transactions = pd.DataFrame(
        {
            "dt": [idx[0], idx[1], idx[2]],
            "symbol": ["AAA", "BBB", "AAA"],
            "amount": [10, -4, 5],
            "price": [1.0, 2.0, 1.5],
        }
    )

    report = create_analysis_report(
        returns,
        benchmark_returns=benchmark,
        positions=positions,
        transactions=transactions,
        output_dir=tmp_path / "analysis_report",
        report_name="demo",
        open_browser=False,
    )

    assert report.html_path.exists()
    assert report.summary_table_path.exists()
    assert report.metric_table_path.exists()
    assert report.figure_paths
    assert report.get_table("metrics").empty is False
    assert report.get_report(open_browser=False).exists()


def test_create_analysis_report_spec_can_disable_exports(tmp_path: Path) -> None:
    idx = pd.bdate_range("2024-01-01", periods=6)
    returns = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01, -0.01], index=idx)
    report = create_analysis_report(
        returns,
        output_dir=tmp_path / "analysis_report_spec",
        report_name="demo_spec",
        open_browser=False,
        spec=MultifactorAnalysisReportSpec(
            save_html=False,
            save_figures=False,
            save_metric_table=False,
            save_compare_table=False,
            save_drawdown_table=False,
            save_monthly_heatmap_table=False,
            save_positions_summary=False,
            save_latest_holdings=False,
            save_transaction_summary=False,
            save_round_trip_summary=False,
            save_round_trips=False,
        ),
    )

    assert report.html_path.exists() is False
    assert report.figure_paths == []
    assert report.metric_table_path is None
    assert report.compare_table_path is None
    assert report.get_table("summary").empty is False
    assert report.report() is None
