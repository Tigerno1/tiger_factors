from __future__ import annotations

import pandas as pd
import pytest

from tiger_factors.factor_evaluation import add_custom_calendar_timedelta
from tiger_factors.factor_evaluation import GridFigure
from tiger_factors.factor_evaluation import infer_trading_calendar
from tiger_factors.factor_evaluation import MaxLossExceededError
from tiger_factors.factor_evaluation.engine import FactorEvaluationEngine
from tiger_factors.factor_store import FactorStore
from tiger_factors.factor_evaluation import compute_forward_returns
from tiger_factors.factor_evaluation import create_portfolio_input
from tiger_factors.factor_evaluation import diff_custom_calendar_timedeltas
from tiger_factors.factor_evaluation import factor_cumulative_returns
from tiger_factors.factor_evaluation import factor_positions
from tiger_factors.factor_evaluation import get_clean_factor
from tiger_factors.factor_evaluation import get_clean_factor_and_forward_returns
from tiger_factors.factor_evaluation.performance import common_start_returns
from tiger_factors.factor_evaluation.performance import positions
from tiger_factors.factor_evaluation.plotting import plot_events_distribution
from tiger_factors.factor_evaluation.plotting import plot_returns_table
from tiger_factors.factor_evaluation.performance import average_cumulative_return_by_quantile


def _sample_factor_series() -> pd.Series:
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    index = pd.MultiIndex.from_product([dates, ["AAA", "BBB", "CCC", "DDD"]], names=["date_", "code"])
    values = [
        -2.0,
        -1.0,
        1.0,
        2.0,
        -1.8,
        -0.8,
        1.2,
        2.1,
        -1.7,
        -0.5,
        1.4,
        2.2,
        -1.5,
        -0.3,
        1.6,
        2.4,
    ]
    return pd.Series(values, index=index, name="alpha_001")


def _sample_prices() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=7, freq="D")
    return pd.DataFrame(
        {
            "AAA": [10.0, 10.2, 10.5, 10.6, 10.8, 11.0, 11.1],
            "BBB": [20.0, 20.1, 20.3, 20.5, 20.8, 21.0, 21.3],
            "CCC": [30.0, 30.4, 30.9, 31.3, 31.8, 32.1, 32.6],
            "DDD": [40.0, 40.6, 41.3, 41.7, 42.1, 42.5, 43.0],
        },
        index=dates,
    )


def _sample_group_map() -> pd.Series:
    return pd.Series({"AAA": "tech", "BBB": "tech", "CCC": "fin", "DDD": "fin"})


def _sample_long_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    factor_series = _sample_factor_series()
    prices = _sample_prices()
    factor_frame = factor_series.rename("alpha_001").reset_index()
    price_frame = (
        prices.stack(future_stack=True)
        .rename("close")
        .reset_index()
        .rename(columns={"level_0": "date_", "level_1": "code"})
    )
    return factor_frame, price_frame


def test_zero_aware_and_group_binning_are_supported() -> None:
    factor = _sample_factor_series()
    prices = _sample_prices()
    forward_returns = compute_forward_returns(factor, prices, periods=(1,))
    clean = get_clean_factor(
        factor,
        forward_returns,
        groupby=_sample_group_map(),
        quantiles=4,
        zero_aware=True,
    )

    first_date = pd.Timestamp("2024-01-01")
    assert clean.loc[(first_date, "AAA"), "factor_quantile"] == 1
    assert clean.loc[(first_date, "BBB"), "factor_quantile"] == 2
    assert clean.loc[(first_date, "CCC"), "factor_quantile"] == 3
    assert clean.loc[(first_date, "DDD"), "factor_quantile"] == 4
    assert set(clean["group"]) == {"tech", "fin"}

    by_group = get_clean_factor(
        factor,
        forward_returns,
        groupby=_sample_group_map(),
        bins=2,
        binning_by_group=True,
    )
    assert by_group["factor_quantile"].isin([1, 2]).all()


def test_max_loss_raises_typed_exception() -> None:
    factor = _sample_factor_series()
    prices = _sample_prices()
    forward_returns = compute_forward_returns(factor, prices, periods=(1,))
    trimmed_forward = forward_returns.iloc[: len(forward_returns) // 2]

    with pytest.raises(MaxLossExceededError):
        get_clean_factor(factor, trimmed_forward, quantiles=4, max_loss=0.1)


def test_compute_forward_returns_supports_filter_zscore_and_non_cumulative() -> None:
    factor = _sample_factor_series()
    prices = _sample_prices().copy()
    prices.loc[pd.Timestamp("2024-01-05"), "DDD"] = 420.0

    filtered = compute_forward_returns(factor, prices, periods=(1,), filter_zscore=0.5)
    non_cumulative = compute_forward_returns(
        factor,
        prices,
        periods=(2,),
        filter_zscore=None,
        cumulative_returns=False,
    )
    cumulative = compute_forward_returns(
        factor,
        prices,
        periods=(2,),
        filter_zscore=None,
        cumulative_returns=True,
    )

    assert filtered["1D"].isna().any()
    assert not non_cumulative["2D"].equals(cumulative["2D"])


def test_compute_forward_returns_supports_timedelta_period_labels() -> None:
    factor = _sample_factor_series()
    prices = _sample_prices()

    forward_returns = compute_forward_returns(
        factor,
        prices,
        periods=("2D", pd.Timedelta("1D1h")),
    )

    assert "2D" in forward_returns.columns
    assert "1D1h" in forward_returns.columns


def test_portfolio_helpers_support_quantile_group_filters_and_capital() -> None:
    factor_data = get_clean_factor_and_forward_returns(
        factor=_sample_factor_series(),
        prices=_sample_prices(),
        groupby=_sample_group_map(),
        quantiles=4,
        periods=(1, 2),
    )

    cumulative = factor_cumulative_returns(
        factor_data.factor_data,
        period="1D",
        quantiles=[4],
        groups=["fin"],
    )
    positions = factor_positions(
        factor_data.factor_data,
        period="2D",
        quantiles=[4],
        groups=["fin"],
    )
    returns, portfolio, benchmark = create_portfolio_input(
        factor_data.factor_data,
        period="1D",
        capital=100_000,
        quantiles=[4],
        groups=["fin"],
        benchmark_period="2D",
    )

    assert not cumulative.empty
    assert not positions.empty
    assert "cash" in portfolio.columns
    assert benchmark.name == "benchmark"
    assert returns.index.isin(portfolio.index).all()


def test_average_cumulative_return_matches_alphalens_style_contract() -> None:
    factor_data = get_clean_factor_and_forward_returns(
        factor=_sample_factor_series(),
        prices=_sample_prices(),
        groupby=_sample_group_map(),
        quantiles=4,
        periods=(1, 2),
    )
    returns = _sample_prices().pct_change(fill_method=None)

    avg_cumulative = average_cumulative_return_by_quantile(
        factor_data.factor_data,
        returns,
        periods_before=2,
        periods_after=2,
        demeaned=True,
        by_group=False,
    )

    assert isinstance(avg_cumulative.index, pd.MultiIndex)
    assert set(avg_cumulative.index.get_level_values(-1)) == {"mean", "std"}


def test_common_start_returns_matches_alphalens_event_alignment() -> None:
    factor_data = get_clean_factor_and_forward_returns(
        factor=_sample_factor_series(),
        prices=_sample_prices(),
        groupby=_sample_group_map(),
        quantiles=4,
        periods=(1,),
    )
    returns = _sample_prices().pct_change(fill_method=None)

    aligned = common_start_returns(
        factor_data.factor_data[["factor_quantile"]],
        returns,
        before=1,
        after=1,
        cumulative=False,
        mean_by_date=True,
    )

    assert list(aligned.index) == [-1, 0, 1]
    assert aligned.shape[1] == len(pd.Index(factor_data.factor_data.index.get_level_values(0).unique()))


def test_positions_keep_explicit_expiry_rows() -> None:
    weights = pd.DataFrame(
        {
            "AAA": [1.0, 0.0],
            "BBB": [0.0, 1.0],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )

    direct = positions(weights, period="2D", freq=None)
    assert direct.index.max() == pd.Timestamp("2024-01-04")
    assert direct.loc[pd.Timestamp("2024-01-04"), ["AAA", "BBB"]].abs().sum() == 0.0


def test_calendar_helpers_follow_custom_business_days() -> None:
    index = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05"])
    freq = infer_trading_calendar(index, index)
    shifted = add_custom_calendar_timedelta(pd.Timestamp("2024-01-02"), pd.Timedelta("1D"), freq)
    diff = diff_custom_calendar_timedeltas(pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04"), freq)

    assert shifted == pd.Timestamp("2024-01-04")
    assert diff == pd.Timedelta("1D")


def test_event_tears_expose_new_alphalens_style_parameters(tmp_path) -> None:
    factor_frame, price_frame = _sample_long_frames()
    engine = FactorEvaluationEngine(
        factor_frame=factor_frame,
        price_frame=price_frame,
        factor_column="alpha_001",
        group_labels=_sample_group_map(),
        factor_store=FactorStore(root_dir=tmp_path),
    )
    daily_returns = _sample_prices().pct_change(fill_method=None)

    event_returns = engine.event_returns(
        periods=(1, 2),
        quantiles=4,
        returns=daily_returns,
        std_bar=True,
        by_group=True,
    )
    event_study = engine.event_study(
        periods=(1, 2),
        quantiles=4,
        returns=daily_returns,
        rate_of_ret=False,
        n_bars=12,
    )

    assert not event_returns.figure_paths
    assert "average_cumulative_return_by_quantile" in event_returns.table_paths
    assert "average_cumulative_return_by_quantile_tech" not in event_returns.table_paths
    assert not event_study.figure_paths
    assert "quantile_statistics" in event_study.table_paths
    assert "mean_return_by_quantile" in event_study.table_paths


def test_returns_tear_sheet_only_adds_cumulative_figures_for_1d(tmp_path) -> None:
    factor_frame, price_frame = _sample_long_frames()
    engine = FactorEvaluationEngine(
        factor_frame=factor_frame,
        price_frame=price_frame,
        factor_column="alpha_001",
        group_labels=_sample_group_map(),
        factor_store=FactorStore(root_dir=tmp_path),
    )

    report = engine.returns(periods=(2,), quantiles=4)

    assert not report.figure_paths


def test_engine_exposes_extended_cleaning_options(tmp_path) -> None:
    factor_frame, price_frame = _sample_long_frames()
    engine = FactorEvaluationEngine(
        factor_frame=factor_frame,
        price_frame=price_frame,
        factor_column="alpha_001",
        group_labels=_sample_group_map(),
        factor_store=FactorStore(root_dir=tmp_path),
    )

    factor_data = engine.get_clean_factor_and_forward_returns(
        periods=(1, 2),
        bins=2,
        binning_by_group=True,
        filter_zscore=5,
        cumulative_returns=False,
    )

    assert factor_data.factor_data["factor_quantile"].isin([1, 2]).all()
    assert "group" in factor_data.factor_data.columns


def test_summary_tear_sheet_includes_turnover_analysis(tmp_path) -> None:
    factor_frame, price_frame = _sample_long_frames()
    engine = FactorEvaluationEngine(
        factor_frame=factor_frame,
        price_frame=price_frame,
        factor_column="alpha_001",
        group_labels=_sample_group_map(),
        factor_store=FactorStore(root_dir=tmp_path),
    )

    report = engine.summary(periods=(1, 2), quantiles=4)

    assert isinstance(report, pd.DataFrame)
    assert not report.empty


def test_returns_tear_sheet_receives_quantile_detail_tables(tmp_path) -> None:
    factor_frame, price_frame = _sample_long_frames()
    engine = FactorEvaluationEngine(
        factor_frame=factor_frame,
        price_frame=price_frame,
        factor_column="alpha_001",
        group_labels=_sample_group_map(),
        factor_store=FactorStore(root_dir=tmp_path),
    )

    report = engine.returns(periods=(1, 2), quantiles=4)

    assert "mean_return_by_quantile_std" in report.table_paths
    assert "quantile_statistics" in report.table_paths


def test_plot_events_distribution_uses_index_dates() -> None:
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=4, freq="D"), ["AAA", "BBB"]],
        names=["date_", "code"],
    )
    events = pd.Series([1] * len(index), index=index)

    ax = plot_events_distribution(events, num_bars=4)

    assert ax.get_ylabel() == "Number of events"


def test_event_study_can_skip_avgretplot(tmp_path) -> None:
    factor_frame, price_frame = _sample_long_frames()
    engine = FactorEvaluationEngine(
        factor_frame=factor_frame,
        price_frame=price_frame,
        factor_column="alpha_001",
        group_labels=_sample_group_map(),
        factor_store=FactorStore(root_dir=tmp_path),
    )

    report = engine.event_study(
        periods=(1, 2),
        quantiles=4,
        avgretplot=None,
        n_bars=6,
    )

    assert not report.figure_paths
    assert "average_cumulative_return_by_quantile" not in report.figure_paths


def test_information_tear_sheet_by_group_matches_alphalens_layout(tmp_path) -> None:
    factor_frame, price_frame = _sample_long_frames()
    engine = FactorEvaluationEngine(
        factor_frame=factor_frame,
        price_frame=price_frame,
        factor_column="alpha_001",
        group_labels=_sample_group_map(),
        factor_store=FactorStore(root_dir=tmp_path),
    )

    report = engine.information(periods=(1, 2), quantiles=4, by_group=True)

    assert not report.figure_paths
    assert "information_coefficient_by_group" in report.table_paths
    assert "mean_information_coefficient_by_group" in report.table_paths


def test_engine_exposes_summary_and_returns_portfolio_flags(tmp_path) -> None:
    factor_frame, price_frame = _sample_long_frames()
    engine = FactorEvaluationEngine(
        factor_frame=factor_frame,
        price_frame=price_frame,
        factor_column="alpha_001",
        group_labels=_sample_group_map(),
        factor_store=FactorStore(root_dir=tmp_path),
    )

    summary = engine.summary(periods=(1, 2), quantiles=4, long_short=False, group_neutral=True)
    returns = engine.returns(
        periods=(1, 2),
        quantiles=4,
        long_short=False,
        group_neutral=True,
        by_group=True,
    )

    assert isinstance(summary, pd.DataFrame)
    assert not summary.empty
    assert "mean_return_by_quantile_by_group" in returns.table_paths


def test_plot_returns_table_supports_return_df_contract() -> None:
    alpha_beta = pd.DataFrame({"1D": [0.1, 1.2]}, index=["Ann. alpha", "beta"])
    mean_quantile_ret = pd.DataFrame({"1D": [0.01, 0.03]}, index=[1, 2])
    mean_spread = pd.Series([0.02], index=["2024-01-01"], name="1D")

    table = plot_returns_table(alpha_beta, mean_quantile_ret, mean_spread, return_df=True)

    assert isinstance(table, pd.DataFrame)
    assert "Mean Period Wise Spread (bps)" in table.index


def test_grid_figure_is_available() -> None:
    grid = GridFigure(2, 2)
    assert grid.next_cell() is not None
    assert grid.next_row() is not None
    grid.close()
