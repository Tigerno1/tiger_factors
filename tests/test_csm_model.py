from __future__ import annotations

import numpy as np
import pandas as pd

from tiger_factors.factor_frame import CSMModel
from tiger_factors.factor_frame import build_csm_model
from tiger_factors.factor_frame import build_csm_training_frame
from tiger_factors.factor_frame import infer_csm_feature_columns
from tiger_factors.multifactor_evaluation import run_csm_backtest
from tiger_factors.multifactor_evaluation import run_csm_factor_frame_selection_backtest
from tiger_factors.multifactor_evaluation import run_csm_selection_backtest
from tiger_factors.multifactor_evaluation.pipeline import run_factor_backtest
from tiger_factors.multifactor_evaluation.reporting.portfolio import run_portfolio_from_backtest


def _sample_csm_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=8)
    codes = ["AAA", "BBB", "CCC", "DDD"]
    rows: list[dict[str, object]] = []
    for date_idx, date in enumerate(dates):
        for code_idx, code in enumerate(codes):
            feat_momentum = float(code_idx) + 0.1 * date_idx
            feat_value = float(len(codes) - code_idx) + 0.05 * date_idx
            noise = 0.01 * (date_idx - code_idx)
            forward_return = 0.6 * feat_momentum - 0.4 * feat_value + noise
            rows.append(
                {
                    "date_": date,
                    "code": code,
                    "momentum": feat_momentum,
                    "value": feat_value,
                    "forward_return": forward_return,
                }
            )
    return pd.DataFrame(rows)


def test_csm_model_fits_predicts_and_selects() -> None:
    frame = _sample_csm_panel()
    train = frame[frame["date_"] <= pd.Timestamp("2024-01-05")].copy()
    test = frame[frame["date_"] > pd.Timestamp("2024-01-05")].copy()

    model = build_csm_model(
        ("momentum", "value"),
        fit_method="rank_ic",
        feature_transform="zscore",
        min_group_size=2,
    )
    model.fit(train)

    assert isinstance(model, CSMModel)
    assert model.is_fitted_
    assert set(model.weights_) == {"momentum", "value"}
    assert model.weights_["momentum"] > 0
    assert model.weights_["value"] < 0
    assert {"feature", "ic_mean", "rank_ic_mean", "weight"}.issubset(model.feature_stats_.columns)

    scored = model.predict(test)
    assert {"csm_score", "csm_rank"}.issubset(scored.columns)
    assert scored.groupby("date_")["csm_rank"].min().ge(1).all()

    selected = model.select(test, top_n=1, bottom_n=1, long_only=False)
    assert {"csm_side", "csm_target_weight"}.issubset(selected.columns)
    assert selected.groupby("date_").size().eq(2).all()
    assert set(selected["csm_side"].unique()) == {1.0, -1.0}


def test_csm_strategy_callable() -> None:
    frame = _sample_csm_panel()
    model = build_csm_model(("momentum", "value"), fit_method="ic", min_group_size=2)
    model.fit(frame)

    strategy = model.to_strategy(top_n=2, long_only=True)

    class _Ctx:
        def __init__(self, combined_frame: pd.DataFrame) -> None:
            self.combined_frame = combined_frame

    selected = strategy(_Ctx(frame))
    assert not selected.empty
    assert {"date_", "code", "csm_score", "csm_rank", "csm_side", "csm_target_weight"}.issubset(selected.columns)
    assert selected.groupby("date_").size().eq(2).all()


def test_csm_ranknet_and_listnet_modes() -> None:
    frame = _sample_csm_panel()
    train = frame[frame["date_"] <= pd.Timestamp("2024-01-05")].copy()

    ranknet_model = build_csm_model(
        ("momentum", "value"),
        fit_method="ranknet",
        feature_transform="zscore",
        min_group_size=2,
        learning_rate=0.1,
        max_iter=50,
    )
    ranknet_model.fit(train)
    assert ranknet_model.is_fitted_
    assert np.isclose(sum(abs(v) for v in ranknet_model.weights_.values()), 1.0)
    assert set(ranknet_model.weights_) == {"momentum", "value"}

    listnet_model = build_csm_model(
        ("momentum", "value"),
        fit_method="listnet",
        feature_transform="zscore",
        min_group_size=2,
        learning_rate=0.1,
        max_iter=50,
    )
    listnet_model.fit(train)
    assert listnet_model.is_fitted_
    assert np.isclose(sum(abs(v) for v in listnet_model.weights_.values()), 1.0)
    assert set(listnet_model.weights_) == {"momentum", "value"}


def test_csm_score_panel_and_backtest() -> None:
    frame = _sample_csm_panel()
    train = frame[frame["date_"] <= pd.Timestamp("2024-01-05")].copy()

    close_panel = (
        frame.assign(close_return=0.001 + 0.001 * frame["momentum"] - 0.0005 * frame["value"])
        .pivot(index="date_", columns="code", values="close_return")
        .sort_index()
    )
    close_panel = (1.0 + close_panel.fillna(0.0)).cumprod() * 100.0

    model = build_csm_model(
        ("momentum", "value"),
        fit_method="rank_ic",
        feature_transform="zscore",
        min_group_size=2,
        normalize_score_by_date=True,
    )
    model.fit(train)

    score_panel = model.score_panel(frame)
    assert score_panel.index.name == "date_"
    assert set(score_panel.columns) == set(frame["code"].unique())

    backtest, stats = run_factor_backtest(
        score_panel,
        close_panel,
        long_pct=0.25,
        rebalance_freq="W-FRI",
        long_short=True,
    )
    assert not backtest.empty
    assert {"portfolio", "benchmark"}.issubset(backtest.columns)
    assert set(stats) == {"portfolio", "benchmark"}


def test_csm_backtest_helper() -> None:
    frame = _sample_csm_panel()
    train = frame[frame["date_"] <= pd.Timestamp("2024-01-05")].copy()
    close_panel = (
        frame.assign(close_return=0.001 + 0.001 * frame["momentum"] - 0.0005 * frame["value"])
        .pivot(index="date_", columns="code", values="close_return")
        .sort_index()
    )
    close_panel = (1.0 + close_panel.fillna(0.0)).cumprod() * 100.0

    model = build_csm_model(
        ("momentum", "value"),
        fit_method="rank_ic",
        feature_transform="zscore",
        min_group_size=2,
        normalize_score_by_date=True,
    )
    model.fit(train)

    result = run_csm_backtest(
        frame,
        close_panel,
        fit_method="rank_ic",
        feature_transform="zscore",
        min_group_size=2,
        normalize_score_by_date=True,
        long_pct=0.25,
        rebalance_freq="W-FRI",
        long_short=True,
    )
    assert result.score_panel.shape[1] == close_panel.shape[1]
    assert not result.backtest_returns.empty
    assert set(result.backtest_stats) == {"portfolio", "benchmark"}


def test_csm_selection_backtest_helper() -> None:
    frame = _sample_csm_panel()
    train = frame[frame["date_"] <= pd.Timestamp("2024-01-05")].copy()
    close_panel = (
        frame.assign(close_return=0.001 + 0.001 * frame["momentum"] - 0.0005 * frame["value"])
        .pivot(index="date_", columns="code", values="close_return")
        .sort_index()
    )
    close_panel = (1.0 + close_panel.fillna(0.0)).cumprod() * 100.0

    model = build_csm_model(
        ("momentum", "value"),
        fit_method="rank_ic",
        feature_transform="zscore",
        min_group_size=2,
        normalize_score_by_date=True,
    )
    model.fit(train)

    result = run_csm_selection_backtest(
        frame,
        close_panel,
        fit_method="rank_ic",
        feature_transform="zscore",
        min_group_size=2,
        normalize_score_by_date=True,
        top_n=1,
        bottom_n=1,
        long_only=False,
    )
    assert result.selection_panel is not None
    assert not result.selection_panel.empty
    assert not result.backtest_returns.empty
    assert set(result.backtest_stats) == {"portfolio", "benchmark"}


def test_csm_backtest_can_generate_portfolio_report(tmp_path) -> None:
    frame = _sample_csm_panel()
    train = frame[frame["date_"] <= pd.Timestamp("2024-01-05")].copy()
    close_panel = (
        frame.assign(close_return=0.001 + 0.001 * frame["momentum"] - 0.0005 * frame["value"])
        .pivot(index="date_", columns="code", values="close_return")
        .sort_index()
    )
    close_panel = (1.0 + close_panel.fillna(0.0)).cumprod() * 100.0

    result = run_csm_selection_backtest(
        frame,
        close_panel,
        ("momentum", "value"),
        fit_method="rank_ic",
        feature_transform="zscore",
        min_group_size=2,
        normalize_score_by_date=True,
        top_n=1,
        bottom_n=1,
        long_only=False,
    )
    report = run_portfolio_from_backtest(
        result.backtest_returns,
        output_dir=tmp_path,
        report_name="csm_test",
    )
    assert report is not None
    assert report.output_dir == tmp_path


def test_csm_factor_frame_selection_backtest_helper() -> None:
    frame = _sample_csm_panel()
    close_panel = (
        frame.assign(close_return=0.001 + 0.001 * frame["momentum"] - 0.0005 * frame["value"])
        .pivot(index="date_", columns="code", values="close_return")
        .sort_index()
    )
    close_panel = (1.0 + close_panel.fillna(0.0)).cumprod() * 100.0
    result = run_csm_factor_frame_selection_backtest(
        frame,
        close_panel,
        label_column="forward_return",
        fit_method="rank_ic",
        feature_transform="zscore",
        min_group_size=2,
        normalize_score_by_date=True,
        top_n=1,
        bottom_n=1,
        long_only=False,
    )
    assert result.selection_panel is not None
    assert not result.selection_panel.empty
    assert not result.backtest_returns.empty


def test_build_csm_training_frame() -> None:
    frame = _sample_csm_panel()
    training = build_csm_training_frame(frame, ("momentum", "value"), label_column="forward_return")
    assert list(training.columns) == ["date_", "code", "momentum", "value", "forward_return"]
    assert training["date_"].is_monotonic_increasing


def test_infer_csm_feature_columns() -> None:
    frame = _sample_csm_panel()
    inferred = infer_csm_feature_columns(frame)
    assert inferred == ("momentum", "value")
