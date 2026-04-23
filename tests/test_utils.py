from __future__ import annotations

import numpy as np
import pandas as pd

from tiger_factors.factor_evaluation.utils import _rowwise_cross_sectional_corr
from tiger_factors.utils.cross_sectional import (
    cs_minmax_pos,
    cs_neutralize,
    group_demean,
    group_dummy_regression_residual,
    group_zscore,
    industry_regression_residual,
    industry_size_regression_residual,
    normalize_cross_section,
    make_industry_dummies,
    neutralize_by_group,
    neutralize_cross_section,
    preprocess_cross_section,
    winsorize_cross_section,
    rank_pct,
    winsorize_quantile,
    zscore,
)
from tiger_factors.utils.time_series import (
    lag,
    rolling_beta,
    rolling_zscore,
    ts_arg_max,
    ts_backfill,
    ts_quantile,
)
from tiger_factors.utils.factor_parallel import run_factor_tasks_parallel


def test_cross_sectional_zscore_rowwise_mean_is_zero():
    panel = pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0],
            "B": [2.0, 3.0, 4.0],
            "C": [3.0, 4.0, 5.0],
        },
        index=pd.date_range("2024-01-01", periods=3),
    )

    out = zscore(panel, axis=1)
    np.testing.assert_allclose(out.mean(axis=1).to_numpy(), np.zeros(3), atol=1e-12)


def test_cross_sectional_rank_pct_bounds():
    row = pd.Series([10.0, 30.0, 20.0], index=["A", "B", "C"])
    out = rank_pct(row)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_cross_sectional_winsorize_quantile_clips_outlier():
    row = pd.Series([1.0, 2.0, 3.0, 100.0])
    out = winsorize_quantile(row, lower=0.0, upper=0.75)
    assert out.max() <= row.quantile(0.75)


def test_neutralize_by_group_demean():
    values = pd.Series([1.0, 3.0, 10.0, 14.0], index=["A", "B", "C", "D"])
    groups = pd.Series(["g1", "g1", "g2", "g2"], index=values.index)
    out = neutralize_by_group(values, groups, method="demean")
    np.testing.assert_allclose(out.groupby(groups).mean().to_numpy(), np.zeros(2), atol=1e-12)


def test_winsorize_cross_section_mad_clips_outlier():
    row = pd.Series([1.0, 2.0, 3.0, 1000.0])
    out = winsorize_cross_section(row, method="mad", n_mad=2.0)
    assert out.iloc[-1] < row.iloc[-1]


def test_normalize_cross_section_rank_centered_is_bounded():
    row = pd.Series([10.0, 30.0, 20.0], index=["A", "B", "C"])
    out = normalize_cross_section(row, method="rank_centered")
    assert out.min() >= -0.5
    assert out.max() <= 0.5


def test_neutralize_cross_section_group_zscore_uses_unit_group_std():
    values = pd.Series([1.0, 3.0, 10.0, 14.0], index=["A", "B", "C", "D"])
    groups = pd.Series(["g1", "g1", "g2", "g2"], index=values.index)
    out = neutralize_cross_section(values, groups=groups, method="group_zscore")
    np.testing.assert_allclose(out.groupby(groups).mean().to_numpy(), np.zeros(2), atol=1e-12)
    np.testing.assert_allclose(out.groupby(groups).std(ddof=0).to_numpy(), np.ones(2), atol=1e-12)


def test_group_demean_matches_group_means_zero():
    values = pd.Series([1.0, 3.0, 7.0, 11.0], index=list("ABCD"))
    groups = pd.Series(["g1", "g1", "g2", "g2"], index=values.index)
    out = group_demean(values, groups)
    np.testing.assert_allclose(out.groupby(groups).mean().to_numpy(), np.zeros(2), atol=1e-12)


def test_group_zscore_matches_population_group_std_one():
    values = pd.Series([1.0, 3.0, 7.0, 11.0], index=list("ABCD"))
    groups = pd.Series(["g1", "g1", "g2", "g2"], index=values.index)
    out = group_zscore(values, groups)
    np.testing.assert_allclose(out.groupby(groups).std(ddof=0).to_numpy(), np.ones(2), atol=1e-12)


def test_industry_regression_residual_removes_industry_level():
    industries = pd.Series(["tech", "tech", "fin", "fin", "energy", "energy"], index=list("ABCDEF"))
    values = pd.Series([10.0, 10.0, 3.0, 3.0, -2.0, -2.0], index=industries.index)
    out = industry_regression_residual(values, industries)
    np.testing.assert_allclose(out.fillna(0.0).to_numpy(), np.zeros(len(values)), atol=1e-12)


def test_group_dummy_regression_residual_removes_group_effect():
    groups = pd.Series(["g1", "g1", "g2", "g2", "g3", "g3"], index=list("ABCDEF"))
    values = pd.Series([5.0, 5.0, 2.0, 2.0, -4.0, -4.0], index=groups.index)
    out = group_dummy_regression_residual(values, groups)
    np.testing.assert_allclose(out.fillna(0.0).to_numpy(), np.zeros(len(values)), atol=1e-12)


def test_industry_size_regression_residual_removes_joint_exposures():
    industries = pd.Series(["tech", "tech", "fin", "fin", "energy", "energy"], index=list("ABCDEF"))
    market_cap = pd.Series([100.0, 200.0, 120.0, 240.0, 80.0, 160.0], index=industries.index)
    log_size = np.log(market_cap)
    industry_component = pd.Series([1.0, 1.0, -2.0, -2.0, 0.5, 0.5], index=industries.index)
    values = 3.0 * log_size + industry_component
    out = industry_size_regression_residual(values, industries, market_cap)
    np.testing.assert_allclose(out.fillna(0.0).to_numpy(), np.zeros(len(values)), atol=1e-10)


def test_preprocess_cross_section_runs_full_pipeline():
    values = pd.Series([1.0, 2.0, 4.0, 100.0], index=["A", "B", "C", "D"])
    groups = pd.Series(["g1", "g1", "g2", "g2"], index=values.index)
    out = preprocess_cross_section(
        values,
        winsorize_method="mad",
        normalize_method="zscore",
        neutralize_method="group_demean",
        groups=groups,
    )
    np.testing.assert_allclose(out.groupby(groups).mean().to_numpy(), np.zeros(2), atol=1e-12)


def test_time_series_lag_and_rolling_zscore_shapes():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    lagged = lag(s, periods=1)
    z = rolling_zscore(s, window=3)
    assert len(lagged) == len(s)
    assert len(z) == len(s)
    assert np.isnan(lagged.iloc[0])


def test_rolling_beta_simple_linear_relation():
    x = pd.Series([1, 2, 3, 4, 5, 6], dtype=float)
    y = 2.0 * x
    beta = rolling_beta(y, x, window=4)
    assert np.isclose(beta.dropna().iloc[-1], 2.0)


def test_ts_backfill_respects_limit():
    s = pd.Series([1.0, np.nan, np.nan, 4.0])
    out = ts_backfill(s, lookback=10, k=1)
    assert out.iloc[0] == 1.0
    assert pd.isna(out.iloc[1])
    assert out.iloc[2] == 4.0
    assert out.iloc[3] == 4.0


def test_ts_quantile_gaussian_is_finite():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = ts_quantile(s, d=3)
    assert np.isfinite(out.dropna().iloc[-1])


def test_ts_arg_max_returns_distance_from_latest():
    s = pd.Series([1.0, 5.0, 3.0, 2.0])
    out = ts_arg_max(s, d=4)
    assert out.dropna().iloc[-1] == 2.0


def test_cs_minmax_pos_uses_half_for_flat_rows():
    panel = pd.DataFrame([[1.0, 1.0, 1.0]], columns=["A", "B", "C"])
    out = cs_minmax_pos(panel)
    np.testing.assert_allclose(out.iloc[0].to_numpy(), np.array([0.5, 0.5, 0.5]))


def test_cs_neutralize_removes_linear_exposure():
    index = pd.to_datetime(["2024-01-01"])
    exposure = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 5.0]], index=index, columns=list("ABCDE"))
    target = 2.0 * exposure + 1.0
    residual = cs_neutralize(target, exposures=[exposure])
    np.testing.assert_allclose(residual.loc[index[0]].fillna(0.0).to_numpy(), np.zeros(5), atol=1e-10)


def test_make_industry_dummies_keeps_expected_columns():
    df = pd.DataFrame({"industry": ["software", "bank", "software"]})
    out = make_industry_dummies(df, all_industries=["bank", "software", "oil"], baseline="bank")
    assert list(out.columns) == ["ind_software", "ind_oil"]
    assert out["ind_oil"].sum() == 0


def test_rowwise_cross_sectional_corr_handles_large_values():
    index = pd.date_range("2024-01-01", periods=3)
    left = pd.DataFrame(
        {
            "A": [1e160, 2e160, 3e160],
            "B": [2e160, 3e160, 4e160],
            "C": [3e160, 4e160, 5e160],
        },
        index=index,
    )
    right = pd.DataFrame(
        {
            "A": [5e160, 4e160, 3e160],
            "B": [4e160, 3e160, 2e160],
            "C": [3e160, 2e160, 1e160],
        },
        index=index,
    )

    corr = _rowwise_cross_sectional_corr(left, right)

    assert not corr.isna().all()
    assert np.isfinite(corr.dropna()).all()


def _parallel_task(task_id: int) -> tuple[int, pd.DataFrame]:
    frame = pd.DataFrame(
        {
            "date_": pd.to_datetime(["2024-01-02"]),
            "code": [f"C{task_id}"],
            "value": [float(task_id)],
        }
    )
    return int(task_id), frame


def test_run_factor_tasks_parallel_computes_and_saves(tmp_path):
    result = run_factor_tasks_parallel(
        [1, 2],
        compute_fn=_parallel_task,
        compute_workers=2,
        save_workers=2,
        save=True,
        output_dir=tmp_path,
        factor_name_for_task=lambda task_id: f"factor_{task_id}",
        metadata_for_task=lambda task_id, mode: {"task_id": int(task_id), "mode": mode},
        prefer_process=False,
    )

    assert set(result.computed_frames) == {1, 2}
    assert result.execution_mode == "thread"
    assert result.saved_factor_paths is not None
    assert result.saved_metadata_paths is not None
    assert (tmp_path / "factor" / "tiger" / "us" / "stock" / "1d" / "factor_1.parquet").exists()
    assert (tmp_path / "factor" / "tiger" / "us" / "stock" / "1d" / "factor_2.parquet").exists()
    assert result.saved_factor_paths["factor_1"].endswith("factor_1.parquet")
    assert result.saved_metadata_paths["factor_2"].endswith("manifest__factor_2.json")
