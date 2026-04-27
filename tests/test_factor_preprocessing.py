from __future__ import annotations

import numpy as np
import pandas as pd

from tiger_factors.factor_preprocessing import FactorPreprocessor
from tiger_factors.factor_preprocessing import bin_factor_panel
from tiger_factors.factor_preprocessing import detect_outliers_factor_panel
from tiger_factors.factor_preprocessing import fill_missing_factor_panel
from tiger_factors.factor_preprocessing import preprocess_factor_panel
from tiger_factors.factor_preprocessing import scale_factor_panel
from tiger_factors.factor_preprocessing import target_encode_binned
from tiger_factors.factor_preprocessing import woe_encode_binned
from tiger_factors.factor_preprocessing import winsorize_factor_panel


def test_fill_missing_factor_panel_median_rowwise() -> None:
    panel = pd.DataFrame(
        {
            "AAA": [1.0, np.nan],
            "BBB": [3.0, 5.0],
            "CCC": [np.nan, 7.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    out = fill_missing_factor_panel(panel, method="median")

    assert float(out.loc[pd.Timestamp("2024-01-01"), "CCC"]) == 2.0
    assert float(out.loc[pd.Timestamp("2024-01-02"), "AAA"]) == 6.0


def test_winsorize_and_detect_outliers_clip_extreme_value() -> None:
    panel = pd.DataFrame(
        {
            "AAA": [1.0, 1.0, 1.0],
            "BBB": [2.0, 2.0, 2.0],
            "CCC": [3.0, 3.0, 100.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )

    mask = detect_outliers_factor_panel(panel, method="mad", n_mad=2.0)
    clipped = winsorize_factor_panel(panel, method="quantile", lower=0.0, upper=0.75)

    assert bool(mask.iloc[-1, -1])
    assert clipped.iloc[-1, -1] <= panel.iloc[:, -1].quantile(0.75)


def test_scale_factor_panel_rowwise_zscore_has_zero_mean() -> None:
    panel = pd.DataFrame(
        {
            "AAA": [1.0, 2.0],
            "BBB": [2.0, 3.0],
            "CCC": [3.0, 4.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    out = scale_factor_panel(panel, method="zscore")

    np.testing.assert_allclose(out.mean(axis=1).to_numpy(), np.zeros(2), atol=1e-12)


def test_bin_factor_panel_quantile_returns_expected_shape() -> None:
    panel = pd.DataFrame(
        {
            "AAA": [1.0, 4.0],
            "BBB": [2.0, 3.0],
            "CCC": [3.0, 2.0],
            "DDD": [4.0, 1.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    out = bin_factor_panel(panel, method="quantile", n_bins=2)

    assert out.shape == panel.shape
    assert set(pd.unique(out.iloc[0].dropna())) <= {0.0, 1.0}


def test_supervised_binning_and_encoding_helpers_work() -> None:
    factor = pd.Series([1.0, 2.0, 3.0, 4.0], index=list("ABCD"))
    target = pd.Series([0, 0, 1, 1], index=list("ABCD"))

    binned = bin_factor_panel(factor, method="chi_merge", target=target, n_bins=2)
    woe = woe_encode_binned(binned, target)
    encoded = target_encode_binned(binned, target)

    assert binned.nunique(dropna=True) <= 2
    assert np.isfinite(woe.dropna().iloc[0])
    assert np.isfinite(encoded.dropna().iloc[0])


def test_factor_preprocessor_pipeline_runs() -> None:
    panel = pd.DataFrame(
        {
            "AAA": [1.0, np.nan],
            "BBB": [2.0, 3.0],
            "CCC": [100.0, 4.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    preprocessor = FactorPreprocessor(
        missing_strategy="median",
        outlier_strategy="winsorize",
        outlier_kwargs={"method": "quantile", "lower": 0.0, "upper": 0.75},
        normalize_strategy="zscore",
        bin_strategy=None,
    )

    out = preprocessor.fit_transform(panel)

    assert out.shape == panel.shape
    np.testing.assert_allclose(out.mean(axis=1).to_numpy(), np.zeros(2), atol=1e-12)


def test_preprocess_factor_panel_with_binning() -> None:
    panel = pd.DataFrame(
        {
            "AAA": [1.0, 4.0],
            "BBB": [2.0, 3.0],
            "CCC": [3.0, 2.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    out = preprocess_factor_panel(
        panel,
        missing_strategy=None,
        outlier_strategy=None,
        normalize_strategy=None,
        bin_strategy="quantile",
        bin_kwargs={"n_bins": 3},
    )

    assert out.shape == panel.shape
