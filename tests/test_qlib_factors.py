from __future__ import annotations

import pandas as pd

from tiger_factors.factor_algorithm.qlib_factors import (
    Alpha158FactorSet,
    Alpha360FactorSet,
    QlibAlphaFactorSetEngine,
    alpha158_feature_config,
    alpha360_feature_config,
    available_qlib_factor_sets,
)


def test_alpha_feature_configs_have_expected_sizes() -> None:
    alpha158_fields, alpha158_names = alpha158_feature_config()
    alpha360_fields, alpha360_names = alpha360_feature_config()

    assert len(alpha158_fields) == 158
    assert len(alpha158_names) == 158
    assert len(alpha360_fields) == 360
    assert len(alpha360_names) == 360
    assert "KMID" in alpha158_names
    assert "VOLUME0" in alpha360_names


def test_factor_set_metadata() -> None:
    alpha158 = Alpha158FactorSet()
    alpha360 = Alpha360FactorSet()

    assert len(alpha158.feature_names) == 158
    assert len(alpha360.feature_names) == 360
    assert set(available_qlib_factor_sets()) == {"alpha158", "alpha360"}


def test_normalize_fetch_frame() -> None:
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-01-02", "2020-01-01"]),
            "instrument": ["AAA", "BBB"],
            "KMID": [1.0, 2.0],
            "VOLUME0": [3.0, 4.0],
        }
    )

    normalized = QlibAlphaFactorSetEngine._normalize_fetch_frame(frame)

    assert list(normalized.columns) == ["date_", "code", "KMID", "VOLUME0"]
    assert normalized.iloc[0]["code"] == "BBB"
    assert pd.api.types.is_datetime64_any_dtype(normalized["date_"])
