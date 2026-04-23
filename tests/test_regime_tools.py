from __future__ import annotations

import numpy as np
import pandas as pd

from tiger_factors.factor_evaluation import build_hmm_regime_group_labels
from tiger_factors.factor_evaluation import build_hmm_regime_labels
from tiger_factors.factor_evaluation import build_kmeans_regime_group_labels
from tiger_factors.factor_evaluation import build_kmeans_regime_labels
from tiger_factors.factor_evaluation import build_market_regime_features
from tiger_factors.factor_evaluation import expand_date_regime_to_group_labels


def _sample_price_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    rows: list[dict[str, object]] = []
    for idx, date_value in enumerate(dates):
        rows.append({"date_": date_value, "code": "AAA", "close": 100 + idx * 0.6})
        rows.append({"date_": date_value, "code": "BBB", "close": 80 + idx * 0.4})
        rows.append({"date_": date_value, "code": "CCC", "close": 60 + idx * 0.3})
    return pd.DataFrame(rows)


def _sample_reference_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-15", periods=12, freq="D")
    return pd.DataFrame(
        {
            "date_": np.repeat(dates, 2),
            "code": ["AAA", "BBB"] * len(dates),
            "alpha_001": np.tile([1.0, -1.0], len(dates)),
        }
    )


def _synthetic_regime_features() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=45, freq="D")
    values = pd.DataFrame(index=dates)
    values["market_return"] = np.r_[
        np.full(15, -0.02),
        np.full(15, 0.0),
        np.full(15, 0.02),
    ]
    values["market_vol"] = np.r_[
        np.full(15, 0.03),
        np.full(15, 0.01),
        np.full(15, 0.02),
    ]
    values["market_momentum"] = np.r_[
        np.full(15, -0.10),
        np.full(15, 0.00),
        np.full(15, 0.10),
    ]
    return values


def test_build_market_regime_features_from_long_prices() -> None:
    features = build_market_regime_features(_sample_price_frame(), vol_window=5, momentum_window=5)

    assert {"market_return", "market_vol", "market_momentum"}.issubset(features.columns)
    assert isinstance(features.index, pd.DatetimeIndex)


def test_kmeans_regime_helpers_expand_to_group_labels() -> None:
    features = _synthetic_regime_features()
    labels = build_kmeans_regime_labels(features, n_regimes=3, random_state=7)
    reference = _sample_reference_frame()
    group_labels = build_kmeans_regime_group_labels(reference, features, n_regimes=3, random_state=7)

    assert set(labels.dropna().unique()) == {"Bear", "Sideways", "Bull"}
    assert len(group_labels) == len(reference[["date_", "code"]].drop_duplicates())
    assert {"date_", "code", "group"} == set(group_labels.columns)


def test_hmm_regime_helpers_expand_to_group_labels() -> None:
    features = _synthetic_regime_features()
    labels = build_hmm_regime_labels(features, n_regimes=3, random_state=11, n_iter=100)
    reference = _sample_reference_frame()
    group_labels = build_hmm_regime_group_labels(reference, features, n_regimes=3, random_state=11, n_iter=100)

    assert labels.dropna().nunique() <= 3
    assert len(group_labels) == len(reference[["date_", "code"]].drop_duplicates())
    assert group_labels["group"].notna().any()


def test_expand_date_regime_to_group_labels_accepts_series_input() -> None:
    reference = _sample_reference_frame()
    regime_by_date = pd.Series(
        ["Bull"] * reference["date_"].nunique(),
        index=pd.Index(sorted(reference["date_"].unique()), name="date_"),
        name="group",
    )

    labels = expand_date_regime_to_group_labels(reference, regime_by_date)

    assert len(labels) == len(reference[["date_", "code"]].drop_duplicates())
    assert labels["group"].eq("Bull").all()
