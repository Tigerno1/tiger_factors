from __future__ import annotations

import numpy as np
import pandas as pd

from tiger_factors.factor_test import build_market_state_features
from tiger_factors.factor_test import build_kmeans_market_state_labels
from tiger_factors.factor_test import market_state_test


def _sample_price_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=40)
    rng = np.random.default_rng(7)
    bull = np.linspace(100.0, 125.0, len(dates))
    vol = np.sin(np.linspace(0.0, 4.0 * np.pi, len(dates))) * 2.0
    price_a = bull + vol + rng.normal(0.0, 0.4, len(dates))
    price_b = bull * 0.98 + vol * 0.8 + rng.normal(0.0, 0.4, len(dates))
    return pd.DataFrame({"AAA": price_a, "BBB": price_b}, index=dates)


def test_market_state_features_and_kmeans_labels_are_available() -> None:
    prices = _sample_price_panel()

    features = build_market_state_features(prices)
    labels = build_kmeans_market_state_labels(features, n_states=3, random_state=11)

    assert {"market_return", "market_vol", "market_momentum"}.issubset(features.columns)
    assert labels.dropna().nunique() <= 3
    assert len(labels) == len(features)


def test_market_state_test_produces_independent_summary() -> None:
    prices = _sample_price_panel()

    result = market_state_test(prices, n_states=3, use_hmm=False, random_state=11)

    assert result.method == "kmeans"
    assert result.current_state is not None
    assert result.state_transition_matrix.shape[0] >= 1
    assert sum(result.state_counts.values()) > 0
    assert "date_" in result.table.columns
    assert "state" in result.table.columns
