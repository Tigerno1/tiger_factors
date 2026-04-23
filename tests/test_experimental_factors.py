from __future__ import annotations

import pandas as pd

from tiger_factors.factor_algorithm import (
    MarketBreathingEngine,
    NewsEntropyEngine,
    experimental_factor_catalog,
    experimental_factor_names,
    experimental_factor_spec,
)


def test_experimental_factor_names_are_registered():
    assert experimental_factor_names() == [
        "market_breathing",
        "vol_surface_torsion",
        "iv_realized_gap",
        "liquidity_pressure",
        "vol_of_vol_regime",
        "news_entropy",
        "event_density",
        "topic_dispersion",
        "news_novelty",
        "attention_spike",
        "information_pressure",
        "regime_anomaly",
        "panic_breath",
    ]
    assert len(experimental_factor_catalog()) == 13


def test_experimental_factor_catalog_contains_expected_metadata():
    spec = experimental_factor_spec("panic_breath")

    assert spec is not None
    assert spec.category == "hybrid_regime"
    assert "iv_surface_skew" in spec.required_fields
    assert "event_sentiment" in spec.optional_fields


def test_market_breathing_engine_computes_score():
    rows = []
    for offset in range(12):
        rows.append(
            {
                "date_": pd.Timestamp("2024-01-01") + pd.Timedelta(days=offset),
                "code": "AAPL",
                "iv_surface_level": 0.20 + 0.01 * offset,
                "iv_surface_skew": -0.10 + 0.005 * offset,
                "iv_surface_curvature": 0.30 + 0.02 * offset,
                "iv_term_slope": 0.15 + 0.003 * offset,
                "option_spread": 0.05 + 0.001 * offset,
                "option_volume": 1000 + 50 * offset,
                "option_open_interest": 5000 + 100 * offset,
                "realized_volatility": 0.18 + 0.002 * offset,
            }
        )
    frame = pd.DataFrame(rows)

    engine = MarketBreathingEngine(frame, window=4)
    result = engine.compute()

    assert not result.frame.empty
    assert set(["date_", "code", "value", "factor_name"]).issubset(result.frame.columns)
    assert result.metadata["factor_name"] == "market_breathing"
    assert result.frame["value"].notna().any()


def test_news_entropy_engine_computes_entropy_features():
    rows = []
    rows.extend(
        [
            {"date_": "2024-01-01", "code": "AAPL", "event_topic": "earnings", "event_sentiment": "positive", "event_novelty": 0.8, "event_weight": 1.0},
            {"date_": "2024-01-01", "code": "AAPL", "event_topic": "earnings", "event_sentiment": "positive", "event_novelty": 0.7, "event_weight": 1.0},
            {"date_": "2024-01-01", "code": "AAPL", "event_topic": "guidance", "event_sentiment": "neutral", "event_novelty": 0.6, "event_weight": 1.0},
            {"date_": "2024-01-02", "code": "AAPL", "event_topic": "earnings", "event_sentiment": "positive", "event_novelty": 0.9, "event_weight": 1.0},
            {"date_": "2024-01-02", "code": "AAPL", "event_topic": "earnings", "event_sentiment": "positive", "event_novelty": 0.9, "event_weight": 1.0},
        ]
    )
    frame = pd.DataFrame(rows)

    engine = NewsEntropyEngine(frame, window=3)
    result = engine.compute()

    assert not result.frame.empty
    assert set(["date_", "code", "value", "topic_entropy", "sentiment_entropy"]).issubset(result.frame.columns)
    assert result.metadata["factor_name"] == "news_entropy"
    mixed_day = result.frame.loc[result.frame["date_"] == pd.Timestamp("2024-01-01"), "topic_entropy"].iloc[0]
    single_topic_day = result.frame.loc[result.frame["date_"] == pd.Timestamp("2024-01-02"), "topic_entropy"].iloc[0]
    assert mixed_day >= single_topic_day
