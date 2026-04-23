from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExperimentalFactorSpec:
    name: str
    category: str
    required_fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = ()
    description: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "category": self.category,
            "required_fields": list(self.required_fields),
            "optional_fields": list(self.optional_fields),
            "description": self.description,
        }


_EXPERIMENTAL_FACTOR_SPECS: tuple[ExperimentalFactorSpec, ...] = (
    ExperimentalFactorSpec(
        name="market_breathing",
        category="market_structure",
        required_fields=(
            "date_",
            "code",
            "iv_surface_level",
            "iv_surface_skew",
            "iv_surface_curvature",
            "iv_term_slope",
            "option_spread",
            "option_volume",
        ),
        optional_fields=("option_open_interest", "realized_volatility"),
        description="Implied-volatility surface motion and liquidity pressure composite.",
    ),
    ExperimentalFactorSpec(
        name="vol_surface_torsion",
        category="market_structure",
        required_fields=(
            "date_",
            "code",
            "iv_surface_level",
            "iv_surface_skew",
            "iv_surface_curvature",
            "iv_term_slope",
        ),
        description="Curvature and slope twist in the IV surface.",
    ),
    ExperimentalFactorSpec(
        name="iv_realized_gap",
        category="market_structure",
        required_fields=(
            "date_",
            "code",
            "iv_surface_level",
            "realized_volatility",
        ),
        optional_fields=("iv_term_slope",),
        description="Gap between implied and realized volatility regimes.",
    ),
    ExperimentalFactorSpec(
        name="liquidity_pressure",
        category="market_structure",
        required_fields=(
            "date_",
            "code",
            "option_spread",
            "option_volume",
        ),
        optional_fields=("option_open_interest", "turnover", "bid_ask_spread"),
        description="Liquidity tightening and trading cost pressure.",
    ),
    ExperimentalFactorSpec(
        name="vol_of_vol_regime",
        category="market_structure",
        required_fields=(
            "date_",
            "code",
            "realized_volatility",
        ),
        optional_fields=("iv_surface_curvature", "iv_surface_skew"),
        description="Volatility of volatility and instability regime proxy.",
    ),
    ExperimentalFactorSpec(
        name="news_entropy",
        category="information_flow",
        required_fields=("date_", "code", "event_topic"),
        optional_fields=("event_sentiment", "event_novelty", "event_weight"),
        description="Shannon entropy of topics, sentiment, density, and novelty.",
    ),
    ExperimentalFactorSpec(
        name="event_density",
        category="information_flow",
        required_fields=("date_", "code", "event_topic"),
        optional_fields=("event_weight", "event_sentiment"),
        description="Event concentration and burst intensity within a window.",
    ),
    ExperimentalFactorSpec(
        name="topic_dispersion",
        category="information_flow",
        required_fields=("date_", "code", "event_topic"),
        optional_fields=("event_weight", "topic_cluster"),
        description="How many different stories are being processed at once.",
    ),
    ExperimentalFactorSpec(
        name="news_novelty",
        category="information_flow",
        required_fields=("date_", "code", "event_topic"),
        optional_fields=("event_embedding", "event_weight"),
        description="Novelty versus repeated narrative pressure.",
    ),
    ExperimentalFactorSpec(
        name="attention_spike",
        category="information_flow",
        required_fields=("date_", "code"),
        optional_fields=("news_count", "search_volume", "social_mentions"),
        description="Abnormal attention / coverage spike proxy.",
    ),
    ExperimentalFactorSpec(
        name="information_pressure",
        category="hybrid_regime",
        required_fields=(
            "date_",
            "code",
            "event_topic",
            "iv_surface_level",
        ),
        optional_fields=("event_sentiment", "option_spread", "realized_volatility"),
        description="Joint information and market-state shock pressure factor.",
    ),
    ExperimentalFactorSpec(
        name="regime_anomaly",
        category="hybrid_regime",
        required_fields=(
            "date_",
            "code",
            "iv_surface_level",
            "event_topic",
        ),
        optional_fields=("option_spread", "news_count", "realized_volatility"),
        description="Composite anomaly factor for non-standard market regimes.",
    ),
    ExperimentalFactorSpec(
        name="panic_breath",
        category="hybrid_regime",
        required_fields=(
            "date_",
            "code",
            "iv_surface_skew",
            "option_spread",
            "event_topic",
        ),
        optional_fields=("news_count", "event_sentiment", "realized_volatility"),
        description="Fear expansion proxy combining skew, spread, and news shock.",
    ),
)


def experimental_factor_catalog() -> list[ExperimentalFactorSpec]:
    return list(_EXPERIMENTAL_FACTOR_SPECS)


def experimental_factor_names() -> list[str]:
    return [spec.name for spec in _EXPERIMENTAL_FACTOR_SPECS]


def experimental_factor_spec(name: str) -> ExperimentalFactorSpec | None:
    normalized = name.strip().casefold()
    for spec in _EXPERIMENTAL_FACTOR_SPECS:
        if spec.name.casefold() == normalized:
            return spec
    return None
