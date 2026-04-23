from __future__ import annotations

from importlib import import_module

_EXPORTS: dict[str, tuple[str, str]] = {
    "ExperimentalFactorSpec": ("tiger_factors.factor_algorithm.experimental.catalog", "ExperimentalFactorSpec"),
    "experimental_factor_catalog": ("tiger_factors.factor_algorithm.experimental.catalog", "experimental_factor_catalog"),
    "MarketBreathingColumns": ("tiger_factors.factor_algorithm.experimental.market_breathing", "MarketBreathingColumns"),
    "MarketBreathingEngine": ("tiger_factors.factor_algorithm.experimental.market_breathing", "MarketBreathingEngine"),
    "MarketBreathingResult": ("tiger_factors.factor_algorithm.experimental.market_breathing", "MarketBreathingResult"),
    "market_breathing_factor_names": (
        "tiger_factors.factor_algorithm.experimental.market_breathing",
        "market_breathing_factor_names",
    ),
    "experimental_factor_spec": ("tiger_factors.factor_algorithm.experimental.catalog", "experimental_factor_spec"),
    "NewsEntropyColumns": ("tiger_factors.factor_algorithm.experimental.news_entropy", "NewsEntropyColumns"),
    "NewsEntropyEngine": ("tiger_factors.factor_algorithm.experimental.news_entropy", "NewsEntropyEngine"),
    "NewsEntropyResult": ("tiger_factors.factor_algorithm.experimental.news_entropy", "NewsEntropyResult"),
    "news_entropy_factor_names": (
        "tiger_factors.factor_algorithm.experimental.news_entropy",
        "news_entropy_factor_names",
    ),
    "experimental_factor_names": ("tiger_factors.factor_algorithm.experimental.catalog", "experimental_factor_names"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(target[0])
    value = getattr(module, target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
