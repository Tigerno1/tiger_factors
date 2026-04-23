from __future__ import annotations

from importlib import import_module

from .valuation_factors import FinancialFactorBundleResult
from .valuation_factors import ValuationFactorEngine

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "DEFAULT_VARIANTS": ("tiger_factors.factor_algorithm.valuation_factors.valuation_factor_recorder", "DEFAULT_VARIANTS"),
    "record_valuation_factors": ("tiger_factors.factor_algorithm.valuation_factors.valuation_factor_recorder", "record_valuation_factors"),
    "resolve_codes": ("tiger_factors.factor_algorithm.valuation_factors.valuation_factor_recorder", "resolve_codes"),
}

__all__ = [
    "FinancialFactorBundleResult",
    "ValuationFactorEngine",
    "DEFAULT_VARIANTS",
    "record_valuation_factors",
    "resolve_codes",
]


def __getattr__(name: str):
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(target[0])
    value = getattr(module, target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_LAZY_ATTRS))
