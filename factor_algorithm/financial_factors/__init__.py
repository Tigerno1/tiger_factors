from __future__ import annotations

from importlib import import_module

from .financial_factors import FinancialFactorBundleResult
from .financial_factors import FinancialFactorEngine
from .financial_factors import QuarterlyFinancialFactorEngine
from .financial_factors import AnnualFinancialFactorEngine
from .financial_factors import TTMFinancialFactorEngine

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "DEFAULT_VARIANTS": ("tiger_factors.factor_algorithm.financial_factors.financial_factor_recorder", "DEFAULT_VARIANTS"),
    "record_financial_factors": ("tiger_factors.factor_algorithm.financial_factors.financial_factor_recorder", "record_financial_factors"),
    "resolve_codes": ("tiger_factors.factor_algorithm.financial_factors.financial_factor_recorder", "resolve_codes"),
}

__all__ = [
    "FinancialFactorBundleResult",
    "FinancialFactorEngine",
    "QuarterlyFinancialFactorEngine",
    "AnnualFinancialFactorEngine",
    "TTMFinancialFactorEngine",
    "DEFAULT_VARIANTS",
    "record_financial_factors",
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
