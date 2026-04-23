from __future__ import annotations

from tiger_factors.factor_algorithm.financial_factors import FinancialFactorBundleResult
from tiger_factors.factor_algorithm.financial_factors import FinancialFactorEngine


class ValuationFactorEngine(FinancialFactorEngine):
    """Valuation-factor wrapper for price-aligned SimFin valuation metrics.

    The implementation is shared with :class:`FinancialFactorEngine`, but this
    module provides the dedicated public home for valuation factor bundles.
    """

    def __init__(self, *, max_factors: int | None = None, **kwargs) -> None:
        kwargs.setdefault("price_provider", "simfin")
        super().__init__(max_factors=max_factors, **kwargs)


__all__ = [
    "FinancialFactorBundleResult",
    "ValuationFactorEngine",
]
