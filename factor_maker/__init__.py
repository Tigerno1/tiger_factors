"""Unified factor maker facade.

This namespace exposes the factor-making entrypoints that are safe to import
eagerly: pipeline-style factor production, vectorization helpers, streaming
utilities, and the algorithm wrappers that do not pull in recorder helpers at
import time.
"""

from __future__ import annotations

from tiger_factors.factor_algorithm.alpha101 import *  # noqa: F401,F403
from tiger_factors.factor_algorithm.alpha101 import __all__ as _alpha101_all
from tiger_factors.factor_algorithm.data_mining import *  # noqa: F401,F403
from tiger_factors.factor_algorithm.data_mining import __all__ as _data_mining_all
from tiger_factors.factor_algorithm.factor_timing import *  # noqa: F401,F403
from tiger_factors.factor_algorithm.factor_timing import __all__ as _factor_timing_all
from tiger_factors.factor_algorithm.financial_factors.financial_factors import *  # noqa: F401,F403
from tiger_factors.factor_algorithm.financial_factors.financial_factors import __all__ as _financial_all
from tiger_factors.factor_algorithm.qlib_factors import *  # noqa: F401,F403
from tiger_factors.factor_algorithm.qlib_factors import __all__ as _qlib_all
from tiger_factors.factor_algorithm.traditional_factors import *  # noqa: F401,F403
from tiger_factors.factor_algorithm.traditional_factors import __all__ as _traditional_all
from tiger_factors.factor_algorithm.valuation_factors.valuation_factors import *  # noqa: F401,F403
from tiger_factors.factor_algorithm.valuation_factors.valuation_factors import __all__ as _valuation_all
from tiger_factors.factor_maker.pipeline import *  # noqa: F401,F403
from tiger_factors.factor_maker.pipeline import __all__ as _factor_pipeline_all
from tiger_factors.factor_maker.vectorization import *  # noqa: F401,F403
from tiger_factors.factor_maker.vectorization import __all__ as _factor_vectorization_all

__all__ = [
    *_alpha101_all,
    *_data_mining_all,
    *_factor_pipeline_all,
    *_factor_timing_all,
    *_factor_vectorization_all,
    *_financial_all,
    *_qlib_all,
    *_traditional_all,
    *_valuation_all,
]
