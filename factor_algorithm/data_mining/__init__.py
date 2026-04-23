"""Cleaned data-mining factors and a small evaluation engine.

The folder name follows the English term for 数据挖掘:
`data_mining`.
"""

from .factors import *  # noqa: F401,F403
from .factors import __all__ as _factors_all
from .practical_factors import *  # noqa: F401,F403
from .practical_factors import __all__ as _practical_factors_all

__all__ = [*_factors_all, *_practical_factors_all]
