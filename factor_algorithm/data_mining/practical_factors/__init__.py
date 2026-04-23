"""Practical factor variants that are ready for direct testing.

This subpackage keeps the cleaned formulas from the screenshots in a
single place, separate from the broader data-mining registry.
"""

from .factors import *  # noqa: F401,F403

from .factors import __all__ as _factors_all

__all__ = [*_factors_all]
