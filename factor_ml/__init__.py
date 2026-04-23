"""Unified factor ML namespace.

This package groups ML / DL factor generation helpers and AlphaGPT-style
factor program synthesis utilities under one umbrella.
"""

from __future__ import annotations

from tiger_factors.factor_ml.alpha_constraints import *  # noqa: F401,F403
from tiger_factors.factor_ml.alpha_constraints import __all__ as _alpha_constraints_all
from tiger_factors.factor_ml.alpha_execution import *  # noqa: F401,F403
from tiger_factors.factor_ml.alpha_execution import __all__ as _alpha_execution_all
from tiger_factors.factor_ml.alpha_ops import *  # noqa: F401,F403
from tiger_factors.factor_ml.alpha_ops import __all__ as _alpha_ops_all
from tiger_factors.factor_ml.data_mining import *  # noqa: F401,F403
from tiger_factors.factor_ml.data_mining import __all__ as _data_mining_all
from tiger_factors.factor_ml.alphagpt import *  # noqa: F401,F403
from tiger_factors.factor_ml.alphagpt import __all__ as _alphagpt_all

__all__ = [
    *_alpha_constraints_all,
    *_alpha_execution_all,
    *_alpha_ops_all,
    *_data_mining_all,
    *_alphagpt_all,
]
