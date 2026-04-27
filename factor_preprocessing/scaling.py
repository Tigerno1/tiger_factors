from __future__ import annotations

import pandas as pd

from tiger_factors.utils.cross_sectional import demean
from tiger_factors.utils.cross_sectional import l1_normalize
from tiger_factors.utils.cross_sectional import l2_normalize
from tiger_factors.utils.cross_sectional import minmax_scale
from tiger_factors.utils.cross_sectional import normalize_cross_section
from tiger_factors.utils.cross_sectional import preprocess_cross_section
from tiger_factors.utils.cross_sectional import rank_centered
from tiger_factors.utils.cross_sectional import rank_pct
from tiger_factors.utils.cross_sectional import robust_zscore
from tiger_factors.utils.cross_sectional import winsorize_cross_section
from tiger_factors.utils.cross_sectional import winsorize_mad
from tiger_factors.utils.cross_sectional import winsorize_quantile
from tiger_factors.utils.cross_sectional import zscore


def scale_factor_panel(
    values: pd.Series | pd.DataFrame,
    *,
    method: str = "zscore",
    axis: int = 1,
    ddof: int = 0,
    clip: tuple[float, float] | None = None,
    feature_range: tuple[float, float] = (0.0, 1.0),
    rank_method: str = "average",
) -> pd.Series | pd.DataFrame:
    return normalize_cross_section(
        values,
        method=method,
        axis=axis,
        ddof=ddof,
        clip=clip,
        feature_range=feature_range,
        rank_method=rank_method,
    )


__all__ = [
    "demean",
    "l1_normalize",
    "l2_normalize",
    "minmax_scale",
    "normalize_cross_section",
    "preprocess_cross_section",
    "rank_centered",
    "rank_pct",
    "robust_zscore",
    "scale_factor_panel",
    "winsorize_cross_section",
    "winsorize_mad",
    "winsorize_quantile",
    "zscore",
]
