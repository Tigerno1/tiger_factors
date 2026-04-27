from __future__ import annotations

import pandas as pd

from tiger_factors.factor_preprocessing._core import coerce_factor_panel
from tiger_factors.utils.cross_sectional import cs_minmax_neg
from tiger_factors.utils.cross_sectional import cs_minmax_pos
from tiger_factors.utils.cross_sectional import cs_neutralize
from tiger_factors.utils.cross_sectional import cs_rank
from tiger_factors.utils.cross_sectional import cs_winsorize
from tiger_factors.utils.cross_sectional import cs_winsorize_mad
from tiger_factors.utils.cross_sectional import cs_zscore
from tiger_factors.utils.cross_sectional import neutralize_cross_section


def neutralize_factor_panel(
    values: pd.Series | pd.DataFrame,
    *,
    groups: pd.Series | pd.DataFrame | None = None,
    exposures: pd.Series | pd.DataFrame | list[pd.Series | pd.DataFrame] | None = None,
    method: str = "group_zscore",
    axis: int = 1,
    add_intercept: bool = True,
) -> pd.Series | pd.DataFrame:
    """Neutralize factor values by group or regression exposures."""

    if isinstance(values, pd.Series):
        return neutralize_cross_section(
            values,
            groups=groups if isinstance(groups, pd.Series) else None,
            exposures=exposures,
            method=method,
            axis=axis,
            add_intercept=add_intercept,
        )

    frame = coerce_factor_panel(values)
    if axis != 1:
        frame = frame.T
        if isinstance(groups, pd.DataFrame):
            groups = groups.T
        if isinstance(exposures, pd.DataFrame):
            exposures = exposures.T

    if groups is None and exposures is None:
        raise ValueError("groups or exposures is required for neutralization.")

    def _neutralize_row(row: pd.Series) -> pd.Series:
        row_groups = None
        if isinstance(groups, pd.DataFrame):
            row_groups = groups.loc[row.name]
        elif isinstance(groups, pd.Series):
            row_groups = groups.reindex(row.index)

        row_exposures = exposures
        if isinstance(exposures, pd.DataFrame):
            row_exposures = exposures.loc[row.name]
        elif isinstance(exposures, list):
            prepared = []
            for item in exposures:
                if isinstance(item, pd.DataFrame):
                    prepared.append(item.loc[row.name])
                else:
                    prepared.append(item)
            row_exposures = prepared

        return neutralize_cross_section(
            row,
            groups=row_groups,
            exposures=row_exposures,
            method=method,
            axis=1,
            add_intercept=add_intercept,
        )

    out = frame.apply(_neutralize_row, axis=1)
    return out.T if axis == 0 else out


__all__ = [
    "cs_minmax_neg",
    "cs_minmax_pos",
    "cs_neutralize",
    "cs_rank",
    "cs_winsorize",
    "cs_winsorize_mad",
    "cs_zscore",
    "neutralize_cross_section",
    "neutralize_factor_panel",
]
